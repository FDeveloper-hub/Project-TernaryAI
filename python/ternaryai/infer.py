"""Inference engine: load an exported model folder and generate with a key/value cache.

    import ternaryai as tn

    eng = tn.Engine("my_model")                 # folder with model.ternary + config.json [+ tokenizer.json]
    s = eng.session()                           # one conversation = one session (own cache, own random generator)
    s.feed(ids)                                 # prefill; the logits of the last token are ready afterwards
    s.generate(32, temperature=0.8, top_k=40)   # ids; ``s.last_stop`` tells why it stopped

    eng = model.engine()                        # from a trained :class:`ternaryai.Model` (exports it to a temp folder)

The engine is a module of libternaryai (``include/ternary_infer.h``): C only, one thread, scalar kernel or AVX-512
(VNNI/BW) chosen at run time (``TERNARYAI_NO_AVX512=1`` or :func:`set_engine_backend` force the scalar one).
Logits are bit-identical to ``Model.predict`` for the same ids, and the two kernels give the same bits.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._backend import get_backend
from .errors import ModelFileError, TernaryAIError

__all__ = ["build_engine_lib", "export_fused", "FUSED_API", "use_engine_lib", "Engine", "Session", "Reply", "InferenceError", "set_engine_backend", "engine_backend", "set_engine_threads", "engine_threads", "export_ternam", "CTX_ERROR", "CTX_TRUNCATE", "CTX_SLIDE"]

CTX_ERROR, CTX_TRUNCATE, CTX_SLIDE = 0, 1, 2
_STOP = {0: "none", 1: "max_tokens", 2: "token", 3: "sequence", 4: "callback", 5: "error"}
_ERR = {-1: "invalid argument", -2: "out of memory", -3: "context full (ctx_policy='error')", -4: "unsupported", -5: "the model has no tokenizer.json",
        -6: "I/O error", -7: "no logits yet: feed at least one token first", -8: "token id outside the vocabulary"}
P = C.c_void_p


class InferenceError(TernaryAIError):
    """The inference engine reported an error."""


class _Info(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("vocab", "dim", "seq_len", "n_residual", "n_attn", "n_ffn", "n_heads", "n_kv_heads", "head_dim", "hidden",
                                       "gate_mode", "smear", "window", "sinks", "has_tokenizer")] + [("weight_bytes", C.c_size_t), ("fingerprint", C.c_uint64)]


class _SessionOpts(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("max_context", "attn_window", "sinks", "ctx_policy", "keep_tokens", "record_attn", "max_history")]


class _Sampler(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_k", C.c_int), ("top_p", C.c_float), ("min_p", C.c_float), ("repeat_penalty", C.c_float),
                ("freq_penalty", C.c_float), ("presence_penalty", C.c_float), ("penalty_last_n", C.c_int), ("logit_bias", P), ("allowed", P),
                ("banned", P), ("n_banned", C.c_int)]


_ON_TOKEN = C.CFUNCTYPE(C.c_int, C.c_int32, P)


class _ChatOpts(C.Structure):
    _fields_ = [("after_role", C.c_char_p), ("after_msg", C.c_char_p), ("stop_on_user", C.c_int), ("force_think", C.c_int), ("think_budget", C.c_int),
                ("show_think", C.c_int)]


@dataclass
class Reply:
    """Result of :meth:`Session.reply`. ``text`` is the answer (the reasoning block is removed unless ``show_thinking=True``);
    the separators the model generates around its turn (``after_role`` at the start, ``after_msg`` at the end) are trimmed from ``text`` (``ids`` keeps everything);
    ``thinking`` is always the reasoning text (empty if there was none); ``stop`` is why generation ended (``'token'`` = the model
    closed its turn, ``'max_tokens'``, ...); ``ids`` are the raw generated ids."""
    text: str
    thinking: str
    stop: str
    ids: List[int] = field(default_factory=list)
    stop_id: int = -1                                        # id of the token that ended the turn (e.g. <|end|>), -1 if none
    prompt_ids: List[int] = field(default_factory=list)      # ids that were fed before generating (for building training data)

    def __str__(self) -> str:
        return self.text


class _Gen(C.Structure):
    _fields_ = [("max_new_tokens", C.c_int), ("sampler", _Sampler), ("stop_ids", P), ("n_stop_ids", C.c_int), ("stop_seq", P), ("stop_seq_len", P),
                ("n_stop_seq", C.c_int), ("include_stop", C.c_int), ("on_token", _ON_TOKEN), ("user", P), ("mask_fn", P),
                ("think_open_id", C.c_int), ("think_close_id", C.c_int), ("think_budget", C.c_int), ("think_inside", C.c_int)]


_lib_cache: Dict[int, Any] = {}


_override: Any = None          # libreria de solo motor elegida con use_engine_lib()


def _bind(L: Any) -> None:
    i32, vp, cp, sz = C.c_int, P, C.c_char_p, C.c_size_t
    sig = {
        "model_load": (vp, [cp, cp, sz]), "model_free": (None, [vp]), "model_info": (None, [vp, vp]),
        "session_opts_default": (None, [vp, vp]), "session_new": (vp, [vp, vp]), "session_fork": (vp, [vp]), "session_free": (None, [vp]),
        "session_reset": (None, [vp]), "session_len": (i32, [vp]), "session_next_position": (i32, [vp]), "session_seed": (None, [vp, C.c_uint64]),
        "session_save": (i32, [vp, cp]), "session_load": (vp, [vp, cp, vp, cp, sz]),
        "feed": (i32, [vp, vp, i32]), "feed1": (i32, [vp, C.c_int32]), "argmax": (i32, [vp, vp]), "logits": (i32, [vp, vp]),
        "top_probs": (i32, [vp, i32, vp, vp]), "gen_params_default": (None, [vp]), "generate": (i32, [vp, vp, vp, i32, vp]),
        "generate_batch": (i32, [vp, i32, vp, vp, vp, vp, vp]), "generate_n": (i32, [vp, i32, vp, vp, vp, vp, vp]),
        "score": (i32, [vp, vp, i32, vp, vp]), "attn_probs": (i32, [vp, i32, i32, vp, i32, vp]),
        "encode": (i32, [vp, cp, i32, vp, i32, vp]), "decode": (i32, [vp, vp, i32, i32, vp, i32, vp]),
        "set_backend": (i32, [i32]), "set_threads": (None, [i32]), "get_threads": (i32, []), "export_ternam": (i32, [cp, cp, i32, cp, sz]), "get_backend": (i32, []), "backend_name": (cp, []), "cpu_has_avx512": (i32, []), "cpu_has_neon": (i32, []),
        "prof_enable": (None, [i32]), "prof_reset": (None, []), "prof_report": (i32, [vp, sz]),
        "fused_model": (vp, []), "special_id": (i32, [vp, cp]),
        "chat_opts_default": (None, [vp]), "chat_encode_fmt": (i32, [vp, vp, vp, vp, i32, i32, vp, i32, vp]), "think_split": (i32, [vp, vp, i32, i32, vp, vp, vp, vp]),
        "chat_reply_ex": (i32, [vp, vp, vp, vp, i32, vp, vp, i32, vp, i32, vp]), "export_fused_c": (i32, [cp, cp, i32, cp, sz]),
    }
    n = 0
    for name, (res, args) in sig.items():
        try:
            f = getattr(L, "ternaryai_infer_" + name)
        except AttributeError:
            continue                      # una libreria fusionada solo trae las funciones que se expusieron
        f.restype, f.argtypes = res, args
        n += 1
    if n == 0:
        raise AttributeError("no ternaryai_infer_* function")


class _LibView:
    """Engine-only library: a function that was not exposed gives a clear error instead of an obscure ctypes one."""

    def __init__(self, L: Any) -> None:
        self._L = L

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._L, name)
        except AttributeError:
            raise InferenceError(f"{name.replace('ternaryai_infer_', '')!r} is not exposed by this engine library")


def _lib(base: bool = False):
    """libternaryai (the same one the rest of the package uses), or the engine-only library chosen with :func:`use_engine_lib`
    (unless ``base``), with the engine signatures bound."""
    if _override is not None and not base:
        return _override
    backend = get_backend()
    L = getattr(backend, "_lib", None)
    if L is None:
        raise InferenceError("the inference engine needs the native backend (libternaryai)")
    if id(L) not in _lib_cache:
        _bind(L)
        _lib_cache[id(L)] = L
    return L


_SRC_ENGINE = ["ternary_threadpool.c", "ternary_infer_cache.c", "ternary_infer_engine.c", "ternary_infer_gen.c", "ternary_infer_load.c", "ternary_infer_ops.c",
               "ternary_infer_ternam.c", "ternary_json.c", "ternary_tokenizer.c"]


def _kernel_files(target: str) -> list:
    """Fuentes del nucleo de un objetivo. NEON incluye tambien el escalar: el backend NEON hibrido usa su matvec (ternaryai_infer_k_scalar)."""
    if target == "avx512":
        return ["ternary_infer_k_avx512.c"]
    if target == "neon":
        return ["ternary_infer_k_neon.c", "ternary_infer_k_scalar.c"]
    return ["ternary_infer_k_scalar.c"]


def _host_has(flag: str) -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    return flag in line.split()
    except OSError:
        pass
    return False


def _host_neon() -> bool:
    """True on an ARM host whose CPU declares NEON (``Features: ... neon`` on 32-bit ARM, ``asimd`` on AArch64)."""
    import platform
    if not (platform.machine().lower().startswith(("arm", "aarch64"))):
        return False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Features"):
                    return bool({"neon", "asimd"} & set(line.split()))
    except OSError:
        pass
    return False


def _neon_flags(cc_cmd, cc, cflags) -> list:
    """Flags for the NEON kernel: 32-bit ARM needs ``-mfpu=neon`` unless the caller already chose an FPU / the compiler has NEON by default."""
    import platform
    if any(c.startswith(("-mfpu", "-march=armv8", "-mcpu")) for c in cflags):
        return []
    joined = " ".join(cc_cmd)
    arm32 = ("arm" in joined and "aarch64" not in joined) if cc is not None else platform.machine().lower().startswith("armv")
    return ["-mfpu=neon"] if arm32 else []


def _source_root(src_dir: Any = None) -> Path:
    """Folder with ``src/`` and ``include/``: the one given, the project tree (running from source), or the copy shipped inside the
    installed package (``ternaryai/_csrc``)."""
    if src_dir:
        return Path(src_dir)
    here = Path(__file__).resolve()
    for cand in (here.parents[2], here.parent / "_csrc"):
        if (cand / "src" / "ternary_infer_engine.c").exists():
            return cand
    return here.parents[2]


def build_engine_lib(out: Any = None, target: str = "auto", cc: Optional[str] = None, vnni: Optional[bool] = None,
                     cflags: Sequence[str] = (), src_dir: Optional[Any] = None) -> str:
    """Compile an **engine-only** shared library (``.so`` / ``.dll`` / ``.dylib``) for one vectorization, right now, for the target platform.

    ``target`` decides the vectorization and only that one is compiled in (never both, no run-time choice):
    ``'avx512'`` (AVX-512; VNNI if ``vnni`` is true, BW otherwise; ``vnni=None`` looks at this CPU), ``'neon'`` (ARM 128-bit NEON; on 32-bit ARM
    ``-mfpu=neon`` is added unless ``cflags`` chooses an FPU), ``'scalar'`` (portable C) or ``'auto'``
    (AVX-512 if this CPU has it, NEON on an ARM CPU that declares it, scalar if not). The library has the flat API of ``include/ternary_infer.h`` (``ternaryai_infer_*``) and
    nothing of training; it opens a model folder or a ``.ternam``. ``cc`` is the compiler command (default ``$CC`` or ``gcc``; a cross compiler
    such as ``aarch64-linux-gnu-gcc`` or ``clang --target=aarch64-linux-android21`` works the same); ``cflags`` are extra flags (e.g. ``-march=...``).
    Needs the library sources (``src/`` and ``include/``). Returns the path of the library.
    :func:`use_engine_lib` makes :class:`Engine` use it."""
    import shlex, subprocess, sys, tempfile
    if target not in ("auto", "scalar", "avx512", "neon"):
        raise ValueError("target must be 'auto', 'scalar', 'avx512' or 'neon'")
    root = _source_root(src_dir)
    srcs = root / "src"
    if not (srcs / "ternary_infer_engine.c").exists() or not (root / "include" / "ternary_infer.h").exists():
        raise InferenceError(f"build_engine_lib needs the library sources (src/ and include/) in {root}; pass src_dir=")
    cc_cmd = shlex.split(cc or os.environ.get("CC") or "gcc")
    if target == "auto":
        target = ("avx512" if (cc is None and _host_has("avx512f") and _host_has("avx512bw") and _host_has("avx512vl") and _host_has("avx512dq"))
                  else "neon" if (cc is None and _host_neon()) else "scalar")
    windows = any(t in " ".join(cc_cmd) for t in ("mingw", "windows", "w64")) or (cc is None and os.name == "nt")
    mac = any(t in " ".join(cc_cmd) for t in ("apple", "darwin")) or (cc is None and sys.platform == "darwin")
    ext = ".dll" if windows else (".dylib" if mac else ".so")
    out = Path(out) if out is not None else Path(f"libternaryai_engine_{target}{ext}")
    files = list(_SRC_ENGINE) + _kernel_files(target)
    flags = ["-std=c11", "-O3", "-I" + str(root / "include"), "-shared", "-fPIC"]
    if target == "avx512":
        use_vnni = _host_has("avx512_vnni") if vnni is None else vnni
        flags += ["-DTERNARYAI_FIXED_AVX512", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"] + (["-mavx512vnni"] if use_vnni else [])
    elif target == "neon":
        flags += ["-DTERNARYAI_FIXED_NEON", "-ffp-contract=off"] + _neon_flags(cc_cmd, cc, cflags)
    else:
        flags += ["-DTERNARYAI_FIXED_SCALAR"]
    flags += list(cflags)
    with tempfile.TemporaryDirectory() as td:
        link = []
        if not windows and not mac:
            vs = Path(td) / "export.map"
            vs.write_text("{ global: ternaryai_infer_*; ternary_tokenizer_*; local: *; };\n")
            link.append("-Wl," + "--version-script=" + str(vs))
        cmd = cc_cmd + flags + [str(srcs / f) for f in files] + ["-o", str(out)] + link + ([] if windows else ["-lm", "-lpthread"])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            raise InferenceError(f"cannot run the compiler {cc_cmd[0]!r}: {e}")
    if r.returncode != 0:
        raise InferenceError("compiling the engine failed:\n" + (r.stderr or r.stdout)[-2000:])
    return os.path.abspath(os.fspath(out))


_FUSED_API = ("set_backend", "get_backend", "set_threads", "get_threads", "backend_name", "cpu_has_avx512", "cpu_has_neon", "model_info",
              "session_opts_default", "session_new", "session_fork", "session_free", "session_reset", "session_len", "session_next_position",
              "session_history", "session_model", "session_seed", "session_save", "session_load", "feed", "feed1", "argmax", "logits", "top_probs",
              "sampler_default", "sample", "gen_params_default", "generate", "generate_batch", "generate_n", "score", "attn_probs",
              "encode", "decode", "special_id", "chat_encode", "chat_reply", "chat_opts_default", "chat_encode_fmt", "think_split", "chat_reply_ex", "prof_enable", "prof_reset", "prof_report", "now_ns")
_FUSED_NEEDS_TOKENIZER = ("encode", "decode", "special_id", "chat_encode", "chat_reply", "chat_encode_fmt", "think_split", "chat_reply_ex")


def export_fused(src: Any, out: Any, expose: Sequence[str], target: str = "auto", cc: Optional[str] = None, vnni: Optional[bool] = None,
                 cflags: Sequence[str] = (), strip: bool = True, src_dir: Optional[Any] = None) -> str:
    """Build a **fused** library: this one model embedded in a minimal engine, for one vectorization, compiled right now for the target platform.

    ``src`` is a model folder or a ``.ternam``. ``expose`` is the list of API functions the library exports, chosen by hand (short names such as
    ``'session_new'``, ``'feed'``, ``'generate'``; see ``export_fused.__doc__`` / ``FUSED_API``); nothing else is exported and the linker drops the rest.
    There is no loading of other models: ``model_load`` and ``export_ternam`` do not exist in the library. The model is reached with
    ``ternaryai_infer_fused_model()`` (always exported) and lives in read-only memory inside the library (no file, no copy of the weights).
    If ``encode``/``decode``/``special_id``/``chat_*`` are not exposed the tokenizer is not embedded either. ``target``, ``cc``, ``vnni`` and
    ``cflags`` are as in :func:`build_engine_lib` (one vectorization only). Returns the path of the library; :func:`use_engine_lib` +
    :meth:`Engine.fused` use it from Python (only the exposed calls work)."""
    import shlex, subprocess, sys, tempfile
    if target not in ("auto", "scalar", "avx512", "neon"):
        raise ValueError("target must be 'auto', 'scalar', 'avx512' or 'neon'")
    names = list(dict.fromkeys(expose))
    bad = [n for n in names if n not in _FUSED_API]
    if bad:
        raise ValueError(f"cannot expose {bad}: valid names are {list(_FUSED_API)} (model_load, model_free and export_ternam do not exist in a fused library)")
    root = _source_root(src_dir)
    srcs = root / "src"
    if not (srcs / "ternary_infer_engine.c").exists() or not (root / "include" / "ternary_infer.h").exists():
        raise InferenceError(f"export_fused needs the library sources (src/ and include/) in {root}; pass src_dir=")
    cc_cmd = shlex.split(cc or os.environ.get("CC") or "gcc")
    if target == "auto":
        target = ("avx512" if (cc is None and _host_has("avx512f") and _host_has("avx512bw") and _host_has("avx512vl") and _host_has("avx512dq"))
                  else "neon" if (cc is None and _host_neon()) else "scalar")
    windows = any(t in " ".join(cc_cmd) for t in ("mingw", "windows", "w64")) or (cc is None and os.name == "nt")
    mac = any(t in " ".join(cc_cmd) for t in ("apple", "darwin")) or (cc is None and sys.platform == "darwin")
    ext = ".dll" if windows else (".dylib" if mac else ".so")
    out = Path(out) if out is not None else Path(f"libternaryai_fused_{target}{ext}")
    need_tok = any(n in _FUSED_NEEDS_TOKENIZER for n in names)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        tn_path = tdp / "model.ternam"
        # el modelo incrustado debe estar en el orden de pesos del nucleo de la libreria: el backend NEON hibrido usa el matvec escalar (orden fila a fila)
        export_ternam(src, tn_path, "scalar" if target == "neon" else target, tokenizer=need_tok)
        blob = tn_path.read_bytes()
        lines = [",".join(str(b) for b in blob[i:i + 32]) for i in range(0, len(blob), 32)]
        (tdp / "fused_blob.c").write_text(
            "#include <stddef.h>\nconst unsigned char ternaryai_fused_blob[%d] __attribute__((aligned(64))) = {\n%s\n};\n"
            "const size_t ternaryai_fused_blob_len = %d;\n" % (len(blob), ",\n".join(lines), len(blob)))
        # estructura del modelo como C estatico: la fusionada no interpreta config.json ni el indice del .ternam
        err = C.create_string_buffer(256)
        rc = _lib(base=True).ternaryai_infer_export_fused_c(os.fsencode(str(tn_path)), os.fsencode(str(tdp / "fused_model.c")), 1 if need_tok else 0, err, 256)
        if rc < 0:
            raise InferenceError("generating the fused model failed: " + (err.value.decode() or _ERR.get(rc, str(rc))))
        # sin .ternam (lector/escritor) ni JSON; el tokenizador (que si usa JSON) solo si se expone encode/decode
        engine_srcs = [f for f in _SRC_ENGINE if f != "ternary_infer_ternam.c" and (need_tok or f not in ("ternary_json.c", "ternary_tokenizer.c"))]
        files = [str(srcs / f) for f in engine_srcs] + [str(srcs / f) for f in _kernel_files(target)] + [str(tdp / "fused_blob.c"), str(tdp / "fused_model.c")]
        flags = ["-std=c11", "-O3", "-I" + str(root / "include"), "-I" + str(srcs), "-shared", "-fPIC", "-ffunction-sections", "-fdata-sections", "-DTERNARYAI_FUSED"]
        if target == "avx512":
            use_vnni = _host_has("avx512_vnni") if vnni is None else vnni
            flags += ["-DTERNARYAI_FIXED_AVX512", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"] + (["-mavx512vnni"] if use_vnni else [])
        elif target == "neon":
            flags += ["-DTERNARYAI_FIXED_NEON", "-ffp-contract=off"] + _neon_flags(cc_cmd, cc, cflags)
        else:
            flags += ["-DTERNARYAI_FIXED_SCALAR"]
        if not need_tok:
            flags += ["-DTERNARYAI_FUSED_NO_TOKENIZER"]
        link = []
        if not windows and not mac:
            vs = tdp / "export.map"
            vs.write_text("{ global: " + " ".join("ternaryai_infer_" + n + ";" for n in ["fused_model"] + names) + " local: *; };\n")
            link += ["-Wl,--version-script=" + str(vs), "-Wl,--gc-sections"] + (["-s"] if strip else [])
        elif mac:
            link += ["-Wl,-dead_strip"] + (["-Wl,-x"] if strip else [])
        else:
            link += ["-Wl,--gc-sections"] + (["-s"] if strip else [])
        try:
            r = subprocess.run(cc_cmd + flags + list(cflags) + files + ["-o", str(out)] + link + ([] if windows else ["-lm", "-lpthread"]), capture_output=True, text=True)
        except OSError as e:
            raise InferenceError(f"cannot run the compiler {cc_cmd[0]!r}: {e}")
    if r.returncode != 0:
        raise InferenceError("compiling the fused library failed:\n" + (r.stderr or r.stdout)[-2000:])
    return os.path.abspath(os.fspath(out))


FUSED_API = _FUSED_API


def use_engine_lib(path: Optional[Any] = None) -> str:
    """Make :class:`Engine` and sessions run on an engine-only library built with :func:`build_engine_lib` (or ``None`` to go back to
    libternaryai). Do it before creating engines; models already loaded keep the library they came from. The system loads a library file once per path: build a new library to a new file name instead of overwriting one already loaded. Returns the vectorization name. A relative ``path`` is resolved against the current directory."""
    global _override
    if path is None:
        _override = None
        return engine_backend()
    L = C.CDLL(os.path.abspath(os.fspath(path)))     # ruta absoluta: dlopen no busca 'lib.so' en la carpeta actual
    try:
        _bind(L)
    except AttributeError as e:
        raise InferenceError(f"{path!r} is not an engine library: {e}")
    _override = _LibView(L)
    try:
        return _override.ternaryai_infer_backend_name().decode()
    except InferenceError:
        return "(backend_name not exposed)"


def _chk(rc: int, what: str) -> int:
    if rc < 0:
        raise InferenceError(f"{what}: {_ERR.get(rc, 'error %d' % rc)}")
    return rc


def set_engine_backend(name: str = "auto") -> str:
    """Choose the engine kernel: ``'auto'``, ``'scalar'``, ``'avx512'`` or ``'neon'`` (ARM). Returns the active kernel's name.

    Call it before creating sessions."""
    L = _lib()
    _chk(L.ternaryai_infer_set_backend({"auto": 0, "scalar": 1, "avx512": 2, "neon": 3}[name]), "set_engine_backend")
    return L.ternaryai_infer_backend_name().decode()


def set_engine_threads(n: int) -> None:
    """Number of threads the engine uses to split each matrix product (and the output head) by blocks of rows. Chosen by hand; 0 or 1 = a single
    thread. The result is the same bit for bit with any number (integer arithmetic). In libternaryai this is the same setting as
    :func:`ternaryai.set_threads`; an engine-only or fused library has its own. Call it with no generation in progress."""
    _lib().ternaryai_infer_set_threads(int(n))


def engine_threads() -> int:
    """Threads set with :func:`set_engine_threads` (1 = a single thread)."""
    return int(_lib().ternaryai_infer_get_threads())


def engine_backend() -> str:
    """Name of the active kernel: ``'scalar'``, ``'avx512-vnni'``, ``'avx512-bw'`` or ``'neon'``."""
    return _lib().ternaryai_infer_backend_name().decode()


_LAYOUTS = {"auto": -1, "scalar": 0, "avx512": 1, "neon": 2}


def export_ternam(src: Any, out: Any, target: str = "auto", tokenizer: bool = True) -> str:
    """Write a self-contained model file (``.ternam``) from a model folder or another ``.ternam``.

    The file holds the configuration, the tokenizer (if any) and the weights already packed to 2 bits in the order of one kernel family:
    ``target`` is ``'scalar'``, ``'avx512'``, ``'neon'`` or ``'auto'`` (the active kernel). :class:`Engine` opens it like a folder, memory-mapped; a file
    in the other order still works (the weights are reordered in memory on first use). Returns ``out``."""
    L = _lib(base=True)                       # exportar siempre con libternaryai, aunque haya una libreria de motor activa
    if target not in _LAYOUTS:
        raise ValueError(f"target must be one of {sorted(_LAYOUTS)}")
    err = C.create_string_buffer(256)
    rc = L.ternaryai_infer_export_ternam(os.fsencode(os.fspath(src)), os.fsencode(os.fspath(out)), _LAYOUTS[target] | (0 if tokenizer or _LAYOUTS[target] < 0 else 0x100), err, 256)
    if rc < 0:
        raise ModelFileError(err.value.decode() or f"cannot export {src!r}: {_ERR.get(rc, rc)}")
    return os.fspath(out)


class Engine:
    """A model loaded for inference (read-only weights; share it among sessions).

    ``path`` is a model folder or a self-contained ``.ternam`` file (see :func:`export_ternam`)."""

    def __init__(self, path: Any) -> None:
        L = _lib()
        err = C.create_string_buffer(256)
        self._h = L.ternaryai_infer_model_load(os.fsencode(os.fspath(path)), err, 256)
        if not self._h:
            raise ModelFileError(err.value.decode() or f"cannot load {path!r}")
        self._info = _Info()
        L.ternaryai_infer_model_info(self._h, C.byref(self._info))
        self._tmp: Optional[str] = None

    @classmethod
    def fused(cls) -> "Engine":
        """The model built into an engine library made with :func:`export_fused` (activate it first with :func:`use_engine_lib`)."""
        L = _lib()
        h = L.ternaryai_infer_fused_model()
        if not h:
            raise ModelFileError("this library has no usable embedded model (is the CPU supported?)")
        self = cls.__new__(cls)
        self._h, self._tmp, self._fused = h, None, True
        self._info = _Info()
        try:
            L.ternaryai_infer_model_info(self._h, C.byref(self._info))
        except InferenceError:
            pass                           # model_info no expuesta: los datos del modelo quedan en cero
        return self

    # -- facts --------------------------------------------------------------
    @property
    def vocab(self) -> int:
        if self._info.vocab <= 0:       # fusionada sin model_info: los buffers de salida no tendrian tamano
            raise InferenceError("'model_info' is not exposed by this engine library: expose it to use logits or generation")
        return self._info.vocab
    @property
    def seq_len(self) -> int: return self._info.seq_len
    @property
    def dim(self) -> int: return self._info.dim
    @property
    def has_tokenizer(self) -> bool: return bool(self._info.has_tokenizer)
    @property
    def fingerprint(self) -> int: return self._info.fingerprint
    @property
    def weight_bytes(self) -> int: return self._info.weight_bytes

    def session(self, **opts: int) -> "Session":
        """New conversation. Options: ``max_context``, ``attn_window``, ``sinks``, ``ctx_policy`` (``CTX_ERROR`` | ``CTX_TRUNCATE`` |
        ``CTX_SLIDE``), ``keep_tokens``, ``record_attn``, ``max_history``."""
        return Session(self, **opts)

    def load_session(self, path: Any, **opts: int) -> "Session":
        """Restore a session saved with :meth:`Session.save` (for example a processed system prompt)."""
        err = C.create_string_buffer(256)
        o = None
        if opts:
            o = _SessionOpts(); _lib().ternaryai_infer_session_opts_default(self._h, C.byref(o))
            for k, v in opts.items(): setattr(o, k, v)
        h = _lib().ternaryai_infer_session_load(self._h, os.fsencode(os.fspath(path)), C.byref(o) if o else None, err, 256)
        if not h:
            raise InferenceError(err.value.decode() or "cannot load the session")
        return Session(self, _h=h)

    # -- text (needs tokenizer.json in the model folder) ---------------------
    def encode(self, text: str, parse_special: bool = True) -> List[int]:
        b = text.encode(); cap = len(b) + 8; a = (C.c_int32 * cap)(); n = C.c_int()
        _chk(_lib().ternaryai_infer_encode(self._h, b, int(parse_special), a, cap, C.byref(n)), "encode")
        return list(a[:n.value])

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        a = (C.c_int32 * max(1, len(ids)))(*ids); cap = 16 * len(ids) + 64; out = C.create_string_buffer(cap); n = C.c_int()
        _chk(_lib().ternaryai_infer_decode(self._h, a, len(ids), int(skip_special), out, cap, C.byref(n)), "decode")
        return out.raw[:n.value].decode("utf-8", "replace")

    def special_id(self, token: str) -> int:
        """Id of a special token such as ``'<|end|>'``, or -1 if the tokenizer does not have it."""
        return _lib().ternaryai_infer_special_id(self._h, token.encode())

    @staticmethod
    def _messages(messages: Any) -> List[Tuple[str, str]]:
        """A string (= one user message), or a list of ``{"role", "content"}`` dicts or ``(role, content)`` pairs."""
        if isinstance(messages, str):
            return [("user", messages)]
        out = []
        for m in messages:
            if isinstance(m, dict):
                out.append((str(m["role"]), str(m["content"])))
            else:
                r, c = m
                out.append((str(r), str(c)))
        return out

    @staticmethod
    def _chat_opts(after_role: str, after_msg: str, stop_on_user: bool, force_think: bool, think_budget: int, show_thinking: bool) -> _ChatOpts:
        o = _ChatOpts()
        o.after_role, o.after_msg = after_role.encode(), after_msg.encode()
        o.stop_on_user, o.force_think, o.think_budget, o.show_think = int(stop_on_user), int(force_think), int(think_budget), int(show_thinking)
        return o

    def chat_encode(self, messages: Any, add_assistant_prompt: bool = True, after_role: str = "\n", after_msg: str = "\n", force_think: bool = False) -> List[int]:
        """Conversation -> ids with the same layout :func:`ternaryai.dataset.load` renders for training (``<|role|>\\ntext\\n``). Each message is encoded as one piece
        (BPE is not independent per piece, so this is what makes the ids equal to the training ones) and the prompt for the answer is just
        ``<|assistant|>``: the model generates the separator itself, as it was trained to. ``after_role`` / ``after_msg`` change the separators
        (use ``""`` for a model trained without them); ``force_think`` appends ``after_role`` and ``<|think|>`` (the model starts thinking)."""
        msgs = self._messages(messages)
        rs = (C.c_char_p * max(1, len(msgs)))(*[r.encode() for r, _ in msgs]); ts = (C.c_char_p * max(1, len(msgs)))(*[t.encode() for _, t in msgs])
        o = self._chat_opts(after_role, after_msg, True, force_think, 0, False)
        n = C.c_int(0)
        L = _lib()
        rc = L.ternaryai_infer_chat_encode_fmt(self._h, C.byref(o), rs, ts, len(msgs), int(add_assistant_prompt), None, 0, C.byref(n))
        if rc not in (0, -2):
            _chk(rc, "chat_encode")
        cap = n.value; a = (C.c_int32 * max(1, cap))()
        _chk(L.ternaryai_infer_chat_encode_fmt(self._h, C.byref(o), rs, ts, len(msgs), int(add_assistant_prompt), a, cap, C.byref(n)), "chat_encode")
        return list(a[:n.value])

    def _chat_prompt(self, messages: Any, force_think: bool, after_role: str, after_msg: str, stop_on_user: bool, extra_stops: Iterable[int]) -> Tuple[List[int], List[int]]:
        ids = self.chat_encode(messages, True, after_role, after_msg, force_think)
        stops = [t for t in (self.special_id("<|end|>"), self.special_id("<|user|>") if stop_on_user else -1) if t >= 0] + list(extra_stops)
        return ids, stops

    def _make_reply(self, gen: List[int], why: str, prompt_ids: List[int], stops: Sequence[int], think_inside: bool, show_thinking: bool,
                    session: Optional["Session"] = None, after_role: str = "\n", after_msg: str = "\n") -> Reply:
        stop_id = -1
        if why == "token" and gen and gen[-1] in stops:      # generation ran with include_stop=True: the stop token is the last id
            stop_id = gen[-1]; gen = gen[:-1]
        answer, thinking = (gen, []) if self.special_id("<|think|>") < 0 else self.split_thinking(gen, think_inside)
        text = self.decode(gen, skip_special=False) if show_thinking else self.decode(answer, skip_special=True)
        if after_role and text.startswith(after_role): text = text[len(after_role):]     # the model generates the separators; they are not part of the answer
        if after_msg and text.endswith(after_msg): text = text[:-len(after_msg)]
        if session is not None: session.last_stop = why
        return Reply(text=text, thinking=self.decode(thinking, skip_special=True) if thinking else "", stop=why, ids=list(gen), stop_id=stop_id, prompt_ids=list(prompt_ids))

    def sample(self, messages: Any, n: int = 1, max_new_tokens: int = 256, *, seed: int = 0, show_thinking: bool = False, think_budget: int = 0,
               force_think: bool = False, after_role: str = "\n", after_msg: str = "\n", stop_on_user: bool = True, **sampling: Any) -> List[Reply]:
        """``n`` independent answers to the same conversation (the prompt is processed once and the session forked; candidates run in one batch,
        so they share the weight reads and the thread pool). Candidate ``i`` uses the random seed ``seed + i``: same arguments, same answers.
        Same options as :meth:`Session.reply`; use ``temperature > 0`` to get different candidates."""
        if n < 1: raise ValueError("n must be >= 1")
        ids, stops = self._chat_prompt(messages, force_think, after_role, after_msg, stop_on_user, sampling.pop("stop_ids", ()))
        base = self.session(); base.feed(ids)
        ss = []
        for i in range(n):
            c = base.fork(); c.seed(seed + i); ss.append(c)
        outs = self.generate_batch(ss, max_new_tokens, stop_ids=stops, think_budget=think_budget, think_inside=force_think, include_stop=True, **sampling)
        res = [self._make_reply(g, c.last_stop, ids, stops, force_think, show_thinking, None, after_role, after_msg) for g, c in zip(outs, ss)]
        for c in ss: c.close()
        base.close()
        return res

    def split_thinking(self, ids: Sequence[int], think_inside: bool = False) -> Tuple[List[int], List[int]]:
        """Split generated ids into ``(answer_ids, thinking_ids)``: what is between ``<|think|>`` and ``<|/think|>`` (tags removed; several blocks are
        joined) is the thinking. An unclosed block runs to the end."""
        n = len(ids); a = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
        ans = np.empty(max(1, n), dtype=np.int32); thk = np.empty(max(1, n), dtype=np.int32); na, nt = C.c_int(), C.c_int()
        _chk(_lib().ternaryai_infer_think_split(self._h, a.ctypes.data if n else None, n, int(think_inside), ans.ctypes.data, C.byref(na), thk.ctypes.data, C.byref(nt)), "split_thinking")
        return list(ans[:na.value]), list(thk[:nt.value])

    # -- batches ---------------------------------------------------------------
    def generate_batch(self, sessions: Sequence["Session"], max_new_tokens: int = 64, **sampling: Any) -> List[List[int]]:
        """Generate for several sessions in rounds (one step each per round). Each must have had ``feed`` called."""
        keep: list = []
        g = _make_gen(self, max_new_tokens, keep, **sampling)
        n = len(sessions)
        hs = (P * n)(*[s._h for s in sessions]); bufs = [(C.c_int32 * max_new_tokens)() for _ in range(n)]
        outs = (P * n)(*[C.cast(b, P) for b in bufs]); caps = (C.c_int * n)(*([max_new_tokens] * n)); cnt = (C.c_int * n)(); why = (C.c_int * n)()
        _chk(_lib().ternaryai_infer_generate_batch(hs, n, C.byref(g), outs, caps, cnt, why), "generate_batch")
        for s, w in zip(sessions, why): s.last_stop = _STOP[w]
        return [list(bufs[i][:cnt[i]]) for i in range(n)]

    def close(self) -> None:
        if getattr(self, "_h", None):
            if not getattr(self, "_fused", False):        # el modelo incrustado vive lo que la libreria
                _lib().ternaryai_infer_model_free(self._h)
            self._h = None
        if getattr(self, "_tmp", None):
            shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None

    def __enter__(self) -> "Engine": return self
    def __exit__(self, *exc: Any) -> None: self.close()

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass


def _make_gen(engine: Engine, max_new_tokens: int, keep: list, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
              repeat_penalty: float = 1.0, freq_penalty: float = 0.0, presence_penalty: float = 0.0, penalty_last_n: int = 0,
              logit_bias: Optional[Dict[int, float]] = None, banned: Optional[Iterable[int]] = None, stop_ids: Iterable[int] = (),
              stop_seqs: Iterable[Sequence[int]] = (), include_stop: bool = False, on_token: Optional[Callable[[int], Any]] = None,
              think_budget: int = 0, think_inside: bool = False) -> _Gen:
    g = _Gen(); _lib().ternaryai_infer_gen_params_default(C.byref(g))
    g.max_new_tokens = max_new_tokens; s = g.sampler
    s.temperature, s.top_k, s.top_p, s.min_p, s.repeat_penalty = temperature, top_k, top_p, min_p, repeat_penalty
    s.freq_penalty, s.presence_penalty, s.penalty_last_n = freq_penalty, presence_penalty, penalty_last_n
    if logit_bias:
        lb = np.zeros(engine.vocab, dtype=np.float32)
        for k, v in dict(logit_bias).items(): lb[k] = v
        keep.append(lb); s.logit_bias = lb.ctypes.data
    banned = list(banned or ())
    if banned:
        bn = np.asarray(banned, dtype=np.int32); keep.append(bn); s.banned = bn.ctypes.data; s.n_banned = len(bn)
    stop_ids = list(stop_ids)
    if stop_ids:
        si = np.asarray(stop_ids, dtype=np.int32); keep.append(si); g.stop_ids = si.ctypes.data; g.n_stop_ids = len(si)
    seqs = [list(q) for q in stop_seqs]
    if seqs:
        flat = np.asarray([t for q in seqs for t in q], dtype=np.int32); lens = np.asarray([len(q) for q in seqs], dtype=np.int32)
        keep += [flat, lens]; g.stop_seq = flat.ctypes.data; g.stop_seq_len = lens.ctypes.data; g.n_stop_seq = len(lens)
    g.include_stop = int(include_stop)
    if think_budget > 0:
        op, cl = engine.special_id("<|think|>"), engine.special_id("<|/think|>")
        if op < 0 or cl < 0:
            raise InferenceError("think_budget needs the special tokens '<|think|>' and '<|/think|>' in the tokenizer")
        g.think_open_id, g.think_close_id, g.think_budget, g.think_inside = op, cl, int(think_budget), int(think_inside)
    if on_token is not None:
        cb = _ON_TOKEN(lambda i, u: 1 if on_token(i) else 0); keep.append(cb); g.on_token = cb
    return g


class Session:
    """One conversation: key/value cache, token history and random generator."""

    def __init__(self, engine: Engine, _h: Any = None, **opts: int) -> None:
        self.engine = engine; self.last_stop = "none"
        L = _lib()
        if _h is None:
            o = _SessionOpts(); L.ternaryai_infer_session_opts_default(engine._h, C.byref(o))
            for k, v in opts.items():
                if not hasattr(o, k): raise TypeError(f"unknown session option {k!r}")
                setattr(o, k, int(v))
            _h = L.ternaryai_infer_session_new(engine._h, C.byref(o))
            if not _h: raise InferenceError("cannot create the session (out of memory?)")
        self._h = _h

    def __len__(self) -> int: return _lib().ternaryai_infer_session_len(self._h)
    @property
    def next_position(self) -> int: return _lib().ternaryai_infer_session_next_position(self._h)
    def seed(self, value: int) -> None: _lib().ternaryai_infer_session_seed(self._h, value)
    def reset(self) -> None: _lib().ternaryai_infer_session_reset(self._h)
    def fork(self) -> "Session":
        h = _lib().ternaryai_infer_session_fork(self._h)
        if not h: raise InferenceError("cannot fork the session (out of memory?)")
        return Session(self.engine, _h=h)

    def feed(self, ids: Any) -> "Session":
        a = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
        _chk(_lib().ternaryai_infer_feed(self._h, a.ctypes.data, len(a)), "feed")
        return self

    def argmax(self) -> int:
        t = C.c_int32(); _chk(_lib().ternaryai_infer_argmax(self._h, C.byref(t)), "argmax"); return t.value

    def logits(self) -> np.ndarray:
        out = np.empty(self.engine.vocab, dtype=np.float32); _chk(_lib().ternaryai_infer_logits(self._h, out.ctypes.data), "logits"); return out

    def top_probs(self, n: int = 5) -> List[Tuple[int, float]]:
        i = (C.c_int32 * n)(); p = (C.c_float * n)(); k = _chk(_lib().ternaryai_infer_top_probs(self._h, n, i, p), "top_probs")
        return list(zip(i[:k], p[:k]))

    def score(self, ids: Any) -> Tuple[np.ndarray, float]:
        """Process ``ids`` and return ``(log p(ids[i] | context), perplexity)``; the first log-probability is NaN on an empty session."""
        a = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1); lp = np.empty(len(a), dtype=np.float32); ppl = C.c_double()
        _chk(_lib().ternaryai_infer_score(self._h, a.ctypes.data, len(a), lp.ctypes.data, C.byref(ppl)), "score")
        return lp, ppl.value

    def attention_probs(self, layer: int, head: int) -> np.ndarray:
        """Attention probabilities of the last token (the session needs ``record_attn=1``)."""
        out = (C.c_float * (self.engine.seq_len * 4 + 64))(); n = C.c_int()
        _chk(_lib().ternaryai_infer_attn_probs(self._h, layer, head, out, len(out), C.byref(n)), "attention_probs")
        return np.array(out[:n.value], dtype=np.float32)

    def save(self, path: Any) -> None:
        _chk(_lib().ternaryai_infer_session_save(self._h, os.fsencode(os.fspath(path))), "save")

    def generate(self, max_new_tokens: int = 64, **sampling: Any) -> List[int]:
        """Continue from the current state. Sampling: ``temperature`` (0 = greedy), ``top_k``, ``top_p``, ``min_p``, ``repeat_penalty``,
        ``freq_penalty``, ``presence_penalty``, ``penalty_last_n``, ``logit_bias`` ({id: bias}), ``banned``; stopping: ``stop_ids``,
        ``stop_seqs``, ``include_stop``, ``on_token`` (called per token; return True to stop); reasoning: ``think_budget`` (maximum tokens
        between ``<|think|>`` and ``<|/think|>``: when reached ``<|/think|>`` is forced) and ``think_inside`` (the context already ends inside the block)."""
        keep: list = []
        g = _make_gen(self.engine, max_new_tokens, keep, **sampling)
        out = (C.c_int32 * max(1, max_new_tokens))(); why = C.c_int()
        n = _chk(_lib().ternaryai_infer_generate(self._h, C.byref(g), out, max_new_tokens, C.byref(why)), "generate")
        self.last_stop = _STOP[why.value]
        return list(out[:n])

    def reply(self, messages: Any, max_new_tokens: int = 256, *, show_thinking: bool = False, think_budget: int = 0, force_think: bool = False,
              after_role: str = "\n", after_msg: str = "\n", stop_on_user: bool = True, **sampling: Any) -> Reply:
        """The assistant's answer to a conversation (the session is reset first).

        ``messages``: a string (one user message) or a list of ``{"role", "content"}`` dicts / ``(role, content)`` pairs, rendered like the
        training data (``<|role|>\\ntext\\n``; see :meth:`Engine.chat_encode`). Generation stops at ``<|end|>`` (and at ``<|user|>`` if ``stop_on_user``).

        Reasoning: the block between ``<|think|>`` and ``<|/think|>`` is returned in ``Reply.thinking`` and removed from ``Reply.text``;
        ``show_thinking=True`` keeps it inline in ``text`` (tags included) so the chain of thought can be read in place. ``force_think`` starts
        the answer already inside ``<|think|>``; ``think_budget`` limits the reasoning tokens (``<|/think|>`` is forced when reached).
        The rest of the keywords are sampling options (see :meth:`generate`)."""
        eng = self.engine
        ids, stops = eng._chat_prompt(messages, force_think, after_role, after_msg, stop_on_user, sampling.pop("stop_ids", ()))
        self.reset(); self.feed(ids)
        gen = self.generate(max_new_tokens, stop_ids=stops, think_budget=think_budget, think_inside=force_think, include_stop=True, **sampling)
        return eng._make_reply(gen, self.last_stop, ids, stops, force_think, show_thinking, self, after_role, after_msg)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib().ternaryai_infer_session_free(self._h); self._h = None

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass
