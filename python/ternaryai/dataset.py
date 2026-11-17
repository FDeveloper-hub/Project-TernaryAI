"""Dataset: fachada delgada en ctypes sobre libternaryai.

Toda la logica de parseo (texto/JSON/JSONL, renderizado de turnos
conversacionales) vive en C (src/ternary_dataset.c + src/ternary_json.c,
expuesta como ternaryai_dataset_load en include/ternaryai.h). Este
modulo solo detecta el formato por extension, empaqueta argumentos y
desempaqueta el resultado -- no reimplementa el parseo.
"""
from __future__ import annotations

import ctypes
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._native import find_library
from .errors import BackendNotFoundError, DataError, TernaryAIError

_ERR_CAP = 512

# Deben coincidir exactamente con el enum TERNARYAI_DATASET_* de ternaryai.h
TEXT_LINES = 0
TEXT_WHOLE = 1
JSON = 2
JSONL = 3

_EXT_TO_FORMAT = {".txt": TEXT_LINES, ".json": JSON, ".jsonl": JSONL}


class DatasetError(TernaryAIError):
    """Algo fallo cargando un dataset (archivo, formato o esquema invalido)."""


def _load_lib() -> ctypes.CDLL:
    path = find_library()
    if path is None:
        raise BackendNotFoundError(
            "libternaryai no se encontro. Compilala (`make libternaryai.so`) y ponela "
            "junto al paquete ternaryai, o apunta $TERNARYAI_LIB a ella.")
    lib = ctypes.CDLL(path)
    c = ctypes
    lib.ternaryai_dataset_targets.restype = c.c_int
    lib.ternaryai_dataset_targets.argtypes = [
        c.POINTER(c.c_int32), c.POINTER(c.c_size_t), c.c_size_t, c.c_int32, c.c_int32, c.POINTER(c.c_int8), c.c_int,
        c.POINTER(c.c_int32), c.c_char_p, c.c_size_t,
    ]
    lib.ternaryai_dataset_load_ex.restype = c.c_int
    lib.ternaryai_dataset_load_ex.argtypes = [
        c.c_char_p, c.c_int, c.c_int, c.c_int,
        c.POINTER(c.c_uint8), c.c_size_t, c.POINTER(c.c_size_t), c.POINTER(c.c_size_t),
        c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t,
    ]
    lib.ternaryai_dataset_load.restype = c.c_int
    lib.ternaryai_dataset_load.argtypes = [
        c.c_char_p, c.c_int, c.c_int,
        c.POINTER(c.c_uint8), c.c_size_t, c.POINTER(c.c_size_t), c.POINTER(c.c_size_t),
        c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t,
    ]
    return lib


def _guess_format(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    fmt = _EXT_TO_FORMAT.get(ext)
    if fmt is None:
        raise DataError(
            f"no se pudo adivinar el formato de '{path}' por su extension ({ext or 'sin extension'}); "
            f"pasa format=ternaryai.dataset.TEXT_LINES/TEXT_WHOLE/JSON/JSONL explicitamente")
    return fmt


def load(path: str, format: int = None, strict: bool = True, reasoning: str = "keep") -> Tuple[List[str], int]:
    """Carga 'path' y devuelve (ejemplos, saltados).

    format: TEXT_LINES, TEXT_WHOLE, JSON o JSONL. Si se omite, se
    adivina por la extension del archivo (.txt -> TEXT_LINES, .json ->
    JSON, .jsonl -> JSONL); para .txt como un solo bloque hay que pedir
    TEXT_WHOLE explicitamente.

    strict: con JSON/JSONL, True (default) hace fallar la carga ante el
    primer registro que no cumpla ningun esquema conocido; False lo
    saltea y sigue -- 'saltados' cuenta cuantos (siempre 0 para texto).

    reasoning: "keep" (default) conserva el razonamiento de los turnos del asistente normalizado a ``<|think|>...<|/think|>``:
    las etiquetas ``<think>...</think>`` (o ``<thinking>``) del texto y los campos ``reasoning_content`` / ``reasoning`` / ``thinking``
    del mensaje (que se escriben antes de la respuesta como ``<|think|>\\n...\\n<|/think|>\\n\\n<respuesta>``). "drop" lo quita todo,
    para entrenar un modelo que no razona con un dataset que si. Las etiquetas deben estar balanceadas; si no, el registro es
    invalido. No afecta a texto plano ni a registros ``text``.

    El formato de cada registro JSON/JSONL se detecta por su clave (un mismo
    archivo puede mezclar formatos):
      "chat"           propio: [{"user": "..."}, {"model": "..."}, ...]
                       (roles: user, model, system, console, terminal, archive, log, result)
      "messages"       ChatML/OpenAI: [{"role":..., "content":...}, ...]
      "conversations"  ShareGPT: [{"from": "human"|"gpt", "value":...}, ...]
      "prompt"+"completion", "instruction"[+"input"]+"output" (Alpaca)
      "text"           texto plano, sin marcadores de turno
    Los turnos se renderizan como "<|rol|>\\ntexto\\n" y cada conversacion
    termina con "<|end|>\\n" (ver include/ternary_dataset.h).
    """
    if format is None:
        format = _guess_format(path)
    if reasoning not in ("keep", "drop"):
        raise ValueError("reasoning debe ser 'keep' o 'drop'")
    flags = 1 if reasoning == "drop" else 0
    lib = _load_lib()
    err = ctypes.create_string_buffer(_ERR_CAP)
    out_len = ctypes.c_size_t(0)
    out_count = ctypes.c_size_t(0)
    out_skipped = ctypes.c_size_t(0)
    path_b = path.encode("utf-8")

    ok = lib.ternaryai_dataset_load_ex(path_b, format, int(strict), flags, None, 0,
                                     ctypes.byref(out_len), ctypes.byref(out_count),
                                     ctypes.byref(out_skipped), err, _ERR_CAP)
    if not ok:
        raise DatasetError(err.value.decode("utf-8", "replace"))

    n = out_len.value
    buf = (ctypes.c_uint8 * n)() if n else None
    ok = lib.ternaryai_dataset_load_ex(path_b, format, int(strict), flags, buf, n,
                                     ctypes.byref(out_len), ctypes.byref(out_count),
                                     ctypes.byref(out_skipped), err, _ERR_CAP)
    if not ok:
        raise DatasetError(err.value.decode("utf-8", "replace"))

    raw = bytes(buf) if n else b""
    # Los ejemplos vienen separados por NUL; el ultimo separador deja un
    # elemento vacio final en split() que hay que descartar.
    parts = raw.split(b"\x00")[:-1] if raw else []
    examples = [p.decode("utf-8", "replace") for p in parts]
    return examples, out_skipped.value


# Roles que el cargador puede escribir como marca "<|rol|>" (ver include/ternary_dataset.h).
CHAT_ROLES = ("user", "assistant", "system", "console", "terminal", "archive", "log", "result", "function_call", "tool")
# Roles cuyo contenido genera el modelo (se entrena); el resto es contexto y se enmascara.
DEFAULT_TRAIN_ROLES = ("assistant", "function_call")


def targets(ids_list: Sequence[Sequence[int]], tokenizer, *, train_roles: Sequence[str] = DEFAULT_TRAIN_ROLES,
            roles: Sequence[str] = CHAT_ROLES, train_text: bool = True) -> List[np.ndarray]:
    """Objetivos de siguiente-token con mascara de perdida para ejemplos ya tokenizados.

    Devuelve, por ejemplo, un arreglo int32 ``y`` del mismo largo que sus ids: ``y[i] = ids[i+1]`` si ese token se entrena y
    ``-1`` (ignorado por ``Model.fit``) si no; el ultimo es ``-1``. Se entrena el contenido de los turnos de ``train_roles`` mas el
    marcador (o ``<|end|>``) que cierra el turno, para que el modelo aprenda a terminarlo; el marcador que abre el turno y el
    contenido de los demas roles quedan enmascarados. ``<|think|>``/``<|/think|>`` y cualquier otro token especial que no sea un
    rol son tokens normales del turno en que aparecen. ``train_text``: el texto fuera de todo turno (registros ``text``, corpus
    plano) se entrena (True) o se enmascara.

    ``roles`` son los nombres que cuentan como marcadores (``"user"`` -> ``<|user|>``); los que no sean tokens especiales del
    tokenizador se ignoran, salvo que esten en ``train_roles`` (error: no se podrian reconocer). Para entrenar sobre TODO el
    ejemplo (sin mascara) pasa ``train_roles=roles``.
    """
    unknown = [r for r in train_roles if r not in roles]
    if unknown:
        raise ValueError(f"train_roles {unknown} no estan en roles {list(roles)}")
    ns = int(tokenizer.num_special)
    kind = np.zeros(max(ns, 1), dtype=np.int8)
    for r in roles:
        try:
            sid = tokenizer.special_id(f"<|{r}|>")
        except TernaryAIError:
            if r in train_roles:
                raise ValueError(f"el rol {r!r} esta en train_roles pero '<|{r}|>' no es un token especial del tokenizador")
            continue
        kind[sid - 256] = 1 if r in train_roles else 2
    try:
        kind[tokenizer.special_id("<|end|>") - 256] = 3
    except TernaryAIError:
        pass
    n_ex = len(ids_list)
    offs = np.zeros(n_ex + 1, dtype=np.uintp)
    if n_ex:
        offs[1:] = np.cumsum([len(x) for x in ids_list], dtype=np.uintp)
    flat = np.fromiter((t for x in ids_list for t in x), dtype=np.int32, count=int(offs[-1])) if n_ex else np.zeros(0, np.int32)
    y = np.empty_like(flat)
    lib = _load_lib()
    err = ctypes.create_string_buffer(_ERR_CAP)
    c = ctypes
    ok = lib.ternaryai_dataset_targets(
        flat.ctypes.data_as(c.POINTER(c.c_int32)), offs.ctypes.data_as(c.POINTER(c.c_size_t)), n_ex,
        256, 256 + ns, kind.ctypes.data_as(c.POINTER(c.c_int8)), int(train_text), y.ctypes.data_as(c.POINTER(c.c_int32)),
        err, _ERR_CAP)
    if not ok:
        raise DatasetError(err.value.decode("utf-8", "replace"))
    return [y[offs[i]:offs[i + 1]] for i in range(n_ex)]


def tokenize(examples: Sequence[str], tokenizer, seq_len: int, *, train_roles: Sequence[str] = DEFAULT_TRAIN_ROLES,
             roles: Sequence[str] = CHAT_ROLES, train_text: bool = True, overflow: str = "truncate", pad_id: int = 0,
             drop_empty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Ejemplos de texto -> ``(x, y)`` listos para ``Model.fit(x, y)``: ids ``x`` y objetivos ``y`` (con -1 donde no se aprende),
    ambos int32 de forma ``(n, seq_len)``.

    ``examples``: lo que devuelve :func:`load`. ``tokenizer``: un ``Tokenizer`` con los tokens especiales de los roles
    (``ternaryai.CHAT_SPECIAL_TOKENS``). Ver :func:`targets` para ``train_roles``, ``roles`` y ``train_text``.

    ``overflow`` decide que pasa con un ejemplo de mas de ``seq_len`` tokens: ``"truncate"`` (default) conserva el principio;
    ``"split"`` lo parte en ventanas consecutivas de ``seq_len`` (cada ventana se entrena con su propia mascara, pero no ve el
    contexto de las anteriores); ``"drop"`` lo descarta. Los ejemplos mas cortos se rellenan con ``pad_id`` en ``x`` y -1 en
    ``y`` (el relleno nunca entra en la perdida). ``drop_empty`` (default True) descarta las ventanas sin ningun objetivo que
    aprender (p. ej. un trozo que es solo contexto de usuario) y los ejemplos de menos de 2 tokens.
    """
    if overflow not in ("truncate", "split", "drop"):
        raise ValueError("overflow debe ser 'truncate', 'split' o 'drop'")
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 2:
        raise ValueError("seq_len debe ser un entero >= 2")
    ids_list = [tokenizer.encode(e) for e in examples]
    ys = targets(ids_list, tokenizer, train_roles=train_roles, roles=roles, train_text=train_text)
    xs_out, ys_out = [], []
    for ids, y in zip(ids_list, ys):
        n = len(ids)
        if n < 2 and drop_empty:
            continue
        if n > seq_len and overflow == "drop":
            continue
        starts = range(0, n, seq_len) if (overflow == "split" and n > seq_len) else (0,)
        for s in starts:
            seg_x, seg_y = ids[s:s + seq_len], y[s:s + seq_len]
            if drop_empty and not (seg_y >= 0).any():
                continue
            xs_out.append(seg_x)
            ys_out.append(seg_y)
    m = len(xs_out)
    x = np.full((m, seq_len), pad_id, dtype=np.int32)
    yy = np.full((m, seq_len), -1, dtype=np.int32)
    for i in range(m):
        x[i, :len(xs_out[i])] = xs_out[i]
        yy[i, :len(ys_out[i])] = ys_out[i]
    return x, yy
