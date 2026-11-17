"""The native backend: libternaryai (the C engine) through ctypes."""
from __future__ import annotations

import ctypes
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._backend import Backend, LayerDesc, ModelDescription
from ._spec import REQUIRED, LayerSpec, Param
from .errors import (ArchitectureError, BackendNotFoundError, DataError, ModelFileError,
                     TernaryAIError)

ABI_VERSION = 30           # must match TERNARYAI_ABI_VERSION in include/ternaryai.h
_ERR_CAP = 512
_MAX_DIMS = 4
_MAX_SLOTS = 16
_NAME_CAP = 64
_HYPER_ORDER = ("scale_lr",)
_KINDS = {0: "int", 1: "float", 2: "bool", 3: "pair"}

_NAMES = {"win32": ["ternaryai.dll"], "darwin": ["libternaryai.dylib", "libternaryai.so"]}


def _candidates() -> List[str]:
    names = _NAMES.get(sys.platform, ["libternaryai.so"])
    here = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(here)
    dirs = [here, python_dir, os.path.dirname(python_dir), os.getcwd()]
    env = os.environ.get("TERNARYAI_LIB")
    found = [env] if env else []
    found += [os.path.join(d, n) for d in dirs for n in names]
    return found


def find_library() -> Optional[str]:
    """Path of libternaryai: ``$TERNARYAI_LIB``, then the package, ``python/``, the project root and
    the current directory."""
    for path in _candidates():
        if path and os.path.isfile(path):
            return path
    return None


def load_default() -> "NativeBackend":
    path = find_library()
    if path is None:
        raise BackendNotFoundError(
            "libternaryai was not found. Build it from the C project (`make libternaryai.so`) and "
            "put it next to the ternaryai package, or point $TERNARYAI_LIB to it.")
    return NativeBackend(path)


def _shape_arg(shape: Sequence[int]):
    return (ctypes.c_size_t * len(shape))(*shape)


class NativeBackend(Backend):
    """Backend backed by libternaryai. Every method is one C call (see include/ternaryai.h)."""

    name = "native"

    def __init__(self, path: str) -> None:
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as exc:
            raise BackendNotFoundError(f"cannot load libternaryai from {path!r}: {exc}") from exc
        self.path = path
        self._bind()
        abi = self._lib.ternaryai_abi_version()
        if abi != ABI_VERSION:
            raise BackendNotFoundError(
                f"{path!r} has ABI version {abi}, but this ternaryai package needs {ABI_VERSION}; "
                "rebuild libternaryai from the matching C sources")
        self._specs = tuple(self._read_registry())
        self._by_type = {s.type_name: s for s in self._specs}
        self._activations = tuple(self._str(self._lib.ternaryai_activation_name(i))
                                  for i in range(self._lib.ternaryai_activation_count()))
        self._live: Dict[int, int] = {}      # handle -> output length (features)
        self._open: Dict[int, List[Optional[int]]] = {}   # handle -> output length before each open residual block

    # -- ctypes plumbing ----------------------------------------------------
    def _bind(self) -> None:
        L, c = self._lib, ctypes
        sig = {
            "ternaryai_abi_version": (c.c_int, []),
            "ternaryai_layer_type_count": (c.c_int, []),
            "ternaryai_layer_type_name": (c.c_char_p, [c.c_int]),
            "ternaryai_layer_type_class": (c.c_char_p, [c.c_int]),
            "ternaryai_layer_type_doc": (c.c_char_p, [c.c_int]),
            "ternaryai_layer_param_count": (c.c_int, [c.c_int]),
            "ternaryai_layer_param_name": (c.c_char_p, [c.c_int, c.c_int]),
            "ternaryai_layer_param_kind": (c.c_int, [c.c_int, c.c_int]),
            "ternaryai_layer_param_required": (c.c_int, [c.c_int, c.c_int]),
            "ternaryai_layer_param_default": (c.c_double, [c.c_int, c.c_int]),
            "ternaryai_layer_param_has_min": (c.c_int, [c.c_int, c.c_int]),
            "ternaryai_layer_param_has_max": (c.c_int, [c.c_int, c.c_int]),
            "ternaryai_layer_param_max": (c.c_double, [c.c_int, c.c_int]),
            "ternaryai_layer_param_min": (c.c_double, [c.c_int, c.c_int]),
            "ternaryai_layer_param_doc": (c.c_char_p, [c.c_int, c.c_int]),
            "ternaryai_activation_count": (c.c_int, []),
            "ternaryai_activation_name": (c.c_char_p, [c.c_int]),
            "ternaryai_model_create": (c.c_void_p, [c.POINTER(c.c_size_t), c.c_int, c.c_uint,
                                                 c.c_char_p, c.c_size_t]),
            "ternaryai_model_add_layer": (c.c_int, [c.c_void_p, c.c_char_p, c.POINTER(c.c_double),
                                                 c.c_int, c.c_char_p, c.POINTER(c.c_size_t),
                                                 c.POINTER(c.c_int), c.c_char_p, c.c_size_t]),
            "ternaryai_model_num_layers": (c.c_int, [c.c_void_p]),
            "ternaryai_model_input_shape": (c.c_int, [c.c_void_p, c.POINTER(c.c_size_t),
                                                   c.POINTER(c.c_int)]),
            "ternaryai_model_layer_info": (c.c_int, [c.c_void_p, c.c_int, c.c_char_p,
                                                  c.POINTER(c.c_double), c.POINTER(c.c_int),
                                                  c.c_char_p, c.POINTER(c.c_size_t),
                                                  c.POINTER(c.c_int)]),
            "ternaryai_model_train": (c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t,
                                             c.c_size_t, c.c_size_t, c.POINTER(c.c_double),
                                             c.c_void_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_predict": (c.c_int, [c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p,
                                               c.c_char_p, c.c_size_t]),
            "ternaryai_model_train_f32": (c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t,
                                                 c.c_size_t, c.c_size_t, c.POINTER(c.c_double),
                                                 c.c_void_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_predict_f32": (c.c_int, [c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p,
                                                   c.c_char_p, c.c_size_t]),
            "ternaryai_model_begin_residual": (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_end_residual": (c.c_int, [c.c_void_p, c.POINTER(c.c_size_t),
                                                    c.POINTER(c.c_int), c.c_char_p, c.c_size_t]),
            "ternaryai_model_cancel_residual": (c.c_int, [c.c_void_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_layer_info_at": (c.c_int, [c.c_void_p, c.POINTER(c.c_int), c.c_int,
                                                     c.c_char_p, c.POINTER(c.c_double),
                                                     c.POINTER(c.c_int), c.c_char_p,
                                                     c.POINTER(c.c_size_t), c.POINTER(c.c_int)]),
            "ternaryai_model_layer_children": (c.c_int, [c.c_void_p, c.POINTER(c.c_int), c.c_int]),
            "ternaryai_model_attention_probs": (c.c_int, [c.c_void_p, c.c_int, c.c_int, c.c_void_p, c.c_size_t]),
            "ternaryai_set_threads": (None, [c.c_int]),
            "ternaryai_get_threads": (c.c_int, []),
            "ternaryai_prof_enable": (None, [c.c_int]),
            "ternaryai_prof_reset": (None, []),
            "ternaryai_prof_dump_json": (c.c_size_t, [c.c_char_p, c.c_size_t]),
            "ternaryai_model_ffn_zero_stats": (c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_int, c.c_int]),
            "ternaryai_model_takes_ids": (c.c_int, [c.c_void_p]),
            "ternaryai_model_set_restore_best": (None, [c.c_void_p, c.c_int]),
            "ternaryai_model_set_batching": (None, [c.c_void_p, c.c_size_t, c.c_int, c.c_uint64]),
            "ternaryai_model_set_update_rule": (c.c_int, [c.c_void_p, c.c_double, c.c_double, c.c_char_p, c.c_size_t]),
            "ternaryai_model_set_r2_params": (c.c_int, [c.c_void_p, c.c_double, c.c_double, c.c_double, c.c_char_p, c.c_size_t]),
            "ternaryai_model_get_update_rule": (c.c_int, [c.c_void_p, c.POINTER(c.c_double)]),
            "ternaryai_model_set_head_beta2": (c.c_int, [c.c_void_p, c.c_double, c.c_char_p, c.c_size_t]),
            "ternaryai_model_get_head_beta2": (c.c_int, [c.c_void_p, c.POINTER(c.c_double)]),
            "ternaryai_model_hold_best": (None, [c.c_void_p, c.c_int]),
            "ternaryai_model_restore_held_best": (c.c_int, [c.c_void_p, c.c_double]),
            "ternaryai_model_set_checkpoint": (c.c_int, [c.c_void_p, c.c_int]),
            "ternaryai_model_train_ids": (c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t,
                                                 c.c_size_t, c.c_size_t, c.POINTER(c.c_double),
                                                 c.c_void_p, c.c_void_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_predict_ids": (c.c_int, [c.c_void_p, c.c_void_p, c.c_size_t,
                                                   c.c_void_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_save": (c.c_int, [c.c_void_p, c.c_char_p, c.c_int, c.c_char_p,
                                            c.c_size_t]),
            "ternaryai_model_load": (c.c_void_p, [c.c_char_p, c.c_char_p, c.c_size_t]),
            "ternaryai_model_free": (None, [c.c_void_p]),
        }
        for name, (restype, argtypes) in sig.items():
            try:
                fn = getattr(L, name)
            except AttributeError as exc:
                raise BackendNotFoundError(f"{self.path!r} does not export {name}") from exc
            fn.restype, fn.argtypes = restype, argtypes

    @staticmethod
    def _str(value: Optional[bytes]) -> str:
        return value.decode() if value else ""

    @staticmethod
    def _msg(buf) -> str:
        return buf.value.decode(errors="replace") or "unknown error"

    def _read_registry(self) -> List[LayerSpec]:
        L = self._lib
        specs = []
        for t in range(L.ternaryai_layer_type_count()):
            params = []
            for p in range(L.ternaryai_layer_param_count(t)):
                kind = _KINDS[L.ternaryai_layer_param_kind(t, p)]
                required = bool(L.ternaryai_layer_param_required(t, p))
                default: Any = REQUIRED
                if not required:
                    raw = L.ternaryai_layer_param_default(t, p)
                    default = ((int(raw),) * 2 if kind == "pair" else bool(raw) if kind == "bool"
                               else float(raw) if kind == "float" else int(raw))
                minimum = L.ternaryai_layer_param_min(t, p) if L.ternaryai_layer_param_has_min(t, p) \
                    else None
                maximum = L.ternaryai_layer_param_max(t, p) if L.ternaryai_layer_param_has_max(t, p) \
                    else None
                params.append(Param(self._str(L.ternaryai_layer_param_name(t, p)), kind, default,
                                    minimum, self._str(L.ternaryai_layer_param_doc(t, p)), maximum))
            specs.append(LayerSpec(self._str(L.ternaryai_layer_type_name(t)),
                                   self._str(L.ternaryai_layer_type_class(t)), tuple(params),
                                   self._str(L.ternaryai_layer_type_doc(t)),
                                   default_activation="uint8"))
        return specs

    def _handle(self, handle: Any) -> int:
        if handle not in self._live:
            raise ValueError("invalid or freed model handle")
        return handle

    # -- Backend contract ---------------------------------------------------
    def layer_types(self) -> Sequence[LayerSpec]:
        return self._specs

    def activations(self) -> Sequence[str]:
        return self._activations

    def create(self, input_shape: Tuple[int, ...], seed: int) -> Any:
        err = ctypes.create_string_buffer(_ERR_CAP)
        handle = self._lib.ternaryai_model_create(_shape_arg(input_shape), len(input_shape),
                                               seed & 0xFFFFFFFF, err, _ERR_CAP)
        if not handle:
            raise TernaryAIError(self._msg(err))
        self._live[handle] = int(math.prod(input_shape))
        return handle

    def add_layer(self, handle: Any, type_name: str, params: Dict[str, Any],
                  activation: str) -> Tuple[int, ...]:
        self._handle(handle)
        spec = self._by_type.get(type_name)
        if spec is None:
            raise ArchitectureError(f"unknown layer type {type_name!r}")
        slots: List[float] = []
        for p in spec.params:
            value = spec.coerce_params(params)[p.name]
            slots.extend(float(v) for v in value) if p.kind == "pair" else slots.append(float(value))
        err = ctypes.create_string_buffer(_ERR_CAP)
        shape = (ctypes.c_size_t * _MAX_DIMS)()
        ndims = ctypes.c_int()
        ok = self._lib.ternaryai_model_add_layer(
            handle, type_name.encode(), (ctypes.c_double * len(slots))(*slots), len(slots),
            activation.encode(), shape, ctypes.byref(ndims), err, _ERR_CAP)
        if not ok:
            raise ArchitectureError(self._msg(err))
        out = tuple(int(shape[i]) for i in range(ndims.value))
        self._live[handle] = int(math.prod(out))
        return out

    def begin_residual(self, handle: Any, activation: str) -> None:
        self._handle(handle)
        err = ctypes.create_string_buffer(_ERR_CAP)
        if not self._lib.ternaryai_model_begin_residual(handle, activation.encode(), err, _ERR_CAP):
            raise ArchitectureError(self._msg(err))
        self._open.setdefault(handle, []).append(self._live.get(handle))

    def end_residual(self, handle: Any) -> Tuple[int, ...]:
        self._handle(handle)
        err = ctypes.create_string_buffer(_ERR_CAP)
        shape = (ctypes.c_size_t * _MAX_DIMS)()
        ndims = ctypes.c_int()
        ok = self._lib.ternaryai_model_end_residual(handle, shape, ctypes.byref(ndims), err, _ERR_CAP)
        before = self._open[handle].pop()          # the C side pops the block whether it worked or not
        if not ok:
            self._set_live(handle, before)
            raise ArchitectureError(self._msg(err))
        out = tuple(int(shape[i]) for i in range(ndims.value))
        self._live[handle] = int(math.prod(out))
        return out

    def cancel_residual(self, handle: Any) -> None:
        self._handle(handle)
        err = ctypes.create_string_buffer(_ERR_CAP)
        if not self._lib.ternaryai_model_cancel_residual(handle, err, _ERR_CAP):
            raise ArchitectureError(self._msg(err))
        self._set_live(handle, self._open[handle].pop())

    def _set_live(self, handle: Any, value: Optional[int]) -> None:
        if value is None:
            self._live.pop(handle, None)
        else:
            self._live[handle] = value

    def _describe_layer(self, handle: Any, path: List[int]) -> LayerDesc:
        L = self._lib
        cpath = (ctypes.c_int * len(path))(*path)
        type_name = ctypes.create_string_buffer(_NAME_CAP)
        activation = ctypes.create_string_buffer(_NAME_CAP)
        slots = (ctypes.c_double * _MAX_SLOTS)()
        shape = (ctypes.c_size_t * _MAX_DIMS)()
        nslots, nd = ctypes.c_int(), ctypes.c_int()
        if not L.ternaryai_model_layer_info_at(handle, cpath, len(path), type_name, slots,
                                               ctypes.byref(nslots), activation, shape,
                                               ctypes.byref(nd)):
            raise TernaryAIError(f"cannot describe layer {path}")
        out_shape = tuple(int(shape[j]) for j in range(nd.value))
        name = type_name.value.decode()
        if name == "residual":
            count = L.ternaryai_model_layer_children(handle, cpath, len(path))
            children = tuple(self._describe_layer(handle, path + [j]) for j in range(count))
            return LayerDesc("residual", {}, activation.value.decode(), out_shape, children)
        spec = self._by_type[name]
        params, pos = {}, 0
        for p in spec.params:
            if p.kind == "pair":
                params[p.name] = (int(slots[pos]), int(slots[pos + 1]))
                pos += 2
            else:
                raw = slots[pos]
                # floats live as float32 in C: 7 significant digits give back what the user typed
                params[p.name] = (bool(raw) if p.kind == "bool" else float(f"{raw:.7g}")
                                  if p.kind == "float" else int(raw))
                pos += 1
        return LayerDesc(spec.type_name, params, activation.value.decode(), out_shape)

    def describe(self, handle: Any) -> ModelDescription:
        self._handle(handle)
        L = self._lib
        in_shape = (ctypes.c_size_t * _MAX_DIMS)()
        in_nd = ctypes.c_int()
        L.ternaryai_model_input_shape(handle, in_shape, ctypes.byref(in_nd))
        input_shape = tuple(int(in_shape[i]) for i in range(in_nd.value))
        layers = [self._describe_layer(handle, [i]) for i in range(L.ternaryai_model_num_layers(handle))]
        output_shape = layers[-1].output_shape if layers else input_shape
        return ModelDescription(input_shape, output_shape, tuple(layers))

    def train_epochs(self, handle: Any, x: np.ndarray, y: np.ndarray, first_epoch: int,
                     count: int, hyper: Dict[str, float]) -> np.ndarray:
        self._handle(handle)
        if x.dtype != np.int8 or x.ndim != 2 or not x.flags.c_contiguous:
            raise DataError("x must be a C-contiguous int8 array (n, features)")
        if y.dtype != np.float32 or y.ndim != 2 or y.shape[0] != x.shape[0] \
                or not y.flags.c_contiguous:
            raise DataError("y must be a C-contiguous float32 array (n, outputs)")
        if y.shape[1] != self._live[handle]:
            raise DataError(f"y has {y.shape[1]} outputs per sample, the model produces "
                            f"{self._live[handle]}")
        h = (ctypes.c_double * len(_HYPER_ORDER))(*[float(hyper[k]) for k in _HYPER_ORDER])
        loss = np.empty(count, dtype=np.float64)
        err = ctypes.create_string_buffer(_ERR_CAP)
        ok = self._lib.ternaryai_model_train(handle, x.ctypes.data, y.ctypes.data, x.shape[0],
                                          first_epoch, count, h, loss.ctypes.data, err, _ERR_CAP)
        if not ok:
            raise TernaryAIError(self._msg(err))
        return loss

    def predict(self, handle: Any, x: np.ndarray) -> np.ndarray:
        self._handle(handle)
        if x.dtype != np.int8 or x.ndim != 2 or not x.flags.c_contiguous:
            raise DataError("x must be a C-contiguous int8 array (n, features)")
        out = np.empty((x.shape[0], self._live[handle]), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_CAP)
        if not self._lib.ternaryai_model_predict(handle, x.ctypes.data, x.shape[0], out.ctypes.data,
                                              err, _ERR_CAP):
            raise TernaryAIError(self._msg(err))
        return out

    def train_epochs_f32(self, handle: Any, x: np.ndarray, y: np.ndarray, first_epoch: int,
                         count: int, hyper: Dict[str, float]) -> np.ndarray:
        self._handle(handle)
        if x.dtype != np.float32 or x.ndim != 2 or not x.flags.c_contiguous:
            raise DataError("x must be a C-contiguous float32 array (n, features)")
        if y.dtype != np.float32 or y.ndim != 2 or y.shape[0] != x.shape[0] \
                or not y.flags.c_contiguous:
            raise DataError("y must be a C-contiguous float32 array (n, outputs)")
        if y.shape[1] != self._live[handle]:
            raise DataError(f"y has {y.shape[1]} outputs per sample, the model produces "
                            f"{self._live[handle]}")
        h = (ctypes.c_double * len(_HYPER_ORDER))(*[float(hyper[k]) for k in _HYPER_ORDER])
        loss = np.empty(count, dtype=np.float64)
        err = ctypes.create_string_buffer(_ERR_CAP)
        ok = self._lib.ternaryai_model_train_f32(handle, x.ctypes.data, y.ctypes.data, x.shape[0],
                                                 first_epoch, count, h, loss.ctypes.data, err, _ERR_CAP)
        if not ok:
            raise TernaryAIError(self._msg(err))
        return loss

    def predict_f32(self, handle: Any, x: np.ndarray) -> np.ndarray:
        self._handle(handle)
        if x.dtype != np.float32 or x.ndim != 2 or not x.flags.c_contiguous:
            raise DataError("x must be a C-contiguous float32 array (n, features)")
        out = np.empty((x.shape[0], self._live[handle]), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_CAP)
        if not self._lib.ternaryai_model_predict_f32(handle, x.ctypes.data, x.shape[0], out.ctypes.data,
                                                     err, _ERR_CAP):
            raise TernaryAIError(self._msg(err))
        return out

    def takes_ids(self, handle: Any) -> bool:
        self._handle(handle)
        return bool(self._lib.ternaryai_model_takes_ids(handle))

    def hold_best(self, handle: Any, on: bool) -> None:
        self._lib.ternaryai_model_hold_best(handle, 1 if on else 0)

    def restore_held_best(self, handle: Any, last_loss: float) -> bool:
        return bool(self._lib.ternaryai_model_restore_held_best(handle, float(last_loss)))

    def set_batching(self, handle: Any, batch_size: int, shuffle: bool, seed: int) -> None:
        self._handle(handle)
        self._lib.ternaryai_model_set_batching(handle, int(batch_size), 1 if shuffle else 0,
                                               int(seed) & 0xFFFFFFFFFFFFFFFF)

    def set_update_rule(self, handle: Any, tau: float, beta: float) -> None:
        self._handle(handle)
        err = ctypes.create_string_buffer(256)
        if not self._lib.ternaryai_model_set_update_rule(handle, float(tau), float(beta), err, len(err)):
            raise TernaryAIError(err.value.decode("utf-8", "replace"))

    def set_r2_params(self, handle: Any, inc: float, zero_target: float, reset: float) -> None:
        self._handle(handle)
        err = ctypes.create_string_buffer(256)
        if not self._lib.ternaryai_model_set_r2_params(handle, float(inc), float(zero_target), float(reset), err, len(err)):
            raise TernaryAIError(err.value.decode("utf-8", "replace"))

    def get_update_rule(self, handle: Any) -> Tuple[float, float, float, float, float]:
        self._handle(handle)
        out = (ctypes.c_double * 5)()
        if not self._lib.ternaryai_model_get_update_rule(handle, out):
            raise TernaryAIError("could not read the update rule")
        return tuple(float(v) for v in out)

    def set_head_beta2(self, handle: Any, head_beta2: float) -> None:
        self._handle(handle)
        err = ctypes.create_string_buffer(256)
        if not self._lib.ternaryai_model_set_head_beta2(handle, float(head_beta2), err, len(err)):
            raise TernaryAIError(err.value.decode("utf-8", "replace"))

    def get_head_beta2(self, handle: Any) -> float:
        self._handle(handle)
        out = ctypes.c_double()
        if not self._lib.ternaryai_model_get_head_beta2(handle, ctypes.byref(out)):
            raise TernaryAIError("could not read head_beta2")
        return float(out.value)

    def set_restore_best(self, handle: Any, on: bool) -> None:
        self._handle(handle)
        self._lib.ternaryai_model_set_restore_best(handle, 1 if on else 0)

    def ffn_zero_stats(self, handle: Any, reset: bool) -> list:
        self._handle(handle)
        n = self._lib.ternaryai_model_ffn_zero_stats(handle, None, None, 0, 0)
        if n < 0:
            raise TernaryAIError("could not read the feed-forward sparsity")
        zeros = (ctypes.c_uint64 * max(n, 1))(); total = (ctypes.c_uint64 * max(n, 1))()
        self._lib.ternaryai_model_ffn_zero_stats(handle, zeros, total, n, 1 if reset else 0)
        return [(int(zeros[i]), int(total[i])) for i in range(n)]

    def attention_probs(self, handle: Any, gqa_index: int, head: int, seq: int) -> np.ndarray:
        self._handle(handle)
        out = np.zeros((seq, seq), dtype=np.float32)
        rc = self._lib.ternaryai_model_attention_probs(handle, gqa_index, head, out.ctypes.data, out.size)
        if rc != 0:
            raise TernaryAIError("could not read the attention probabilities (window attention, bad layer/head index, or no forward yet)")
        return out

    def num_gqa(self, handle: Any) -> int:
        self._handle(handle)
        return int(self._lib.ternaryai_model_attention_probs(handle, -1, 0, None, 0))

    def set_threads(self, n: int) -> None:
        self._lib.ternaryai_set_threads(int(n))

    def get_threads(self) -> int:
        return int(self._lib.ternaryai_get_threads())

    def prof_enable(self, on: bool) -> None:
        self._lib.ternaryai_prof_enable(1 if on else 0)

    def prof_reset(self) -> None:
        self._lib.ternaryai_prof_reset()

    def prof_dump(self) -> str:
        need = int(self._lib.ternaryai_prof_dump_json(None, 0))
        buf = ctypes.create_string_buffer(need + 1)
        self._lib.ternaryai_prof_dump_json(buf, need + 1)
        return buf.value.decode("utf-8")

    def set_checkpoint(self, handle: Any, on: bool) -> None:
        self._handle(handle)
        if self._lib.ternaryai_model_set_checkpoint(handle, 1 if on else 0) != 0:
            raise TernaryAIError("could not enable block checkpointing")

    def train_epochs_ids(self, handle: Any, ids: np.ndarray, targets: np.ndarray, first_epoch: int,
                         count: int, hyper: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        self._handle(handle)
        for name, arr in (("ids", ids), ("targets", targets)):
            if arr.dtype != np.int32 or arr.ndim != 2 or not arr.flags.c_contiguous:
                raise DataError(f"{name} must be a C-contiguous int32 array (n, seq)")
        if ids.shape != targets.shape:
            raise DataError(f"ids {ids.shape} and targets {targets.shape} must have the same shape")
        h = (ctypes.c_double * len(_HYPER_ORDER))(*[float(hyper[k]) for k in _HYPER_ORDER])
        loss = np.empty(count, dtype=np.float64)
        acc = np.empty(count, dtype=np.float64)
        err = ctypes.create_string_buffer(_ERR_CAP)
        ok = self._lib.ternaryai_model_train_ids(handle, ids.ctypes.data, targets.ctypes.data,
                                              ids.shape[0], first_epoch, count, h,
                                              loss.ctypes.data, acc.ctypes.data, err, _ERR_CAP)
        if not ok:
            raise self._data_or_generic(self._msg(err))
        return loss, acc

    def predict_ids(self, handle: Any, ids: np.ndarray) -> np.ndarray:
        self._handle(handle)
        if ids.dtype != np.int32 or ids.ndim != 2 or not ids.flags.c_contiguous:
            raise DataError("ids must be a C-contiguous int32 array (n, seq)")
        out = np.empty((ids.shape[0], self._live[handle]), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_CAP)
        if not self._lib.ternaryai_model_predict_ids(handle, ids.ctypes.data, ids.shape[0],
                                                  out.ctypes.data, err, _ERR_CAP):
            raise self._data_or_generic(self._msg(err))
        return out

    @staticmethod
    def _data_or_generic(message: str) -> TernaryAIError:
        """Bad ids/targets are the caller's data; anything else is a generic engine error."""
        if "outside" in message or "nothing to learn" in message:
            return DataError(message)
        return TernaryAIError(message)

    def save(self, handle: Any, path: str, full_checkpoint: bool) -> None:
        self._handle(handle)
        err = ctypes.create_string_buffer(_ERR_CAP)
        if not self._lib.ternaryai_model_save(handle, os.fsencode(path), int(full_checkpoint), err,
                                           _ERR_CAP):
            raise ModelFileError(self._msg(err))

    def load(self, path: str) -> Any:
        err = ctypes.create_string_buffer(_ERR_CAP)
        handle = self._lib.ternaryai_model_load(os.fsencode(path), err, _ERR_CAP)
        if not handle:
            raise ModelFileError(self._msg(err))
        self._live[handle] = 0
        self._live[handle] = int(math.prod(self.describe(handle).output_shape))
        return handle

    def free(self, handle: Any) -> None:
        if self._live.pop(handle, None) is not None:
            self._lib.ternaryai_model_free(handle)

