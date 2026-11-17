"""A backend that validates architectures and data but does not learn.

``DryRunBackend`` implements the full backend contract in pure Python: layer
registry, shape inference, save/load, epoch bookkeeping. ``predict`` returns
zeros and the reported loss is that of an all-zero predictor. It exists to
try the API without the C library, and as the executable reference for the
contract that the C backend must also satisfy (see python/tests/contract.py).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ._backend import Backend, LayerDesc, ModelDescription
from ._spec import LayerSpec, Param
from .errors import ArchitectureError, DataError, ModelFileError

_DENSE = LayerSpec(
    "dense", "Dense",
    (Param("units", "int", min=1, doc="number of outputs"),
     Param("bias", "bool", True, doc="trainable bias")),
    doc="Fully connected ternary layer.")

_CONV2D = LayerSpec(
    "conv2d", "Conv2D",
    (Param("filters", "int", min=1, doc="output channels"),
     Param("kernel", "pair", min=1, doc="kernel size (int or (h, w))"),
     Param("stride", "int", 1, min=1),
     Param("bias", "bool", True, doc="trainable bias")),
    doc="2D ternary convolution over (height, width, channels) input. "
        "Must be the first layer.")

_SPECS = (_DENSE, _CONV2D)
_BY_TYPE = {s.type_name: s for s in _SPECS}


class _DryModel:
    def __init__(self, input_shape: Tuple[int, ...], seed: int) -> None:
        self.input_shape = input_shape
        self.seed = seed
        self.layers: List[LayerDesc] = []
        self.freed = False
        self.rule = [0.7, 0.999, 0.03, 0.2, 0.5]   # tau, beta, inc, zero_target, reset
        self.head_beta2 = 0.99                      # normalizacion por elemento de la cabeza (0 = apagada)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self.layers[-1].output_shape if self.layers else self.input_shape


class DryRunBackend(Backend):
    name = "dry-run"

    def layer_types(self) -> Sequence[LayerSpec]:
        return _SPECS

    def activations(self) -> Sequence[str]:
        return ("uint8",)

    def _live(self, handle: Any) -> _DryModel:
        if not isinstance(handle, _DryModel) or handle.freed:
            raise ValueError("invalid or freed model handle")
        return handle

    def create(self, input_shape: Tuple[int, ...], seed: int) -> Any:
        return _DryModel(tuple(input_shape), seed)

    def add_layer(self, handle: Any, type_name: str, params: Dict[str, Any],
                  activation: str) -> Tuple[int, ...]:
        model = self._live(handle)
        spec = _BY_TYPE.get(type_name)
        if spec is None:
            raise ArchitectureError(f"unknown layer type {type_name!r}")
        if activation not in self.activations():
            raise ArchitectureError(f"unknown activation {activation!r}")
        params = spec.coerce_params(params)
        current = model.output_shape
        if type_name == "dense":
            out: Tuple[int, ...] = (params["units"],)
        else:
            if model.layers:
                raise ArchitectureError(
                    "Conv2D does not propagate gradient to its input, so it can only be "
                    "the first layer")
            if len(current) != 3:
                raise ArchitectureError(
                    f"Conv2D needs a (height, width, channels) input, got {current}")
            (h, w, _), (kh, kw), s = current, params["kernel"], params["stride"]
            if kh > h or kw > w:
                raise ArchitectureError(f"kernel {(kh, kw)} is larger than the input {(h, w)}")
            out = ((h - kh) // s + 1, (w - kw) // s + 1, params["filters"])
        model.layers.append(LayerDesc(type_name, params, activation, out))
        return out

    def describe(self, handle: Any) -> ModelDescription:
        model = self._live(handle)
        return ModelDescription(model.input_shape, model.output_shape, tuple(model.layers))

    def train_epochs(self, handle: Any, x: np.ndarray, y: np.ndarray, first_epoch: int,
                     count: int, hyper: Dict[str, float]) -> np.ndarray:
        model = self._live(handle)
        features = int(np.prod(model.input_shape))
        outputs = int(np.prod(model.output_shape))
        if (x.dtype != np.int8 or x.ndim != 2 or x.shape[1] != features
                or not x.flags.c_contiguous):
            raise DataError(f"x must be a C-contiguous int8 array (n, {features})")
        if (y.dtype != np.float32 or y.shape != (x.shape[0], outputs)
                or not y.flags.c_contiguous):
            raise DataError(f"y must be a C-contiguous float32 array (n, {outputs})")
        return np.full(count, float(np.mean(y.astype(np.float64) ** 2)))

    def train_epochs_f32(self, handle: Any, x: np.ndarray, y: np.ndarray, first_epoch: int,
                         count: int, hyper: Dict[str, float]) -> np.ndarray:
        model = self._live(handle)
        features = int(np.prod(model.input_shape))
        outputs = int(np.prod(model.output_shape))
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != features or not x.flags.c_contiguous:
            raise DataError(f"x must be a C-contiguous float32 array (n, {features})")
        if y.dtype != np.float32 or y.shape != (x.shape[0], outputs) or not y.flags.c_contiguous:
            raise DataError(f"y must be a C-contiguous float32 array (n, {outputs})")
        return np.full(count, float(np.mean(y.astype(np.float64) ** 2)))

    def predict_f32(self, handle: Any, x: np.ndarray) -> np.ndarray:
        model = self._live(handle)
        return np.zeros((x.shape[0], int(np.prod(model.output_shape))), dtype=np.float32)

    def predict(self, handle: Any, x: np.ndarray) -> np.ndarray:
        model = self._live(handle)
        return np.zeros((x.shape[0], int(np.prod(model.output_shape))), dtype=np.float32)

    def save(self, handle: Any, path: str, full_checkpoint: bool) -> None:
        model = self._live(handle)
        doc = {"format": "ternaryai-dry-run", "version": 1, "full_checkpoint": full_checkpoint,
               "input_shape": list(model.input_shape),
               "layers": [{"type_name": l.type_name, "params": l.params,
                           "activation": l.activation, "output_shape": list(l.output_shape)}
                          for l in model.layers]}
        if full_checkpoint:
            doc["r2"] = list(model.rule)    # solo el checkpoint de entrenamiento lleva la regla
            doc["head_beta2"] = float(model.head_beta2)
        with open(path, "w") as fh:
            json.dump(doc, fh)

    def load(self, path: str) -> Any:
        try:
            with open(path) as fh:
                doc = json.load(fh)
            if doc["format"] != "ternaryai-dry-run" or doc["version"] != 1:
                raise ValueError("unsupported format")
            model = _DryModel(tuple(doc["input_shape"]), 0)
            if "r2" in doc:
                model.rule = [float(v) for v in doc["r2"]]
                if "head_beta2" in doc:
                    model.head_beta2 = float(doc["head_beta2"])
            for l in doc["layers"]:
                spec = _BY_TYPE[l["type_name"]]
                model.layers.append(LayerDesc(l["type_name"], spec.coerce_params(l["params"]),
                                              l["activation"], tuple(l["output_shape"])))
            return model
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelFileError(f"cannot load model from {path!r}: {exc}") from exc

    def set_update_rule(self, handle: Any, tau: float, beta: float) -> None:
        self._live(handle).rule[0:2] = [float(tau), float(beta)]   # no entrena: solo guarda los valores

    def set_r2_params(self, handle: Any, inc: float, zero_target: float, reset: float) -> None:
        self._live(handle).rule[2:5] = [float(inc), float(zero_target), float(reset)]

    def get_update_rule(self, handle: Any):
        return tuple(self._live(handle).rule)

    def set_head_beta2(self, handle: Any, head_beta2: float) -> None:
        self._live(handle).head_beta2 = float(head_beta2)

    def get_head_beta2(self, handle: Any) -> float:
        return float(self._live(handle).head_beta2)

    def free(self, handle: Any) -> None:
        if isinstance(handle, _DryModel):
            handle.freed = True
