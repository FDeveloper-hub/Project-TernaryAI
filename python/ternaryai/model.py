"""The Model class: the whole public workflow (build, fit, predict, save, load)."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._backend import Backend, ModelDescription, get_backend
from ._registry import classes_for
from .errors import ArchitectureError, DataError, TernaryAIError
from .layers import Layer, Residual

Shape = Tuple[int, ...]


def _norm_shape(inputs: Union[int, Sequence[int]]) -> Shape:
    if isinstance(inputs, (int, np.integer)) and not isinstance(inputs, bool):
        shape: Shape = (int(inputs),)
    elif isinstance(inputs, (tuple, list)) and inputs and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in inputs):
        shape = tuple(int(v) for v in inputs)
    else:
        raise TypeError(f"inputs must be an int or a tuple of ints, got {inputs!r}")
    if any(v < 1 for v in shape):
        raise ValueError(f"inputs must be positive, got {shape}")
    return shape


class History:
    """Result of :meth:`Model.fit`: the loss of every epoch that was run.

    For ``loss="xent"`` (token-id models) ``accuracy`` also holds, per epoch, the
    fraction of non-ignored positions whose highest logit was the right id;
    it is ``None`` for ``loss="mse"``.
    """

    def __init__(self, loss: Sequence[float], first_epoch: int,
                 accuracy: Optional[Sequence[float]] = None) -> None:
        self.loss = [float(v) for v in loss]
        self.accuracy = None if accuracy is None else [float(v) for v in accuracy]
        self.first_epoch = first_epoch
        self.last_epoch = first_epoch + len(self.loss)  # exclusive

    def __repr__(self) -> str:
        final = f"{self.loss[-1]:.6g}" if self.loss else "n/a"
        acc = f" final_accuracy={self.accuracy[-1]:.4g}" if self.accuracy else ""
        return (f"<ternaryai.History epochs {self.first_epoch}..{self.last_epoch} "
                f"final_loss={final}{acc}>")


# Default scale learning rate per loss. 0.1 for "xent" (not 0.02): with residual blocks the branch scales are corrected
# too slowly with 0.02 (measured with 1-2 blocks, see README, "Calibracion de xent en multicapa").
_LOSS_DEFAULTS = {
    "mse": dict(scale_lr=0.02),
    "xent": dict(scale_lr=0.1),
}


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return value


class Model:
    """A stack of ternary layers.

    >>> model = ternaryai.Model(2, [ternaryai.Dense(16), ternaryai.Dense(1)])
    >>> model.fit(X, y, epochs=100)
    >>> model.predict(X, sign=True)

    ``inputs`` is the input shape: an int, or a tuple such as ``(28, 28, 1)``
    for images (height, width, channels). Input data may be any float array;
    values above ``threshold`` become +1, below ``-threshold`` become -1 and
    the rest 0. Data that is already made of -1/0/1 is used as is. With
    ``input="uint8"`` the data is kept as float and the first layer reads it as
    uint8 (255 levels, scale and zero-point per sample, exact zero); the first
    layer must then accept uint8 (``Dense``, ``RMSNorm``-free heads, ...).
    """

    def __init__(self, inputs: Union[int, Sequence[int]], layers: Iterable[Layer] = (), *,
                 seed: int = 0, threshold: float = 0.5, input: str = "ternary",
                 backend: Optional[Backend] = None) -> None:
        self._handle: Any = None
        self._backend = backend or get_backend()
        self._input_shape = _norm_shape(inputs)
        self._threshold = self._check_threshold(threshold)
        self._input = self._check_input(input)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an int, got {seed!r}")
        self._layers: List[Layer] = []
        self._shapes: List[Shape] = []
        self._epoch = 0
        self._handle = self._backend.create(self._input_shape, int(seed))
        try:
            for layer in layers:
                self.add(layer)
        except BaseException:
            self.close()
            raise

    # -- construction -------------------------------------------------------
    @classmethod
    def _from_description(cls, backend: Backend, handle: Any, desc: ModelDescription,
                          threshold: float, input: str = "ternary") -> "Model":
        self = object.__new__(cls)
        self._backend = backend
        self._handle = handle
        self._input_shape = tuple(desc.input_shape)
        self._threshold = cls._check_threshold(threshold)
        self._input = cls._check_input(input)
        self._epoch = 0
        classes = classes_for(backend)
        self._layers, self._shapes = [], []
        for ld in desc.layers:
            self._layers.append(cls._layer_from_desc(classes, ld))
            self._shapes.append(tuple(ld.output_shape))
        return self

    @staticmethod
    def _layer_from_desc(classes: Dict[str, type], ld: Any) -> Layer:
        if ld.type_name == "residual":
            return Residual([Model._layer_from_desc(classes, c) for c in ld.children],
                            activation=ld.activation)
        if ld.type_name not in classes:
            raise TernaryAIError(f"the backend loaded an unknown layer type {ld.type_name!r}")
        return classes[ld.type_name](**ld.params, activation=ld.activation)

    @staticmethod
    def _check_input(mode: str) -> str:
        if mode not in ("ternary", "uint8"):
            raise ValueError(f"input must be 'ternary' or 'uint8', got {mode!r}")
        return mode

    @staticmethod
    def _check_threshold(threshold: float) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
            raise TypeError(f"threshold must be a number, got {threshold!r}")
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold!r}")
        return threshold

    def add(self, layer: Layer) -> "Model":
        """Append a layer (returns the model so calls can be chained).

        A :class:`Residual` is added whole, with the layers inside it; if any of them
        breaks a rule the model is left exactly as it was.
        """
        self._check_open()
        if not isinstance(layer, Layer):
            raise TypeError(f"expected a layer such as ternaryai.Dense(16), got {layer!r}")
        if self._epoch > 0:
            raise ArchitectureError("layers cannot be added after training has started")
        shape = self._add_layer(layer)
        self._layers.append(layer)
        self._shapes.append(tuple(shape))
        return self

    def _add_layer(self, layer: Layer) -> Shape:
        """Send ``layer`` (and, for a Residual, everything inside it) to the backend; return its output shape."""
        if layer.activation in ("sign", "band"):
            raise ArchitectureError(
                f"activation {layer.activation!r} was removed: the only activation is 'uint8' "
                f"(the default; just drop the argument)")
        if layer.activation not in self._backend.activations():
            raise ArchitectureError(
                f"unknown activation {layer.activation!r}; "
                f"available: {list(self._backend.activations())}")
        if not isinstance(layer, Residual):
            return self._backend.add_layer(self._handle, layer.type_name, layer.params,
                                           layer.activation)
        self._backend.begin_residual(self._handle, layer.activation)
        try:
            for inner in layer.layers:
                self._add_layer(inner)
        except BaseException:
            self._backend.cancel_residual(self._handle)    # a rejected inner layer: back to before the block
            raise
        return self._backend.end_residual(self._handle)    # on failure the backend has already discarded it

    # -- introspection ------------------------------------------------------
    @property
    def input_shape(self) -> Shape:
        return self._input_shape

    @property
    def output_shape(self) -> Shape:
        return self._shapes[-1] if self._shapes else self._input_shape

    @property
    def layers(self) -> Tuple[Layer, ...]:
        return tuple(self._layers)

    @property
    def epoch(self) -> int:
        """Epochs trained so far (drives the learning schedule)."""
        return self._epoch

    @property
    def r2(self) -> Dict[str, float]:
        """The R2 rule parameters of the model: ``tau``, ``beta``, ``inc``, ``zero_target``, ``reset``, ``head_beta2`` (see :meth:`fit`)."""
        self._check_open()
        vals = tuple(self._backend.get_update_rule(self._handle)) + (self._backend.get_head_beta2(self._handle),)
        # stored as float32: shown with 7 significant digits
        return {n: float(f"{v:.7g}") for n, v in zip(("tau", "beta", "inc", "zero_target", "reset", "head_beta2"), vals)}

    @property
    def input(self) -> str:
        """``"ternary"`` (default: x is ternarized with ``threshold``) or ``"uint8"`` (x is kept as float and
        quantized to uint8, with a scale and zero-point per sample)."""
        return self._input

    @input.setter
    def input(self, value: str) -> None:
        self._input = self._check_input(value)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = self._check_threshold(value)

    @property
    def closed(self) -> bool:
        return self._handle is None

    def summary(self) -> str:
        rows = [("#", "layer", "output", "activation")]

        def add_rows(prefix: str, layers: Sequence[Layer], shapes: Optional[Sequence[Shape]], indent: str) -> None:
            for i, layer in enumerate(layers, 1):
                number = f"{prefix}{i}"
                shape = str(shapes[i - 1]) if shapes is not None else ""
                if isinstance(layer, Residual):
                    rows.append((number, f"{indent}Residual", shape, layer.activation))
                    add_rows(number + ".", layer.layers, None, indent + "  ")
                else:
                    rows.append((number, indent + repr(layer), shape, layer.activation))

        add_rows("", self._layers, self._shapes, "")
        widths = [max(len(r[c]) for r in rows) for c in range(4)]
        lines = [f"Model  input={self._input_shape}  epoch={self._epoch}"]
        lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in rows]
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"epoch={self._epoch}"
        return (f"<ternaryai.Model input={self._input_shape} layers={len(self._layers)} {state}>")

    # -- data preparation ---------------------------------------------------
    def _prepare_x(self, x: Any) -> np.ndarray:
        if self._input == "uint8":
            return self._prepare_xf(x)
        arr = np.asarray(x)
        if arr.dtype == object or not (np.issubdtype(arr.dtype, np.number)
                                       or arr.dtype == np.bool_):
            raise DataError("x must be a numeric array")
        features = int(np.prod(self._input_shape))
        if arr.ndim >= 1 and arr.shape[1:] in (self._input_shape, (features,)) \
                and arr.shape[0] >= 1:
            n = arr.shape[0]
        else:
            raise DataError(f"x must have shape (n, {features}) or (n, *{self._input_shape}); "
                            f"got {arr.shape} (for a single sample use x[None])")
        arr = arr.reshape(n, features)
        if not np.all(np.isfinite(arr)):
            raise DataError("x contains NaN or infinite values")
        if np.all((arr == -1) | (arr == 0) | (arr == 1)):
            out = arr.astype(np.int8)
        else:
            out = np.where(arr > self._threshold, 1, np.where(arr < -self._threshold, -1, 0))
        return np.ascontiguousarray(out, dtype=np.int8)

    def _prepare_xf(self, x: Any) -> np.ndarray:
        """x as C-contiguous float32 (n, features) for the uint8 input."""
        arr = np.asarray(x)
        if arr.dtype == object or not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
            raise DataError("x must be a numeric array")
        features = int(np.prod(self._input_shape))
        if arr.ndim >= 1 and arr.shape[1:] in (self._input_shape, (features,)) and arr.shape[0] >= 1:
            n = arr.shape[0]
        else:
            raise DataError(f"x must have shape (n, {features}) or (n, *{self._input_shape}); "
                            f"got {arr.shape} (for a single sample use x[None])")
        arr = arr.reshape(n, features)
        if not np.all(np.isfinite(arr)):
            raise DataError("x contains NaN or infinite values")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _takes_ids(self) -> bool:
        return bool(self._layers) and self._backend.takes_ids(self._handle)

    def _prepare_ids(self, a: Any, name: str, like: Optional[np.ndarray] = None) -> np.ndarray:
        """Integer ids/targets as C-contiguous int32 (n, seq); ``like`` is the ids array to match."""
        arr = np.asarray(a)
        if arr.dtype == object or not (np.issubdtype(arr.dtype, np.integer)
                                       or arr.dtype == np.bool_):
            raise DataError(f"{name} must be an integer array of token ids, got dtype {arr.dtype}")
        seq = int(np.prod(self._input_shape))
        if like is not None:
            if arr.shape != like.shape:
                raise DataError(f"y must have the same shape as x {like.shape}, got {arr.shape}")
        elif arr.ndim != 2 or arr.shape[1] != seq or arr.shape[0] < 1:
            raise DataError(f"{name} must have shape (n, {seq}); got {arr.shape} "
                            "(for a single sequence use x[None])")
        info = np.iinfo(np.int32)
        if arr.size and (arr.min() < info.min or arr.max() > info.max):
            raise DataError(f"{name} has values that do not fit in int32")
        return np.ascontiguousarray(arr, dtype=np.int32)

    def _prepare_y(self, y: Any, n: int) -> np.ndarray:
        arr = np.asarray(y)
        if arr.dtype == object or not (np.issubdtype(arr.dtype, np.number)
                                       or arr.dtype == np.bool_):
            raise DataError("y must be a numeric array")
        outputs = int(np.prod(self.output_shape))
        if arr.ndim == 1 and outputs == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim >= 1 and arr.shape[1:] in (self.output_shape, (outputs,)):
            arr = arr.reshape(arr.shape[0], outputs)
        else:
            raise DataError(f"y must have shape (n, {outputs}) or (n, *{self.output_shape}); "
                            f"got {np.asarray(y).shape}")
        if arr.shape[0] != n:
            raise DataError(f"x has {n} samples but y has {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise DataError("y contains NaN or infinite values")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _check_open(self) -> None:
        if self._handle is None:
            raise TernaryAIError("this model is closed")

    def _check_ready(self) -> None:
        self._check_open()
        if not self._layers:
            raise ArchitectureError("the model has no layers; add some with model.add(...)")

    # -- training and inference --------------------------------------------
    def fit(self, x: Any, y: Any, epochs: int = 100, *, initial_epoch: Optional[int] = None,
            verbose: int = 0, loss: Optional[str] = None, scale_lr: Optional[float] = None,
            restore_best: Optional[bool] = None, batch_size: Optional[int] = None, shuffle: bool = True,
            shuffle_seed: int = 0, tau: Optional[float] = None, beta: Optional[float] = None,
            inc: Optional[float] = None, zero_target: Optional[float] = None,
            reset: Optional[float] = None, head_beta2: Optional[float] = None) -> History:
        """Train for ``epochs`` epochs and return a :class:`History`.

        By default each epoch is one full-batch step. ``batch_size=B`` (``B < n``) makes it mini-batch: every epoch
        walks the samples in steps of ``B`` (the last step may be smaller); each step averages the gradient of its own
        batch and applies its own R2 update and scale update, so there are ``ceil(n / B)`` updates per epoch. ``shuffle`` (default True) reshuffles the
        samples every epoch with a permutation that depends only on ``(shuffle_seed, global epoch)``: the result does
        not depend on how the fit is split (``verbose``) and is reproducible. ``batch_size=None`` or ``>= n`` is the
        full batch, bit for bit what earlier versions did (no shuffle). The reported loss/accuracy of an epoch is the
        mean over all its samples as they were seen along the epoch. With ``restore_best`` the state an epoch's loss is
        attributed to is the one at the *start* of that epoch.

        ``loss`` is ``"mse"`` (float targets, one row per sample; the default for
        ternary-input models) or ``"xent"`` (cross-entropy, the default and only
        choice for token-id models): then ``x`` is an integer array of ids
        ``(n, seq)`` and ``y`` the id expected at each position ``(n, seq)``
        -- normally the next token; a negative value ignores that position
        (padding). The hyperparameters left as ``None`` take the defaults of the
        chosen loss (see ``_LOSS_DEFAULTS``).

        Successive calls continue the learning schedule; pass ``initial_epoch``
        to override where it resumes (e.g. after ``load``). ``verbose=N`` prints
        the loss every N epochs.

        Weights are updated with the R2 rule: an int8 counter per weight (1 byte) integrates the descent direction, fed with
        the gradient normalized by the RMS of its row (clipped to +-3), almost without decay (``beta``, default 0.999). When the
        counter reaches the threshold ``tau`` (a fraction of its range, default 0.7) the weight moves one step and the counter
        restarts on the *opposite* side (hysteresis, at ``reset`` times the threshold, default 0.5); a per-row target of
        ``zero_target`` zeros (default 0.2, negative = off) adjusts the thresholds of 0 -> +-1 and +-1 -> 0, and ``inc``
        (default 0.03) is the counter step per unit of normalized gradient. It covers every ternary layer (Dense, Conv2D, GQA,
        SwiGLU, RMSNorm, LMHead, Embedding) and the biases. The LMHead is the exception to the per-row normalization: its gradient is
        normalized per *weight* by the RMS of its own history (``head_beta2``, default 0.99 = memory of the moving average of g^2,
        like RMSprop with bias correction; 0 turns it off and leaves the per-row R2), which removes the common-mode drift of the
        rows of rare tokens. That state is kept in 4 bits per weight (logarithmic code, one float scale per row) and travels in the
        training checkpoint. The six values live in the model: an argument left as ``None``
        keeps the current value (the defaults for a new model, the saved ones after :func:`load` of a training checkpoint);
        a value that is passed replaces it from this call on. :attr:`Model.r2` reads them. They are saved by ``save()`` (the
        training checkpoint), not by ``export()``.

        ``restore_best`` (``loss="xent"`` only; default True): at the end of each
        ``fit`` call the model returns to the state that had the lowest training
        loss (``verbose`` only splits the printing: the best state is kept across
        its chunks, so the result does not depend on ``verbose``). It guards against the late collapse
        that can follow the threshold valley (attention solutions that had
        converged get destroyed). Selection uses the training loss on the same
        data; pass ``restore_best=False`` to keep the last state.
        """
        self._check_ready()
        ids_model = self._takes_ids()
        if loss is None:
            loss = "xent" if ids_model else "mse"
        if loss not in _LOSS_DEFAULTS:
            raise ValueError(f"loss must be 'mse' or 'xent', got {loss!r}")
        if ids_model and loss != "xent":
            raise DataError("this model takes token ids (its first layer is an Embedding): "
                            "train it with loss='xent'")
        if not ids_model and loss == "xent":
            raise DataError("loss='xent' needs a model whose first layer takes token ids "
                            "(an Embedding); this one takes ternary inputs")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer))
                                       or batch_size < 1):
            raise ValueError(f"batch_size must be an int >= 1 (or None for the full batch), got {batch_size!r}")
        if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, (int, np.integer)) or shuffle_seed < 0:
            raise ValueError(f"shuffle_seed must be an int >= 0, got {shuffle_seed!r}")
        defaults = _LOSS_DEFAULTS[loss]
        if scale_lr is None:
            scale_lr = defaults["scale_lr"]
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError(f"epochs must be an int >= 1, got {epochs!r}")
        epochs = int(epochs)
        if isinstance(verbose, bool):
            verbose = 10 if verbose else 0
        if not isinstance(verbose, (int, np.integer)) or verbose < 0:
            raise ValueError(f"verbose must be an int >= 0, got {verbose!r}")
        first = self._epoch if initial_epoch is None else initial_epoch
        if isinstance(first, bool) or not isinstance(first, (int, np.integer)) or first < 0:
            raise ValueError(f"initial_epoch must be an int >= 0, got {first!r}")
        first = int(first)
        hyper = {"scale_lr": _positive("scale_lr", scale_lr)}

        if ids_model:
            xs = self._prepare_ids(x, "x")
            ys = self._prepare_ids(y, "y", like=xs)
        else:
            xs = self._prepare_x(x)
            ys = self._prepare_y(y, xs.shape[0])
        if restore_best is True and not ids_model:
            raise ValueError("restore_best only applies to loss='xent' (token-id models)")
        use_restore = ids_model and (True if restore_best is None else bool(restore_best))
        cur = tuple(self._backend.get_update_rule(self._handle)) + (self._backend.get_head_beta2(self._handle),)
        new = {"tau": tau, "beta": beta, "inc": inc, "zero_target": zero_target, "reset": reset, "head_beta2": head_beta2}
        for (name, v), c in zip(new.items(), cur):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating))
                                  or not math.isfinite(float(v))):
                raise ValueError(f"{name} must be a number, got {v!r}")
        vals = {n: (c if v is None else float(v)) for (n, v), c in zip(new.items(), cur)}
        if not (0.0 < vals["tau"] <= 1.0):
            raise ValueError(f"tau must be a number in (0, 1], got {tau!r}")
        if not (0.0 <= vals["beta"] < 1.0):
            raise ValueError(f"beta must be a number in [0, 1), got {beta!r}")
        if not vals["inc"] > 0.0:
            raise ValueError(f"inc must be > 0, got {inc!r}")
        if not vals["zero_target"] < 1.0:
            raise ValueError(f"zero_target must be < 1 (negative turns it off), got {zero_target!r}")
        if not (0.0 <= vals["reset"] <= 1.0):
            raise ValueError(f"reset must be a number in [0, 1], got {reset!r}")
        if not (0.0 <= vals["head_beta2"] < 1.0):
            raise ValueError(f"head_beta2 must be a number in [0, 1) (0 turns it off), got {head_beta2!r}")
        self._backend.set_update_rule(self._handle, vals["tau"], vals["beta"])
        self._backend.set_head_beta2(self._handle, vals["head_beta2"])
        self._backend.set_r2_params(self._handle, vals["inc"], vals["zero_target"], vals["reset"])
        # always set (also None -> 0): the setting lives in the model and must not leak from an earlier fit
        self._backend.set_batching(self._handle, 0 if batch_size is None else int(batch_size), bool(shuffle),
                                   int(shuffle_seed))
        if ids_model:
            self._backend.set_restore_best(self._handle, use_restore)
        chunk = int(verbose) or epochs
        # With verbose the epochs run in chunks. The best state is held ACROSS the chunks and restored once at the
        # end of the whole fit, so verbose never changes the result (restoring per chunk would replay the same
        # chunk forever whenever one chunk fails to improve on its starting state).
        held = bool(use_restore and chunk < epochs)
        if ids_model:
            self._backend.hold_best(self._handle, held)   # also clears anything left by an interrupted fit
        losses: List[np.ndarray] = []
        accs: List[np.ndarray] = []
        done = 0
        try:
            while done < epochs:
                count = min(chunk, epochs - done)
                if ids_model:
                    part, acc = self._backend.train_epochs_ids(self._handle, xs, ys, first + done,
                                                               count, hyper)
                    acc = np.asarray(acc, dtype=np.float64)
                    if acc.shape != (count,):
                        raise TernaryAIError(f"backend returned {acc.shape} accuracies for {count} epochs")
                    accs.append(acc)
                else:
                    train = (self._backend.train_epochs_f32 if self._input == "uint8"
                             else self._backend.train_epochs)
                    part = train(self._handle, xs, ys, first + done, count, hyper)
                part = np.asarray(part, dtype=np.float64)
                if part.shape != (count,):
                    raise TernaryAIError(f"backend returned {part.shape} losses for {count} epochs")
                losses.append(part)
                done += count
                if verbose:
                    extra = f"  accuracy {accs[-1].mean():.4f}" if ids_model else ""
                    print(f"epoch {done}/{epochs}  loss {part.mean():.6f}{extra}")
        except BaseException:
            if held:
                self._backend.hold_best(self._handle, False)
            raise
        if held:
            self._backend.restore_held_best(self._handle, float(losses[-1][-1]))
        self._epoch = first + epochs
        return History(np.concatenate(losses), first, np.concatenate(accs) if ids_model else None)

    def ffn_sparsity(self, reset: bool = False) -> list:
        """Fraction of the feed-forward's uint8 activation (the input of W_down) that is exactly its zero-point,
        one value per SwiGLU layer, accumulated over the forwards run since the last reset (``reset=True`` clears).
        Same definition for every ``gate_mode``. Layers that have not run yet give ``nan``."""
        self._check_ready()
        return [(z / t if t else float("nan")) for z, t in self._backend.ffn_zero_stats(self._handle, bool(reset))]

    def attention_probs(self, layer: int, head: int) -> np.ndarray:
        """Attention probabilities ``(seq, seq)`` of GQA layer ``layer`` (traversal order) and query head ``head`` as used by
        the LAST forward (call ``predict`` with ONE sequence first). Row i holds the causal weights over keys j <= i."""
        self._check_ready()
        seq = self.input_shape[0] if hasattr(self, "input_shape") else None
        if seq is None: raise TernaryAIError("unknown sequence length")
        return self._backend.attention_probs(self._handle, int(layer), int(head), int(seq))

    def num_gqa(self) -> int:
        """Number of GQA layers in the model (traversal order, residual blocks included)."""
        self._check_ready()
        return self._backend.num_gqa(self._handle)

    def prof_enable(self, on: bool = True) -> None:
        """Turn the scope profiler on/off. Call it from the thread that trains. It never changes results."""
        self._backend.prof_enable(bool(on))

    def prof_reset(self) -> None:
        """Zero the profiler's call counts and times (the tree is kept)."""
        self._backend.prof_reset()

    def prof_report(self) -> dict:
        """The profiler's call tree: ``{"overhead_ns_per_scope": x, "nodes": [{id, parent, name, idx, calls,
        total_ns, self_ns}, ...]}``. ``total_ns`` includes children, ``self_ns`` does not."""
        import json
        return json.loads(self._backend.prof_dump())

    def set_checkpoint(self, on: bool = True) -> None:
        """Enable/disable block checkpointing (less activation memory, same bits)."""
        self._check_ready()
        self._backend.set_checkpoint(self._handle, bool(on))

    def predict(self, x: Any, *, sign: bool = False) -> np.ndarray:
        """Outputs for ``x``: float32 (n, *output_shape), or int8 -1/0/1 with ``sign=True``.

        For token-id models ``x`` is an integer array of ids ``(n, seq)`` and the
        result holds the logits (no softmax) ``(n, seq, vocab)``; the predicted id
        at each position is ``argmax`` of the last axis.
        """
        self._check_ready()
        if self._takes_ids():
            xs = self._prepare_ids(x, "x")
            out = np.asarray(self._backend.predict_ids(self._handle, xs), dtype=np.float32)
        else:
            xs = self._prepare_x(x)
            run = self._backend.predict_f32 if self._input == "uint8" else self._backend.predict
            out = np.asarray(run(self._handle, xs), dtype=np.float32)
        outputs = int(np.prod(self.output_shape))
        if out.shape != (xs.shape[0], outputs):
            raise TernaryAIError(f"backend returned {out.shape}, expected {(xs.shape[0], outputs)}")
        out = out.reshape((xs.shape[0],) + self.output_shape)
        return np.sign(out).astype(np.int8) if sign else out

    # -- persistence --------------------------------------------------------
    def save(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """Full checkpoint into the folder ``path``: ``model.ternary`` + ``config.json`` + ``training_state.ternary``.

        It can be loaded and trained further."""
        self._check_ready()
        self._backend.save(self._handle, os.fspath(path), True)

    def export(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """Inference-only export into the folder ``path``: ``model.ternary`` (weights, 5 trits per byte) + ``config.json``."""
        self._check_ready()
        self._backend.save(self._handle, os.fspath(path), False)

    def engine(self, tokenizer: "Any" = None) -> "Any":
        """An inference :class:`ternaryai.Engine` for this model (exports it to a temporary folder that the engine removes on close).
        Pass the :class:`ternaryai.Tokenizer` to get text features (``encode``/``decode``/``reply``/``sample``).

        The engine is a snapshot: later training does not change it."""
        import tempfile
        from .infer import Engine
        self._check_ready()
        tmp = tempfile.mkdtemp(prefix="ternaryai_engine_")
        try:
            self.export(tmp)
            if tokenizer is not None:
                tokenizer.save(os.path.join(tmp, "tokenizer.json"))
            eng = Engine(tmp)
        except BaseException:
            import shutil; shutil.rmtree(tmp, ignore_errors=True); raise
        eng._tmp = tmp
        return eng

    def export_ternam(self, path: Any, target: str = "auto", tokenizer: "Any" = None) -> str:
        """Write this model as a self-contained ``.ternam`` file for the inference engine (see :func:`ternaryai.export_ternam`).
        Pass the :class:`ternaryai.Tokenizer` to embed it in the file: without it the engine cannot use text features (``reply``, ``encode``, ...)."""
        import shutil
        import tempfile
        from .infer import export_ternam
        self._check_ready()
        tmp = tempfile.mkdtemp(prefix="ternaryai_export_")
        try:
            self.export(tmp)
            if tokenizer is not None:
                tokenizer.save(os.path.join(tmp, "tokenizer.json"))
            return export_ternam(tmp, path, target)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def close(self) -> None:
        """Release the native model. Safe to call more than once."""
        handle, self._handle = self._handle, None
        if handle is not None:
            self._backend.free(handle)

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load(path: Union[str, "os.PathLike[str]"], *, threshold: float = 0.5, input: str = "ternary",
         backend: Optional[Backend] = None) -> Model:
    """Load a model folder saved with :meth:`Model.save` or :meth:`Model.export`.

    The input ``threshold`` and the input mode (``input="ternary"`` or ``"uint8"``) are not stored in model files;
    pass them again if you used non-default ones. Models saved with the removed ``sign`` / ``band`` activations
    do not load (retrain them).
    """
    backend = backend or get_backend()
    handle = backend.load(os.fspath(path))
    try:
        return Model._from_description(backend, handle, backend.describe(handle), threshold, input)
    except BaseException:
        backend.free(handle)
        raise
