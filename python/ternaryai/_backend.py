"""Backend contract: everything TernaryAI needs from the compute engine.

The production backend is the C library (libternaryai) reached through ctypes.
Every method below maps to one flat C entry point (see python/README.md).
A backend owns all model state; Python only holds an opaque ``handle``.

Data conventions (already prepared by :class:`ternaryai.Model`):
  * ``x``: int8 array (n, features), C-contiguous, values in {-1, 0, 1} (ternary input, the default);
    ``train_epochs_f32`` / ``predict_f32`` take float32 instead (uint8 input: the model quantizes each sample).
  * ``y``: float32 array (n, outputs), C-contiguous.
  * shapes are plain tuples of ints, HWC for images.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ._spec import LayerSpec
from .errors import ArchitectureError, BackendNotFoundError, TernaryAIError


class LayerDesc(NamedTuple):
    type_name: str
    params: Dict[str, Any]
    activation: str
    output_shape: Tuple[int, ...]
    children: Tuple["LayerDesc", ...] = ()      # the chain inside a "residual" block


class ModelDescription(NamedTuple):
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    layers: Tuple[LayerDesc, ...]


class Backend(ABC):
    """Abstract compute backend."""

    name = "abstract"

    @abstractmethod
    def layer_types(self) -> Sequence[LayerSpec]:
        """Every layer type this backend implements (the C layer registry)."""

    @abstractmethod
    def activations(self) -> Sequence[str]:
        """Names of the available activations (e.g. ``"sign"``)."""

    @abstractmethod
    def create(self, input_shape: Tuple[int, ...], seed: int) -> Any:
        """New empty model; ``seed`` drives the initial weights."""

    @abstractmethod
    def add_layer(self, handle: Any, type_name: str, params: Dict[str, Any],
                  activation: str) -> Tuple[int, ...]:
        """Append a layer and return its output shape.

        Must raise :class:`ternaryai.ArchitectureError` (leaving the model
        unchanged) if the layer cannot be added.
        """

    # -- residual blocks: x + f(x) -------------------------------------------
    # Optional, like the token-id methods below. ``begin_residual`` opens a block; the layers
    # added with :meth:`add_layer` until :meth:`end_residual` form its chain f (blocks can be
    # nested). While one is open the model cannot train, predict or be saved.
    def begin_residual(self, handle: Any, activation: str) -> None:
        raise ArchitectureError(f"the {self.name!r} backend does not support residual blocks")

    def end_residual(self, handle: Any) -> Tuple[int, ...]:
        """Close the innermost open block and add it to the model; return its output shape
        (the shape of x). On failure raise :class:`ternaryai.ArchitectureError`: the block is
        discarded and the model is as it was before :meth:`begin_residual`."""
        raise ArchitectureError(f"the {self.name!r} backend does not support residual blocks")

    def cancel_residual(self, handle: Any) -> None:
        """Discard the innermost open block, leaving the model as before :meth:`begin_residual`."""
        raise ArchitectureError(f"the {self.name!r} backend does not support residual blocks")

    @abstractmethod
    def describe(self, handle: Any) -> ModelDescription:
        """Architecture of the model (used after ``load`` and by ``summary``)."""

    @abstractmethod
    def train_epochs(self, handle: Any, x: np.ndarray, y: np.ndarray,
                     first_epoch: int, count: int, hyper: Dict[str, float]) -> np.ndarray:
        """Run ``count`` full-batch epochs, numbered from ``first_epoch``.

        ``hyper`` has ``scale_lr``. Returns a float64 array of
        length ``count``: the mean squared error of each epoch, measured on
        its forward passes (before that epoch's update).
        """

    @abstractmethod
    def predict(self, handle: Any, x: np.ndarray) -> np.ndarray:
        """Scaled float outputs of the last layer, float32 (n, outputs)."""

    # -- uint8 input (optional) -----------------------------------------------
    def train_epochs_f32(self, handle: Any, x: np.ndarray, y: np.ndarray,
                         first_epoch: int, count: int, hyper: Dict[str, float]) -> np.ndarray:
        """Like ``train_epochs`` but ``x`` is float32 (n, features): the model quantizes it to uint8."""
        raise TernaryAIError("this backend does not support uint8 input")

    def predict_f32(self, handle: Any, x: np.ndarray) -> np.ndarray:
        """Like ``predict`` but ``x`` is float32 (n, features), quantized to uint8 by the model."""
        raise TernaryAIError("this backend does not support uint8 input")

    # -- token-id models (language models) ----------------------------------
    # Optional: a backend without id support keeps these defaults, and
    # Model reports a clear error if someone builds an id model on it.
    def takes_ids(self, handle: Any) -> bool:
        """True if the model's first layer takes token ids (``Embedding``)."""
        return False

    def set_update_rule(self, handle: Any, tau: float, beta: float) -> None:
        """R2 rule parameters (``tau`` in (0, 1], ``beta`` in [0, 1)). A backend that trains must implement it."""
        raise TernaryAIError(f"the {self.name!r} backend does not support the R2 rule")

    def set_r2_params(self, handle: Any, inc: float, zero_target: float, reset: float) -> None:
        """Other R2 parameters (``inc`` > 0, ``zero_target`` < 1 (negative = off), ``reset`` in [0, 1])."""
        raise TernaryAIError(f"the {self.name!r} backend does not support the R2 rule")

    def get_update_rule(self, handle: Any) -> Tuple[float, float, float, float, float]:
        """Current ``(tau, beta, inc, zero_target, reset)``."""
        raise TernaryAIError(f"the {self.name!r} backend does not support the R2 rule")

    def set_head_beta2(self, handle: Any, head_beta2: float) -> None:
        """Memory (beta2, in [0, 1)) of the per-element normalization of the lm_head gradient (0 = off: per-row R2)."""
        raise TernaryAIError(f"the {self.name!r} backend does not support the R2 rule")

    def get_head_beta2(self, handle: Any) -> float:
        raise TernaryAIError(f"the {self.name!r} backend does not support the R2 rule")

    def set_batching(self, handle: Any, batch_size: int, shuffle: bool, seed: int) -> None:
        """Mini-batches for the next training calls: ``batch_size`` samples per weight update (0 = the whole set),
        reshuffled every epoch when ``shuffle``. A backend without support must refuse, not silently train full-batch."""
        if batch_size:
            raise TernaryAIError(f"the {self.name!r} backend does not support mini-batches")

    def set_restore_best(self, handle: Any, on: bool) -> None:
        """Id models: restore the lowest-training-loss state at the end of each training call."""

    def hold_best(self, handle: Any, on: bool) -> None:
        """Id models: keep the best state across training calls (the chunks of one ``fit`` with ``verbose``)
        instead of restoring it at the end of each call; ``on=False`` discards what was held."""

    def restore_held_best(self, handle: Any, last_loss: float) -> bool:
        """Id models: go back to the held best state if ``last_loss`` is worse, and stop holding.
        Returns True if the model was restored."""
        return False

    def ffn_zero_stats(self, handle: Any, reset: bool) -> list:
        """[(zeros, total)] per swiglu layer: uint8 activations of the feed-forward equal to their zero-point."""
        return []

    def attention_probs(self, handle: Any, gqa_index: int, head: int, seq: int) -> np.ndarray:
        """Attention probabilities (seq, seq) of one GQA layer/head from the last forward (native backend only)."""
        raise NotImplementedError("attention_probs needs the native backend")

    def num_gqa(self, handle: Any) -> int:
        return 0

    def set_threads(self, n: int) -> None:
        """Compute threads (native backend only; other backends run on one thread)."""

    def get_threads(self) -> int:
        return 1

    def prof_enable(self, on: bool) -> None:
        """Scope profiler on/off (native backend only)."""

    def prof_reset(self) -> None:
        """Zero the profiler counters."""

    def prof_dump(self) -> str:
        """The profiler tree as JSON text."""
        return '{"overhead_ns_per_scope":0,"nodes":[]}'

    def set_checkpoint(self, handle: Any, on: bool) -> None:
        """Block checkpointing: layers share one workspace and the backward recomputes each block's forward."""

    def train_epochs_ids(self, handle: Any, ids: np.ndarray, targets: np.ndarray,
                         first_epoch: int, count: int,
                         hyper: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Like :meth:`train_epochs` for id models, with cross-entropy loss.

        ``ids``: int32 (n, seq), C-contiguous. ``targets``: int32 (n, seq), the
        expected id at each position; a negative value ignores that position.
        Returns ``(loss, accuracy)``, float64 arrays of length ``count``: the
        cross-entropy and the fraction of non-ignored positions whose argmax is
        right, both measured on that epoch's forward passes.
        """
        raise TernaryAIError(f"the {self.name!r} backend does not support token-id models")

    def predict_ids(self, handle: Any, ids: np.ndarray) -> np.ndarray:
        """Logits (no softmax) of an id model, float32 (n, outputs)."""
        raise TernaryAIError(f"the {self.name!r} backend does not support token-id models")

    @abstractmethod
    def save(self, handle: Any, path: str, full_checkpoint: bool) -> None:
        """Write the model: a resumable checkpoint, or a minimal export."""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Read a model file; raise :class:`ternaryai.ModelFileError` if unusable."""

    @abstractmethod
    def free(self, handle: Any) -> None:
        """Release the model. Must be safe to call more than once."""


_current: Optional[Backend] = None


def set_backend(backend: Optional[Backend]) -> None:
    """Use ``backend`` for everything created from now on (``None`` resets to the default)."""
    global _current
    if backend is not None and not isinstance(backend, Backend):
        raise TypeError("backend must be a ternaryai.Backend instance or None")
    _current = backend


def get_backend() -> Backend:
    """The active backend; loads the default (C library) on first use."""
    global _current
    if _current is None:
        from . import _native
        _current = _native.load_default()
    if _current is None:  # pragma: no cover - load_default raises instead
        raise BackendNotFoundError("no backend available")
    return _current
