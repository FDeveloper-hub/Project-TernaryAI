"""TernaryAI: neural networks whose weights are strictly -1, 0 or +1.

    import ternaryai as tn

    model = tn.Model(2, [tn.Dense(16), tn.Dense(1)])
    model.fit(X, y, epochs=100)
    model.predict(X, sign=True)

Layer classes (``tn.Dense``, ``tn.Conv2D``, ...) come from the compute
backend's layer registry, so a layer added in C is usable from Python as soon
as the library is rebuilt. See :func:`available_layers`.
"""
from ._backend import Backend, LayerDesc, ModelDescription, get_backend, set_backend
from ._registry import available_layers, classes_for, layer
from ._spec import LayerSpec, Param
from .errors import (ArchitectureError, BackendNotFoundError, DataError, ModelFileError,
                     TernaryAIError)
from .blocks import transformer_block
from .layers import Layer, Residual
from .model import History, Model, load
from .tokenizer import Tokenizer, TokenizerError, CHAT_SPECIAL_TOKENS
from . import reasoning
from .infer import Engine, Session, Reply, InferenceError, set_engine_backend, engine_backend, set_engine_threads, engine_threads, export_ternam, build_engine_lib, export_fused, FUSED_API, use_engine_lib, CTX_ERROR, CTX_TRUNCATE, CTX_SLIDE
from . import dataset


def set_threads(n: int) -> None:
    """Number of compute threads for the whole library (training and the inference engine). ``n <= 1`` = one thread (default;
    ``TERNARYAI_THREADS`` sets the initial value). Results are identical for any number of threads. Do not change it while training."""
    get_backend().set_threads(int(n))


def get_threads() -> int:
    """Current number of compute threads."""
    return get_backend().get_threads()

from .dataset import DatasetError

__version__ = "1.0.0a1"

__all__ = [
    "Model", "History", "load", "Layer", "Residual", "transformer_block", "layer", "available_layers",
    "Backend", "LayerSpec", "Param", "LayerDesc", "ModelDescription",
    "get_backend", "set_backend", "set_threads", "get_threads", "Tokenizer", "CHAT_SPECIAL_TOKENS", "dataset",
    "reasoning", "Engine", "Session", "Reply", "InferenceError", "set_engine_backend", "engine_backend", "set_engine_threads", "engine_threads", "export_ternam", "build_engine_lib", "export_fused", "FUSED_API", "use_engine_lib", "CTX_ERROR", "CTX_TRUNCATE", "CTX_SLIDE",
    "TernaryAIError", "ArchitectureError", "BackendNotFoundError", "DataError", "ModelFileError",
    "TokenizerError", "DatasetError",
]


def __getattr__(name):
    # Public layer classes (Dense, Conv2D, ...) are generated from the backend.
    if name.startswith("_") or not name[:1].isupper():
        raise AttributeError(f"module 'ternaryai' has no attribute {name!r}")
    classes = classes_for(get_backend())
    if name in classes and classes[name].__name__ == name:
        return classes[name]
    raise AttributeError(f"module 'ternaryai' has no attribute {name!r}; "
                         f"available layers: {available_layers()}")


def __dir__():
    names = set(globals())
    try:
        names.update(available_layers())
    except BackendNotFoundError:
        pass
    return sorted(names)
