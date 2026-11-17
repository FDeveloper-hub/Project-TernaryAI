"""Layer classes generated from the active backend's layer specs."""
from __future__ import annotations

import weakref
from typing import Dict, List, Optional

from ._backend import Backend, get_backend
from .layers import Layer, make_layer_class

_cache: "weakref.WeakKeyDictionary[Backend, Dict[str, type]]" = weakref.WeakKeyDictionary()


def classes_for(backend: Backend) -> Dict[str, type]:
    """Map class_name and type_name to the generated layer class."""
    classes = _cache.get(backend)
    if classes is None:
        classes = {}
        for spec in backend.layer_types():
            cls = make_layer_class(spec)
            classes[spec.class_name] = cls
            classes[spec.type_name] = cls
        _cache[backend] = classes
    return classes


def available_layers(backend: Optional[Backend] = None) -> List[str]:
    """Class names of every layer type the backend implements."""
    backend = backend or get_backend()
    return sorted({cls.__name__ for cls in classes_for(backend).values()})


def layer(name: str, *args, **kwargs) -> Layer:
    """Build a layer by class or type name: ``layer("dense", 16)``."""
    classes = classes_for(get_backend())
    try:
        cls = classes[name]
    except KeyError:
        raise ValueError(f"unknown layer {name!r}; available: {available_layers()}") from None
    return cls(*args, **kwargs)
