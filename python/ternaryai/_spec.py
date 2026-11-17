"""Layer specifications.

A backend (the C library, in production) describes every layer type it
implements with a :class:`LayerSpec`. Python builds ``ternaryai.Dense``,
``ternaryai.Conv2D``, ... from those specs, so adding a layer type in C makes it
available in Python without writing any Python code for it.
"""
from __future__ import annotations

import inspect
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

REQUIRED = inspect.Parameter.empty
KINDS = ("int", "float", "bool", "str", "pair")


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class Param:
    """One constructor parameter of a layer.

    kind: "int", "float", "bool", "str", or "pair" (an int or a pair of ints,
    always normalised to a 2-tuple, e.g. a kernel size).
    """

    name: str
    kind: str
    default: Any = REQUIRED
    min: Optional[float] = None
    doc: str = ""
    max: Optional[float] = None

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def coerce(self, value: Any) -> Any:
        kind = self.kind
        if kind == "int":
            if not _is_int(value):
                raise TypeError(f"{self.name} must be an int, got {value!r}")
            value = int(value)
        elif kind == "float":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise TypeError(f"{self.name} must be a number, got {value!r}")
            value = float(value)
        elif kind == "bool":
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{self.name} must be a bool, got {value!r}")
            value = bool(value)
        elif kind == "str":
            if not isinstance(value, str):
                raise TypeError(f"{self.name} must be a str, got {value!r}")
        else:  # pair
            if _is_int(value):
                value = (int(value), int(value))
            elif (isinstance(value, (tuple, list)) and len(value) == 2
                  and all(_is_int(v) for v in value)):
                value = (int(value[0]), int(value[1]))
            else:
                raise TypeError(f"{self.name} must be an int or a pair of ints, got {value!r}")
        if self.min is not None:
            parts = value if kind == "pair" else (value,)
            if any(v < self.min for v in parts):
                raise ValueError(f"{self.name} must be >= {self.min}, got {value!r}")
        if self.max is not None:
            parts = value if kind == "pair" else (value,)
            if any(v > self.max for v in parts):
                raise ValueError(f"{self.name} must be <= {self.max}, got {value!r}")
        return value


@dataclass(frozen=True)
class LayerSpec:
    """Description of one layer type: its C name, Python class name and parameters."""

    type_name: str            # name stored in model files, e.g. "dense"
    class_name: str           # Python class name, e.g. "Dense"
    params: Tuple[Param, ...]
    doc: str = ""
    default_activation: str = "uint8"

    def __post_init__(self) -> None:
        if not self.class_name.isidentifier():
            raise ValueError(f"class_name {self.class_name!r} is not a valid identifier")
        seen = set()
        optional_seen = False
        for p in self.params:
            if p.kind not in KINDS:
                raise ValueError(f"param {p.name!r}: unknown kind {p.kind!r}")
            if not p.name.isidentifier() or p.name == "activation" or p.name in seen:
                raise ValueError(f"param name {p.name!r} is invalid or duplicated")
            seen.add(p.name)
            if p.required and optional_seen:
                raise ValueError(f"required param {p.name!r} cannot follow an optional one")
            optional_seen = optional_seen or not p.required

    def signature(self) -> inspect.Signature:
        """Python signature of the generated layer class (params + keyword-only activation)."""
        params = [
            inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=p.default)
            for p in self.params
        ]
        params.append(inspect.Parameter("activation", inspect.Parameter.KEYWORD_ONLY,
                                        default=self.default_activation))
        return inspect.Signature(params)

    def coerce_params(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a dict of parameter values; fill defaults, return them in spec order."""
        unknown = set(values) - {p.name for p in self.params}
        if unknown:
            raise TypeError(f"{self.class_name}: unknown parameter(s) {sorted(unknown)}")
        out = {}
        for p in self.params:
            if p.name in values:
                out[p.name] = p.coerce(values[p.name])
            elif p.required:
                raise TypeError(f"{self.class_name}: missing required parameter {p.name!r}")
            else:
                out[p.name] = p.default
        return out
