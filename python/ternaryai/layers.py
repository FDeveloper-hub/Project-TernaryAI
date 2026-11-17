"""Layer objects, generated from backend specs."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from ._spec import LayerSpec


#: Parameters that only matter when a layer is built (e.g. the initial scale ``out_scale``): the library does not
#: store them in model files, so a loaded layer reports their default and they are ignored when comparing layers.
BUILD_ONLY_PARAMS = frozenset({"out_scale"})


class Layer:
    """Immutable description of a layer to add to a :class:`ternaryai.Model`.

    Concrete classes (``ternaryai.Dense``, ...) are generated from the layer
    specs the backend reports; see :func:`ternaryai.available_layers`.
    """

    def __init__(self, spec: LayerSpec, params: Dict[str, Any], activation: str) -> None:
        self._spec = spec
        self._params = dict(params)
        self._activation = activation

    @property
    def spec(self) -> LayerSpec:
        return self._spec

    @property
    def type_name(self) -> str:
        return self._spec.type_name

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def activation(self) -> str:
        return self._activation

    def __repr__(self) -> str:
        parts = [f"{p.name}={self._params[p.name]!r}" for p in self._spec.params
                 if p.required or self._params[p.name] != p.default]
        if self._activation != self._spec.default_activation:
            parts.append(f"activation={self._activation!r}")
        return f"{self._spec.class_name}({', '.join(parts)})"

    def _identity(self) -> Dict[str, Any]:
        """Parameters that identify the layer: those that survive save/load (not the build-only ones)."""
        return {k: v for k, v in self._params.items() if k not in BUILD_ONLY_PARAMS}

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Layer) and self.type_name == other.type_name
                and self._identity() == other._identity() and self._activation == other._activation)

    def __hash__(self) -> int:
        return hash((self.type_name, tuple(self._identity().items()), self._activation))


def make_layer_class(spec: LayerSpec) -> type:
    """Build the public layer class for ``spec``."""
    sig = spec.signature()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        activation = values.pop("activation")
        if not isinstance(activation, str):
            raise TypeError(f"activation must be a str (its name), got {activation!r}")
        Layer.__init__(self, spec, spec.coerce_params(values), activation)

    lines = [spec.doc or f"{spec.class_name} layer.", "", "Parameters:"]
    for p in spec.params:
        default = "required" if p.required else f"default {p.default!r}"
        lines.append(f"    {p.name} ({p.kind}, {default}) {p.doc}".rstrip())
    lines.append(f"    activation (str, default {spec.default_activation!r}) "
                 "applied to this layer's output")
    return type(spec.class_name, (Layer,), {
        "__init__": __init__,
        "__doc__": "\n".join(lines),
        "__module__": "ternaryai",
        "__signature__": sig,
    })


class Residual(Layer):
    """A residual block: ``out = x + f(x)``, where ``f`` is the chain of ``layers``.

    >>> tn.Residual([tn.RMSNorm(), tn.GQA(4, 2)])

    The input of the block is the float output of the layer before it (a pre-norm
    transformer sub-block, for example). The rules are checked when the block is added
    to a model and break with a message that says which one: the first layer of the
    chain must take float input (like ``RMSNorm``), every layer must pass gradient to its
    input, none may take token ids, and the last one must return as many values as ``x``.
    ``activation`` is applied to the block's output before the next layer (``"uint8"`` by
    default, as in the rest of the library; it is not used if the next layer takes float
    input). Blocks can be nested: a ``Residual`` can be one of the ``layers`` of another.
    """

    def __init__(self, layers: Iterable[Layer], activation: str = "uint8") -> None:
        chain = tuple(layers)
        if not chain:
            raise ValueError("a Residual needs at least one layer")
        for item in chain:
            if not isinstance(item, Layer):
                raise TypeError(f"Residual layers must be layers such as ternaryai.RMSNorm(), got {item!r}")
        if not isinstance(activation, str):
            raise TypeError(f"activation must be a str (its name), got {activation!r}")
        self._layers: Tuple[Layer, ...] = chain
        self._spec = None
        self._params: Dict[str, Any] = {}
        self._activation = activation

    @property
    def type_name(self) -> str:
        return "residual"

    @property
    def layers(self) -> Tuple[Layer, ...]:
        return self._layers

    def __repr__(self) -> str:
        inner = ", ".join(repr(l) for l in self._layers)
        act = "" if self._activation == "uint8" else f", activation={self._activation!r}"
        return f"Residual([{inner}]{act})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Residual) and self._layers == other._layers
                and self._activation == other._activation)

    def __hash__(self) -> int:
        return hash(("residual", self._layers, self._activation))
