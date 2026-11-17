"""Ready-made blocks built from the generic pieces (see :class:`ternaryai.Residual`)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Initial scale of each branch's output matrix when ``out_scale`` is not given (see ``transformer_block``).
TRANSFORMER_OUT_SCALE = 0.03

from ._backend import Backend, get_backend
from ._registry import classes_for
from .layers import Layer, Residual


def transformer_block(heads: int, hidden: int, *, kv_heads: Optional[int] = None,
                      rope: Optional[int] = None,
                      rope_base: Optional[float] = None, score_shift: Optional[int] = None,
                      gate_mode: Optional[int] = None,
                      band: Optional[float] = None, eps: Optional[float] = None,
                      out_scale: Optional[float] = None,
                      window: Optional[int] = None, sinks: Optional[int] = None, smear: Optional[bool] = None,
                      norm_activation: str = "uint8",
                      backend: Optional[Backend] = None) -> List[Layer]:
    """The two sub-blocks of a pre-norm transformer layer, ready to put in a model.

    Returns ``[Residual([RMSNorm, GQA]), Residual([RMSNorm, SwiGLU])]``, i.e.

        x = x + GQA(norm(x))
        x = x + SwiGLU(norm(x))

    >>> model = tn.Model(seq, [tn.Embedding(vocab, 64),
    ...                        *tn.transformer_block(heads=4, hidden=256, kv_heads=2),
    ...                        *tn.transformer_block(heads=4, hidden=256, kv_heads=2),
    ...                        tn.RMSNorm(), tn.LMHead(vocab)])

    The width of the block is that of its input (the embedding dimension). Measured: with 3 or more blocks use an
    embedding dimension of at least 32 (at 16 most seeds collapse late in training; measured with hidden=32, see the README). Every
    argument left as ``None`` takes the default of the layer it belongs to:

    * ``heads``, ``kv_heads``, ``rope``, ``rope_base``, ``score_shift``: :class:`GQA`
      (``kv_heads=None`` = same as ``heads``);
    * ``window``, ``sinks``: sliding-window attention (see :class:`GQA`; needs ``rope=1``, the default);
    * ``smear``: smeared keys, K'_j = (K_j + K_{j-1}) >> 1 before RoPE (see :class:`GQA`); lets one layer match a token against the
      one that precedes each key, which is what an induction head needs. Needs ``rope=1`` and ``window=0``. Default off;
    * ``hidden``, ``gate_mode``, ``band``: :class:`SwiGLU` (``gate_mode=2`` = ReLU² without a gate, BitNet style:
      two matrices instead of three; optional, the default stays ``0`` = silu);
    * ``eps``: both :class:`RMSNorm`; ``norm_activation`` is the activation of their output
      (``"uint8"``, the only one: 255 levels with a scale and zero-point per token);
    * ``out_scale``: initial scale of the output matrix of each branch (GQA's Wo, SwiGLU's down projection);
      the block's default is 0.03 (not the layers' own 1.0): a branch that starts small next to the residual stream
      makes deeper models train far more reliably, and the scale is learned from there.

    The blocks are plain descriptions, so the list can be repeated for a deeper model
    (``[*tn.transformer_block(...)] * 12``); each one gets its own weights.
    """
    classes = classes_for(backend or get_backend())
    if out_scale is None:
        out_scale = TRANSFORMER_OUT_SCALE

    def pick(**kw: Any) -> Dict[str, Any]:
        return {k: v for k, v in kw.items() if v is not None}

    norm = lambda: classes["RMSNorm"](**pick(eps=eps), activation=norm_activation)
    attention = classes["GQA"](**pick(n_heads=heads, n_kv_heads=kv_heads,
                                      out_scale=out_scale, rope=rope, rope_base=rope_base,
                                      score_shift=score_shift, window=window, sinks=sinks, smear=smear))
    ffn = classes["SwiGLU"](**pick(hidden=hidden, gate_mode=gate_mode, band=band, out_scale=out_scale))
    return [Residual([norm(), attention]), Residual([norm(), ffn])]
