"""Tokenizer: fachada delgada en ctypes sobre libternaryai.

Toda la logica (BPE por bytes, entrenamiento, encode/decode, guardado/
carga) vive en C (src/ternary_tokenizer.c, expuesta como
ternaryai_tokenizer_* en include/ternaryai.h). Este modulo solo empaqueta
argumentos y desempaqueta errores -- no reimplementa nada.
"""
from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence, Union

from ._native import find_library
from .errors import BackendNotFoundError, TernaryAIError

_ERR_CAP = 512


# Tokens especiales de la conversacion: todo lo que el cargador de datasets (dataset.load) puede escribir como marca, mas los de
# razonamiento. Ids fijos 256, 257, ... en este orden (los tres primeros no cambian al agregar mas al final). Se pasan al construir:
#   tn.Tokenizer(1024, special_tokens=tn.CHAT_SPECIAL_TOKENS)
# <|think|> ... <|/think|> delimitan el razonamiento dentro del turno del asistente.
CHAT_SPECIAL_TOKENS = (
    "<|user|>", "<|assistant|>", "<|end|>", "<|system|>",
    "<|console|>", "<|terminal|>", "<|archive|>", "<|log|>", "<|result|>",
    "<|function_call|>", "<|tool|>",
    "<|think|>", "<|/think|>",
)


class TokenizerError(TernaryAIError):
    """Algo fallo dentro del tokenizador nativo (crear/entrenar/encode/decode/save/load)."""


def _load_lib() -> ctypes.CDLL:
    path = find_library()
    if path is None:
        raise BackendNotFoundError(
            "libternaryai no se encontro. Compilala (`make libternaryai.so`) y ponela "
            "junto al paquete ternaryai, o apunta $TERNARYAI_LIB a ella.")
    lib = ctypes.CDLL(path)
    c = ctypes
    lib.ternaryai_tokenizer_create.restype = c.c_void_p
    lib.ternaryai_tokenizer_create.argtypes = [c.c_size_t, c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_train.restype = c.c_int
    lib.ternaryai_tokenizer_train.argtypes = [c.c_void_p, c.POINTER(c.c_uint8), c.c_size_t,
                                               c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_vocab_size.restype = c.c_size_t
    lib.ternaryai_tokenizer_vocab_size.argtypes = [c.c_void_p]
    lib.ternaryai_tokenizer_num_merges.restype = c.c_size_t
    lib.ternaryai_tokenizer_num_merges.argtypes = [c.c_void_p]
    lib.ternaryai_tokenizer_encode.restype = c.c_int
    lib.ternaryai_tokenizer_encode.argtypes = [c.c_void_p, c.POINTER(c.c_uint8), c.c_size_t,
                                                c.POINTER(c.c_int32), c.c_size_t,
                                                c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_decode.restype = c.c_int
    lib.ternaryai_tokenizer_decode.argtypes = [c.c_void_p, c.POINTER(c.c_int32), c.c_size_t,
                                                c.POINTER(c.c_uint8), c.c_size_t,
                                                c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_add_special.restype = c.c_int
    lib.ternaryai_tokenizer_add_special.argtypes = [c.c_void_p, c.POINTER(c.c_uint8), c.c_size_t,
                                                     c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_num_special.restype = c.c_size_t
    lib.ternaryai_tokenizer_num_special.argtypes = [c.c_void_p]
    lib.ternaryai_tokenizer_special_text.restype = c.c_int
    lib.ternaryai_tokenizer_special_text.argtypes = [c.c_void_p, c.c_size_t, c.POINTER(c.c_uint8), c.c_size_t,
                                                      c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_special_id.restype = c.c_int32
    lib.ternaryai_tokenizer_special_id.argtypes = [c.c_void_p, c.POINTER(c.c_uint8), c.c_size_t]
    lib.ternaryai_tokenizer_encode_ex.restype = c.c_int
    lib.ternaryai_tokenizer_encode_ex.argtypes = [c.c_void_p, c.POINTER(c.c_uint8), c.c_size_t, c.c_int,
                                                   c.POINTER(c.c_int32), c.c_size_t,
                                                   c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_decode_ex.restype = c.c_int
    lib.ternaryai_tokenizer_decode_ex.argtypes = [c.c_void_p, c.POINTER(c.c_int32), c.c_size_t, c.c_int,
                                                   c.POINTER(c.c_uint8), c.c_size_t,
                                                   c.POINTER(c.c_size_t), c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_save.restype = c.c_int
    lib.ternaryai_tokenizer_save.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_load.restype = c.c_void_p
    lib.ternaryai_tokenizer_load.argtypes = [c.c_char_p, c.c_char_p, c.c_size_t]
    lib.ternaryai_tokenizer_free.restype = None
    lib.ternaryai_tokenizer_free.argtypes = [c.c_void_p]
    return lib


def _err_buf() -> ctypes.Array:
    return ctypes.create_string_buffer(_ERR_CAP)


def _raise(err: ctypes.Array) -> None:
    raise TokenizerError(err.value.decode("utf-8", "replace"))


def _to_bytes(text: Union[str, bytes, bytearray]) -> bytes:
    if isinstance(text, str):
        return text.encode("utf-8")
    if isinstance(text, (bytes, bytearray)):
        return bytes(text)
    raise TypeError(f"se esperaba str o bytes, se recibio {type(text).__name__}")


class Tokenizer:
    """Tokenizador BPE por bytes: sin OOV, se construye dando solo un
    tamano de vocabulario y un corpus para entrenar.

        tok = ternaryai.Tokenizer(vocab_size=4000)
        tok.train(texto_o_bytes)
        ids = tok.encode("hola mundo")
        tok.decode(ids) == "hola mundo"
        tok.save("tokenizer.json")
        tok2 = ternaryai.Tokenizer.load("tokenizer.json")

    Tokens especiales: cadenas reservadas con ids fijos (256, 257, ... en el
    orden dado) que el BPE nunca parte ni fusiona; se indican al construir,
    antes de entrenar. ``vocab_size`` cuenta 256 bytes + especiales + fusiones.

        tok = ternaryai.Tokenizer(512, special_tokens=["<|user|>", "<|assistant|>"])
        tok.train(corpus)
        tok.special_id("<|user|>")            # 256
        tok.encode("<|user|>\nHola")          # [256, ...]
        tok.encode(texto, parse_special=False) # trata "<|user|>" como texto corriente
        tok.decode(ids, skip_special=True)     # omite las marcas
    """

    def __init__(self, vocab_size: int, special_tokens: Optional[Sequence[str]] = None) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError(f"vocab_size debe ser int, se recibio {vocab_size!r}")
        self._lib = _load_lib()
        err = _err_buf()
        handle = self._lib.ternaryai_tokenizer_create(vocab_size, err, _ERR_CAP)
        if not handle:
            _raise(err)
        self._handle = handle
        self._trained = False
        for tok in (special_tokens or ()):
            if isinstance(tok, str) and tok == "":
                raise TokenizerError("un token especial no puede ser vacio")
            data = _to_bytes(tok)
            buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else None
            if not self._lib.ternaryai_tokenizer_add_special(self._handle, buf, len(data), err, _ERR_CAP):
                _raise(err)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        lib = getattr(self, "_lib", None)
        if handle and lib:
            lib.ternaryai_tokenizer_free(handle)

    def train(self, corpus: Union[str, bytes, bytearray]) -> "Tokenizer":
        data = _to_bytes(corpus)
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else None
        err = _err_buf()
        ok = self._lib.ternaryai_tokenizer_train(self._handle, buf, len(data), err, _ERR_CAP)
        if not ok:
            _raise(err)
        self._trained = True
        return self

    @property
    def vocab_size(self) -> int:
        return int(self._lib.ternaryai_tokenizer_vocab_size(self._handle))

    @property
    def num_merges(self) -> int:
        return int(self._lib.ternaryai_tokenizer_num_merges(self._handle))

    @property
    def num_special(self) -> int:
        return int(self._lib.ternaryai_tokenizer_num_special(self._handle))

    @property
    def special_tokens(self) -> List[str]:
        """Los tokens especiales en orden de id (el i-esimo tiene id 256+i)."""
        out = []
        err = _err_buf()
        for i in range(self.num_special):
            n = ctypes.c_size_t(0)
            if not self._lib.ternaryai_tokenizer_special_text(self._handle, i, None, 0, ctypes.byref(n), err, _ERR_CAP):
                _raise(err)
            buf = (ctypes.c_uint8 * n.value)()
            if not self._lib.ternaryai_tokenizer_special_text(self._handle, i, buf, n.value, ctypes.byref(n), err, _ERR_CAP):
                _raise(err)
            out.append(bytes(buf).decode("utf-8", "replace"))
        return out

    def special_id(self, token: Union[str, bytes, bytearray]) -> int:
        """Id del token especial 'token'; TokenizerError si no lo es."""
        data = _to_bytes(token)
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else None
        tid = int(self._lib.ternaryai_tokenizer_special_id(self._handle, buf, len(data))) if data else -1
        if tid < 0:
            raise TokenizerError(f"{token!r} no es un token especial de este tokenizador")
        return tid

    def encode(self, text: Union[str, bytes, bytearray], parse_special: bool = True) -> List[int]:
        """Ids de 'text'. Con parse_special=True (por defecto) los tokens
        especiales que aparezcan en el texto se emiten como su id; con False
        se tratan como texto corriente (util para texto no confiable)."""
        data = _to_bytes(text)
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else None
        out_len = ctypes.c_size_t(0)
        err = _err_buf()
        flag = 1 if parse_special else 0
        if not self._lib.ternaryai_tokenizer_encode_ex(self._handle, buf, len(data), flag, None, 0,
                                                        ctypes.byref(out_len), err, _ERR_CAP):
            _raise(err)
        n = out_len.value
        out = (ctypes.c_int32 * n)() if n else None
        if not self._lib.ternaryai_tokenizer_encode_ex(self._handle, buf, len(data), flag, out, n,
                                                        ctypes.byref(out_len), err, _ERR_CAP):
            _raise(err)
        return list(out) if n else []

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Texto de 'ids'. Los especiales se escriben como su texto, o se
        omiten con skip_special=True."""
        ids = list(ids)
        buf = (ctypes.c_int32 * len(ids))(*ids) if ids else None
        out_len = ctypes.c_size_t(0)
        err = _err_buf()
        flag = 1 if skip_special else 0
        if not self._lib.ternaryai_tokenizer_decode_ex(self._handle, buf, len(ids), flag, None, 0,
                                                        ctypes.byref(out_len), err, _ERR_CAP):
            _raise(err)
        n = out_len.value
        out = (ctypes.c_uint8 * n)() if n else None
        if not self._lib.ternaryai_tokenizer_decode_ex(self._handle, buf, len(ids), flag, out, n,
                                                        ctypes.byref(out_len), err, _ERR_CAP):
            _raise(err)
        return bytes(out).decode("utf-8", "replace") if n else ""

    def save(self, path: str) -> None:
        err = _err_buf()
        if not self._lib.ternaryai_tokenizer_save(self._handle, path.encode("utf-8"), err, _ERR_CAP):
            _raise(err)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        lib = _load_lib()
        err = _err_buf()
        handle = lib.ternaryai_tokenizer_load(path.encode("utf-8"), err, _ERR_CAP)
        if not handle:
            _raise(err)
        self = cls.__new__(cls)
        self._lib = lib
        self._handle = handle
        self._trained = True
        return self

    def __repr__(self) -> str:
        sp = f" special={self.num_special}" if self.num_special else ""
        return f"<ternaryai.Tokenizer vocab_size={self.vocab_size} merges={self.num_merges}{sp}>"
