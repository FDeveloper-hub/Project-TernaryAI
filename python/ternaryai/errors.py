"""Exceptions raised by TernaryAI."""


class TernaryAIError(Exception):
    """Base class for every error raised by TernaryAI itself."""


class BackendNotFoundError(TernaryAIError, RuntimeError):
    """No compute backend is available (libternaryai was not found or not connected)."""


class ArchitectureError(TernaryAIError, ValueError):
    """A layer cannot be added to the model (bad shape, unsupported position, ...)."""


class DataError(TernaryAIError, ValueError):
    """Input or target data does not fit the model."""


class ModelFileError(TernaryAIError):
    """A model file is missing, corrupt, or from an unknown version/layer type."""
