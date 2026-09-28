"""Public exceptions raised by the inference-only detector."""


class DetectorError(Exception):
    """Base error for model, input, and output failures."""


class DetectorBusyError(DetectorError):
    """Raised when one detector is asked to process concurrent jobs."""


class InputError(DetectorError):
    """Raised when the supplied PDF input is missing or invalid."""


class ModelError(DetectorError):
    """Raised when a configured model is missing, changed, or incompatible."""


class OutputError(DetectorError):
    """Raised when the requested output destination or selection is invalid."""
