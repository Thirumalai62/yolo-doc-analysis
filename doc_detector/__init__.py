"""Inference-only public API for legal-notice detection."""

from .detector import DetectionResult, LegalNoticeDetector, get_detector
from .errors import (
    DetectorBusyError,
    DetectorError,
    InputError,
    ModelError,
    OutputError,
)

__all__ = [
    "DetectionResult",
    "DetectorBusyError",
    "DetectorError",
    "InputError",
    "LegalNoticeDetector",
    "ModelError",
    "OutputError",
    "get_detector",
]
