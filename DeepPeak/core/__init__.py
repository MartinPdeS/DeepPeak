"""Lightweight domain types shared across DeepPeak workflows."""

from .types import DetectionResult, MetricResult, SeriesResult, Trace
from .config import DetectionConfig, PlotConfig, SeriesConfig, TraceConfig
from .exceptions import (
    AnalysisInputError,
    AnalysisStateError,
    DeepPeakError,
    InvalidConfigurationError,
    InvalidDetectorError,
    InvalidTraceError,
    MissingDetectorError,
    MissingOptionalDependencyError,
)

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "AnalysisInputError",
    "AnalysisStateError",
    "DeepPeakError",
    "InvalidConfigurationError",
    "InvalidDetectorError",
    "InvalidTraceError",
    "MissingDetectorError",
    "MissingOptionalDependencyError",
    "MetricResult",
    "PlotConfig",
    "SeriesConfig",
    "SeriesResult",
    "Trace",
    "TraceConfig",
]
