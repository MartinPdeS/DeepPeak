"""Lightweight domain types shared across DeepPeak workflows."""

from .types import DetectionResult, MetricResult, SeriesResult, Trace
from .config import (
    AnalysisConfig,
    DetectionConfig,
    GenerationConfig,
    ModelConfig,
    NoiseConfig,
    PlotConfig,
    SeriesConfig,
    TraceConfig,
)
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
    "AnalysisConfig",
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
    "GenerationConfig",
    "ModelConfig",
    "NoiseConfig",
    "PlotConfig",
    "SeriesConfig",
    "SeriesResult",
    "Trace",
    "TraceConfig",
]
