"""Stable metrics namespace."""

from ..analysis.metrics import (
    EventArrivalDistribution,
    PeakAmplitudeDistribution,
    PeakCountSeriesResult,
    PeakWidthDistribution,
    PoissonSeriesDiagnostics,
    TraceRecord,
    WaveNetAnalyzerConfig,
    resolve_series_or_result,
)
from ..core.types import DetectionResult, MetricResult, SeriesResult

__all__ = [
    "DetectionResult",
    "EventArrivalDistribution",
    "MetricResult",
    "PeakAmplitudeDistribution",
    "PeakCountSeriesResult",
    "PeakWidthDistribution",
    "PoissonSeriesDiagnostics",
    "SeriesResult",
    "TraceRecord",
    "WaveNetAnalyzerConfig",
    "resolve_series_or_result",
]
