"""Canonical result models for DeepPeak analysis workflows."""

from .config import WaveNetAnalyzerConfig
from .detection import PeakDetectionResult
from .distributions import (
    EventArrivalDistribution,
    PeakAmplitudeDistribution,
    PeakWidthDistribution,
    PoissonSeriesDiagnostics,
)
from .series_result import PeakCountSeriesResult
from .trace_record import TraceRecord
from .utils import resolve_series_or_result

__all__ = [
    "EventArrivalDistribution",
    "PeakAmplitudeDistribution",
    "PeakWidthDistribution",
    "PeakCountSeriesResult",
    "PeakDetectionResult",
    "PoissonSeriesDiagnostics",
    "TraceRecord",
    "WaveNetAnalyzerConfig",
    "resolve_series_or_result",
]
