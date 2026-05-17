"""Public analysis API for processed-signal, WaveNet, and dilution-series workflows.

The package re-exports the small set of analysis types and helpers intended for
notebooks and application code: CSV loading, single-trace analysis,
dilution-series orchestration, distribution diagnostics, and specific plotting
functions.
"""

from .trace_io import CsvTrace
from . import metrics
from .distributions import (
    compute_event_arrival_distribution_metrics,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
)

from .dilution_series import (
    DilutionSeries,
    PeakCountSeries,
)
from .metrics import (
    EventArrivalDistribution,
    PeakAmplitudeDistribution,
    PeakWidthDistribution,
    PeakCountSeriesResult,
    PeakDetectionResult,
    PoissonSeriesDiagnostics,
    TraceRecord,
    WaveNetAnalyzerConfig,
)
from .dead_time import (
    correct_observed_flow,
    expected_observed_flow,
    fraction_missed,
    plot_dead_time_saturation,
    throughput_tick_formatter,
)
from .triggers import BasePeakTrigger, HeightPeakTrigger, SigmaPeakTrigger
from .wavenet_trace import CNNTraceAnalyzer, StandardTraceAnalyzer, WaveNetTraceAnalyzer

__all__ = [
    "CsvTrace",
    "DilutionSeries",
    "metrics",
    "EventArrivalDistribution",
    "PeakAmplitudeDistribution",
    "PeakWidthDistribution",
    "PeakCountSeries",
    "PeakCountSeriesResult",
    "PeakDetectionResult",
    "BasePeakTrigger",
    "CNNTraceAnalyzer",
    "HeightPeakTrigger",
    "SigmaPeakTrigger",
    "PoissonSeriesDiagnostics",
    "StandardTraceAnalyzer",
    "TraceRecord",
    "WaveNetAnalyzerConfig",
    "WaveNetTraceAnalyzer",
    "compute_event_arrival_distribution_metrics",
    "compute_peak_amplitude_distribution_metrics",
    "compute_peak_width_distribution_metrics",
    "correct_observed_flow",
    "expected_observed_flow",
    "fraction_missed",
    "plot_dead_time_saturation",
    "throughput_tick_formatter",
]
