"""Public analysis API for processed-signal, WaveNet, and dilution-series workflows.

The package re-exports the small set of analysis types and helpers intended for
notebooks and application code: CSV loading, single-trace analysis,
dilution-series orchestration, distribution diagnostics, and specific plotting
functions.
"""

from ..io.trace_io import CsvTrace
from . import metrics
from .distributions import (
    compute_event_arrival_distribution_metrics,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
)

from .dilution_series import (
    FlashDilutionSeries,
    StandardDilutionSeries,
)
from .metrics import (
    EventArrivalDistribution,
    PeakAmplitudeDistribution,
    PeakWidthDistribution,
    PeakCountSeriesResult,
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
from .noise_analysis import NoiseAnalyzer
from .pulse_shape import PulseShapeAnalyzer
from ..detection.triggers import (
    BasePeakTrigger,
    HeightPeakTrigger,
    ProminencePeakTrigger,
    SigmaPeakTrigger,
)
from .wavenet_trace import CNNTraceAnalyzer, StandardTraceAnalyzer, WaveNetTraceAnalyzer
from .comparison import (
    SeriesComparisonResult,
    TraceComparisonAnalyzer,
    TraceComparisonResult,
)

__all__ = [
    "CsvTrace",
    "FlashDilutionSeries",
    "metrics",
    "EventArrivalDistribution",
    "PeakAmplitudeDistribution",
    "PeakWidthDistribution",
    "PeakCountSeriesResult",
    "NoiseAnalyzer",
    "PulseShapeAnalyzer",
    "BasePeakTrigger",
    "CNNTraceAnalyzer",
    "HeightPeakTrigger",
    "ProminencePeakTrigger",
    "SigmaPeakTrigger",
    "PoissonSeriesDiagnostics",
    "StandardTraceAnalyzer",
    "StandardDilutionSeries",
    "TraceRecord",
    "WaveNetAnalyzerConfig",
    "WaveNetTraceAnalyzer",
    "SeriesComparisonResult",
    "TraceComparisonAnalyzer",
    "TraceComparisonResult",
    "compute_event_arrival_distribution_metrics",
    "compute_peak_amplitude_distribution_metrics",
    "compute_peak_width_distribution_metrics",
    "correct_observed_flow",
    "expected_observed_flow",
    "fraction_missed",
    "plot_dead_time_saturation",
    "throughput_tick_formatter",
]
