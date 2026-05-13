"""Public analysis API for processed-signal, WaveNet, and dilution-series workflows.

The package re-exports the small set of analysis types and helpers intended for
notebooks and application code: CSV loading, single-trace analysis,
dilution-series orchestration, distribution diagnostics, and specific plotting
functions.
"""

from .trace_io import CsvTrace, Data
from .trace_plots import plot_standard_detection_trace, plot_wavenet_detection_trace
from .distributions import (
    compute_event_arrival_distribution_metrics,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
    plot_count_distribution,
    plot_event_raster,
    plot_inter_arrival_histogram,
    plot_peak_amplitude_ecdf,
    plot_peak_amplitude_histogram,
    plot_peak_amplitude_qq,
    plot_peak_width_ecdf,
    plot_peak_width_histogram,
    plot_peak_width_qq,
    plot_rescaled_uniform_qq,
)

from .dilution_series import (
    DilutionSeries,
    PeakCountSeries,
)
from .results import (
    EventArrivalDistributionMetrics,
    PeakAmplitudeDistributionMetrics,
    PeakWidthDistributionMetrics,
    PeakCountSeriesResult,
    PeakDetectionResult,
    PoissonSeriesDiagnostics,
    TraceRecord,
    WaveNetAnalyzerConfig,
)
from .triggers import PeakTrigger
from .wavenet_trace import WaveNetTraceAnalyzer

__all__ = [
    "CsvTrace",
    "Data",
    "DilutionSeries",
    "EventArrivalDistributionMetrics",
    "PeakAmplitudeDistributionMetrics",
    "PeakWidthDistributionMetrics",
    "PeakCountSeries",
    "PeakCountSeriesResult",
    "PeakDetectionResult",
    "PeakTrigger",
    "PoissonSeriesDiagnostics",
    "TraceRecord",
    "WaveNetAnalyzerConfig",
    "WaveNetTraceAnalyzer",
    "compute_event_arrival_distribution_metrics",
    "compute_peak_amplitude_distribution_metrics",
    "compute_peak_width_distribution_metrics",
    "plot_count_distribution",
    "plot_event_raster",
    "plot_inter_arrival_histogram",
    "plot_peak_amplitude_ecdf",
    "plot_peak_amplitude_histogram",
    "plot_peak_amplitude_qq",
    "plot_peak_width_ecdf",
    "plot_peak_width_histogram",
    "plot_peak_width_qq",
    "plot_rescaled_uniform_qq",
    "plot_standard_detection_trace",
    "plot_wavenet_detection_trace",
]
