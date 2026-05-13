"""Distribution-oriented statistical analysis helpers.

This subpackage groups analysis tools that compare observed detector outputs to
simple distribution models. Event-arrival diagnostics live alongside peak-
amplitude diagnostics so notebook users can discover them in one place.
"""

from .amplitude import (
    compute_peak_amplitude_distribution_metrics,
    plot_peak_amplitude_ecdf,
    plot_peak_amplitude_histogram,
    plot_peak_amplitude_qq,
)
from .event_arrival import (
    compute_event_arrival_distribution_metrics,
    plot_count_distribution,
    plot_event_raster,
    plot_inter_arrival_histogram,
    plot_rescaled_uniform_qq,
)
from .width import (
    compute_peak_width_distribution_metrics,
    plot_peak_width_ecdf,
    plot_peak_width_histogram,
    plot_peak_width_qq,
)

__all__ = [
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
]
