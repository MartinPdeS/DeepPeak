"""Distribution-oriented statistical analysis helpers.

This subpackage groups analysis tools that compare observed detector outputs to
simple distribution models. Event-arrival diagnostics live alongside peak-
amplitude diagnostics so notebook users can discover them in one place.
"""

from .amplitude import (
    compute_peak_amplitude_distribution_metrics,
)
from .event_arrival import (
    compute_event_arrival_distribution_metrics,
)
from .width import (
    compute_peak_width_distribution_metrics,
)

event_arrival = compute_event_arrival_distribution_metrics
peak_amplitude = compute_peak_amplitude_distribution_metrics
peak_width = compute_peak_width_distribution_metrics

__all__ = [
    "compute_event_arrival_distribution_metrics",
    "compute_peak_amplitude_distribution_metrics",
    "compute_peak_width_distribution_metrics",
    "event_arrival",
    "peak_amplitude",
    "peak_width",
]
