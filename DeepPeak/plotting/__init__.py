"""Plotting entry points for detection and analysis results."""

from ..core.config import PlotConfig
from .trace_plots import (
    get_detection_threshold,
    plot_detection_comparison,
    reconstruct_gaussian_trace,
)
from .trace_record import (
    standard_detection,
    standard_detection_with_histogram,
    wavenet_detection,
    wavenet_detection_with_histogram,
)

__all__ = [
    "PlotConfig",
    "get_detection_threshold",
    "plot_detection_comparison",
    "reconstruct_gaussian_trace",
    "standard_detection",
    "standard_detection_with_histogram",
    "wavenet_detection",
    "wavenet_detection_with_histogram",
]
