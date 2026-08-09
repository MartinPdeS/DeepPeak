"""Plotting entry points for canonical trace analysis records.

The analysis package still exposes methods on ``TraceRecord`` for notebook
compatibility. New code can keep plotting orchestration in this module and
depend only on the record and ``PlotConfig`` contracts.
"""

from typing import Any

from ..core.config import PlotConfig
from ..analysis.metrics.trace_record import TraceRecord


def standard_detection(
    record: TraceRecord, *, config: PlotConfig | None = None, **kwargs: Any
):
    """Render standard detection annotations for one trace record."""

    if config is not None:
        kwargs["config"] = config
    return record.plot_standard_detection(**kwargs)


def wavenet_detection(
    record: TraceRecord, *, config: PlotConfig | None = None, **kwargs: Any
):
    """Render CNN/WaveNet detection annotations for one trace record."""

    if config is not None:
        kwargs["config"] = config
    return record.plot_wavenet_detection(**kwargs)


def standard_detection_with_histogram(
    record: TraceRecord, *, config: PlotConfig | None = None, **kwargs: Any
):
    """Render standard detection and its amplitude histogram."""

    if config is not None:
        kwargs["config"] = config
    return record.plot_standard_detection_with_histogram(**kwargs)


def wavenet_detection_with_histogram(
    record: TraceRecord, *, config: PlotConfig | None = None, **kwargs: Any
):
    """Render CNN/WaveNet detection and its amplitude histogram."""

    if config is not None:
        kwargs["config"] = config
    return record.plot_wavenet_detection_with_histogram(**kwargs)


__all__ = [
    "standard_detection",
    "standard_detection_with_histogram",
    "wavenet_detection",
    "wavenet_detection_with_histogram",
]
