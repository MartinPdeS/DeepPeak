"""Trace-level plotting utilities for standard and WaveNet detections.

These helpers deliberately keep one plotting function per view so notebooks can
compose figures explicitly instead of relying on large multi-purpose plotters.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from MPSPlots.styles import scientific

from .results import TraceRecord


@contextmanager
def plot_style_context() -> Iterator[None]:
    """Apply the shared MPSPlots scientific style for a plotting block."""

    with plt.style.context(scientific):
        yield


def make_plot_figure(
    *,
    figsize: Tuple[float, float],
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kwargs,
):
    """Create styled Matplotlib subplots with a small common wrapper."""

    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **subplot_kwargs)


def make_or_reuse_single_axis(
    *,
    figsize: Tuple[float, float],
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes, bool]:
    """Return a single axis, creating one only when the caller did not supply it."""

    if ax is not None:
        return ax.figure, ax, False

    figure, axis = make_plot_figure(figsize=figsize)
    return figure, axis, True


def style_plot_axis(
    axis: plt.Axes,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_title: bool = True,
    show_grid: bool = False,
    show_legend: bool = False,
    legend_loc: str = "upper right",
) -> None:
    """Apply common axis presentation settings in place."""

    apply_axis_limits(axis, xlim, ylim)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if show_title and title is not None:
        axis.set_title(title)
    if show_grid:
        axis.grid(True, which="both", alpha=0.2, zorder=0)
    if show_legend:
        axis.legend(
            loc=legend_loc,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
        )


def finalize_single_axis_figure(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_title: bool = True,
    show_grid: bool = False,
    show_legend: bool = False,
    legend_loc: str = "upper right",
    tight_layout: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
    close: bool = False,
) -> plt.Figure:
    """Apply common single-axis layout/legend/finalization steps."""

    style_plot_axis(
        axis,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        title=title,
        show_title=show_title,
        show_grid=show_grid,
        show_legend=show_legend,
        legend_loc=legend_loc,
    )
    if tight_layout:
        figure.tight_layout()
    return finalize_figure(
        figure=figure,
        show=show,
        close=close,
        save_path=save_path,
        dpi=dpi,
        bbox_inches=None,
    )


def make_trace_x_axis(
    number_of_samples: int, dx: float, x_axis: str
) -> Tuple[np.ndarray, str]:
    """Build x-axis coordinates in sample units or physical time.

    Parameters
    ----------
    number_of_samples : int
        Number of samples in the trace.
    dx : float
        Sampling interval of the trace.
    x_axis : {"sample", "time"}
        Coordinate system used for the x-axis.

    Returns
    -------
    x_values : numpy.ndarray
        Coordinates for each sample.
    x_label : str
        Human-readable axis label.
    """

    if x_axis == "sample":
        return np.arange(number_of_samples, dtype=float), "Sample index"
    if x_axis == "time":
        return np.arange(number_of_samples, dtype=float) * float(dx), "Time"
    raise ValueError('x_axis must be either "sample" or "time".')


def peak_x_values(peak_indices: np.ndarray, dx: float, x_axis: str) -> np.ndarray:
    """Convert peak indices into sample positions or times.

    Parameters
    ----------
    peak_indices : numpy.ndarray
        Integer peak indices.
    dx : float
        Sampling interval of the trace.
    x_axis : {"sample", "time"}
        Coordinate system used for the conversion.

    Returns
    -------
    numpy.ndarray
        Peak positions in sample units or physical time.
    """

    peak_indices = np.asarray(peak_indices, dtype=int)
    if x_axis == "sample":
        return peak_indices.astype(float)
    if x_axis == "time":
        return peak_indices.astype(float) * float(dx)
    raise ValueError('x_axis must be either "sample" or "time".')


def valid_peak_indices(peak_indices: np.ndarray, array_size: int) -> np.ndarray:
    """Clip peak indices to the range addressable by an array.

    Parameters
    ----------
    peak_indices : numpy.ndarray
        Candidate peak indices.
    array_size : int
        Length of the target array.

    Returns
    -------
    numpy.ndarray
        Peak indices that fall inside the valid range.
    """

    peak_indices = np.asarray(peak_indices, dtype=int)
    return peak_indices[(peak_indices >= 0) & (peak_indices < int(array_size))]


def get_detection_threshold(namespace: object, key: str = "height") -> Optional[float]:
    """Extract the effective detection threshold from a detection result.

    Parameters
    ----------
    namespace : object
        Detection result or compatible object carrying threshold metadata.
    key : str, default="height"
        Threshold key to look up in stored detection kwargs.

    Returns
    -------
    float or None
        Effective scalar detection threshold when one is available.
    """

    threshold = getattr(namespace, "threshold", None)
    if threshold is not None and np.isfinite(float(threshold)):
        return float(threshold)

    detection_kwargs = getattr(namespace, "detection_kwargs", None)
    if isinstance(detection_kwargs, dict) and key in detection_kwargs:
        threshold = detection_kwargs[key]
        if threshold is not None and np.isfinite(float(threshold)):
            return float(threshold)

    for attribute_name in ("std_kwargs", "cnn_kwargs"):
        kwargs = getattr(namespace, attribute_name, None)
        if isinstance(kwargs, dict) and key in kwargs:
            threshold = kwargs[key]
            if threshold is not None and np.isfinite(float(threshold)):
                return float(threshold)

    return None


def apply_axis_limits(
    ax: plt.Axes,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> None:
    """Apply optional axis bounds in place.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes modified in place.
    xlim, ylim : tuple of float, optional
        Optional lower and upper axis bounds.

    Returns
    -------
    None
        The axes are modified in place.
    """

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def save_figure(
    figure: plt.Figure,
    save_path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: Optional[str] = None,
) -> None:
    """Save a figure using export settings suitable for vector and raster output.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    save_path : str or pathlib.Path
        Destination file path.
    dpi : int, default=300
        Raster resolution used by Matplotlib when applicable.
    bbox_inches : str, optional
        Optional ``bbox_inches`` argument forwarded to ``savefig``.

    Returns
    -------
    None
        The figure is written to disk.
    """

    save_path = Path(save_path)
    save_kwargs = {
        "dpi": dpi,
        "bbox_inches": bbox_inches,
        "facecolor": "white",
        "edgecolor": "none",
    }
    if save_path.suffix.lower() == ".svg":
        rc_export = {
            "svg.fonttype": "none",
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
            "agg.path.chunksize": 20_000,
        }
    elif save_path.suffix.lower() == ".pdf":
        rc_export = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
            "agg.path.chunksize": 20_000,
        }
    else:
        rc_export = {}
    with plt.rc_context(rc_export):
        figure.savefig(save_path, **save_kwargs)


def finalize_figure(
    figure: plt.Figure,
    show: bool = False,
    close: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    bbox_inches: Optional[str] = None,
) -> plt.Figure:
    """Apply optional save/show/close actions and return the figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to finalize.
    show : bool, default=False
        If ``True``, display the figure with Matplotlib.
    close : bool, default=False
        If ``True``, close the figure before returning it.
    save_path : str or pathlib.Path, optional
        Optional output file path.
    dpi : int, default=300
        Raster resolution used when saving.
    bbox_inches : str, optional
        Optional ``bbox_inches`` argument forwarded to :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
        The finalized figure object.
    """

    if save_path is not None:
        save_figure(
            figure=figure, save_path=save_path, dpi=dpi, bbox_inches=bbox_inches
        )
    if show:
        plt.show()
    if close:
        plt.close(figure)
    return figure


def plot_standard_detection_trace(
    record: TraceRecord,
    x_axis: str = "sample",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10.0, 4.0),
    line_width: float = 0.8,
    marker_size: float = 18.0,
    threshold_line_width: float = 0.8,
    show_threshold: bool = True,
    show_peaks: bool = True,
    show_legend: bool = True,
    show_grid: bool = False,
    show_title: bool = True,
    title: Optional[str] = None,
    signal_color: str = "C0",
    marker_color: str = "black",
    threshold_color: str = "black",
    rasterize_dense_artists: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    close: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot one processed trace with its standard-detector annotations.

    Parameters
    ----------
    record : TraceRecord
        Trace record to visualize.
    x_axis : {"sample", "time"}, default="sample"
        Coordinate system used for the x-axis.
    xlim, ylim : tuple of float, optional
        Optional axis bounds.
    figsize : tuple of float, default=(10.0, 4.0)
        Figure size passed to Matplotlib.
    line_width, marker_size, threshold_line_width : float
        Styling parameters for lines and markers.
    show_threshold, show_peaks, show_legend, show_grid, show_title : bool
        Toggles controlling displayed plot elements.
    title : str, optional
        Explicit title override.
    signal_color, marker_color, threshold_color : str
        Matplotlib colors used for the plotted artists.
    rasterize_dense_artists : bool, default=True
        Rasterize dense line artists when exporting vector graphics.
    ax : matplotlib.axes.Axes, optional
        Existing axis on which to draw the trace.
    show, close : bool, default=False
        Whether to display or close the figure.
    save_path : str or pathlib.Path, optional
        Optional output file path.
    dpi : int, default=300
        Raster resolution used when saving.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the standard detector trace view.
    """

    signal = np.asarray(record.signal).ravel()
    if signal.size == 0:
        raise ValueError("Cannot plot an empty signal.")

    x_values, x_label = make_trace_x_axis(signal.size, record.dx, x_axis)
    standard_peak_indices = valid_peak_indices(record.standard.peaks, signal.size)
    standard_threshold = get_detection_threshold(record.standard)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        axis.plot(
            x_values,
            signal,
            color=signal_color,
            linewidth=line_width,
            label="Signal",
            zorder=1,
            rasterized=rasterize_dense_artists,
        )

        if show_threshold and standard_threshold is not None:
            axis.axhline(
                standard_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=threshold_line_width,
                label="Threshold",
                zorder=2,
            )

        if show_peaks and standard_peak_indices.size > 0:
            axis.scatter(
                peak_x_values(standard_peak_indices, record.dx, x_axis),
                signal[standard_peak_indices],
                color=marker_color,
                edgecolors=marker_color,
                s=marker_size,
                marker="o",
                label=f"Peaks ({standard_peak_indices.size})",
                zorder=10,
                rasterized=rasterize_dense_artists,
            )

        figure.subplots_adjust(
            left=0.10, right=0.97, bottom=0.14, top=0.88 if show_title else 0.95
        )
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel=x_label,
            ylabel="Signal amplitude",
            xlim=xlim,
            ylim=ylim,
            title=title,
            show_title=show_title,
            show_grid=show_grid,
            show_legend=show_legend,
            legend_loc="upper right",
            tight_layout=created_figure,
            save_path=save_path,
            dpi=dpi,
            show=show,
            close=close,
        )


def plot_wavenet_detection_trace(
    record: TraceRecord,
    x_axis: str = "sample",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10.0, 4.0),
    line_width: float = 0.8,
    marker_size: float = 18.0,
    threshold_line_width: float = 0.8,
    show_signal: bool = True,
    show_prediction: bool = True,
    show_cnn_signal_peaks: bool = False,
    show_cnn_prediction_peaks: bool = True,
    show_cnn_threshold: bool = True,
    show_legend: bool = True,
    show_grid: bool = False,
    show_title: bool = True,
    title: Optional[str] = None,
    signal_color: str = "C0",
    prediction_color: str = "C1",
    marker_color: str = "black",
    threshold_color: str = "black",
    rasterize_dense_artists: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    close: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot one trace together with its WaveNet prediction and CNN peaks.

    Parameters
    ----------
    record : TraceRecord
        Trace record to visualize.
    x_axis : {"sample", "time"}, default="sample"
        Coordinate system used for the x-axis.
    xlim, ylim : tuple of float, optional
        Optional axis bounds.
    figsize : tuple of float, default=(10.0, 4.0)
        Figure size passed to Matplotlib.
    line_width, marker_size, threshold_line_width : float
        Styling parameters for lines and markers.
    show_signal, show_prediction, show_cnn_signal_peaks, show_cnn_prediction_peaks, show_cnn_threshold : bool
        Toggles controlling displayed plot elements. By default, CNN detections
        are shown on the prediction rather than projected onto the signal.
    show_legend, show_grid, show_title : bool
        Axes presentation toggles.
    title : str, optional
        Explicit title override.
    signal_color, prediction_color, marker_color, threshold_color : str
        Matplotlib colors used for the plotted artists.
    rasterize_dense_artists : bool, default=True
        Rasterize dense line artists when exporting vector graphics.
    ax : matplotlib.axes.Axes, optional
        Existing axis on which to draw the trace.
    show, close : bool, default=False
        Whether to display or close the figure.
    save_path : str or pathlib.Path, optional
        Optional output file path.
    dpi : int, default=300
        Raster resolution used when saving.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the WaveNet detector trace view.
    """

    signal = np.asarray(record.signal).ravel()
    prediction = np.asarray(record.prediction).ravel()
    if signal.size == 0:
        raise ValueError("Cannot plot an empty signal.")
    if prediction.size == 0:
        raise ValueError("Cannot plot an empty prediction.")

    x_values, x_label = make_trace_x_axis(signal.size, record.dx, x_axis)
    prediction_x_values, _ = make_trace_x_axis(prediction.size, record.dx, x_axis)
    cnn_peak_indices_for_signal = valid_peak_indices(record.cnn.peaks, signal.size)
    cnn_peak_indices_for_prediction = valid_peak_indices(
        record.cnn.peaks, prediction.size
    )
    cnn_threshold = get_detection_threshold(record.cnn)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        if show_signal:
            axis.plot(
                x_values,
                signal,
                color=signal_color,
                linewidth=line_width,
                label="Signal",
                zorder=1,
                rasterized=rasterize_dense_artists,
            )
        if show_prediction:
            axis.plot(
                prediction_x_values,
                prediction,
                color=prediction_color,
                linewidth=line_width,
                label="Prediction",
                zorder=2,
                rasterized=rasterize_dense_artists,
            )
        if show_cnn_threshold and cnn_threshold is not None:
            axis.axhline(
                cnn_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=threshold_line_width,
                label="CNN threshold",
                zorder=3,
            )
        if show_cnn_signal_peaks and cnn_peak_indices_for_signal.size > 0:
            axis.scatter(
                peak_x_values(cnn_peak_indices_for_signal, record.dx, x_axis),
                signal[cnn_peak_indices_for_signal],
                color=marker_color,
                s=marker_size,
                marker="x",
                label=f"CNN positions on signal ({cnn_peak_indices_for_signal.size})",
                zorder=20,
                rasterized=rasterize_dense_artists,
            )
        if show_cnn_prediction_peaks and cnn_peak_indices_for_prediction.size > 0:
            axis.scatter(
                peak_x_values(cnn_peak_indices_for_prediction, record.dx, x_axis),
                prediction[cnn_peak_indices_for_prediction],
                color=marker_color,
                edgecolors=marker_color,
                s=marker_size,
                marker="o",
                label=f"CNN peaks on prediction ({cnn_peak_indices_for_prediction.size})",
                zorder=21,
                rasterized=rasterize_dense_artists,
            )

        figure.subplots_adjust(
            left=0.10, right=0.97, bottom=0.14, top=0.88 if show_title else 0.95
        )
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel=x_label,
            ylabel="Signal / prediction amplitude",
            xlim=xlim,
            ylim=ylim,
            title=title,
            show_title=show_title,
            show_grid=show_grid,
            show_legend=show_legend,
            legend_loc="upper right",
            tight_layout=created_figure,
            save_path=save_path,
            dpi=dpi,
            show=show,
            close=close,
        )


def plot_detection_comparison(record: TraceRecord, **kwargs) -> plt.Figure:
    """Plot standard and CNN detections on vertically stacked axes."""

    x_axis = kwargs.pop("x_axis", "sample")
    xlim = kwargs.pop("xlim", None)
    standard_ylim = kwargs.pop("standard_ylim", None)
    cnn_ylim = kwargs.pop("cnn_ylim", None)
    figsize = kwargs.pop("figsize", (10.0, 6.0))
    line_width = kwargs.pop("line_width", 0.8)
    marker_size = kwargs.pop("marker_size", 18.0)
    threshold_line_width = kwargs.pop("threshold_line_width", 0.8)
    show_standard_peaks = kwargs.pop("show_standard_peaks", True)
    show_cnn_signal_peaks = kwargs.pop("show_cnn_signal_peaks", False)
    show_cnn_prediction_peaks = kwargs.pop("show_cnn_prediction_peaks", True)
    show_standard_threshold = kwargs.pop("show_standard_threshold", True)
    show_cnn_threshold = kwargs.pop("show_cnn_threshold", True)
    show_legend = kwargs.pop("show_legend", True)
    show_grid = kwargs.pop("show_grid", False)
    show_title = kwargs.pop("show_title", True)
    title = kwargs.pop("title", None)
    standard_signal_color = kwargs.pop("standard_signal_color", "C0")
    cnn_signal_color = kwargs.pop("cnn_signal_color", "C0")
    prediction_color = kwargs.pop("prediction_color", "C1")
    marker_color = kwargs.pop("marker_color", "black")
    threshold_color = kwargs.pop("threshold_color", "black")
    rasterize_dense_artists = kwargs.pop("rasterize_dense_artists", True)
    show = kwargs.pop("show", False)
    close = kwargs.pop("close", False)
    save_path = kwargs.pop("save_path", None)
    dpi = kwargs.pop("dpi", 300)

    signal = np.asarray(record.signal).ravel()
    prediction = np.asarray(record.prediction).ravel()
    if signal.size == 0:
        raise ValueError("Cannot plot an empty signal.")

    x_values, x_label = make_trace_x_axis(signal.size, record.dx, x_axis)
    prediction_x_values, _ = make_trace_x_axis(prediction.size, record.dx, x_axis)
    standard_peak_indices = valid_peak_indices(record.standard.peaks, signal.size)
    cnn_peak_indices_for_signal = valid_peak_indices(record.cnn.peaks, signal.size)
    cnn_peak_indices_for_prediction = valid_peak_indices(
        record.cnn.peaks, prediction.size
    )
    standard_threshold = get_detection_threshold(record.standard)
    cnn_threshold = get_detection_threshold(record.cnn)

    with plot_style_context():
        figure, axes = make_plot_figure(
            nrows=2, ncols=1, sharex=True, figsize=figsize, constrained_layout=False
        )
        standard_axis, cnn_axis = axes

        standard_axis.plot(
            x_values,
            signal,
            color=standard_signal_color,
            linewidth=line_width,
            label="Signal",
            zorder=1,
            rasterized=rasterize_dense_artists,
        )
        if show_standard_threshold and standard_threshold is not None:
            standard_axis.axhline(
                standard_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=threshold_line_width,
                label="Threshold",
                zorder=2,
            )
        if show_standard_peaks and standard_peak_indices.size > 0:
            standard_axis.scatter(
                peak_x_values(standard_peak_indices, record.dx, x_axis),
                signal[standard_peak_indices],
                color=marker_color,
                edgecolors=marker_color,
                s=marker_size,
                marker="o",
                label=f"Peaks ({standard_peak_indices.size})",
                zorder=10,
                rasterized=rasterize_dense_artists,
            )

        cnn_axis.plot(
            x_values,
            signal,
            color=cnn_signal_color,
            linewidth=line_width,
            label="Signal",
            zorder=1,
            rasterized=rasterize_dense_artists,
        )
        cnn_axis.plot(
            prediction_x_values,
            prediction,
            color=prediction_color,
            linewidth=line_width,
            label="Prediction",
            zorder=2,
            rasterized=rasterize_dense_artists,
        )
        if show_cnn_threshold and cnn_threshold is not None:
            cnn_axis.axhline(
                cnn_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=threshold_line_width,
                label="CNN threshold",
                zorder=3,
            )
        if show_cnn_signal_peaks and cnn_peak_indices_for_signal.size > 0:
            cnn_axis.scatter(
                peak_x_values(cnn_peak_indices_for_signal, record.dx, x_axis),
                signal[cnn_peak_indices_for_signal],
                color=marker_color,
                s=marker_size,
                marker="x",
                label=f"CNN positions on signal ({cnn_peak_indices_for_signal.size})",
                zorder=20,
                rasterized=rasterize_dense_artists,
            )
        if show_cnn_prediction_peaks and cnn_peak_indices_for_prediction.size > 0:
            cnn_axis.scatter(
                peak_x_values(cnn_peak_indices_for_prediction, record.dx, x_axis),
                prediction[cnn_peak_indices_for_prediction],
                color=marker_color,
                edgecolors=marker_color,
                s=marker_size,
                marker="o",
                label=f"CNN peaks on prediction ({cnn_peak_indices_for_prediction.size})",
                zorder=21,
                rasterized=rasterize_dense_artists,
            )

        style_plot_axis(
            standard_axis,
            ylabel="Signal amplitude",
            xlim=xlim,
            ylim=standard_ylim,
            title="Standard detection",
            show_grid=show_grid,
            show_legend=show_legend,
            legend_loc="upper right",
        )
        style_plot_axis(
            cnn_axis,
            xlabel=x_label,
            ylabel="Signal / prediction amplitude",
            xlim=xlim,
            ylim=cnn_ylim,
            title="CNN detection",
            show_grid=show_grid,
            show_legend=show_legend,
            legend_loc="upper right",
        )

        if show_title and title is not None:
            figure.suptitle(title, fontsize=12)

        figure.subplots_adjust(
            left=0.10,
            right=0.97,
            bottom=0.09,
            top=0.90 if show_title else 0.95,
            hspace=0.35,
        )
        return finalize_figure(
            figure=figure,
            show=show,
            close=close,
            save_path=save_path,
            dpi=dpi,
            bbox_inches=None,
        )


__all__ = [
    "apply_axis_limits",
    "finalize_figure",
    "finalize_single_axis_figure",
    "get_detection_threshold",
    "make_plot_figure",
    "make_or_reuse_single_axis",
    "make_trace_x_axis",
    "peak_x_values",
    "plot_detection_comparison",
    "plot_standard_detection_trace",
    "style_plot_axis",
    "plot_style_context",
    "plot_wavenet_detection_trace",
    "save_figure",
    "valid_peak_indices",
]
