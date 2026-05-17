"""Trace-level plotting utilities for standard and WaveNet detections.

These helpers deliberately keep one plotting function per view so notebooks can
compose figures explicitly instead of relying on large multi-purpose plotters.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from . import metrics


def reconstruct_gaussian_trace(
    number_of_samples: int,
    peak_indices: np.ndarray,
    amplitudes: np.ndarray,
    sigma_samples: float,
    baseline: float = 0.0,
) -> np.ndarray:
    """Rebuild a trace from fixed Gaussian centers and recovered amplitudes."""

    x_values = np.arange(int(number_of_samples), dtype=float)
    peak_indices = np.asarray(peak_indices, dtype=float).ravel()
    amplitudes = np.asarray(amplitudes, dtype=float).ravel()

    if peak_indices.size == 0 or amplitudes.size == 0:
        return np.zeros(int(number_of_samples), dtype=float)

    gaussian_matrix = np.exp(
        -0.5 * ((x_values[:, None] - peak_indices[None, :]) / float(sigma_samples)) ** 2
    )
    return float(baseline) + gaussian_matrix @ amplitudes


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


def plot_detection_comparison(record: metrics.TraceRecord, **kwargs) -> plt.Figure:
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
    show_cnn_recovered_signal_peaks = kwargs.pop(
        "show_cnn_recovered_signal_peaks", False
    )
    show_cnn_reconstructed_trace = kwargs.pop("show_cnn_reconstructed_trace", False)
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
    reconstructed_trace_color = kwargs.pop("reconstructed_trace_color", "C3")
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

    if x_axis == "sample":
        x_values = np.arange(signal.size, dtype=float)
        prediction_x_values = np.arange(prediction.size, dtype=float)
        x_label = "Sample index"
    elif x_axis == "time":
        x_values = np.arange(signal.size, dtype=float) * float(record.dx)
        prediction_x_values = np.arange(prediction.size, dtype=float) * float(record.dx)
        x_label = "Time"
    else:
        raise ValueError('x_axis must be either "sample" or "time".')

    standard_peak_indices = np.asarray(record.standard.peaks, dtype=int)
    standard_peak_indices = standard_peak_indices[
        (standard_peak_indices >= 0) & (standard_peak_indices < signal.size)
    ]
    cnn_peak_indices_for_signal = np.asarray(record.cnn.peaks, dtype=int)
    cnn_peak_indices_for_signal = cnn_peak_indices_for_signal[
        (cnn_peak_indices_for_signal >= 0) & (cnn_peak_indices_for_signal < signal.size)
    ]
    cnn_peak_indices_for_prediction = np.asarray(record.cnn.peaks, dtype=int)
    cnn_peak_indices_for_prediction = cnn_peak_indices_for_prediction[
        (cnn_peak_indices_for_prediction >= 0)
        & (cnn_peak_indices_for_prediction < prediction.size)
    ]
    cnn_signal_peak_values = signal[cnn_peak_indices_for_signal]
    cnn_signal_peak_label = (
        f"CNN positions on signal ({cnn_peak_indices_for_signal.size})"
    )
    cnn_recovered_amplitudes = getattr(record.cnn, "amplitudes", None)
    reconstructed_trace = None
    if (
        (show_cnn_recovered_signal_peaks or show_cnn_reconstructed_trace)
        and cnn_recovered_amplitudes is not None
        and cnn_peak_indices_for_signal.size > 0
    ):
        cnn_recovered_amplitudes = np.asarray(
            cnn_recovered_amplitudes, dtype=float
        ).ravel()
        if (
            cnn_recovered_amplitudes.size
            == np.asarray(record.cnn.peaks, dtype=int).size
        ):
            valid_signal_mask = (np.asarray(record.cnn.peaks, dtype=int) >= 0) & (
                np.asarray(record.cnn.peaks, dtype=int) < signal.size
            )
            recovered_signal_peak_values = cnn_recovered_amplitudes[valid_signal_mask]
            finite_recovered_mask = np.isfinite(recovered_signal_peak_values)
            cnn_peak_indices_for_signal = cnn_peak_indices_for_signal[
                finite_recovered_mask
            ]
            recovered_baseline = float(
                getattr(record.cnn, "properties", {}).get("recovered_baseline", 0.0)
            )
            cnn_signal_peak_values = (
                recovered_signal_peak_values[finite_recovered_mask] + recovered_baseline
            )
            cnn_signal_peak_label = f"CNN recovered amplitudes on signal ({cnn_peak_indices_for_signal.size})"
            if show_cnn_reconstructed_trace:
                sigma_samples = getattr(record.cnn, "properties", {}).get(
                    "recovered_sigma_samples"
                )
                if sigma_samples is not None and np.isfinite(float(sigma_samples)):
                    reconstructed_trace = reconstruct_gaussian_trace(
                        number_of_samples=signal.size,
                        peak_indices=cnn_peak_indices_for_signal,
                        amplitudes=cnn_signal_peak_values - recovered_baseline,
                        sigma_samples=float(sigma_samples),
                        baseline=recovered_baseline,
                    )
    standard_threshold = get_detection_threshold(record.standard)
    cnn_threshold = get_detection_threshold(record.cnn)

    figure, axes = plt.subplots(
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
        if x_axis == "sample":
            standard_peak_x_values = standard_peak_indices.astype(float)
        else:
            standard_peak_x_values = standard_peak_indices.astype(float) * float(
                record.dx
            )
        standard_axis.scatter(
            standard_peak_x_values,
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
    if reconstructed_trace is not None:
        cnn_axis.plot(
            x_values,
            reconstructed_trace,
            color=reconstructed_trace_color,
            linewidth=line_width,
            linestyle="-.",
            label="CNN reconstructed trace",
            zorder=3,
            rasterized=rasterize_dense_artists,
        )
    if show_cnn_threshold and cnn_threshold is not None:
        cnn_axis.axhline(
            cnn_threshold,
            color=threshold_color,
            linestyle="--",
            linewidth=threshold_line_width,
            label="CNN threshold",
            zorder=4,
        )
    if (
        show_cnn_signal_peaks or show_cnn_recovered_signal_peaks
    ) and cnn_peak_indices_for_signal.size > 0:
        if x_axis == "sample":
            cnn_signal_peak_x_values = cnn_peak_indices_for_signal.astype(float)
        else:
            cnn_signal_peak_x_values = cnn_peak_indices_for_signal.astype(
                float
            ) * float(record.dx)
        cnn_axis.scatter(
            cnn_signal_peak_x_values,
            cnn_signal_peak_values,
            color=marker_color,
            s=marker_size,
            marker="x",
            label=cnn_signal_peak_label,
            zorder=20,
            rasterized=rasterize_dense_artists,
        )
    if show_cnn_prediction_peaks and cnn_peak_indices_for_prediction.size > 0:
        if x_axis == "sample":
            cnn_prediction_peak_x_values = cnn_peak_indices_for_prediction.astype(float)
        else:
            cnn_prediction_peak_x_values = cnn_peak_indices_for_prediction.astype(
                float
            ) * float(record.dx)
        cnn_axis.scatter(
            cnn_prediction_peak_x_values,
            prediction[cnn_peak_indices_for_prediction],
            color=marker_color,
            edgecolors=marker_color,
            s=marker_size,
            marker="o",
            label=f"CNN peaks on prediction ({cnn_peak_indices_for_prediction.size})",
            zorder=21,
            rasterized=rasterize_dense_artists,
        )

    if xlim is not None:
        standard_axis.set_xlim(xlim)
    if standard_ylim is not None:
        standard_axis.set_ylim(standard_ylim)
    standard_axis.set_ylabel("Signal amplitude")
    standard_axis.set_title("Standard detection")
    if show_grid:
        standard_axis.grid(True, which="both", alpha=0.2, zorder=0)
    if show_legend:
        standard_axis.legend(
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
        )

    if xlim is not None:
        cnn_axis.set_xlim(xlim)
    if cnn_ylim is not None:
        cnn_axis.set_ylim(cnn_ylim)
    cnn_axis.set_xlabel(x_label)
    cnn_axis.set_ylabel("Signal / prediction amplitude")
    cnn_axis.set_title("CNN detection")
    if show_grid:
        cnn_axis.grid(True, which="both", alpha=0.2, zorder=0)
    if show_legend:
        cnn_axis.legend(
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
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
    if save_path is not None:
        figure.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close(figure)
    return figure


__all__ = [
    "get_detection_threshold",
    "plot_detection_comparison",
]
