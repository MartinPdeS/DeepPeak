"""Peak-amplitude distribution diagnostics and plots.

These helpers mirror the event-arrival analysis workflow for peak amplitudes.
They operate on the canonical trace records produced by the analysis pipeline,
compute summary statistics for detected peak amplitudes on the processed trace,
and provide small, explicit plotting helpers for notebooks.
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import colors as mcolors

from ..results import PeakAmplitudeDistributionMetrics, resolve_series_or_result
from ..trace_plots import (
    finalize_single_axis_figure,
    make_or_reuse_single_axis,
    plot_style_context,
)


DetectorLabel = Literal["standard", "cnn", "both"]


def _resolve_single_label_metrics(
    metrics_by_label: Dict[str, PeakAmplitudeDistributionMetrics],
    label: Optional[str],
) -> PeakAmplitudeDistributionMetrics:
    if label is None:
        if len(metrics_by_label) != 1:
            raise ValueError(
                "label must be provided when multiple detector entries are present."
            )
        return next(iter(metrics_by_label.values()))

    try:
        return metrics_by_label[label]
    except KeyError as error:
        available_labels = ", ".join(sorted(metrics_by_label))
        raise ValueError(
            f"Unknown label {label!r}. Available labels: {available_labels}."
        ) from error


def _detector_labels(detector: DetectorLabel) -> tuple[str, ...]:
    if detector == "standard":
        return ("standard",)
    if detector == "cnn":
        return ("cnn",)
    if detector == "both":
        return ("standard", "cnn")
    raise ValueError('detector must be either "standard", "cnn", or "both".')


def _extract_peak_amplitudes(values: np.ndarray, detection: object) -> np.ndarray:
    peak_indices = np.asarray(getattr(detection, "peaks", np.asarray([])), dtype=int)
    peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < values.size)]

    amplitudes = np.asarray(values, dtype=float).ravel()[peak_indices]
    return amplitudes[np.isfinite(amplitudes)]


def compute_peak_amplitude_distribution_metrics(
    series_or_result,
    index: int,
    detector: DetectorLabel = "both",
) -> Dict[str, PeakAmplitudeDistributionMetrics]:
    """Compute peak-amplitude diagnostics for one trace and one or more detectors.

    Parameters
    ----------
    series_or_result:
        A series instance or previously computed series result.
    index:
        Index of the trace record inside the sorted series result.
    detector:
        Which detector output to analyze: standard, CNN, or both.

    Returns
    -------
    dict
        Mapping from detector label to amplitude diagnostics.
    """

    resolved_result = resolve_series_or_result(series_or_result)
    record = resolved_result.records[index]

    metrics_by_label: Dict[str, PeakAmplitudeDistributionMetrics] = {}

    for label in _detector_labels(detector):
        values = np.asarray(record.signal, dtype=float).ravel()
        if label == "standard":
            detection = record.standard
        else:
            detection = record.cnn

        amplitudes = _extract_peak_amplitudes(values=values, detection=detection)
        if amplitudes.size == 0:
            raise ValueError(
                f"{label}: at least one detected peak amplitude is required."
            )

        mean_amplitude = float(np.mean(amplitudes))
        median_amplitude = float(np.median(amplitudes))
        minimum_amplitude = float(np.min(amplitudes))
        maximum_amplitude = float(np.max(amplitudes))
        standard_deviation_amplitude = (
            float(np.std(amplitudes, ddof=1)) if amplitudes.size > 1 else np.nan
        )
        coefficient_of_variation_amplitude = (
            float(standard_deviation_amplitude / mean_amplitude)
            if np.isfinite(standard_deviation_amplitude) and mean_amplitude != 0.0
            else np.nan
        )
        skewness_amplitude = (
            float(stats.skew(amplitudes, bias=False)) if amplitudes.size > 2 else np.nan
        )
        kurtosis_amplitude = (
            float(stats.kurtosis(amplitudes, fisher=True, bias=False))
            if amplitudes.size > 3
            else np.nan
        )

        fitted_normal_mean, fitted_normal_standard_deviation = stats.norm.fit(
            amplitudes
        )
        fitted_normal_mean = float(fitted_normal_mean)
        fitted_normal_standard_deviation = float(fitted_normal_standard_deviation)

        if amplitudes.size >= 5 and fitted_normal_standard_deviation > 0.0:
            ks_normal_statistic, ks_normal_p_value = stats.kstest(
                amplitudes,
                "norm",
                args=(fitted_normal_mean, fitted_normal_standard_deviation),
            )
            ks_normal_statistic = float(ks_normal_statistic)
            ks_normal_p_value = float(ks_normal_p_value)
        else:
            ks_normal_statistic = np.nan
            ks_normal_p_value = np.nan

        metrics_by_label[label] = PeakAmplitudeDistributionMetrics(
            label=label,
            number_of_peaks=int(amplitudes.size),
            mean_amplitude=mean_amplitude,
            median_amplitude=median_amplitude,
            minimum_amplitude=minimum_amplitude,
            maximum_amplitude=maximum_amplitude,
            standard_deviation_amplitude=standard_deviation_amplitude,
            coefficient_of_variation_amplitude=coefficient_of_variation_amplitude,
            skewness_amplitude=skewness_amplitude,
            kurtosis_amplitude=kurtosis_amplitude,
            fitted_normal_mean=fitted_normal_mean,
            fitted_normal_standard_deviation=fitted_normal_standard_deviation,
            ks_normal_statistic=ks_normal_statistic,
            ks_normal_p_value=ks_normal_p_value,
            amplitudes=amplitudes,
        )

    return metrics_by_label


def plot_peak_amplitude_histogram(
    metrics_by_label: Dict[str, PeakAmplitudeDistributionMetrics],
    *,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.0),
    bins: int = 40,
    histogram_alpha: float = 0.75,
    histogram_color: str = "C0",
    edge_color: str = "black",
    edge_line_width: float = 0.8,
    line_width: float = 1.2,
    show_legend: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
    close: bool = False,
) -> plt.Figure:
    """Plot the peak-amplitude histogram against a fitted normal density.

    Parameters
    ----------
    histogram_color : str, default="C0"
        Face color used for the histogram bars.
    edge_color : str, default="black"
        Edge color used for the histogram bars.
    edge_line_width : float, default=0.8
        Edge line width used for the histogram bars.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the peak-amplitude histogram.
    """

    metrics = _resolve_single_label_metrics(metrics_by_label, label)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        axis.hist(
            metrics.amplitudes,
            bins=bins,
            density=True,
            color=mcolors.to_rgba(histogram_color, histogram_alpha),
            edgecolor=edge_color,
            linewidth=edge_line_width,
            label="Observed",
            zorder=2,
        )

        if metrics.fitted_normal_standard_deviation > 0.0:
            x_values = np.linspace(
                metrics.minimum_amplitude, metrics.maximum_amplitude, 400
            )
            density = stats.norm.pdf(
                x_values,
                loc=metrics.fitted_normal_mean,
                scale=metrics.fitted_normal_standard_deviation,
            )
            axis.plot(
                x_values,
                density,
                color="C1",
                linewidth=line_width,
                label="Fitted normal",
                zorder=3,
            )

        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Peak amplitude",
            ylabel="Density",
            title=title,
            show_legend=show_legend,
            legend_loc="upper right",
            show_grid=True,
            tight_layout=created_figure,
            save_path=save_path,
            dpi=dpi,
            show=show,
            close=close,
        )


def plot_peak_amplitude_qq(
    metrics_by_label: Dict[str, PeakAmplitudeDistributionMetrics],
    *,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    marker_size: float = 18.0,
    line_width: float = 1.2,
    show_legend: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
    close: bool = False,
) -> plt.Figure:
    """Plot a normal Q-Q diagnostic for detected peak amplitudes."""

    metrics = _resolve_single_label_metrics(metrics_by_label, label)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        amplitudes = np.sort(metrics.amplitudes)

        if amplitudes.size > 0:
            probabilities = (
                np.arange(1, amplitudes.size + 1, dtype=float) - 0.5
            ) / amplitudes.size

            if metrics.fitted_normal_standard_deviation > 0.0:
                expected_quantiles = stats.norm.ppf(
                    probabilities,
                    loc=metrics.fitted_normal_mean,
                    scale=metrics.fitted_normal_standard_deviation,
                )
            else:
                expected_quantiles = np.full_like(
                    amplitudes, metrics.fitted_normal_mean
                )

            axis.scatter(
                expected_quantiles,
                amplitudes,
                s=marker_size,
                color="black",
                zorder=3,
                rasterized=True,
            )
            line_min = float(min(np.min(expected_quantiles), np.min(amplitudes)))
            line_max = float(max(np.max(expected_quantiles), np.max(amplitudes)))
            axis.plot(
                [line_min, line_max],
                [line_min, line_max],
                color="C1",
                linewidth=line_width,
                linestyle="--",
                label="Ideal normal",
                zorder=2,
            )

        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Expected normal quantile",
            ylabel="Observed peak amplitude",
            title=title,
            show_legend=show_legend,
            legend_loc="upper left",
            show_grid=True,
            tight_layout=created_figure,
            save_path=save_path,
            dpi=dpi,
            show=show,
            close=close,
        )


def plot_peak_amplitude_ecdf(
    metrics_by_label: Dict[str, PeakAmplitudeDistributionMetrics],
    *,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.0),
    line_width: float = 1.2,
    show_legend: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
    close: bool = False,
) -> plt.Figure:
    """Plot the empirical peak-amplitude CDF against the fitted normal CDF."""

    metrics = _resolve_single_label_metrics(metrics_by_label, label)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        amplitudes = np.sort(metrics.amplitudes)
        empirical_cdf = (
            np.arange(1, amplitudes.size + 1, dtype=float) - 0.5
        ) / amplitudes.size
        axis.step(
            amplitudes,
            empirical_cdf,
            where="mid",
            linewidth=line_width,
            color="black",
            label="Empirical CDF",
            zorder=3,
        )

        if metrics.fitted_normal_standard_deviation > 0.0:
            x_values = np.linspace(
                metrics.minimum_amplitude, metrics.maximum_amplitude, 400
            )
            fitted_cdf = stats.norm.cdf(
                x_values,
                loc=metrics.fitted_normal_mean,
                scale=metrics.fitted_normal_standard_deviation,
            )
            axis.plot(
                x_values,
                fitted_cdf,
                color="C1",
                linewidth=line_width,
                label="Fitted normal CDF",
                zorder=2,
            )

        axis.set_ylim(0.0, 1.0)
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Peak amplitude",
            ylabel="Cumulative probability",
            title=title,
            show_legend=show_legend,
            legend_loc="lower right",
            show_grid=True,
            tight_layout=created_figure,
            save_path=save_path,
            dpi=dpi,
            show=show,
            close=close,
        )


__all__ = [
    "compute_peak_amplitude_distribution_metrics",
    "plot_peak_amplitude_ecdf",
    "plot_peak_amplitude_histogram",
    "plot_peak_amplitude_qq",
]
