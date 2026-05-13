"""Peak-width distribution diagnostics and plots."""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import colors as mcolors

from ..results import PeakWidthDistributionMetrics, resolve_series_or_result
from ..trace_plots import (
    finalize_single_axis_figure,
    make_or_reuse_single_axis,
    plot_style_context,
)


DetectorLabel = Literal["standard", "cnn", "both"]
XAxisLabel = Literal["sample", "time"]


def _resolve_single_label_metrics(
    metrics_by_label: Dict[str, PeakWidthDistributionMetrics],
    label: Optional[str],
) -> PeakWidthDistributionMetrics:
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


def _extract_peak_widths(
    detection: object, dx: float, x_axis: XAxisLabel
) -> np.ndarray:
    properties = getattr(detection, "properties", {})
    widths = None
    if isinstance(properties, dict):
        widths = properties.get("widths_pixels")

    if widths is None:
        start_indices = np.asarray(
            properties.get("start_indices", np.asarray([])), dtype=float
        )
        end_indices = np.asarray(
            properties.get("end_indices", np.asarray([])), dtype=float
        )
        if (
            start_indices.size
            and end_indices.size
            and start_indices.size == end_indices.size
        ):
            widths = end_indices - start_indices + 1.0
        else:
            widths = np.asarray([])

    widths = np.asarray(widths, dtype=float).ravel()
    widths = widths[np.isfinite(widths) & (widths > 0.0)]

    if x_axis == "time":
        widths = widths * float(dx)
    elif x_axis != "sample":
        raise ValueError('x_axis must be either "sample" or "time".')

    return widths


def compute_peak_width_distribution_metrics(
    series_or_result,
    index: int,
    detector: DetectorLabel = "both",
    x_axis: XAxisLabel = "sample",
) -> Dict[str, PeakWidthDistributionMetrics]:
    """Compute peak-width diagnostics for one trace and one or more detectors."""

    resolved_result = resolve_series_or_result(series_or_result)
    record = resolved_result.records[index]

    metrics_by_label: Dict[str, PeakWidthDistributionMetrics] = {}
    width_unit = "Time" if x_axis == "time" else "Samples"

    for label in _detector_labels(detector):
        detection = record.standard if label == "standard" else record.cnn
        widths = _extract_peak_widths(detection=detection, dx=record.dx, x_axis=x_axis)
        if widths.size == 0:
            raise ValueError(f"{label}: at least one detected peak width is required.")

        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))
        minimum_width = float(np.min(widths))
        maximum_width = float(np.max(widths))
        standard_deviation_width = (
            float(np.std(widths, ddof=1)) if widths.size > 1 else np.nan
        )
        coefficient_of_variation_width = (
            float(standard_deviation_width / mean_width)
            if np.isfinite(standard_deviation_width) and mean_width != 0.0
            else np.nan
        )
        skewness_width = (
            float(stats.skew(widths, bias=False)) if widths.size > 2 else np.nan
        )
        kurtosis_width = (
            float(stats.kurtosis(widths, fisher=True, bias=False))
            if widths.size > 3
            else np.nan
        )

        fitted_shape, fitted_loc, fitted_scale = stats.lognorm.fit(widths, floc=0.0)
        fitted_shape = float(fitted_shape)
        fitted_loc = float(fitted_loc)
        fitted_scale = float(fitted_scale)

        if widths.size >= 5 and fitted_shape > 0.0 and fitted_scale > 0.0:
            ks_lognormal_statistic, ks_lognormal_p_value = stats.kstest(
                widths,
                "lognorm",
                args=(fitted_shape, fitted_loc, fitted_scale),
            )
            ks_lognormal_statistic = float(ks_lognormal_statistic)
            ks_lognormal_p_value = float(ks_lognormal_p_value)
        else:
            ks_lognormal_statistic = np.nan
            ks_lognormal_p_value = np.nan

        metrics_by_label[label] = PeakWidthDistributionMetrics(
            label=label,
            x_axis=x_axis,
            width_unit_label=width_unit,
            number_of_peaks=int(widths.size),
            mean_width=mean_width,
            median_width=median_width,
            minimum_width=minimum_width,
            maximum_width=maximum_width,
            standard_deviation_width=standard_deviation_width,
            coefficient_of_variation_width=coefficient_of_variation_width,
            skewness_width=skewness_width,
            kurtosis_width=kurtosis_width,
            fitted_lognormal_shape=fitted_shape,
            fitted_lognormal_loc=fitted_loc,
            fitted_lognormal_scale=fitted_scale,
            ks_lognormal_statistic=ks_lognormal_statistic,
            ks_lognormal_p_value=ks_lognormal_p_value,
            widths=widths,
        )

    return metrics_by_label


def plot_peak_width_histogram(
    metrics_by_label: Dict[str, PeakWidthDistributionMetrics],
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
    """Plot the peak-width histogram against a fitted lognormal density.

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
        Figure containing the peak-width histogram.
    """

    metrics = _resolve_single_label_metrics(metrics_by_label, label)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        axis.hist(
            metrics.widths,
            bins=bins,
            density=True,
            color=mcolors.to_rgba(histogram_color, histogram_alpha),
            edgecolor=edge_color,
            linewidth=edge_line_width,
            label="Observed",
            zorder=2,
        )

        if (
            metrics.fitted_lognormal_shape > 0.0
            and metrics.fitted_lognormal_scale > 0.0
        ):
            x_values = np.linspace(metrics.minimum_width, metrics.maximum_width, 400)
            density = stats.lognorm.pdf(
                x_values,
                metrics.fitted_lognormal_shape,
                loc=metrics.fitted_lognormal_loc,
                scale=metrics.fitted_lognormal_scale,
            )
            axis.plot(
                x_values,
                density,
                color="C1",
                linewidth=line_width,
                label="Fitted lognormal",
                zorder=3,
            )

        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel=f"Peak width [{metrics.width_unit_label}]",
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


def plot_peak_width_qq(
    metrics_by_label: Dict[str, PeakWidthDistributionMetrics],
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
    metrics = _resolve_single_label_metrics(metrics_by_label, label)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        widths = np.sort(metrics.widths)

        if widths.size > 0:
            probabilities = (
                np.arange(1, widths.size + 1, dtype=float) - 0.5
            ) / widths.size
            if (
                metrics.fitted_lognormal_shape > 0.0
                and metrics.fitted_lognormal_scale > 0.0
            ):
                expected_quantiles = stats.lognorm.ppf(
                    probabilities,
                    metrics.fitted_lognormal_shape,
                    loc=metrics.fitted_lognormal_loc,
                    scale=metrics.fitted_lognormal_scale,
                )
            else:
                expected_quantiles = np.full_like(widths, metrics.mean_width)

            axis.scatter(
                expected_quantiles,
                widths,
                s=marker_size,
                color="black",
                zorder=3,
                rasterized=True,
            )
            line_min = float(min(np.min(expected_quantiles), np.min(widths)))
            line_max = float(max(np.max(expected_quantiles), np.max(widths)))
            axis.plot(
                [line_min, line_max],
                [line_min, line_max],
                color="C1",
                linewidth=line_width,
                linestyle="--",
                label="Ideal lognormal",
                zorder=2,
            )

        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel=f"Expected lognormal quantile [{metrics.width_unit_label}]",
            ylabel=f"Observed peak width [{metrics.width_unit_label}]",
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


def plot_peak_width_ecdf(
    metrics_by_label: Dict[str, PeakWidthDistributionMetrics],
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
    metrics = _resolve_single_label_metrics(metrics_by_label, label)

    with plot_style_context():
        figure, axis, created_figure = make_or_reuse_single_axis(figsize=figsize, ax=ax)
        widths = np.sort(metrics.widths)
        empirical_cdf = (np.arange(1, widths.size + 1, dtype=float) - 0.5) / widths.size
        axis.step(
            widths,
            empirical_cdf,
            where="mid",
            linewidth=line_width,
            color="black",
            label="Empirical CDF",
            zorder=3,
        )

        if (
            metrics.fitted_lognormal_shape > 0.0
            and metrics.fitted_lognormal_scale > 0.0
        ):
            x_values = np.linspace(metrics.minimum_width, metrics.maximum_width, 400)
            fitted_cdf = stats.lognorm.cdf(
                x_values,
                metrics.fitted_lognormal_shape,
                loc=metrics.fitted_lognormal_loc,
                scale=metrics.fitted_lognormal_scale,
            )
            axis.plot(
                x_values,
                fitted_cdf,
                color="C1",
                linewidth=line_width,
                label="Fitted lognormal CDF",
                zorder=2,
            )

        axis.set_ylim(0.0, 1.0)
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel=f"Peak width [{metrics.width_unit_label}]",
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
    "compute_peak_width_distribution_metrics",
    "plot_peak_width_ecdf",
    "plot_peak_width_histogram",
    "plot_peak_width_qq",
]
