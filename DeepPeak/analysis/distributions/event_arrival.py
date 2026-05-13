"""Event-arrival diagnostics and single-purpose Poisson-style plots.

The functions in this module operate on detected event times and summarize how
closely the observed arrivals resemble a Poisson process, both numerically and
through dedicated plotting helpers.
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import colors as mcolors

from ..trace_plots import (
    finalize_single_axis_figure,
    make_or_reuse_single_axis,
    plot_style_context,
)
from ..results import EventArrivalDistributionMetrics, resolve_series_or_result


def _clean_sorted_event_time(event_time: np.ndarray) -> np.ndarray:
    event_time = np.asarray(event_time, dtype=float)
    event_time = event_time[np.isfinite(event_time)]
    return np.sort(event_time)


def _inter_arrival_time(event_time: np.ndarray) -> np.ndarray:
    inter_arrival_time = np.diff(event_time)
    return inter_arrival_time[inter_arrival_time > 0.0]


def _poisson_counts_chi2(
    event_time: np.ndarray,
    observation_start: float,
    observation_end: float,
    number_of_bins: int,
    lambda_hat: float,
) -> Tuple[float, int, float, float, float, np.ndarray]:
    bin_edges = np.linspace(observation_start, observation_end, number_of_bins + 1)
    counts_per_bin, _ = np.histogram(event_time, bins=bin_edges)

    bin_width = float(bin_edges[1] - bin_edges[0])
    mu_hat_per_bin = float(lambda_hat * bin_width)

    max_count = int(np.max(counts_per_bin)) if counts_per_bin.size else 0
    count_values = np.arange(0, max_count + 1)

    expected_frequency = (
        stats.poisson.pmf(count_values, mu_hat_per_bin) * number_of_bins
    )
    observed_frequency = np.bincount(counts_per_bin, minlength=max_count + 1).astype(
        float
    )

    grouped_observed_frequency = []
    grouped_expected_frequency = []
    running_observed = 0.0
    running_expected = 0.0

    for observed_k, expected_k in zip(observed_frequency, expected_frequency):
        running_observed += float(observed_k)
        running_expected += float(expected_k)

        if running_expected >= 5.0:
            grouped_observed_frequency.append(running_observed)
            grouped_expected_frequency.append(running_expected)
            running_observed = 0.0
            running_expected = 0.0

    if running_expected > 0.0:
        grouped_observed_frequency.append(running_observed)
        grouped_expected_frequency.append(running_expected)

    grouped_observed_frequency = np.asarray(grouped_observed_frequency, dtype=float)
    grouped_expected_frequency = np.asarray(grouped_expected_frequency, dtype=float)

    if grouped_observed_frequency.size < 3:
        return np.nan, 0, np.nan, mu_hat_per_bin, bin_width, counts_per_bin

    chi2_counts_stat = float(
        np.sum(
            (grouped_observed_frequency - grouped_expected_frequency) ** 2
            / grouped_expected_frequency
        )
    )
    chi2_counts_dof = max(int(grouped_observed_frequency.size - 2), 1)
    chi2_counts_p_value = float(stats.chi2.sf(chi2_counts_stat, chi2_counts_dof))

    return (
        chi2_counts_stat,
        chi2_counts_dof,
        chi2_counts_p_value,
        mu_hat_per_bin,
        bin_width,
        counts_per_bin,
    )


def compute_event_arrival_distribution_metrics(
    series_or_result,
    index: int,
    detector: Literal["standard", "cnn", "both"] = "both",
    x_axis: Literal["sample", "time"] = "sample",
    number_of_count_bins: int = 50,
    observation_start: Optional[float] = None,
    observation_end: Optional[float] = None,
) -> Dict[str, EventArrivalDistributionMetrics]:
    resolved_result = resolve_series_or_result(series_or_result)
    record = resolved_result.records[index]

    if detector == "standard":
        detector_labels = ("standard",)
    elif detector == "cnn":
        detector_labels = ("cnn",)
    elif detector == "both":
        detector_labels = ("standard", "cnn")
    else:
        raise ValueError('detector must be either "standard", "cnn", or "both".')

    def compute_metrics_for_label(
        label: str, raw_event_times: np.ndarray, dx: float
    ) -> EventArrivalDistributionMetrics:
        event_times = _clean_sorted_event_time(raw_event_times)

        if x_axis == "time":
            event_times = event_times * float(dx)
        elif x_axis != "sample":
            raise ValueError('x_axis must be either "sample" or "time".')

        if event_times.size < 2:
            raise ValueError(
                f"{label}: at least two events are required to compute arrival statistics."
            )

        observation_start_value = float(
            event_times[0] if observation_start is None else observation_start
        )
        observation_end_value = float(
            event_times[-1] if observation_end is None else observation_end
        )
        if observation_end_value <= observation_start_value:
            raise ValueError(
                f"{label}: observation_end must be greater than observation_start."
            )

        event_times = event_times[
            (event_times >= observation_start_value)
            & (event_times <= observation_end_value)
        ]
        if event_times.size < 2:
            raise ValueError(
                f"{label}: fewer than two events remain inside the observation window."
            )

        observation_duration = float(observation_end_value - observation_start_value)
        number_of_events = int(event_times.size)
        lambda_hat = float(number_of_events / observation_duration)

        inter_arrival_times = _inter_arrival_time(event_times)
        number_of_inter_arrival_times = int(inter_arrival_times.size)
        if number_of_inter_arrival_times == 0:
            raise ValueError(f"{label}: no positive inter-arrival times were found.")

        mean_inter_arrival_time = float(np.mean(inter_arrival_times))
        standard_deviation_inter_arrival_time = (
            float(np.std(inter_arrival_times, ddof=1))
            if number_of_inter_arrival_times > 1
            else np.nan
        )
        if mean_inter_arrival_time > 0.0 and np.isfinite(
            standard_deviation_inter_arrival_time
        ):
            coefficient_of_variation_inter_arrival_time = float(
                standard_deviation_inter_arrival_time / mean_inter_arrival_time
            )
        else:
            coefficient_of_variation_inter_arrival_time = np.nan

        if number_of_inter_arrival_times >= 5 and lambda_hat > 0.0:
            ks_exponential_statistic, ks_exponential_p_value = stats.kstest(
                inter_arrival_times, "expon", args=(0.0, 1.0 / lambda_hat)
            )
            rescaled_uniform_values = 1.0 - np.exp(-lambda_hat * inter_arrival_times)
            ks_rescaled_uniform_statistic, ks_rescaled_uniform_p_value = stats.kstest(
                rescaled_uniform_values, "uniform", args=(0.0, 1.0)
            )
        else:
            ks_exponential_statistic = np.nan
            ks_exponential_p_value = np.nan
            ks_rescaled_uniform_statistic = np.nan
            ks_rescaled_uniform_p_value = np.nan

        (
            chi2_count_statistic,
            chi2_count_degrees_of_freedom,
            chi2_count_p_value,
            count_bin_width,
            counts_per_bin,
        ) = _compute_count_chi_square_test(
            event_times=event_times,
            observation_start_value=observation_start_value,
            observation_end_value=observation_end_value,
            number_of_bins=number_of_count_bins,
            lambda_hat_value=lambda_hat,
        )

        mean_count_per_bin = (
            float(np.mean(counts_per_bin)) if counts_per_bin.size else np.nan
        )
        variance_count_per_bin = (
            float(np.var(counts_per_bin, ddof=1)) if counts_per_bin.size > 1 else np.nan
        )
        if mean_count_per_bin > 0.0 and np.isfinite(variance_count_per_bin):
            fano_factor_count_per_bin = float(
                variance_count_per_bin / mean_count_per_bin
            )
        else:
            fano_factor_count_per_bin = np.nan

        return EventArrivalDistributionMetrics(
            label=label,
            number_of_events=number_of_events,
            observation_start=observation_start_value,
            observation_end=observation_end_value,
            observation_duration=observation_duration,
            lambda_hat=lambda_hat,
            number_of_inter_arrival_times=number_of_inter_arrival_times,
            mean_inter_arrival_time=mean_inter_arrival_time,
            standard_deviation_inter_arrival_time=standard_deviation_inter_arrival_time,
            coefficient_of_variation_inter_arrival_time=coefficient_of_variation_inter_arrival_time,
            ks_exponential_statistic=float(ks_exponential_statistic),
            ks_exponential_p_value=float(ks_exponential_p_value),
            ks_rescaled_uniform_statistic=float(ks_rescaled_uniform_statistic),
            ks_rescaled_uniform_p_value=float(ks_rescaled_uniform_p_value),
            number_of_count_bins=int(number_of_count_bins),
            count_bin_width=float(count_bin_width),
            mean_count_per_bin=mean_count_per_bin,
            variance_count_per_bin=variance_count_per_bin,
            fano_factor_count_per_bin=fano_factor_count_per_bin,
            chi2_count_statistic=float(chi2_count_statistic),
            chi2_count_degrees_of_freedom=int(chi2_count_degrees_of_freedom),
            chi2_count_p_value=float(chi2_count_p_value),
            event_times=event_times,
            inter_arrival_times=inter_arrival_times,
            counts_per_bin=counts_per_bin,
        )

    metrics_by_label: Dict[str, EventArrivalDistributionMetrics] = {}
    for label in detector_labels:
        raw_event_times = np.asarray(
            record.standard.peaks if label == "standard" else record.cnn.peaks,
            dtype=float,
        )
        metrics_by_label[label] = compute_metrics_for_label(
            label, raw_event_times, record.dx
        )

    return metrics_by_label


def _compute_count_chi_square_test(
    event_times: np.ndarray,
    observation_start_value: float,
    observation_end_value: float,
    number_of_bins: int,
    lambda_hat_value: float,
) -> Tuple[float, int, float, float, np.ndarray]:
    (
        chi2_statistic,
        degrees_of_freedom,
        chi2_p_value,
        _,
        count_bin_width,
        counts_per_bin,
    ) = _poisson_counts_chi2(
        event_time=event_times,
        observation_start=observation_start_value,
        observation_end=observation_end_value,
        number_of_bins=number_of_bins,
        lambda_hat=lambda_hat_value,
    )
    return (
        chi2_statistic,
        degrees_of_freedom,
        chi2_p_value,
        count_bin_width,
        counts_per_bin,
    )


def _resolve_single_label_metrics(
    metrics_by_label: Dict[str, EventArrivalDistributionMetrics],
    label: Optional[str],
) -> EventArrivalDistributionMetrics:
    if label is None:
        if len(metrics_by_label) != 1:
            raise ValueError(
                "label is required when metrics_by_label contains multiple entries."
            )
        return next(iter(metrics_by_label.values()))
    if label not in metrics_by_label:
        raise ValueError(
            f'Unknown label "{label}". Available labels are {tuple(metrics_by_label.keys())}.'
        )
    return metrics_by_label[label]


def plot_event_raster(
    metrics_by_label: Dict[str, EventArrivalDistributionMetrics],
    *,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 2.5),
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
        axis.vlines(
            metrics.event_times, 0.0, 1.0, linewidth=0.8, color="black", zorder=2
        )
        axis.set_xlim(metrics.observation_start, metrics.observation_end)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([])
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Event time",
            ylabel="Events",
            title=title,
            show_grid=True,
            tight_layout=created_figure,
            save_path=save_path,
            dpi=dpi,
            show=show,
            close=close,
        )


def plot_inter_arrival_histogram(
    metrics_by_label: Dict[str, EventArrivalDistributionMetrics],
    *,
    label: Optional[str] = None,
    expected_lambda_hat: Optional[float] = None,
    expected_label: str = "Expected Poisson",
    figsize: Tuple[float, float] = (8.0, 4.0),
    bins: int = 40,
    max_percentile: float = 99.0,
    line_width: float = 1.2,
    histogram_alpha: float = 0.75,
    histogram_color: str = "C0",
    edge_color: str = "black",
    edge_line_width: float = 0.8,
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
        inter_arrival_times = metrics.inter_arrival_times
        if inter_arrival_times.size > 0:
            histogram_upper = float(np.percentile(inter_arrival_times, max_percentile))
            histogram_upper = max(
                histogram_upper, float(np.max(inter_arrival_times)), 1e-12
            )
            x_values = np.linspace(0.0, histogram_upper, 400)
            axis.hist(
                inter_arrival_times,
                bins=bins,
                density=True,
                color=mcolors.to_rgba(histogram_color, histogram_alpha),
                edgecolor=edge_color,
                linewidth=edge_line_width,
                label="Observed",
                zorder=2,
            )
            if metrics.lambda_hat > 0.0:
                exponential_density = stats.expon.pdf(
                    x_values, loc=0.0, scale=1.0 / metrics.lambda_hat
                )
                axis.plot(
                    x_values,
                    exponential_density,
                    linewidth=line_width,
                    color="C1",
                    label="Fitted exponential",
                    zorder=3,
                )
            if expected_lambda_hat is not None:
                expected_lambda_hat = float(expected_lambda_hat)
                if expected_lambda_hat > 0.0:
                    expected_density = stats.expon.pdf(
                        x_values, loc=0.0, scale=1.0 / expected_lambda_hat
                    )
                    axis.plot(
                        x_values,
                        expected_density,
                        linewidth=line_width,
                        color="C2",
                        linestyle=":",
                        label=expected_label,
                        zorder=4,
                    )
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Inter-arrival time",
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


def plot_rescaled_uniform_qq(
    metrics_by_label: Dict[str, EventArrivalDistributionMetrics],
    *,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    line_width: float = 1.2,
    marker_size: float = 18.0,
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
        if metrics.inter_arrival_times.size > 0 and metrics.lambda_hat > 0.0:
            rescaled_uniform_values = 1.0 - np.exp(
                -metrics.lambda_hat * metrics.inter_arrival_times
            )
            rescaled_uniform_values = np.sort(rescaled_uniform_values)
            expected_uniform_quantiles = (
                np.arange(1, rescaled_uniform_values.size + 1, dtype=float) - 0.5
            ) / rescaled_uniform_values.size
            axis.scatter(
                expected_uniform_quantiles,
                rescaled_uniform_values,
                s=marker_size,
                color="black",
                zorder=3,
                rasterized=True,
            )
            axis.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color="C1",
                linewidth=line_width,
                linestyle="--",
                label="Ideal Uniform(0, 1)",
                zorder=2,
            )
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Expected Uniform quantile",
            ylabel="Observed rescaled quantile",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
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


def plot_count_distribution(
    metrics_by_label: Dict[str, EventArrivalDistributionMetrics],
    *,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.0),
    line_width: float = 1.2,
    histogram_alpha: float = 0.75,
    histogram_color: str = "C0",
    edge_color: str = "black",
    edge_line_width: float = 0.8,
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
        counts_per_bin = np.asarray(metrics.counts_per_bin, dtype=int)
        if counts_per_bin.size > 0:
            observed_frequency = np.bincount(counts_per_bin)
            count_values = np.arange(observed_frequency.size)
            axis.bar(
                count_values,
                observed_frequency,
                color=mcolors.to_rgba(histogram_color, histogram_alpha),
                edgecolor=edge_color,
                linewidth=edge_line_width,
                label="Observed",
                zorder=2,
            )
            expected_frequency = (
                stats.poisson.pmf(count_values, metrics.mean_count_per_bin)
                * metrics.number_of_count_bins
            )
            axis.plot(
                count_values,
                expected_frequency,
                marker="o",
                linewidth=line_width,
                color="C1",
                label="Fitted Poisson",
                zorder=3,
            )
        return finalize_single_axis_figure(
            figure=figure,
            axis=axis,
            xlabel="Count per bin",
            ylabel="Frequency",
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


__all__ = [
    "compute_event_arrival_distribution_metrics",
    "plot_count_distribution",
    "plot_event_raster",
    "plot_inter_arrival_histogram",
    "plot_rescaled_uniform_qq",
]
