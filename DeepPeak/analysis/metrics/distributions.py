"""Distribution-style metric models and their plotting helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import colors as mcolors


@dataclass(frozen=True)
class PoissonSeriesDiagnostics:
    """Compact Poisson diagnostic summary for one detector."""

    label: str
    lambda_hat: float
    observation_start: float
    observation_end: float
    observation_duration: float
    number_of_events: int
    number_of_inter_arrival_times: int
    ks_inter_arrival_stat: float
    ks_inter_arrival_p_value: float
    chi2_counts_stat: float
    chi2_counts_dof: int
    chi2_counts_p_value: float
    mu_hat_per_bin: float
    bin_width: float
    ks_rescaled_uniform_stat: float
    ks_rescaled_uniform_p_value: float


@dataclass(frozen=True)
class EventArrivalDistribution:
    """Event-arrival diagnostics for one detector on one trace."""

    label: str
    number_of_events: int
    observation_start: float
    observation_end: float
    observation_duration: float
    lambda_hat: float
    number_of_inter_arrival_times: int
    mean_inter_arrival_time: float
    standard_deviation_inter_arrival_time: float
    coefficient_of_variation_inter_arrival_time: float
    ks_exponential_statistic: float
    ks_exponential_p_value: float
    ks_rescaled_uniform_statistic: float
    ks_rescaled_uniform_p_value: float
    number_of_count_bins: int
    count_bin_width: float
    mean_count_per_bin: float
    variance_count_per_bin: float
    fano_factor_count_per_bin: float
    chi2_count_statistic: float
    chi2_count_degrees_of_freedom: int
    chi2_count_p_value: float
    event_times: np.ndarray
    inter_arrival_times: np.ndarray
    counts_per_bin: np.ndarray

    class PlotAccessor:
        """Namespace exposing event-arrival plots for one metrics object."""

        def __init__(self, metrics: "EventArrivalDistribution") -> None:
            self._metrics = metrics

        def raster(
            self,
            *,
            figsize: tuple[float, float] = (8.0, 2.5),
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            axis.vlines(
                self._metrics.event_times,
                0.0,
                1.0,
                linewidth=0.8,
                color="black",
                zorder=2,
            )
            axis.set_xlim(
                self._metrics.observation_start, self._metrics.observation_end
            )
            axis.set_ylim(0.0, 1.0)
            axis.set_yticks([])
            axis.set_xlabel("Event time")
            axis.set_ylabel("Events")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def histogram(
            self,
            *,
            expected_lambda_hat: Optional[float] = None,
            expected_label: str = "Expected Poisson",
            figsize: tuple[float, float] = (8.0, 4.0),
            bins: int = 40,
            max_percentile: float = 99.0,
            xlim_quantile: Optional[float] = None,
            line_width: float = 1.2,
            histogram_alpha: float = 0.75,
            histogram_color: str = "C0",
            edge_color: str = "black",
            edge_line_width: float = 0.8,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            inter_arrival_times = self._metrics.inter_arrival_times
            if inter_arrival_times.size > 0:
                histogram_upper = float(
                    np.percentile(inter_arrival_times, max_percentile)
                )
                histogram_upper = max(histogram_upper, 1e-12)
                x_limit = histogram_upper
                if xlim_quantile is not None:
                    if not 0.0 < xlim_quantile <= 1.0:
                        raise ValueError(
                            "xlim_quantile must be in the interval (0, 1]."
                        )
                    x_limit = float(np.quantile(inter_arrival_times, xlim_quantile))
                    x_limit = max(x_limit, 1e-12)
                x_values = np.linspace(0.0, histogram_upper, 400)
                axis.hist(
                    inter_arrival_times,
                    bins=bins,
                    range=(0.0, x_limit),
                    density=True,
                    color=mcolors.to_rgba(histogram_color, histogram_alpha),
                    edgecolor=edge_color,
                    linewidth=edge_line_width,
                    label="Observed",
                    zorder=2,
                )
                if self._metrics.lambda_hat > 0.0:
                    exponential_density = stats.expon.pdf(
                        x_values, loc=0.0, scale=1.0 / self._metrics.lambda_hat
                    )
                    axis.plot(
                        x_values,
                        exponential_density,
                        linewidth=line_width,
                        color="black",
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
                            color="black",
                            linestyle=":",
                            label=expected_label,
                            zorder=4,
                        )
                axis.set_xlim(0.0, x_limit)

            axis.set_xlabel("Inter-arrival time")
            axis.set_ylabel("Density")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper right",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def qq(
            self,
            *,
            figsize: tuple[float, float] = (5.0, 5.0),
            line_width: float = 1.2,
            marker_size: float = 18.0,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            if (
                self._metrics.inter_arrival_times.size > 0
                and self._metrics.lambda_hat > 0.0
            ):
                rescaled_uniform_values = 1.0 - np.exp(
                    -self._metrics.lambda_hat * self._metrics.inter_arrival_times
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

            axis.set_xlabel("Expected Uniform quantile")
            axis.set_ylabel("Observed rescaled quantile")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper left",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def count_distribution(
            self,
            *,
            figsize: tuple[float, float] = (8.0, 4.0),
            line_width: float = 1.2,
            histogram_alpha: float = 0.75,
            histogram_color: str = "C0",
            edge_color: str = "black",
            edge_line_width: float = 0.8,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            counts_per_bin = np.asarray(self._metrics.counts_per_bin, dtype=int)
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
                    stats.poisson.pmf(count_values, self._metrics.mean_count_per_bin)
                    * self._metrics.number_of_count_bins
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

            axis.set_xlabel("Count per bin")
            axis.set_ylabel("Frequency")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper right",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

    @property
    def plot(self) -> "EventArrivalDistribution.PlotAccessor":
        return self.PlotAccessor(self)


@dataclass(frozen=True)
class PeakAmplitudeDistribution:
    """Peak-amplitude diagnostics for one detector on one trace."""

    label: str
    number_of_peaks: int
    mean_amplitude: float
    median_amplitude: float
    minimum_amplitude: float
    maximum_amplitude: float
    standard_deviation_amplitude: float
    coefficient_of_variation_amplitude: float
    skewness_amplitude: float
    kurtosis_amplitude: float
    fitted_normal_mean: float
    fitted_normal_standard_deviation: float
    ks_normal_statistic: float
    ks_normal_p_value: float
    amplitudes: np.ndarray

    class PlotAccessor:
        """Namespace exposing amplitude plots for one metrics object."""

        def __init__(self, metrics: "PeakAmplitudeDistribution") -> None:
            self._metrics = metrics

        def histogram(
            self,
            *,
            figsize: tuple[float, float] = (8.0, 4.0),
            bins: int = 40,
            xlim_quantile: Optional[float] = None,
            histogram_alpha: float = 0.75,
            histogram_color: str = "C0",
            edge_color: str = "black",
            edge_line_width: float = 0.8,
            line_width: float = 1.2,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            x_limit: Optional[float] = None
            if self._metrics.amplitudes.size > 0 and xlim_quantile is not None:
                if not 0.0 < xlim_quantile <= 1.0:
                    raise ValueError("xlim_quantile must be in the interval (0, 1].")
                x_limit = float(np.quantile(self._metrics.amplitudes, xlim_quantile))
                x_limit = max(x_limit, 1e-12)
            axis.hist(
                self._metrics.amplitudes,
                bins=bins,
                range=(0.0, x_limit) if x_limit is not None else None,
                density=True,
                color=mcolors.to_rgba(histogram_color, histogram_alpha),
                edgecolor=edge_color,
                linewidth=edge_line_width,
                label="Observed",
                zorder=2,
            )
            if self._metrics.fitted_normal_standard_deviation > 0.0:
                x_values = np.linspace(
                    self._metrics.minimum_amplitude,
                    self._metrics.maximum_amplitude,
                    400,
                )
                density = stats.norm.pdf(
                    x_values,
                    loc=self._metrics.fitted_normal_mean,
                    scale=self._metrics.fitted_normal_standard_deviation,
                )
                axis.plot(
                    x_values,
                    density,
                    color="C1",
                    linewidth=line_width,
                    label="Fitted normal",
                    zorder=3,
                )
            if x_limit is not None:
                axis.set_xlim(0.0, x_limit)
            axis.set_xlabel("Peak amplitude")
            axis.set_ylabel("Density")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper right",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def qq(
            self,
            *,
            figsize: tuple[float, float] = (5.0, 5.0),
            marker_size: float = 18.0,
            line_width: float = 1.2,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            amplitudes = np.sort(self._metrics.amplitudes)
            if amplitudes.size > 0:
                probabilities = (
                    np.arange(1, amplitudes.size + 1, dtype=float) - 0.5
                ) / amplitudes.size
                if self._metrics.fitted_normal_standard_deviation > 0.0:
                    expected_quantiles = stats.norm.ppf(
                        probabilities,
                        loc=self._metrics.fitted_normal_mean,
                        scale=self._metrics.fitted_normal_standard_deviation,
                    )
                else:
                    expected_quantiles = np.full_like(
                        amplitudes, self._metrics.fitted_normal_mean
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
            axis.set_xlabel("Expected normal quantile")
            axis.set_ylabel("Observed peak amplitude")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper left",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def ecdf(
            self,
            *,
            figsize: tuple[float, float] = (8.0, 4.0),
            line_width: float = 1.2,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            amplitudes = np.sort(self._metrics.amplitudes)
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
            if self._metrics.fitted_normal_standard_deviation > 0.0:
                x_values = np.linspace(
                    self._metrics.minimum_amplitude,
                    self._metrics.maximum_amplitude,
                    400,
                )
                fitted_cdf = stats.norm.cdf(
                    x_values,
                    loc=self._metrics.fitted_normal_mean,
                    scale=self._metrics.fitted_normal_standard_deviation,
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
            axis.set_xlabel("Peak amplitude")
            axis.set_ylabel("Cumulative probability")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="lower right",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

    @property
    def plot(self) -> "PeakAmplitudeDistribution.PlotAccessor":
        return self.PlotAccessor(self)


@dataclass(frozen=True)
class PeakWidthDistribution:
    """Peak-width diagnostics for one detector on one trace."""

    label: str
    x_axis: str
    width_unit_label: str
    number_of_peaks: int
    mean_width: float
    median_width: float
    minimum_width: float
    maximum_width: float
    standard_deviation_width: float
    coefficient_of_variation_width: float
    skewness_width: float
    kurtosis_width: float
    fitted_lognormal_shape: float
    fitted_lognormal_loc: float
    fitted_lognormal_scale: float
    ks_lognormal_statistic: float
    ks_lognormal_p_value: float
    widths: np.ndarray

    class PlotAccessor:
        """Namespace exposing width plots for one metrics object."""

        def __init__(self, metrics: "PeakWidthDistribution") -> None:
            self._metrics = metrics

        def histogram(
            self,
            *,
            figsize: tuple[float, float] = (8.0, 4.0),
            bins: int = 40,
            xlim_quantile: Optional[float] = None,
            histogram_alpha: float = 0.75,
            histogram_color: str = "C0",
            edge_color: str = "black",
            edge_line_width: float = 0.8,
            line_width: float = 1.2,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            x_limit: Optional[float] = None
            if self._metrics.widths.size > 0 and xlim_quantile is not None:
                if not 0.0 < xlim_quantile <= 1.0:
                    raise ValueError("xlim_quantile must be in the interval (0, 1].")
                x_limit = float(np.quantile(self._metrics.widths, xlim_quantile))
                x_limit = max(x_limit, 1e-12)
            axis.hist(
                self._metrics.widths,
                bins=bins,
                range=(0.0, x_limit) if x_limit is not None else None,
                density=True,
                color=mcolors.to_rgba(histogram_color, histogram_alpha),
                edgecolor=edge_color,
                linewidth=edge_line_width,
                label="Observed",
                zorder=2,
            )
            if (
                self._metrics.fitted_lognormal_shape > 0.0
                and self._metrics.fitted_lognormal_scale > 0.0
            ):
                x_values = np.linspace(
                    self._metrics.minimum_width,
                    self._metrics.maximum_width,
                    400,
                )
                density = stats.lognorm.pdf(
                    x_values,
                    self._metrics.fitted_lognormal_shape,
                    loc=self._metrics.fitted_lognormal_loc,
                    scale=self._metrics.fitted_lognormal_scale,
                )
                axis.plot(
                    x_values,
                    density,
                    color="C1",
                    linewidth=line_width,
                    label="Fitted lognormal",
                    zorder=3,
                )
            if x_limit is not None:
                axis.set_xlim(0.0, x_limit)
            axis.set_xlabel(f"Peak width [{self._metrics.width_unit_label}]")
            axis.set_ylabel("Density")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper right",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def qq(
            self,
            *,
            figsize: tuple[float, float] = (5.0, 5.0),
            marker_size: float = 18.0,
            line_width: float = 1.2,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            widths = np.sort(self._metrics.widths)
            if widths.size > 0:
                probabilities = (
                    np.arange(1, widths.size + 1, dtype=float) - 0.5
                ) / widths.size
                if (
                    self._metrics.fitted_lognormal_shape > 0.0
                    and self._metrics.fitted_lognormal_scale > 0.0
                ):
                    expected_quantiles = stats.lognorm.ppf(
                        probabilities,
                        self._metrics.fitted_lognormal_shape,
                        loc=self._metrics.fitted_lognormal_loc,
                        scale=self._metrics.fitted_lognormal_scale,
                    )
                else:
                    expected_quantiles = np.full_like(widths, self._metrics.mean_width)
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
            axis.set_xlabel(
                f"Expected lognormal quantile [{self._metrics.width_unit_label}]"
            )
            axis.set_ylabel(f"Observed peak width [{self._metrics.width_unit_label}]")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="upper left",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def ecdf(
            self,
            *,
            figsize: tuple[float, float] = (8.0, 4.0),
            line_width: float = 1.2,
            show_legend: bool = True,
            title: Optional[str] = None,
            ax: Optional[Any] = None,
            save_path: Optional[Path | str] = None,
            dpi: int = 300,
            show: bool = False,
            close: bool = False,
        ) -> Any:
            if ax is not None:
                figure, axis = ax.figure, ax
                created_figure = False
            else:
                figure, axis = plt.subplots(figsize=figsize)
                created_figure = True
            widths = np.sort(self._metrics.widths)
            empirical_cdf = (
                np.arange(1, widths.size + 1, dtype=float) - 0.5
            ) / widths.size
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
                self._metrics.fitted_lognormal_shape > 0.0
                and self._metrics.fitted_lognormal_scale > 0.0
            ):
                x_values = np.linspace(
                    self._metrics.minimum_width,
                    self._metrics.maximum_width,
                    400,
                )
                fitted_cdf = stats.lognorm.cdf(
                    x_values,
                    self._metrics.fitted_lognormal_shape,
                    loc=self._metrics.fitted_lognormal_loc,
                    scale=self._metrics.fitted_lognormal_scale,
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
            axis.set_xlabel(f"Peak width [{self._metrics.width_unit_label}]")
            axis.set_ylabel("Cumulative probability")
            if title is not None:
                axis.set_title(title)
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            if show_legend:
                axis.legend(
                    loc="lower right",
                    frameon=True,
                    framealpha=1.0,
                    facecolor="white",
                    edgecolor="black",
                )
            if created_figure:
                figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

    @property
    def plot(self) -> "PeakWidthDistribution.PlotAccessor":
        return self.PlotAccessor(self)
