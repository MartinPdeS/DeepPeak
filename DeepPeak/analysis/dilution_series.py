"""Dilution-series orchestration for processed-signal and WaveNet analysis.

This module handles the folder-level workflow: loading traces, associating them
with dilutions, running the single-trace analyzer, and exposing series-level
plots and diagnostics.
"""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .distributions import (
    compute_event_arrival_distribution_metrics,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
)
from .metrics import (
    EventArrivalDistribution,
    PeakAmplitudeDistribution,
    PeakCountSeriesResult,
    PeakDetectionResult,
    PeakWidthDistribution,
    TraceRecord,
    resolve_series_or_result,
)

from .triggers import BasePeakTrigger
from .wavenet_trace import CNNTraceAnalyzer, StandardTraceAnalyzer


def _iterate_explicit_trace_files(
    folder: Path,
    trace_files: List[Tuple[Union[str, Path], float]],
) -> Iterator[Tuple[float, Path]]:
    """Yield explicitly provided ``(dilution, filename)`` pairs.

    Relative filenames are resolved against ``folder`` so notebook code can stay
    concise while avoiding brittle filename parsing.

    Parameters
    ----------
    folder : pathlib.Path
        Base folder used to resolve relative filenames.
    trace_files : list of tuple
        Explicit ``(filename, dilution)`` pairs.

    Returns
    -------
    iterator of tuple
        Iterator yielding resolved ``(dilution, filename)`` pairs.
    """
    for filename, dilution in trace_files:
        resolved_filename = Path(filename)
        if not resolved_filename.is_absolute():
            resolved_filename = folder / resolved_filename

        yield float(dilution), resolved_filename


class DilutionSeries:
    """Compute and visualize standard and WaveNet-based peak diagnostics over a dilution series."""

    @staticmethod
    def _resolve_single_label_metrics(
        metrics_by_label: Mapping[str, Any],
        label: Optional[str],
    ) -> Any:
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

    class PoissonAnalysisAccessor:
        """Namespace exposing Poisson-style diagnostics for one dilution series."""

        class PlotAccessor:
            """Namespace exposing Poisson-style plots for one dilution series."""

            def __init__(
                self, accessor: "DilutionSeries.PoissonAnalysisAccessor"
            ) -> None:
                self._accessor = accessor

            def histogram(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                number_of_count_bins: int = 50,
                observation_start: Optional[float] = None,
                observation_end: Optional[float] = None,
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                    number_of_count_bins=number_of_count_bins,
                    observation_start=observation_start,
                    observation_end=observation_end,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.histogram(ax=ax, **plot_kwargs)

            def expected_histogram(
                self,
                index: int,
                base_index: int = 0,
                reference_indices: Optional[Sequence[int]] = None,
                use_water_baseline: bool = True,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                number_of_count_bins: int = 50,
                observation_start: Optional[float] = None,
                observation_end: Optional[float] = None,
                *,
                label: Optional[str] = None,
                expected_label: str = "Expected Poisson",
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                    number_of_count_bins=number_of_count_bins,
                    observation_start=observation_start,
                    observation_end=observation_end,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                expected_particle_flow = (
                    self._accessor._series.get_expected_particle_flow(
                        index=index,
                        base_index=base_index,
                        reference_indices=reference_indices,
                        use_water_baseline=use_water_baseline,
                    )
                )
                record = self._accessor._series.get_record(index=index)
                expected_lambda_hat = (
                    expected_particle_flow
                    if x_axis == "time"
                    else expected_particle_flow * float(record.dx)
                )
                return metrics.plot.histogram(
                    expected_lambda_hat=expected_lambda_hat,
                    expected_label=expected_label,
                    ax=ax,
                    **plot_kwargs,
                )

            def qq(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                number_of_count_bins: int = 50,
                observation_start: Optional[float] = None,
                observation_end: Optional[float] = None,
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                    number_of_count_bins=number_of_count_bins,
                    observation_start=observation_start,
                    observation_end=observation_end,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.qq(ax=ax, **plot_kwargs)

            def count_distribution(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                number_of_count_bins: int = 50,
                observation_start: Optional[float] = None,
                observation_end: Optional[float] = None,
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                    number_of_count_bins=number_of_count_bins,
                    observation_start=observation_start,
                    observation_end=observation_end,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.count_distribution(ax=ax, **plot_kwargs)

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series
            self.plot = self.PlotAccessor(self)

        def diagnose(
            self,
            index: int,
            detector: Literal["standard", "cnn"] = "standard",
            x_axis: Literal["sample", "time"] = "sample",
            number_of_count_bins: int = 50,
            observation_start: Optional[float] = None,
            observation_end: Optional[float] = None,
        ) -> Dict[str, EventArrivalDistribution]:
            return self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )

    class AmplitudeAnalysisAccessor:
        """Namespace exposing peak-amplitude diagnostics for one dilution series."""

        class PlotAccessor:
            """Namespace exposing peak-amplitude plots for one dilution series."""

            def __init__(
                self, accessor: "DilutionSeries.AmplitudeAnalysisAccessor"
            ) -> None:
                self._accessor = accessor

            def histogram(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.histogram(ax=ax, **plot_kwargs)

            def qq(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.qq(ax=ax, **plot_kwargs)

            def ecdf(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.ecdf(ax=ax, **plot_kwargs)

            def local_crowding(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                *,
                ax: Optional[plt.Axes] = None,
                figsize: tuple[float, float] = (6.5, 4.0),
                marker_size: float = 18.0,
                marker_alpha: float = 0.7,
                marker_color: Optional[str] = None,
                title: Optional[str] = None,
                show_grid: bool = True,
                save_path: Optional[Union[str, Path]] = None,
                dpi: int = 300,
                show: bool = False,
                close: bool = False,
            ) -> plt.Figure:
                record = self._accessor._series.get_record(index=index)
                amplitude_sources = self._accessor.compare_sources(index=index)

                if detector == "standard":
                    peak_indices = np.asarray(
                        amplitude_sources["standard_peak_indices"], dtype=float
                    )
                    amplitudes = np.asarray(
                        amplitude_sources["standard_amplitudes"], dtype=float
                    )
                    resolved_color = marker_color or "C0"
                elif detector == "cnn":
                    peak_indices = np.asarray(
                        amplitude_sources["cnn_peak_indices"], dtype=float
                    )
                    recovered_amplitudes = amplitude_sources.get(
                        "cnn_recovered_amplitudes"
                    )
                    if recovered_amplitudes is not None:
                        amplitudes = np.asarray(recovered_amplitudes, dtype=float)
                    else:
                        amplitudes = np.asarray(
                            amplitude_sources["cnn_raw_signal_amplitudes"], dtype=float
                        )
                    resolved_color = marker_color or "C1"
                else:
                    raise ValueError('detector must be either "standard" or "cnn".')

                peak_indices = peak_indices[np.isfinite(peak_indices)]
                amplitudes = amplitudes[np.isfinite(amplitudes)]
                pair_count = min(peak_indices.size, amplitudes.size)
                peak_indices = peak_indices[:pair_count]
                amplitudes = amplitudes[:pair_count]

                if pair_count < 2:
                    nearest_neighbor_distance = np.full(pair_count, np.nan, dtype=float)
                else:
                    order = np.argsort(peak_indices)
                    sorted_peaks = peak_indices[order]
                    sorted_amplitudes = amplitudes[order]
                    left_distance = np.full(sorted_peaks.size, np.inf, dtype=float)
                    right_distance = np.full(sorted_peaks.size, np.inf, dtype=float)
                    left_distance[1:] = np.diff(sorted_peaks)
                    right_distance[:-1] = np.diff(sorted_peaks)
                    nearest_neighbor_distance = np.minimum(
                        left_distance, right_distance
                    )
                    peak_indices = sorted_peaks
                    amplitudes = sorted_amplitudes

                finite_mask = np.isfinite(nearest_neighbor_distance) & np.isfinite(
                    amplitudes
                )
                if x_axis == "time":
                    nearest_neighbor_distance = nearest_neighbor_distance * float(
                        record.dx
                    )
                    x_label = "Nearest-neighbor distance [Time]"
                elif x_axis == "sample":
                    x_label = "Nearest-neighbor distance [Samples]"
                else:
                    raise ValueError('x_axis must be either "sample" or "time".')

                if ax is not None:
                    figure, axis = ax.figure, ax
                    created_figure = False
                else:
                    figure, axis = plt.subplots(figsize=figsize)
                    created_figure = True

                if np.any(finite_mask):
                    axis.scatter(
                        nearest_neighbor_distance[finite_mask],
                        amplitudes[finite_mask],
                        s=marker_size,
                        alpha=marker_alpha,
                        color=resolved_color,
                        rasterized=True,
                    )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "Need at least two detected peaks",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )

                axis.set_xlabel(x_label)
                axis.set_ylabel("Peak amplitude")
                axis.set_title(
                    title or f"{detector.capitalize()} amplitude vs local crowding"
                )
                if show_grid:
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

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series
            self.plot = self.PlotAccessor(self)

        def diagnose(
            self,
            index: int,
            detector: Literal["standard", "cnn"] = "standard",
        ) -> Dict[str, PeakAmplitudeDistribution]:
            return self._series.compute_peak_amplitude_distribution_metrics(
                index=index,
                detector=detector,
            )

        def compare_sources(self, index: int) -> Dict[str, Any]:
            """Return the raw and recovered amplitude sources for one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.

            Returns
            -------
            dict
                Dictionary containing the standard amplitudes, CNN raw signal
                amplitudes at CNN peak positions, CNN recovered amplitudes, and
                the peak indices used to extract them.
            """

            record = self._series.get_record(index=index)
            signal = np.asarray(record.signal, dtype=float).ravel()
            standard_peak_indices = np.asarray(record.standard.peaks, dtype=int)
            standard_peak_indices = standard_peak_indices[
                (standard_peak_indices >= 0) & (standard_peak_indices < signal.size)
            ]
            cnn_peak_indices = np.asarray(record.cnn.peaks, dtype=int)
            cnn_peak_indices = cnn_peak_indices[
                (cnn_peak_indices >= 0) & (cnn_peak_indices < signal.size)
            ]

            cnn_recovered_amplitudes = getattr(record.cnn, "amplitudes", None)
            if cnn_recovered_amplitudes is not None:
                cnn_recovered_amplitudes = np.asarray(
                    cnn_recovered_amplitudes, dtype=float
                ).ravel()

            return {
                "standard_peak_indices": standard_peak_indices,
                "standard_amplitudes": signal[standard_peak_indices],
                "cnn_peak_indices": cnn_peak_indices,
                "cnn_raw_signal_amplitudes": signal[cnn_peak_indices],
                "cnn_recovered_amplitudes": cnn_recovered_amplitudes,
            }

    class WidthAnalysisAccessor:
        """Namespace exposing peak-width diagnostics for one dilution series."""

        class PlotAccessor:
            """Namespace exposing peak-width plots for one dilution series."""

            def __init__(
                self, accessor: "DilutionSeries.WidthAnalysisAccessor"
            ) -> None:
                self._accessor = accessor

            def histogram(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.histogram(ax=ax, **plot_kwargs)

            def qq(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.qq(ax=ax, **plot_kwargs)

            def ecdf(
                self,
                index: int,
                detector: Literal["standard", "cnn"] = "standard",
                x_axis: Literal["sample", "time"] = "sample",
                *,
                label: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                **plot_kwargs,
            ) -> plt.Figure:
                metrics_by_label = self._accessor.diagnose(
                    index=index,
                    detector=detector,
                    x_axis=x_axis,
                )
                metrics = DilutionSeries._resolve_single_label_metrics(
                    metrics_by_label, label
                )
                return metrics.plot.ecdf(ax=ax, **plot_kwargs)

            def compare(
                self,
                index: int,
                x_axis: Literal["sample", "time"] = "sample",
                plot: Literal["histogram", "ecdf"] = "histogram",
                *,
                figsize: tuple[float, float] = (10.0, 4.0),
                axes: Optional[tuple[plt.Axes, plt.Axes]] = None,
                standard_plot_kwargs: Optional[dict[str, Any]] = None,
                cnn_plot_kwargs: Optional[dict[str, Any]] = None,
                show: bool = False,
                close: bool = False,
                save_path: Optional[Union[str, Path]] = None,
                dpi: int = 300,
            ) -> plt.Figure:
                if axes is not None:
                    standard_axis, cnn_axis = axes
                    figure = standard_axis.figure
                    created_figure = False
                else:
                    figure, (standard_axis, cnn_axis) = plt.subplots(
                        ncols=2,
                        figsize=figsize,
                        sharey=True,
                    )
                    created_figure = True

                standard_kwargs = dict(standard_plot_kwargs or {})
                cnn_kwargs = dict(cnn_plot_kwargs or {})
                standard_kwargs.setdefault("show", False)
                standard_kwargs.setdefault("close", False)
                cnn_kwargs.setdefault("show", False)
                cnn_kwargs.setdefault("close", False)

                if plot == "histogram":
                    self.histogram(
                        index=index,
                        detector="standard",
                        x_axis=x_axis,
                        ax=standard_axis,
                        **standard_kwargs,
                    )
                    self.histogram(
                        index=index,
                        detector="cnn",
                        x_axis=x_axis,
                        ax=cnn_axis,
                        **cnn_kwargs,
                    )
                elif plot == "ecdf":
                    self.ecdf(
                        index=index,
                        detector="standard",
                        x_axis=x_axis,
                        ax=standard_axis,
                        **standard_kwargs,
                    )
                    self.ecdf(
                        index=index,
                        detector="cnn",
                        x_axis=x_axis,
                        ax=cnn_axis,
                        **cnn_kwargs,
                    )
                else:
                    raise ValueError('plot must be either "histogram" or "ecdf".')

                standard_axis.set_title("Standard")
                cnn_axis.set_title("WaveNet")
                if created_figure:
                    figure.tight_layout()
                if save_path is not None:
                    figure.savefig(save_path, dpi=dpi)
                if show:
                    plt.show()
                if close:
                    plt.close(figure)
                return figure

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series
            self.plot = self.PlotAccessor(self)

        def diagnose(
            self,
            index: int,
            detector: Literal["standard", "cnn"] = "standard",
            x_axis: Literal["sample", "time"] = "sample",
        ) -> Dict[str, PeakWidthDistribution]:
            return self._series.compute_peak_width_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
            )

    class PlotAccessor:
        """Namespace exposing trace-level plotting views for one dilution series."""

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series

        def particle_flows(self, **plot_kwargs) -> plt.Figure:
            """Plot particle flows for the most recent series result."""
            result = self._series.get_last_result()
            x_axis = plot_kwargs.pop("x_axis", "concentration")
            figsize = plot_kwargs.pop("figsize", (10.0, 4.0))
            marker_size = plot_kwargs.pop("marker_size", 42.0)
            line_width = plot_kwargs.pop("line_width", 1.2)
            show_water_baseline = plot_kwargs.pop("show_water_baseline", True)
            show = plot_kwargs.pop("show", False)
            close = plot_kwargs.pop("close", False)
            save_path = plot_kwargs.pop("save_path", None)
            dpi = plot_kwargs.pop("dpi", 300)
            if plot_kwargs:
                unexpected = ", ".join(sorted(plot_kwargs))
                raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

            if x_axis == "concentration":
                x_values = np.asarray(result.concentration, dtype=float)
                x_label = "Concentration"
            elif x_axis == "dilution":
                x_values = np.asarray(result.dilution, dtype=float)
                x_label = "Dilution"
            else:
                raise ValueError('x_axis must be either "concentration" or "dilution".')

            figure, axis = plt.subplots(figsize=figsize)
            axis.plot(
                x_values,
                np.asarray(result.standard_particle_flow, dtype=float),
                marker="o",
                linewidth=line_width,
                markersize=np.sqrt(marker_size),
                label="Standard flow",
            )
            axis.plot(
                x_values,
                np.asarray(result.cnn_particle_flow, dtype=float),
                marker="s",
                linewidth=line_width,
                markersize=np.sqrt(marker_size),
                label="WaveNet flow",
            )
            if show_water_baseline and result.water_record is not None:
                axis.axhline(
                    result.water_record.cnn_particle_flow,
                    color="C1",
                    linestyle=":",
                    linewidth=line_width,
                    label="Water CNN flow",
                )
                axis.axhline(
                    result.water_record.standard_particle_flow,
                    color="C0",
                    linestyle=":",
                    linewidth=line_width,
                    label="Water standard flow",
                )
            axis.set_xlabel(x_label)
            axis.set_ylabel("Particle flow")
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def measured_particle_flows(self, **plot_kwargs) -> plt.Figure:
            """Plot measured particle flows against dilution or concentration."""
            result = self._series.get_last_result()
            x_axis = plot_kwargs.pop("x_axis", "dilution")
            figsize = plot_kwargs.pop("figsize", (6.0, 6.0))
            marker_size = plot_kwargs.pop("marker_size", 42.0)
            show = plot_kwargs.pop("show", False)
            close = plot_kwargs.pop("close", False)
            save_path = plot_kwargs.pop("save_path", None)
            dpi = plot_kwargs.pop("dpi", 300)
            if plot_kwargs:
                unexpected = ", ".join(sorted(plot_kwargs))
                raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

            if x_axis == "dilution":
                x_values = np.asarray(result.dilution, dtype=float)
                x_label = "Dilution"
            elif x_axis == "concentration":
                x_values = np.asarray(result.concentration, dtype=float)
                x_label = "Concentration"
            else:
                raise ValueError('x_axis must be either "dilution" or "concentration".')

            figure, axis = plt.subplots(figsize=figsize)
            axis.plot(
                x_values,
                np.asarray(result.standard_particle_flow, dtype=float),
                linestyle="None",
                marker="o",
                markersize=np.sqrt(marker_size),
                label="Standard flow",
            )
            axis.plot(
                x_values,
                np.asarray(result.cnn_particle_flow, dtype=float),
                linestyle="None",
                marker="s",
                markersize=np.sqrt(marker_size),
                label="WaveNet flow",
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel("Measured particle flow")
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def measured_vs_expected_particle_flows(self, **plot_kwargs) -> plt.Figure:
            """Plot measured particle flows against dilution-scaled expectations."""
            result = self._series.get_last_result()
            base_index = plot_kwargs.pop("base_index", 0)
            reference_indices = plot_kwargs.pop("reference_indices", None)
            use_water_baseline = plot_kwargs.pop("use_water_baseline", True)
            figsize = plot_kwargs.pop("figsize", (6.0, 6.0))
            marker_size = plot_kwargs.pop("marker_size", 42.0)
            line_width = plot_kwargs.pop("line_width", 1.2)
            show_ideal_line = plot_kwargs.pop("show_ideal_line", True)
            ideal_label = plot_kwargs.pop("ideal_label", "Ideal")
            show = plot_kwargs.pop("show", False)
            close = plot_kwargs.pop("close", False)
            save_path = plot_kwargs.pop("save_path", None)
            dpi = plot_kwargs.pop("dpi", 300)
            if plot_kwargs:
                unexpected = ", ".join(sorted(plot_kwargs))
                raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

            expected_flows = np.asarray(
                [
                    DilutionSeries.get_expected_particle_flow_for_result(
                        result,
                        index=record_index,
                        base_index=base_index,
                        reference_indices=reference_indices,
                        use_water_baseline=use_water_baseline,
                    )
                    for record_index in range(len(result.records))
                ],
                dtype=float,
            )
            standard_flows = np.asarray(result.standard_particle_flow, dtype=float)
            cnn_flows = np.asarray(result.cnn_particle_flow, dtype=float)
            finite_values = np.concatenate(
                [
                    expected_flows[np.isfinite(expected_flows)],
                    standard_flows[np.isfinite(standard_flows)],
                    cnn_flows[np.isfinite(cnn_flows)],
                ]
            )

            figure, axis = plt.subplots(figsize=figsize)
            axis.plot(
                expected_flows,
                standard_flows,
                linestyle="None",
                marker="o",
                markersize=np.sqrt(marker_size),
                label="Standard flow",
            )
            axis.plot(
                expected_flows,
                cnn_flows,
                linestyle="None",
                marker="s",
                markersize=np.sqrt(marker_size),
                label="WaveNet flow",
            )
            if show_ideal_line and finite_values.size > 0:
                axis_min = float(np.min(finite_values))
                axis_max = float(np.max(finite_values))
                axis.plot(
                    [axis_min, axis_max],
                    [axis_min, axis_max],
                    linestyle="--",
                    linewidth=line_width,
                    color="0.35",
                    label=ideal_label,
                )
            axis.set_xlabel("Expected particle flow")
            axis.set_ylabel("Measured particle flow")
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

        def detected_peaks_per_second_vs_expected_throughput(
            self, **plot_kwargs
        ) -> plt.Figure:
            """Alias for measured-vs-expected detector flow comparison.

            This makes the notebook intent explicit: compare detected peaks per
            second against the throughput predicted from the dilution series.
            """

            return self.measured_vs_expected_particle_flows(**plot_kwargs)

        def peak_counts(self, **plot_kwargs) -> plt.Figure:
            """Plot standard and WaveNet peak counts for the series."""
            result = self._series.get_last_result()
            x_axis = plot_kwargs.pop("x_axis", "concentration")
            figsize = plot_kwargs.pop("figsize", (10.0, 4.0))
            marker_size = plot_kwargs.pop("marker_size", 42.0)
            line_width = plot_kwargs.pop("line_width", 1.2)
            show = plot_kwargs.pop("show", False)
            close = plot_kwargs.pop("close", False)
            save_path = plot_kwargs.pop("save_path", None)
            dpi = plot_kwargs.pop("dpi", 300)
            if plot_kwargs:
                unexpected = ", ".join(sorted(plot_kwargs))
                raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

            if x_axis == "concentration":
                x_values = np.asarray(result.concentration, dtype=float)
                x_label = "Concentration"
            elif x_axis == "dilution":
                x_values = np.asarray(result.dilution, dtype=float)
                x_label = "Dilution"
            else:
                raise ValueError('x_axis must be either "concentration" or "dilution".')

            figure, axis = plt.subplots(figsize=figsize)
            axis.plot(
                x_values,
                np.asarray(result.standard_particle_count, dtype=float),
                marker="o",
                linewidth=line_width,
                markersize=np.sqrt(marker_size),
                label="Standard count",
            )
            axis.plot(
                x_values,
                np.asarray(result.cnn_particle_count, dtype=float),
                marker="s",
                linewidth=line_width,
                markersize=np.sqrt(marker_size),
                label="WaveNet count",
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel("Peak count")
            axis.grid(True, which="both", alpha=0.2, zorder=0)
            axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            figure.tight_layout()
            if save_path is not None:
                figure.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(figure)
            return figure

    def __init__(
        self,
        folder: Union[str, Path],
        wavenet: Any,
        initial_concentration: float,
        nrows: int,
        water_filename: str = "_water_1.csv",
        low_pass: float = None,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        cnn_low_pass: Optional[float] = None,
        cnn_amplitude_sigma_samples: Optional[float] = None,
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
        dilution_parser: Optional[Callable[[Path], float]] = None,
        trace_files: Optional[List[Tuple[Union[str, Path], float]]] = None,
        *,
        std_trigger: Optional[BasePeakTrigger] = None,
        cnn_trigger: Optional[BasePeakTrigger] = None,
    ) -> None:
        self.folder = Path(folder)
        self.wavenet = wavenet
        self.initial_concentration = float(initial_concentration)
        self.nrows = int(nrows)
        self.std_trigger = std_trigger
        self.cnn_trigger = cnn_trigger
        self.low_pass = low_pass
        self._signal_normalization = str(signal_normalization)
        self._prediction_sampling_rate_hz = float(prediction_sampling_rate_hz)
        self._cnn_low_pass = cnn_low_pass
        self._cnn_amplitude_sigma_samples = cnn_amplitude_sigma_samples
        self._cnn_amplitude_baseline = cnn_amplitude_baseline
        self._sequence_length_override = sequence_length
        self.dilution_parser = dilution_parser
        self.trace_files = None if trace_files is None else list(trace_files)
        self.water_filename = str(water_filename)
        self.sequence_length = self._resolve_sequence_length()
        self.standard_analyzer, self.cnn_analyzer = self._build_analyzers()
        self.analyzer = self.cnn_analyzer or self.standard_analyzer
        self._last_result: Optional[PeakCountSeriesResult] = None
        self.poisson = self.PoissonAnalysisAccessor(self)
        self.amplitude = self.AmplitudeAnalysisAccessor(self)
        self.width = self.WidthAnalysisAccessor(self)
        self.plot = self.PlotAccessor(self)

    def _load_signal(self, filename: Path) -> Tuple[np.ndarray, float]:
        """Load one trace file and convert it into processed windows.

        Parameters
        ----------
        filename : pathlib.Path
            Trace file to load.

        Returns
        -------
        signal : numpy.ndarray
            Processed signal windows.
        dx : float
            Sampling interval of the trace.
        """

        analyzer = self.standard_analyzer or self.cnn_analyzer
        if analyzer is None:
            raise RuntimeError(
                "No analyzer is configured. Provide a detector trigger before running the series."
            )

        return analyzer.load_processed_signal(
            filename=filename,
            nrows=self.nrows,
            low_pass=self.low_pass,
        )

    def _resolve_sequence_length(self) -> int:
        return CNNTraceAnalyzer._infer_sequence_length(
            self.wavenet,
            sequence_length=self._sequence_length_override,
        )

    def _build_analyzers(
        self,
    ) -> Tuple[Optional[StandardTraceAnalyzer], Optional[CNNTraceAnalyzer]]:
        standard_analyzer = (
            None
            if self.std_trigger is None
            else StandardTraceAnalyzer(
                std_trigger=self.std_trigger,
                wavenet=self.wavenet,
                sequence_length=self.sequence_length,
                signal_normalization=self._signal_normalization,
                prediction_sampling_rate_hz=self._prediction_sampling_rate_hz,
            )
        )
        cnn_analyzer = (
            None
            if self.cnn_trigger is None
            else CNNTraceAnalyzer(
                wavenet=self.wavenet,
                cnn_trigger=self.cnn_trigger,
                sequence_length=self.sequence_length,
                signal_normalization=self._signal_normalization,
                prediction_sampling_rate_hz=self._prediction_sampling_rate_hz,
                cnn_low_pass=self._cnn_low_pass,
                cnn_amplitude_sigma_samples=self._cnn_amplitude_sigma_samples,
                cnn_amplitude_baseline=self._cnn_amplitude_baseline,
            )
        )
        return standard_analyzer, cnn_analyzer

    def _has_standard_configuration(self) -> bool:
        return self.std_trigger is not None

    def _has_cnn_configuration(self) -> bool:
        return self.cnn_trigger is not None

    def _validate_requested_detectors(
        self,
        *,
        include_standard: bool,
        include_cnn: bool,
    ) -> None:
        if include_standard and not self._has_standard_configuration():
            if include_cnn:
                raise ValueError(
                    "run() requires both detector configurations. Provide std_trigger "
                    "and cnn_trigger, or call run_standard(...) / run_cnn(...)."
                )
            raise ValueError(
                "Standard detection is not configured. Provide std_trigger or "
                "call run_standard(std_trigger=...)."
            )

        if include_cnn and not self._has_cnn_configuration():
            if include_standard:
                raise ValueError(
                    "run() requires both detector configurations. Provide std_trigger "
                    "and cnn_trigger, or call run_standard(...) / run_cnn(...)."
                )
            raise ValueError(
                "CNN detection is not configured. Provide cnn_trigger or "
                "call run_cnn(cnn_trigger=...)."
            )

    def _reconfigure_analyzer(
        self,
        *,
        std_trigger: Optional[BasePeakTrigger] = None,
        cnn_trigger: Optional[BasePeakTrigger] = None,
        cnn_low_pass: Optional[float] = None,
        cnn_amplitude_sigma_samples: Optional[float] = None,
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
    ) -> None:
        if std_trigger is not None:
            self.std_trigger = std_trigger

        if cnn_trigger is not None:
            self.cnn_trigger = cnn_trigger

        if cnn_low_pass is not None:
            self._cnn_low_pass = cnn_low_pass
        if cnn_amplitude_sigma_samples is not None:
            self._cnn_amplitude_sigma_samples = cnn_amplitude_sigma_samples
        if cnn_amplitude_baseline is not None:
            self._cnn_amplitude_baseline = cnn_amplitude_baseline

        self.sequence_length = self._resolve_sequence_length()
        self.standard_analyzer, self.cnn_analyzer = self._build_analyzers()
        self.analyzer = self.cnn_analyzer or self.standard_analyzer

    @staticmethod
    def get_expected_particle_flow_for_result(
        series_or_result: Any,
        index: int,
        base_index: int = 0,
        reference_indices: Optional[Sequence[int]] = None,
        use_water_baseline: bool = True,
    ) -> float:
        """Estimate target particle flow from one or more reference traces.

        When a water-control trace is available and ``use_water_baseline`` is
        true, its measured standard particle flow is treated as the blank
        background and only the dilution-dependent throughput term is fit.

        When multiple reference traces are provided, the expected flow is
        estimated with an affine dilution model,

        ``particle_flow = throughput / dilution + background_flow``

        so dilution-independent carry-over can be absorbed into the fitted
        background term.
        """

        result = resolve_series_or_result(series_or_result)
        if len(result.records) == 0:
            raise IndexError(
                "No trace records are available. Call run() first and make sure files were found."
            )

        base_record = result.records[base_index]
        current_record = result.records[index]
        current_dilution = float(current_record.dilution)

        if current_dilution == 0.0:
            raise ValueError(
                "Expected particle flow is undefined when the target dilution is zero."
            )

        water_record = getattr(result, "water_record", None)
        background_flow = (
            float(water_record.standard_particle_flow)
            if use_water_baseline and water_record is not None
            else 0.0
        )

        indices = (
            [base_index]
            if reference_indices is None
            else [int(value) for value in reference_indices]
        )
        if reference_indices is not None and len(indices) > 1:
            indices = [
                reference_index
                for reference_index in indices
                if reference_index != index
            ]

        if len(indices) == 0:
            raise ValueError(
                "reference_indices must contain at least one usable reference index. "
                "When multiple indices are provided, the target index is excluded "
                "from the fit."
            )

        if len(indices) == 1:
            return float(
                (base_record.standard_particle_flow - background_flow)
                * float(base_record.dilution)
                / current_dilution
                + background_flow
            )

        reference_dilutions = []
        reference_flows = []
        for reference_index in indices:
            reference_record = result.records[reference_index]
            dilution = float(reference_record.dilution)
            if dilution == 0.0:
                raise ValueError(
                    "Expected particle flow is undefined when a reference dilution is zero."
                )
            reference_dilutions.append(dilution)
            reference_flows.append(float(reference_record.standard_particle_flow))

        if use_water_baseline and water_record is not None:
            design_matrix = (
                1.0 / np.asarray(reference_dilutions, dtype=float)
            ).reshape(-1, 1)
            throughput = np.linalg.lstsq(
                design_matrix,
                np.asarray(reference_flows, dtype=float) - background_flow,
                rcond=None,
            )[0][0]
            return float(throughput / current_dilution + background_flow)

        design_matrix = np.column_stack(
            (
                1.0 / np.asarray(reference_dilutions, dtype=float),
                np.ones(len(reference_dilutions), dtype=float),
            )
        )
        throughput, background_flow = np.linalg.lstsq(
            design_matrix,
            np.asarray(reference_flows, dtype=float),
            rcond=None,
        )[0]

        return float(throughput / current_dilution + background_flow)

    def get_record(self, index: int) -> TraceRecord:
        return self._get_record_by_index(index=index)

    def get_record_with_expected_particle_flow(
        self,
        index: int,
        base_index: int = 0,
        reference_indices: Optional[Sequence[int]] = None,
        use_water_baseline: bool = True,
    ) -> tuple[TraceRecord, float]:
        """Return one sorted record together with its expected particle flow."""

        return (
            self.get_record(index=index),
            self.get_expected_particle_flow(
                index=index,
                base_index=base_index,
                reference_indices=reference_indices,
                use_water_baseline=use_water_baseline,
            ),
        )

    @staticmethod
    def _empty_detection_result() -> PeakDetectionResult:
        return PeakDetectionResult(
            peaks=np.asarray([], dtype=int),
            properties={},
            peak_count=0,
            detection_kwargs={},
            threshold=None,
            amplitudes=np.asarray([], dtype=float),
        )

    def _merge_trace_records(
        self,
        *,
        filename: Path,
        dilution: float,
        concentration: float,
        dx: float,
        standard_record: Optional[TraceRecord],
        cnn_record: Optional[TraceRecord],
    ) -> TraceRecord:
        source_record = standard_record if standard_record is not None else cnn_record
        if source_record is None:
            raise RuntimeError(
                "At least one detector result is required to build a trace record."
            )

        return TraceRecord(
            filename=Path(filename),
            dilution=float(dilution),
            concentration=float(concentration),
            dx=float(dx),
            signal=np.asarray(source_record.signal, dtype=float),
            standard=(
                self._empty_detection_result()
                if standard_record is None
                else standard_record.standard
            ),
            prediction=(
                np.asarray([], dtype=float)
                if cnn_record is None
                else np.asarray(cnn_record.prediction, dtype=float)
            ),
            cnn=(
                self._empty_detection_result() if cnn_record is None else cnn_record.cnn
            ),
        )

    def compute_record(
        self,
        dilution: float,
        filename: Path,
        *,
        include_standard: bool = True,
        include_cnn: bool = True,
    ) -> TraceRecord:
        """Analyze one trace file and attach its dilution-derived concentration.

        Parameters
        ----------
        dilution : float
            Dilution factor assigned to the trace.
        filename : pathlib.Path
            Trace file to analyze.

        Returns
        -------
        TraceRecord
            Analysis record for the trace.
        """

        concentration = self.initial_concentration / float(dilution)
        signal, dx = self._load_signal(filename=filename)
        standard_record = (
            None
            if not include_standard
            else self.standard_analyzer.analyze_processed_signal(
                signal,
                dx=dx,
                filename=filename,
                dilution=float(dilution),
                concentration=float(concentration),
            )
        )
        cnn_record = (
            None
            if not include_cnn
            else self.cnn_analyzer.analyze_processed_signal(
                signal,
                dx=dx,
                filename=filename,
                dilution=float(dilution),
                concentration=float(concentration),
            )
        )
        return self._merge_trace_records(
            filename=filename,
            dilution=float(dilution),
            concentration=float(concentration),
            dx=dx,
            standard_record=standard_record,
            cnn_record=cnn_record,
        )

    def compute_water_record(
        self,
        *,
        include_standard: bool = True,
        include_cnn: bool = True,
    ) -> Optional[TraceRecord]:
        """Analyze the optional water-control trace if it exists.

        Returns
        -------
        TraceRecord or None
            Analyzed control trace, or ``None`` if the file does not exist.
        """

        water_path = self.folder / self.water_filename
        if not water_path.exists():
            return None

        signal, dx = self._load_signal(filename=water_path)
        standard_record = (
            None
            if not include_standard
            else self.standard_analyzer.analyze_processed_signal(
                signal,
                dx=dx,
                filename=water_path,
                dilution=float("nan"),
                concentration=float("nan"),
            )
        )
        cnn_record = (
            None
            if not include_cnn
            else self.cnn_analyzer.analyze_processed_signal(
                signal,
                dx=dx,
                filename=water_path,
                dilution=float("nan"),
                concentration=float("nan"),
            )
        )
        return self._merge_trace_records(
            filename=water_path,
            dilution=float("nan"),
            concentration=float("nan"),
            dx=dx,
            standard_record=standard_record,
            cnn_record=cnn_record,
        )

    def _build_result_bundle(
        self,
        records: List[TraceRecord],
        water_record: Optional[TraceRecord],
        *,
        include_standard: bool,
        include_cnn: bool,
    ) -> PeakCountSeriesResult:
        if len(records) == 0:
            return PeakCountSeriesResult(
                dilution=np.asarray([]),
                concentration=np.asarray([]),
                standard_particle_count=np.asarray([]),
                standard_particle_flow=np.asarray([]),
                cnn_particle_count=np.asarray([]),
                cnn_particle_flow=np.asarray([]),
                water_record=water_record,
                records=[],
            )

        dilution_series = np.asarray(
            [record.dilution for record in records], dtype=float
        )
        sort_index = np.argsort(dilution_series)[::-1]
        records_sorted = [records[index] for index in sort_index]
        dilution_values = np.asarray(
            [record.dilution for record in records_sorted], dtype=float
        )
        concentration_values = np.asarray(
            [record.concentration for record in records_sorted], dtype=float
        )
        standard_particle_count = (
            np.asarray(
                [record.standard.peak_count for record in records_sorted], dtype=float
            )
            if include_standard
            else np.full(len(records_sorted), np.nan, dtype=float)
        )
        standard_particle_flow = (
            np.asarray(
                [record.standard_particle_flow for record in records_sorted],
                dtype=float,
            )
            if include_standard
            else np.full(len(records_sorted), np.nan, dtype=float)
        )
        cnn_particle_count = (
            np.asarray(
                [record.cnn.peak_count for record in records_sorted], dtype=float
            )
            if include_cnn
            else np.full(len(records_sorted), np.nan, dtype=float)
        )
        cnn_particle_flow = (
            np.asarray(
                [record.cnn_particle_flow for record in records_sorted], dtype=float
            )
            if include_cnn
            else np.full(len(records_sorted), np.nan, dtype=float)
        )

        return PeakCountSeriesResult(
            dilution=dilution_values,
            concentration=concentration_values,
            standard_particle_count=standard_particle_count,
            standard_particle_flow=standard_particle_flow,
            cnn_particle_count=cnn_particle_count,
            cnn_particle_flow=cnn_particle_flow,
            water_record=water_record,
            records=records_sorted,
        )

    def _run_selected_detectors(
        self,
        *,
        include_standard: bool,
        include_cnn: bool,
    ) -> PeakCountSeriesResult:
        self._validate_requested_detectors(
            include_standard=include_standard,
            include_cnn=include_cnn,
        )
        records: List[TraceRecord] = []
        water_record = self.compute_water_record(
            include_standard=include_standard,
            include_cnn=include_cnn,
        )

        trace_file_iterator = _iterate_explicit_trace_files(
            self.folder, self.trace_files
        )
        for dilution, filename in trace_file_iterator:
            record = self.compute_record(
                dilution=dilution,
                filename=Path(filename),
                include_standard=include_standard,
                include_cnn=include_cnn,
            )
            standard_flow = (
                f"{record.standard_particle_flow}" if include_standard else "disabled"
            )
            cnn_flow = f"{record.cnn_particle_flow}" if include_cnn else "disabled"
            print(
                f"Standard particle flow: {standard_flow}\tCNN particle flow: {cnn_flow}"
            )
            records.append(record)

        result = self._build_result_bundle(
            records,
            water_record,
            include_standard=include_standard,
            include_cnn=include_cnn,
        )
        self._last_result = result
        return result

    def run(self) -> PeakCountSeriesResult:
        """Analyze the entire series and cache the sorted result bundle.

        Returns
        -------
        PeakCountSeriesResult
            Sorted result bundle containing arrays and per-trace records.
        """

        return self._run_selected_detectors(
            include_standard=True,
            include_cnn=True,
        )

    def run_standard(
        self,
        *,
        std_trigger: Optional[BasePeakTrigger] = None,
    ) -> PeakCountSeriesResult:
        """Analyze the series with the standard detector only."""

        self._reconfigure_analyzer(std_trigger=std_trigger)
        return self._run_selected_detectors(
            include_standard=True,
            include_cnn=False,
        )

    def run_cnn(
        self,
        *,
        cnn_trigger: Optional[BasePeakTrigger] = None,
        cnn_low_pass: Optional[float] = None,
        cnn_amplitude_sigma_samples: Optional[float] = None,
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
    ) -> PeakCountSeriesResult:
        """Analyze the series with the CNN detector only."""

        self._reconfigure_analyzer(
            cnn_trigger=cnn_trigger,
            cnn_low_pass=cnn_low_pass,
            cnn_amplitude_sigma_samples=cnn_amplitude_sigma_samples,
            cnn_amplitude_baseline=cnn_amplitude_baseline,
        )
        return self._run_selected_detectors(
            include_standard=False,
            include_cnn=True,
        )

    def get_last_result(self) -> PeakCountSeriesResult:
        """Return the most recent series result produced by :meth:`run`.

        Returns
        -------
        PeakCountSeriesResult
            Cached result bundle from the latest successful series run.
        """

        if self._last_result is None:
            raise RuntimeError("No result available. Call run() first.")
        return self._last_result

    def _get_record_by_index(
        self,
        index: int,
        result: Optional[PeakCountSeriesResult] = None,
    ) -> TraceRecord:
        """Return one sorted trace record by index.

        Parameters
        ----------
        index : int
            Index of the desired record in the sorted result.
        result : PeakCountSeriesResult, optional
            Result bundle to read from. If omitted, the cached result is used.

        Returns
        -------
        TraceRecord
            Record at the requested index.
        """

        if result is None:
            result = self.get_last_result()
        if len(result.records) == 0:
            raise IndexError(
                "No trace records are available. Call run() first and make sure files were found."
            )
        return result.records[index]

    def get_expected_particle_flow(
        self,
        index: int,
        base_index: int = 0,
        reference_indices: Optional[Sequence[int]] = None,
        use_water_baseline: bool = True,
    ) -> float:
        """Estimate standard-detector particle flow from one or more reference traces.

        Parameters
        ----------
        index : int
            Index of the target trace in the sorted series result.
        base_index : int, default=0
            Index of the reference trace in the sorted series result.
            Used when ``reference_indices`` is not provided.
        reference_indices : sequence of int, optional
            Indices of traces used to fit the expected flow. Supplying two or
            more indices fits a dilution-dependent throughput term together
            with a dilution-independent background flow.
        use_water_baseline : bool, default=True
            When a water-control trace is available, use its measured standard
            particle flow as the blank/background level.

        Returns
        -------
        float
            Expected standard-detector particle flow at the target dilution.
        """

        return self.get_expected_particle_flow_for_result(
            series_or_result=self,
            index=index,
            base_index=base_index,
            reference_indices=reference_indices,
            use_water_baseline=use_water_baseline,
        )

    def compute_event_arrival_distribution_metrics(
        self,
        index: int,
        detector: Literal["standard", "cnn"] = "standard",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
    ) -> Dict[str, EventArrivalDistribution]:
        """Compute Poisson-style arrival diagnostics for one trace in the series.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn"}, default="standard"
            Detector outputs included in the diagnostics.
        x_axis : {"sample", "time"}, default="sample"
            Units used for the event times.
        number_of_count_bins : int, default=50
            Number of bins used for the counts-per-bin diagnostics.
        observation_start, observation_end : float, optional
            Optional bounds of the observation window.

        Returns
        -------
        dict
            Mapping from detector label to arrival diagnostics.
        """

        return compute_event_arrival_distribution_metrics(
            series_or_result=self,
            index=index,
            detector=detector,
            x_axis=x_axis,
            number_of_count_bins=number_of_count_bins,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def compute_peak_amplitude_distribution_metrics(
        self,
        index: int,
        detector: Literal["standard", "cnn"] = "standard",
    ) -> Dict[str, PeakAmplitudeDistribution]:
        """Compute peak-amplitude diagnostics for one trace in the series.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn"}, default="standard"
            Detector outputs included in the diagnostics.

        Returns
        -------
        dict
            Mapping from detector label to amplitude diagnostics.
        """

        return compute_peak_amplitude_distribution_metrics(
            series_or_result=self,
            index=index,
            detector=detector,
        )

    def compute_peak_width_distribution_metrics(
        self,
        index: int,
        detector: Literal["standard", "cnn"] = "standard",
        x_axis: Literal["sample", "time"] = "sample",
    ) -> Dict[str, PeakWidthDistribution]:
        """Compute peak-width diagnostics for one trace in the series."""

        return compute_peak_width_distribution_metrics(
            series_or_result=self,
            index=index,
            detector=detector,
            x_axis=x_axis,
        )


class PeakCountSeries(DilutionSeries):
    """Backward-compatible alias for :class:`DilutionSeries`."""
