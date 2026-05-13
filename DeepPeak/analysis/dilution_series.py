"""Dilution-series orchestration for processed-signal and WaveNet analysis.

This module handles the folder-level workflow: loading traces, associating them
with dilutions, running the single-trace analyzer, and exposing series-level
plots and diagnostics.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .distributions import (
    compute_event_arrival_distribution_metrics,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
    plot_count_distribution,
    plot_event_raster,
    plot_inter_arrival_histogram,
    plot_peak_amplitude_ecdf,
    plot_peak_amplitude_histogram,
    plot_peak_amplitude_qq,
    plot_peak_width_ecdf,
    plot_peak_width_histogram,
    plot_peak_width_qq,
    plot_rescaled_uniform_qq,
)
from .trace_plots import (
    finalize_single_axis_figure,
    make_plot_figure,
    plot_standard_detection_trace,
    plot_style_context,
    plot_wavenet_detection_trace,
)
from .results import (
    EventArrivalDistributionMetrics,
    PeakAmplitudeDistributionMetrics,
    PeakCountSeriesResult,
    PeakWidthDistributionMetrics,
    TraceRecord,
    resolve_series_or_result,
)
from .triggers import PeakTrigger, TriggerLike
from .wavenet_trace import WaveNetTraceAnalyzer


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
    """
    Compute and visualize standard and WaveNet-based peak counts over a dilution series.

    Parameters
    ----------
    folder:
        Base folder containing the trace CSV files.
    wavenet:
        Trained model exposing a ``predict(signal=...)`` interface.
    initial_concentration:
        Concentration associated with dilution factor 1.
    nrows : int
        Maximum number of CSV rows loaded per trace.
    std_trigger, cnn_trigger : PeakTrigger or mapping, optional
        Preferred typed trigger configurations for the standard and WaveNet-based detectors.
    std_kwargs, cnn_kwargs : mapping, optional
        Legacy dictionary-based trigger configurations kept for compatibility.
    water_filename : str, default="_water_1.csv"
        Filename of the optional control trace.
    low_pass : float, optional
        Optional low-pass cutoff applied before signal processing.
    sequence_length : int, optional
        Explicit WaveNet window length override.
    signal_normalization : str, default="zscore"
        Normalization strategy used before model inference.
    prediction_sampling_rate_hz : float, default=125_000_000.0
        Sampling rate used when low-pass filtering the WaveNet prediction.
    dilution_parser : callable, optional
        Custom filename-to-dilution parser used when ``trace_files`` is not provided.
    trace_files:
        Optional explicit ``(filename, dilution)`` pairs. When provided, this is
        preferred over filename parsing.
    """

    class PoissonAnalysisAccessor:
        """Namespace exposing Poisson-style diagnostics for one dilution series."""

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series

        def diagnose(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            number_of_count_bins: int = 50,
            observation_start: Optional[float] = None,
            observation_end: Optional[float] = None,
        ) -> Dict[str, EventArrivalDistributionMetrics]:
            """Compute Poisson-style event-arrival diagnostics for one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the diagnostics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for event times and derived arrival metrics.
            number_of_count_bins : int, default=50
                Number of bins used for counts-per-bin diagnostics.
            observation_start, observation_end : float, optional
                Optional observation-window bounds.

            Returns
            -------
            dict
                Mapping from detector label to event-arrival diagnostics.
            """

            return self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )

        def plot_event_raster(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            number_of_count_bins: int = 50,
            observation_start: Optional[float] = None,
            observation_end: Optional[float] = None,
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot detected event times for one trace as a raster.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the computed metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the event times.
            number_of_count_bins : int, default=50
                Number of bins used for the underlying count diagnostics.
            observation_start, observation_end : float, optional
                Optional observation-window bounds.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the raster.
            **plot_kwargs
                Additional keyword arguments forwarded to :func:`plot_event_raster`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the event raster plot.
            """

            metrics = self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            return plot_event_raster(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_inter_arrival_histogram(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            number_of_count_bins: int = 50,
            observation_start: Optional[float] = None,
            observation_end: Optional[float] = None,
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the inter-arrival histogram for one trace and detector selection.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the computed metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the event times.
            number_of_count_bins : int, default=50
                Number of bins used for the underlying count diagnostics.
            observation_start, observation_end : float, optional
                Optional observation-window bounds.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the histogram.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_inter_arrival_histogram`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the inter-arrival histogram.
            """

            metrics = self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            return plot_inter_arrival_histogram(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_expected_inter_arrival_histogram(
            self,
            index: int,
            base_index: int = 0,
            detector: Literal["standard", "cnn", "both"] = "standard",
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
            """Overlay measured inter-arrival times with an expected Poisson model.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            base_index : int, default=0
                Reference trace index used to derive the expected particle flow.
            detector : {"standard", "cnn", "both"}, default="standard"
                Detector outputs included in the computed metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the event times.
            number_of_count_bins : int, default=50
                Number of bins used for the underlying count diagnostics.
            observation_start, observation_end : float, optional
                Optional observation-window bounds.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            expected_label : str, default="Expected Poisson"
                Legend label used for the expected Poisson overlay.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the plot.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_inter_arrival_histogram`.

            The expected Poisson rate is derived from the dilution-scaled standard
            particle flow returned by :meth:`DilutionSeries.get_expected_particle_flow`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the measured and expected inter-arrival plot.
            """

            metrics = self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            expected_particle_flow = self._series.get_expected_particle_flow(
                index=index,
                base_index=base_index,
            )
            record = self._series.get_record(index=index)
            expected_lambda_hat = (
                expected_particle_flow
                if x_axis == "time"
                else expected_particle_flow * float(record.dx)
            )
            return plot_inter_arrival_histogram(
                metrics_by_label=metrics,
                label=label,
                expected_lambda_hat=expected_lambda_hat,
                expected_label=expected_label,
                ax=ax,
                **plot_kwargs,
            )

        def plot_rescaled_uniform_qq(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            number_of_count_bins: int = 50,
            observation_start: Optional[float] = None,
            observation_end: Optional[float] = None,
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the time-rescaling Uniform(0, 1) Q-Q diagnostic for one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the computed metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the event times.
            number_of_count_bins : int, default=50
                Number of bins used for the underlying count diagnostics.
            observation_start, observation_end : float, optional
                Optional observation-window bounds.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the Q-Q plot.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_rescaled_uniform_qq`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the time-rescaling Q-Q plot.
            """

            metrics = self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            return plot_rescaled_uniform_qq(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_count_distribution(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            number_of_count_bins: int = 50,
            observation_start: Optional[float] = None,
            observation_end: Optional[float] = None,
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot counts-per-bin against the fitted Poisson count model for one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the computed metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the event times.
            number_of_count_bins : int, default=50
                Number of bins used for the underlying count diagnostics.
            observation_start, observation_end : float, optional
                Optional observation-window bounds.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the count-distribution plot.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_count_distribution`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the count-distribution plot.
            """

            metrics = self._series.compute_event_arrival_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
                number_of_count_bins=number_of_count_bins,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            return plot_count_distribution(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

    class AmplitudeAnalysisAccessor:
        """Namespace exposing peak-amplitude diagnostics for one dilution series."""

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series

        def diagnose(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
        ) -> Dict[str, PeakAmplitudeDistributionMetrics]:
            """Compute peak-amplitude diagnostics for one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the diagnostics.

            Returns
            -------
            dict
                Mapping from detector label to amplitude diagnostics.
            """

            return self._series.compute_peak_amplitude_distribution_metrics(
                index=index,
                detector=detector,
            )

        def plot_histogram(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the amplitude histogram for one trace and detector selection.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the amplitude metrics.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the histogram.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_peak_amplitude_histogram`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the peak-amplitude histogram.
            """

            metrics = self._series.compute_peak_amplitude_distribution_metrics(
                index=index, detector=detector
            )
            return plot_peak_amplitude_histogram(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_qq(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the normal Q-Q diagnostic for peak amplitudes on one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the amplitude metrics.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the Q-Q plot.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_peak_amplitude_qq`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the peak-amplitude Q-Q plot.
            """

            metrics = self._series.compute_peak_amplitude_distribution_metrics(
                index=index, detector=detector
            )
            return plot_peak_amplitude_qq(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_ecdf(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the empirical amplitude CDF for one trace and detector selection.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the amplitude metrics.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the CDF.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_peak_amplitude_ecdf`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the peak-amplitude empirical CDF.
            """

            metrics = self._series.compute_peak_amplitude_distribution_metrics(
                index=index, detector=detector
            )
            return plot_peak_amplitude_ecdf(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

    class WidthAnalysisAccessor:
        """Namespace exposing peak-width diagnostics for one dilution series."""

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series

        def diagnose(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
        ) -> Dict[str, PeakWidthDistributionMetrics]:
            """Compute peak-width diagnostics for one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the diagnostics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the peak widths.

            Returns
            -------
            dict
                Mapping from detector label to width diagnostics.
            """

            return self._series.compute_peak_width_distribution_metrics(
                index=index,
                detector=detector,
                x_axis=x_axis,
            )

        def plot_histogram(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the width histogram for one trace and detector selection.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the width metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the peak widths.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the histogram.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_peak_width_histogram`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the peak-width histogram.
            """

            metrics = self._series.compute_peak_width_distribution_metrics(
                index=index, detector=detector, x_axis=x_axis
            )
            return plot_peak_width_histogram(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_qq(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the Q-Q diagnostic for detected peak widths on one trace.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the width metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the peak widths.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the Q-Q plot.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_peak_width_qq`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the peak-width Q-Q plot.
            """

            metrics = self._series.compute_peak_width_distribution_metrics(
                index=index, detector=detector, x_axis=x_axis
            )
            return plot_peak_width_qq(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

        def plot_ecdf(
            self,
            index: int,
            detector: Literal["standard", "cnn", "both"] = "both",
            x_axis: Literal["sample", "time"] = "sample",
            *,
            label: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            **plot_kwargs,
        ) -> plt.Figure:
            """Plot the empirical width CDF for one trace and detector selection.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            detector : {"standard", "cnn", "both"}, default="both"
                Detector outputs included in the width metrics.
            x_axis : {"sample", "time"}, default="sample"
                Units used for the peak widths.
            label : str, optional
                Detector label to plot when multiple detector metrics are present.
            ax : matplotlib.axes.Axes, optional
                Existing axis on which to draw the empirical CDF.
            **plot_kwargs
                Additional keyword arguments forwarded to
                :func:`plot_peak_width_ecdf`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the peak-width empirical CDF.
            """

            metrics = self._series.compute_peak_width_distribution_metrics(
                index=index, detector=detector, x_axis=x_axis
            )
            return plot_peak_width_ecdf(
                metrics_by_label=metrics, label=label, ax=ax, **plot_kwargs
            )

    class TraceAnalysisAccessor:
        """Namespace exposing trace-level detector views for one dilution series."""

        def __init__(self, series: "DilutionSeries") -> None:
            self._series = series

        def standard(self, index: int, **plot_kwargs) -> plt.Figure:
            """Plot the standard-detector trace view for one dilution-series record.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            **plot_kwargs
                Keyword arguments forwarded to
                :meth:`DilutionSeries.plot_standard_detection_for_record`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the standard-detector trace view.
            """

            record = self._series.get_record(index=index)
            return self._series.plot_standard_detection_for_record(
                record=record, **plot_kwargs
            )

        def wavenet(self, index: int, **plot_kwargs) -> plt.Figure:
            """Plot the WaveNet prediction view for one dilution-series record.

            Parameters
            ----------
            index : int
                Trace index in the sorted dilution-series result.
            **plot_kwargs
                Keyword arguments forwarded to
                :meth:`DilutionSeries.plot_wavenet_detection_for_record`.

            Returns
            -------
            matplotlib.figure.Figure
                Figure containing the WaveNet detector trace view.
            """

            record = self._series.get_record(index=index)
            return self._series.plot_wavenet_detection_for_record(
                record=record, **plot_kwargs
            )

    def __init__(
        self,
        folder: Union[str, Path],
        wavenet: Any,
        initial_concentration: float,
        nrows: int,
        std_kwargs: Optional[Mapping[str, Any]] = None,
        cnn_kwargs: Optional[Mapping[str, Any]] = None,
        water_filename: str = "_water_1.csv",
        low_pass: float = None,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        dilution_parser: Optional[Callable[[Path], float]] = None,
        trace_files: Optional[List[Tuple[Union[str, Path], float]]] = None,
        *,
        std_trigger: Optional[TriggerLike] = None,
        cnn_trigger: Optional[TriggerLike] = None,
    ) -> None:
        self.folder = Path(folder)
        self.wavenet = wavenet
        self.initial_concentration = float(initial_concentration)
        self.nrows = int(nrows)
        self.std_kwargs = dict(std_kwargs) if std_kwargs is not None else None
        self.cnn_kwargs = dict(cnn_kwargs) if cnn_kwargs is not None else None
        self.std_trigger = std_trigger
        self.cnn_trigger = cnn_trigger
        self.low_pass = low_pass
        self.dilution_parser = dilution_parser
        self.trace_files = None if trace_files is None else list(trace_files)
        self.water_filename = str(water_filename)
        self.analyzer = WaveNetTraceAnalyzer(
            wavenet=wavenet,
            std_kwargs=self.std_kwargs,
            cnn_kwargs=self.cnn_kwargs,
            std_trigger=self.std_trigger,
            cnn_trigger=self.cnn_trigger,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
        )
        self.sequence_length = self.analyzer.config.sequence_length
        self._last_result: Optional[PeakCountSeriesResult] = None
        self.poisson = self.PoissonAnalysisAccessor(self)
        self.amplitude = self.AmplitudeAnalysisAccessor(self)
        self.width = self.WidthAnalysisAccessor(self)
        self.trace = self.TraceAnalysisAccessor(self)

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

        return self.analyzer.load_processed_signal(
            filename=filename,
            nrows=self.nrows,
            low_pass=self.low_pass,
        )

    @staticmethod
    def plot_peak_counts_for_result(
        series_or_result: Any,
        x_axis: Literal["concentration", "dilution"] = "concentration",
        figsize: Tuple[float, float] = (10.0, 4.0),
        marker_size: float = 42.0,
        line_width: float = 1.2,
        show: bool = False,
        close: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot standard and WaveNet peak counts against dilution or concentration."""

        result = resolve_series_or_result(series_or_result)

        if x_axis == "concentration":
            x_values = np.asarray(result.concentration, dtype=float)
            x_label = "Concentration"
        elif x_axis == "dilution":
            x_values = np.asarray(result.dilution, dtype=float)
            x_label = "Dilution"
        else:
            raise ValueError('x_axis must be either "concentration" or "dilution".')

        with plot_style_context():
            figure, axis = make_plot_figure(figsize=figsize)
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
            return finalize_single_axis_figure(
                figure=figure,
                axis=axis,
                xlabel=x_label,
                ylabel="Peak count",
                show_legend=True,
                show_grid=True,
                save_path=save_path,
                dpi=dpi,
                show=show,
                close=close,
            )

    @staticmethod
    def plot_particle_flows_for_result(
        series_or_result: Any,
        x_axis: Literal["concentration", "dilution"] = "concentration",
        figsize: Tuple[float, float] = (10.0, 4.0),
        marker_size: float = 42.0,
        line_width: float = 1.2,
        show_water_baseline: bool = True,
        show: bool = False,
        close: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot standard and WaveNet particle-flow estimates over the series."""

        result = resolve_series_or_result(series_or_result)

        if x_axis == "concentration":
            x_values = np.asarray(result.concentration, dtype=float)
            x_label = "Concentration"
        elif x_axis == "dilution":
            x_values = np.asarray(result.dilution, dtype=float)
            x_label = "Dilution"
        else:
            raise ValueError('x_axis must be either "concentration" or "dilution".')

        with plot_style_context():
            figure, axis = make_plot_figure(figsize=figsize)
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

            return finalize_single_axis_figure(
                figure=figure,
                axis=axis,
                xlabel=x_label,
                ylabel="Particle flow",
                show_legend=True,
                show_grid=True,
                save_path=save_path,
                dpi=dpi,
                show=show,
                close=close,
            )

    @staticmethod
    def plot_measured_vs_expected_particle_flows_for_result(
        series_or_result: Any,
        base_index: int = 0,
        figsize: Tuple[float, float] = (6.0, 6.0),
        marker_size: float = 42.0,
        line_width: float = 1.2,
        show_ideal_line: bool = True,
        ideal_label: str = "Ideal",
        show: bool = False,
        close: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot measured particle flow against the dilution-scaled expected flow."""

        result = resolve_series_or_result(series_or_result)
        expected_flows = np.asarray(
            [
                DilutionSeries.get_expected_particle_flow_for_result(
                    result, index=index, base_index=base_index
                )
                for index in range(len(result.records))
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

        with plot_style_context():
            figure, axis = make_plot_figure(figsize=figsize)
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

            return finalize_single_axis_figure(
                figure=figure,
                axis=axis,
                xlabel="Expected particle flow",
                ylabel="Measured particle flow",
                show_legend=True,
                show_grid=True,
                save_path=save_path,
                dpi=dpi,
                show=show,
                close=close,
            )

    @staticmethod
    def get_expected_particle_flow_for_result(
        series_or_result: Any,
        index: int,
        base_index: int = 0,
    ) -> float:
        """Estimate target particle flow from a reference dilution trace."""

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

        return float(
            base_record.standard_particle_flow
            * float(base_record.dilution)
            / current_dilution
        )

    def get_record(self, index: int) -> TraceRecord:
        return self._get_record_by_index(index=index)

    def compute_record(self, dilution: float, filename: Path) -> TraceRecord:
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
        return self.analyzer.analyze_processed_signal(
            signal,
            dx=dx,
            filename=filename,
            dilution=float(dilution),
            concentration=float(concentration),
        )

    def compute_water_record(self) -> Optional[TraceRecord]:
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
        return self.analyzer.analyze_processed_signal(
            signal,
            dx=dx,
            filename=water_path,
            dilution=float("nan"),
            concentration=float("nan"),
        )

    def run(self) -> PeakCountSeriesResult:
        """Analyze the entire series and cache the sorted result bundle.

        Returns
        -------
        PeakCountSeriesResult
            Sorted result bundle containing arrays and per-trace records.
        """

        records: List[TraceRecord] = []
        water_record = self.compute_water_record()

        trace_file_iterator = _iterate_explicit_trace_files(
            self.folder, self.trace_files
        )

        for dilution, filename in trace_file_iterator:
            record = self.compute_record(dilution=dilution, filename=Path(filename))
            print(
                f"Standard particle flow: {record.standard_particle_flow}\t"
                f"CNN particle flow: {record.cnn_particle_flow}"
            )
            records.append(record)

        if len(records) == 0:
            result = PeakCountSeriesResult(
                dilution=np.asarray([]),
                concentration=np.asarray([]),
                standard_particle_count=np.asarray([]),
                standard_particle_flow=np.asarray([]),
                cnn_particle_count=np.asarray([]),
                cnn_particle_flow=np.asarray([]),
                water_record=water_record,
                records=[],
            )
            self._last_result = result
            return result

        dilution_series = np.asarray(
            [record.dilution for record in records], dtype=float
        )
        sort_index = np.argsort(dilution_series)[::-1]
        records_sorted = [records[index] for index in sort_index]

        result = PeakCountSeriesResult(
            dilution=np.asarray(
                [record.dilution for record in records_sorted], dtype=float
            ),
            concentration=np.asarray(
                [record.concentration for record in records_sorted], dtype=float
            ),
            standard_particle_count=np.asarray(
                [record.standard.peak_count for record in records_sorted], dtype=float
            ),
            standard_particle_flow=np.asarray(
                [record.standard_particle_flow for record in records_sorted],
                dtype=float,
            ),
            cnn_particle_count=np.asarray(
                [record.cnn.peak_count for record in records_sorted], dtype=float
            ),
            cnn_particle_flow=np.asarray(
                [record.cnn_particle_flow for record in records_sorted], dtype=float
            ),
            water_record=water_record,
            records=records_sorted,
        )
        self._last_result = result
        return result

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

    def plot_peak_counts(self, *args, **kwargs) -> plt.Figure:
        """Plot peak counts for the most recent series result.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :meth:`plot_peak_counts_for_result`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the peak-count series plot.
        """

        return self.plot_peak_counts_for_result(series_or_result=self, *args, **kwargs)

    def plot_particle_flows(self, *args, **kwargs) -> plt.Figure:
        """Plot particle flows for the most recent series result.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :meth:`plot_particle_flows_for_result`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the particle-flow series plot.
        """

        return self.plot_particle_flows_for_result(
            series_or_result=self, *args, **kwargs
        )

    def plot_measured_vs_expected_particle_flows(self, *args, **kwargs) -> plt.Figure:
        """Plot measured particle flow against the dilution-scaled expected flow.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :meth:`plot_measured_vs_expected_particle_flows_for_result`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the measured-vs-expected particle-flow plot.
        """

        return self.plot_measured_vs_expected_particle_flows_for_result(
            series_or_result=self, *args, **kwargs
        )

    def get_expected_particle_flow(self, index: int, base_index: int = 0) -> float:
        """Estimate standard-detector particle flow from a reference trace.

        Parameters
        ----------
        index : int
            Index of the target trace in the sorted series result.
        base_index : int, default=0
            Index of the reference trace in the sorted series result.

        Returns
        -------
        float
            Expected standard-detector particle flow at the target dilution.
        """

        return self.get_expected_particle_flow_for_result(
            series_or_result=self, index=index, base_index=base_index
        )

    @staticmethod
    def plot_standard_detection_for_record(
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
        return record.plot_standard_detection(
            x_axis=x_axis,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            line_width=line_width,
            marker_size=marker_size,
            threshold_line_width=threshold_line_width,
            show_threshold=show_threshold,
            show_peaks=show_peaks,
            show_legend=show_legend,
            show_grid=show_grid,
            show_title=show_title,
            title=title,
            signal_color=signal_color,
            marker_color=marker_color,
            threshold_color=threshold_color,
            rasterize_dense_artists=rasterize_dense_artists,
            ax=ax,
            show=show,
            close=close,
            save_path=save_path,
            dpi=dpi,
        )

    @staticmethod
    def plot_wavenet_detection_for_record(
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
        return record.plot_wavenet_detection(
            x_axis=x_axis,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            line_width=line_width,
            marker_size=marker_size,
            threshold_line_width=threshold_line_width,
            show_signal=show_signal,
            show_prediction=show_prediction,
            show_cnn_signal_peaks=show_cnn_signal_peaks,
            show_cnn_prediction_peaks=show_cnn_prediction_peaks,
            show_cnn_threshold=show_cnn_threshold,
            show_legend=show_legend,
            show_grid=show_grid,
            show_title=show_title,
            title=title,
            signal_color=signal_color,
            prediction_color=prediction_color,
            marker_color=marker_color,
            threshold_color=threshold_color,
            rasterize_dense_artists=rasterize_dense_artists,
            ax=ax,
            show=show,
            close=close,
            save_path=save_path,
            dpi=dpi,
        )

    def plot_standard_detection_by_index(self, index: int, **plot_kwargs) -> plt.Figure:
        """
        Plot the standard detector view for one trace in the series.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        **plot_kwargs
            Forwarded to :meth:`plot_standard_detection_for_record`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the standard detector trace view.
        """

        record = self._get_record_by_index(index=index)
        return self.plot_standard_detection_for_record(record=record, **plot_kwargs)

    def plot_wavenet_detection_by_index(self, index: int, **plot_kwargs) -> plt.Figure:
        """
        Plot the WaveNet detector view for one trace in the series.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        **plot_kwargs
            Forwarded to :meth:`plot_wavenet_detection_for_record`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the WaveNet detector trace view.
        """

        record = self._get_record_by_index(index=index)
        return self.plot_wavenet_detection_for_record(record=record, **plot_kwargs)

    def plot_event_raster(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        Plot event times for one trace and one detector label.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn", "both"}, default="both"
            Detector outputs included in the computed metrics.
        x_axis : {"sample", "time"}, default="sample"
            Units used for the event times.
        number_of_count_bins : int, default=50
            Number of bins used when computing the underlying diagnostics.
        observation_start, observation_end : float, optional
            Optional bounds of the observation window.
        label : str, optional
            Detector label to plot when multiple entries are present.
        **plot_kwargs
            Forwarded to :func:`plot_event_raster`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the event raster.
        """

        metrics = self.compute_event_arrival_distribution_metrics(
            index=index,
            detector=detector,
            x_axis=x_axis,
            number_of_count_bins=number_of_count_bins,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        return plot_event_raster(metrics_by_label=metrics, label=label, **plot_kwargs)

    def plot_inter_arrival_histogram(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        Plot inter-arrival times for one trace and one detector label.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn", "both"}, default="both"
            Detector outputs included in the computed metrics.
        x_axis : {"sample", "time"}, default="sample"
            Units used for the event times.
        number_of_count_bins : int, default=50
            Number of bins used when computing the underlying diagnostics.
        observation_start, observation_end : float, optional
            Optional bounds of the observation window.
        label : str, optional
            Detector label to plot when multiple entries are present.
        **plot_kwargs
            Forwarded to :func:`plot_inter_arrival_histogram`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the inter-arrival histogram.
        """

        metrics = self.compute_event_arrival_distribution_metrics(
            index=index,
            detector=detector,
            x_axis=x_axis,
            number_of_count_bins=number_of_count_bins,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        return plot_inter_arrival_histogram(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def plot_expected_poisson_inter_arrival_histogram(
        self,
        index: int,
        base_index: int = 0,
        detector: Literal["standard", "cnn", "both"] = "standard",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
        *,
        label: Optional[str] = None,
        expected_label: str = "Expected Poisson",
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot measured inter-arrival data against an expected Poisson model.

        The expected Poisson rate is derived from
        :meth:`get_expected_particle_flow` using the dilution-scaled standard
        particle flow of a reference trace.
        """

        metrics = self.compute_event_arrival_distribution_metrics(
            index=index,
            detector=detector,
            x_axis=x_axis,
            number_of_count_bins=number_of_count_bins,
            observation_start=observation_start,
            observation_end=observation_end,
        )

        expected_particle_flow = self.get_expected_particle_flow(
            index=index,
            base_index=base_index,
        )
        record = self._get_record_by_index(index=index)
        if x_axis == "time":
            expected_lambda_hat = expected_particle_flow
        else:
            expected_lambda_hat = expected_particle_flow * float(record.dx)

        return plot_inter_arrival_histogram(
            metrics_by_label=metrics,
            label=label,
            expected_lambda_hat=expected_lambda_hat,
            expected_label=expected_label,
            **plot_kwargs,
        )

    def plot_rescaled_uniform_qq(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        Plot the time-rescaling Q-Q diagnostic for one trace and detector label.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn", "both"}, default="both"
            Detector outputs included in the computed metrics.
        x_axis : {"sample", "time"}, default="sample"
            Units used for the event times.
        number_of_count_bins : int, default=50
            Number of bins used when computing the underlying diagnostics.
        observation_start, observation_end : float, optional
            Optional bounds of the observation window.
        label : str, optional
            Detector label to plot when multiple entries are present.
        **plot_kwargs
            Forwarded to :func:`plot_rescaled_uniform_qq`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the time-rescaling Q-Q plot.
        """

        metrics = self.compute_event_arrival_distribution_metrics(
            index=index,
            detector=detector,
            x_axis=x_axis,
            number_of_count_bins=number_of_count_bins,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        return plot_rescaled_uniform_qq(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def plot_count_distribution(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        Plot counts-per-bin against the fitted Poisson count model.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn", "both"}, default="both"
            Detector outputs included in the computed metrics.
        x_axis : {"sample", "time"}, default="sample"
            Units used for the event times.
        number_of_count_bins : int, default=50
            Number of bins used when computing the underlying diagnostics.
        observation_start, observation_end : float, optional
            Optional bounds of the observation window.
        label : str, optional
            Detector label to plot when multiple entries are present.
        **plot_kwargs
            Forwarded to :func:`plot_count_distribution`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the counts-per-bin plot.
        """

        metrics = self.compute_event_arrival_distribution_metrics(
            index=index,
            detector=detector,
            x_axis=x_axis,
            number_of_count_bins=number_of_count_bins,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        return plot_count_distribution(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def compute_event_arrival_distribution_metrics(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        number_of_count_bins: int = 50,
        observation_start: Optional[float] = None,
        observation_end: Optional[float] = None,
    ) -> Dict[str, EventArrivalDistributionMetrics]:
        """Compute Poisson-style arrival diagnostics for one trace in the series.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn", "both"}, default="both"
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
        detector: Literal["standard", "cnn", "both"] = "both",
    ) -> Dict[str, PeakAmplitudeDistributionMetrics]:
        """Compute peak-amplitude diagnostics for one trace in the series.

        Parameters
        ----------
        index : int
            Index of the trace in the sorted series result.
        detector : {"standard", "cnn", "both"}, default="both"
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

    def plot_peak_amplitude_histogram(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot the peak-amplitude histogram for one trace and detector label."""

        metrics = self.compute_peak_amplitude_distribution_metrics(
            index=index, detector=detector
        )
        return plot_peak_amplitude_histogram(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def plot_peak_amplitude_qq(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot the normal Q-Q diagnostic for peak amplitudes on one trace."""

        metrics = self.compute_peak_amplitude_distribution_metrics(
            index=index, detector=detector
        )
        return plot_peak_amplitude_qq(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def plot_peak_amplitude_ecdf(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot the empirical peak-amplitude CDF for one trace and detector label."""

        metrics = self.compute_peak_amplitude_distribution_metrics(
            index=index, detector=detector
        )
        return plot_peak_amplitude_ecdf(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def compute_peak_width_distribution_metrics(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
    ) -> Dict[str, PeakWidthDistributionMetrics]:
        """Compute peak-width diagnostics for one trace in the series."""

        return compute_peak_width_distribution_metrics(
            series_or_result=self,
            index=index,
            detector=detector,
            x_axis=x_axis,
        )

    def plot_peak_width_histogram(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot the peak-width histogram for one trace and detector label."""

        metrics = self.compute_peak_width_distribution_metrics(
            index=index, detector=detector, x_axis=x_axis
        )
        return plot_peak_width_histogram(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )

    def plot_peak_width_qq(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot the peak-width Q-Q diagnostic for one trace and detector label."""

        metrics = self.compute_peak_width_distribution_metrics(
            index=index, detector=detector, x_axis=x_axis
        )
        return plot_peak_width_qq(metrics_by_label=metrics, label=label, **plot_kwargs)

    def plot_peak_width_ecdf(
        self,
        index: int,
        detector: Literal["standard", "cnn", "both"] = "both",
        x_axis: Literal["sample", "time"] = "sample",
        *,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot the empirical peak-width CDF for one trace and detector label."""

        metrics = self.compute_peak_width_distribution_metrics(
            index=index, detector=detector, x_axis=x_axis
        )
        return plot_peak_width_ecdf(
            metrics_by_label=metrics, label=label, **plot_kwargs
        )


class PeakCountSeries(DilutionSeries):
    """Backward-compatible alias for :class:`DilutionSeries`."""
