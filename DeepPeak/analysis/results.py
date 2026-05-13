"""Result models for WaveNet trace analysis and dilution-series workflows.

This module centralizes the light-weight dataclasses exchanged between the
single-trace analyzer, dilution-series orchestration, and plotting helpers.
Keeping them in one place makes the analysis API easier to document, test, and
reuse from notebooks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class PoissonSeriesDiagnostics:
    """Backward-compatible Poisson diagnostic summary.

    This type is retained for compatibility with older notebook code. New code
    should usually prefer :class:`EventArrivalDistributionMetrics`, which stores
    the same core statistics together with the underlying event-time arrays.

    Attributes
    ----------
    label : str
        Detector label associated with the metrics.
    lambda_hat : float
        Estimated event rate over the observation window.
    observation_start, observation_end : float
        Bounds of the observation window used for the diagnostics.
    observation_duration : float
        Duration of the observation window.
    number_of_events : int
        Number of events retained inside the observation window.
    number_of_inter_arrival_times : int
        Number of strictly positive inter-arrival times used in the tests.
    ks_inter_arrival_stat, ks_inter_arrival_p_value : float
        Kolmogorov-Smirnov statistic and p-value against an exponential model.
    chi2_counts_stat, chi2_counts_dof, chi2_counts_p_value : float or int
        Chi-square summary for the counts-per-bin test.
    mu_hat_per_bin : float
        Estimated expected count per bin under the Poisson model.
    bin_width : float
        Width of the counting bins.
    ks_rescaled_uniform_stat, ks_rescaled_uniform_p_value : float
        Time-rescaling uniformity statistic and p-value.
    """

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
class EventArrivalDistributionMetrics:
    """Event-arrival diagnostics for one detector on one trace.

    The fields combine rate estimates, goodness-of-fit statistics, and the raw
    arrays used by the plotting helpers. The event times are expressed either in
    samples or physical time depending on how the metrics were computed.

    Attributes
    ----------
    label : str
        Detector label associated with the metrics.
    number_of_events : int
        Number of events used in the diagnostics.
    observation_start, observation_end : float
        Bounds of the observation window.
    observation_duration : float
        Duration of the observation window.
    lambda_hat : float
        Estimated event rate over the observation window.
    number_of_inter_arrival_times : int
        Number of strictly positive inter-arrival intervals.
    mean_inter_arrival_time, standard_deviation_inter_arrival_time : float
        Summary statistics of the inter-arrival distribution.
    coefficient_of_variation_inter_arrival_time : float
        Ratio between the standard deviation and the mean inter-arrival time.
    ks_exponential_statistic, ks_exponential_p_value : float
        Goodness-of-fit statistics against an exponential inter-arrival model.
    ks_rescaled_uniform_statistic, ks_rescaled_uniform_p_value : float
        Goodness-of-fit statistics for the time-rescaled uniform model.
    number_of_count_bins : int
        Number of bins used for counts-per-bin diagnostics.
    count_bin_width : float
        Width of one count bin in the chosen x-axis units.
    mean_count_per_bin, variance_count_per_bin : float
        Summary statistics of the counts-per-bin distribution.
    fano_factor_count_per_bin : float
        Variance-to-mean ratio of the counts-per-bin distribution.
    chi2_count_statistic, chi2_count_degrees_of_freedom, chi2_count_p_value : float or int
        Chi-square summary for the counts-per-bin comparison to a Poisson model.
    event_times : numpy.ndarray
        Event times retained inside the observation window.
    inter_arrival_times : numpy.ndarray
        Strictly positive time gaps between successive events.
    counts_per_bin : numpy.ndarray
        Event counts in equally spaced bins over the observation window.
    """

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


@dataclass(frozen=True)
class PeakAmplitudeDistributionMetrics:
    """Peak-amplitude diagnostics for one detector on one trace.

    Attributes
    ----------
    label : str
        Detector label associated with the metrics.
    number_of_peaks : int
        Number of detected peaks contributing amplitudes.
    mean_amplitude, median_amplitude : float
        Central tendency summary statistics.
    minimum_amplitude, maximum_amplitude : float
        Extremal observed peak amplitudes.
    standard_deviation_amplitude : float
        Sample standard deviation of the peak amplitudes.
    coefficient_of_variation_amplitude : float
        Standard deviation divided by the mean amplitude.
    skewness_amplitude, kurtosis_amplitude : float
        Shape diagnostics of the amplitude distribution.
    fitted_normal_mean, fitted_normal_standard_deviation : float
        Parameters of the fitted normal reference model.
    ks_normal_statistic, ks_normal_p_value : float
        Kolmogorov-Smirnov comparison to the fitted normal model.
    amplitudes : numpy.ndarray
        Observed peak amplitudes used in the diagnostics.
    """

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


@dataclass(frozen=True)
class PeakWidthDistributionMetrics:
    """Peak-width diagnostics for one detector on one trace.

    Attributes
    ----------
    label : str
        Detector label associated with the metrics.
    x_axis : {"sample", "time"}
        Unit system used for the stored widths.
    width_unit_label : str
        Human-readable unit label used in plots.
    number_of_peaks : int
        Number of detected peaks contributing widths.
    mean_width, median_width : float
        Central tendency summary statistics.
    minimum_width, maximum_width : float
        Extremal observed peak widths.
    standard_deviation_width : float
        Sample standard deviation of the peak widths.
    coefficient_of_variation_width : float
        Standard deviation divided by the mean width.
    skewness_width, kurtosis_width : float
        Shape diagnostics of the width distribution.
    fitted_lognormal_shape, fitted_lognormal_loc, fitted_lognormal_scale : float
        Parameters of the fitted lognormal reference model.
    ks_lognormal_statistic, ks_lognormal_p_value : float
        Kolmogorov-Smirnov comparison to the fitted lognormal model.
    widths : numpy.ndarray
        Observed peak widths used in the diagnostics.
    """

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


@dataclass
class PeakDetectionResult:
    """Detected peaks and the parameters used to obtain them.

    Attributes
    ----------
    peaks : numpy.ndarray
        Peak indices returned by the detector.
    properties : dict
        Detector-specific metadata associated with each peak.
    peak_count : int
        Number of detected peaks.
    detection_kwargs : dict
        Concrete keyword arguments passed to the detector.
    threshold : float or None
        Effective scalar threshold used for detection when one is available.
    """

    peaks: np.ndarray
    properties: Dict[str, Any]
    peak_count: int
    detection_kwargs: Dict[str, Any]
    threshold: Optional[float] = None

    @property
    def std_kwargs(self) -> Dict[str, Any]:
        """Return the stored detection kwargs for compatibility.

        Returns
        -------
        dict
            The concrete detection keyword arguments.
        """

        return self.detection_kwargs

    @property
    def cnn_kwargs(self) -> Dict[str, Any]:
        """Return the stored detection kwargs for compatibility.

        Returns
        -------
        dict
            The concrete detection keyword arguments.
        """

        return self.detection_kwargs


@dataclass(frozen=True)
class WaveNetAnalyzerConfig:
    """Configuration shared by a :class:`WaveNetTraceAnalyzer` instance.

    Attributes
    ----------
    sequence_length : int
        Window length expected by the model.
    signal_normalization : str
        Name of the normalization strategy applied before inference.
    prediction_sampling_rate_hz : float
        Sampling rate used when low-pass filtering predictions.
    """

    sequence_length: int
    signal_normalization: str = "zscore"
    prediction_sampling_rate_hz: float = 125_000_000.0


@dataclass
class TraceRecord:
    """Canonical analysis bundle for one processed trace.

    A record stores the processed signal, the WaveNet prediction, the standard
    detector result, and the CNN-based detector result for the same acquisition.
    Plotting and series-level aggregation functions consume this object.

    Attributes
    ----------
    filename : pathlib.Path
        Source file associated with the trace.
    dilution : float
        Dilution factor assigned to the trace.
    concentration : float
        Concentration associated with the dilution factor.
    dx : float
        Sampling interval in the trace's x-axis units.
    signal : numpy.ndarray
        Processed signal windows used for standard detection and model inference.
    standard : PeakDetectionResult
        Standard peak-detection result on the processed signal.
    prediction : numpy.ndarray
        WaveNet prediction after optional postprocessing.
    cnn : PeakDetectionResult
        Peak-detection result computed on the WaveNet prediction.
    """

    filename: Path
    dilution: float
    concentration: float
    dx: float
    signal: np.ndarray
    standard: PeakDetectionResult
    prediction: np.ndarray
    cnn: PeakDetectionResult

    @property
    def processed_signal(self) -> np.ndarray:
        """Return the processed signal stored in the record.

        Returns
        -------
        numpy.ndarray
            The processed signal windows.
        """

        return self.signal

    @property
    def wavenet_prediction(self) -> np.ndarray:
        """Return the WaveNet prediction stored in the record.

        Returns
        -------
        numpy.ndarray
            The postprocessed WaveNet prediction.
        """

        return self.prediction

    @property
    def delta_x(self) -> float:
        """Return the total duration covered by the processed signal.

        Returns
        -------
        float
            Product of the sampling interval and the number of samples.
        """

        return float(self.dx * self.signal.size)

    @property
    def standard_particle_flow(self) -> float:
        """Return the standard detector particle flow.

        Returns
        -------
        float
            Number of standard peaks divided by the trace duration.
        """

        return float(self.standard.peaks.size / self.delta_x)

    @property
    def cnn_particle_flow(self) -> float:
        """Return the WaveNet detector particle flow.

        Returns
        -------
        float
            Number of CNN peaks divided by the trace duration.
        """

        return float(self.cnn.peak_count / self.delta_x)

    def plot_standard_detection(
        self,
        x_axis: str = "sample",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        figsize: tuple[float, float] = (10.0, 4.0),
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
        ax: Optional[Any] = None,
        show: bool = False,
        close: bool = False,
        save_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> Any:
        """Plot the processed trace with its standard-detector annotations.

        Parameters
        ----------
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

        if isinstance(x_axis, TraceRecord):
            raise TypeError(
                "record.plot_standard_detection(...) is a bound method, so do not pass the record again. "
                "Use record.plot_standard_detection(figsize=(12, 4)) or set x_axis='time'."
            )

        from .trace_plots import plot_standard_detection_trace

        return plot_standard_detection_trace(
            record=self,
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

    def plot_wavenet_detection(
        self,
        x_axis: str = "sample",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        figsize: tuple[float, float] = (10.0, 4.0),
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
        ax: Optional[Any] = None,
        show: bool = False,
        close: bool = False,
        save_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> Any:
        """Plot the trace together with its WaveNet prediction and CNN peaks.

        Parameters
        ----------
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

        if isinstance(x_axis, TraceRecord):
            raise TypeError(
                "record.plot_wavenet_detection(...) is a bound method, so do not pass the record again. "
                "Use record.plot_wavenet_detection(figsize=(12, 4)) or set x_axis='time'."
            )

        from .trace_plots import plot_wavenet_detection_trace

        return plot_wavenet_detection_trace(
            record=self,
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


@dataclass(frozen=True)
class PeakCountSeriesResult:
    """Aggregated peak-count and particle-flow arrays over a dilution series.

    Attributes
    ----------
    dilution, concentration : numpy.ndarray
        Sorted dilution factors and associated concentrations.
    standard_particle_count, cnn_particle_count : numpy.ndarray
        Peak counts per trace for the standard and WaveNet detectors.
    standard_particle_flow, cnn_particle_flow : numpy.ndarray
        Particle-flow estimates per trace for the standard and WaveNet detectors.
    water_record : TraceRecord or None
        Optional analyzed control trace.
    records : list of TraceRecord
        Sorted per-trace records used to build the arrays.
    """

    dilution: np.ndarray
    concentration: np.ndarray
    standard_particle_count: np.ndarray
    standard_particle_flow: np.ndarray
    cnn_particle_count: np.ndarray
    cnn_particle_flow: np.ndarray
    water_record: Optional[TraceRecord]
    records: List[TraceRecord]


def resolve_series_or_result(series_or_result: Any) -> Any:
    """Return a result object from either a series instance or a result bundle.

    Parameters
    ----------
    series_or_result:
        Either a :class:`PeakCountSeriesResult`-like object exposing ``records``
        or a series object exposing ``get_last_result()``.

    Returns
    -------
    object
        The resolved result object exposing ``records`` and the aggregated
        series arrays.
    """

    if hasattr(series_or_result, "records"):
        return series_or_result

    if hasattr(series_or_result, "get_last_result"):
        return series_or_result.get_last_result()

    raise TypeError(
        "series_or_result must be either a result object with `.records` "
        "or a PeakCountSeries-like object with `.get_last_result()`."
    )
