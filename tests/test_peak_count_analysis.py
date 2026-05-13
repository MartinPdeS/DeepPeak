import matplotlib.pyplot as plt
import numpy as np
import pytest

from DeepPeak.analysis import (
    CsvTrace,
    DilutionSeries,
    EventArrivalDistributionMetrics,
    PeakAmplitudeDistributionMetrics,
    PeakCountSeries,
    PeakCountSeriesResult,
    PeakTrigger,
    PeakWidthDistributionMetrics,
    WaveNetTraceAnalyzer,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
    plot_count_distribution,
    plot_event_raster,
    plot_inter_arrival_histogram,
    plot_peak_amplitude_ecdf,
    plot_peak_amplitude_histogram,
    plot_peak_amplitude_qq,
    plot_standard_detection_trace,
    plot_peak_width_ecdf,
    plot_peak_width_histogram,
    plot_peak_width_qq,
    plot_wavenet_detection_trace,
)


class DummyWaveNet:
    sequence_length = 8

    def predict(self, signal):
        return np.asarray(signal, dtype=float)


def test_wavenet_trace_analyzer_analyzes_processed_signal():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    assert record.filename.name == "trace.csv"
    assert record.signal.shape == (1, 8)
    assert record.standard.peaks.tolist() == [3]
    assert record.cnn.peaks.tolist() == [3]
    assert record.cnn.peak_count == 1
    assert record.standard_particle_flow == record.cnn_particle_flow


def test_wavenet_trace_analyzer_segments_one_dimensional_traces():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    assert record.signal.shape == (2, 8)
    assert record.standard.peaks.tolist() == [2, 11]
    assert record.cnn.peaks.tolist() == [2, 11]


def test_analysis_package_exports_series_result_type():
    result = PeakCountSeriesResult(
        dilution=np.array([10.0]),
        concentration=np.array([1.0]),
        standard_particle_count=np.array([2.0]),
        standard_particle_flow=np.array([4.0]),
        cnn_particle_count=np.array([3.0]),
        cnn_particle_flow=np.array([6.0]),
        water_record=None,
        records=[],
    )

    assert result.cnn_particle_flow.tolist() == [6.0]


def test_analysis_package_exports_renamed_loader_and_specific_plotters():
    assert CsvTrace.__name__ == "CsvTrace"
    assert DilutionSeries.__name__ == "DilutionSeries"
    assert callable(DilutionSeries.plot_peak_counts_for_result)
    assert callable(DilutionSeries.plot_particle_flows_for_result)
    assert callable(DilutionSeries.plot_measured_vs_expected_particle_flows_for_result)
    assert callable(DilutionSeries.get_expected_particle_flow_for_result)
    assert callable(DilutionSeries.plot_standard_detection_for_record)
    assert callable(DilutionSeries.plot_wavenet_detection_for_record)
    assert callable(plot_standard_detection_trace)
    assert callable(plot_wavenet_detection_trace)
    assert callable(compute_peak_amplitude_distribution_metrics)
    assert callable(compute_peak_width_distribution_metrics)
    assert callable(plot_event_raster)
    assert callable(plot_inter_arrival_histogram)
    assert callable(plot_peak_amplitude_histogram)
    assert callable(plot_peak_amplitude_qq)
    assert callable(plot_peak_amplitude_ecdf)
    assert callable(plot_peak_width_histogram)
    assert callable(plot_peak_width_qq)
    assert callable(plot_peak_width_ecdf)


def test_analysis_package_exports_amplitude_metrics_type():
    metrics = PeakAmplitudeDistributionMetrics(
        label="standard",
        number_of_peaks=2,
        mean_amplitude=2.0,
        median_amplitude=2.0,
        minimum_amplitude=1.0,
        maximum_amplitude=3.0,
        standard_deviation_amplitude=1.0,
        coefficient_of_variation_amplitude=0.5,
        skewness_amplitude=0.0,
        kurtosis_amplitude=np.nan,
        fitted_normal_mean=2.0,
        fitted_normal_standard_deviation=1.0,
        ks_normal_statistic=np.nan,
        ks_normal_p_value=np.nan,
        amplitudes=np.array([1.0, 3.0]),
    )

    assert metrics.number_of_peaks == 2


def test_analysis_package_exports_width_metrics_type():
    metrics = PeakWidthDistributionMetrics(
        label="standard",
        x_axis="sample",
        width_unit_label="Samples",
        number_of_peaks=2,
        mean_width=2.0,
        median_width=2.0,
        minimum_width=1.0,
        maximum_width=3.0,
        standard_deviation_width=1.0,
        coefficient_of_variation_width=0.5,
        skewness_width=0.0,
        kurtosis_width=np.nan,
        fitted_lognormal_shape=0.1,
        fitted_lognormal_loc=0.0,
        fitted_lognormal_scale=2.0,
        ks_lognormal_statistic=np.nan,
        ks_lognormal_p_value=np.nan,
        widths=np.array([1.0, 3.0]),
    )

    assert metrics.number_of_peaks == 2


def test_histogram_plots_accept_edge_styling_and_have_no_title_by_default():
    amplitude_metrics = PeakAmplitudeDistributionMetrics(
        label="standard",
        number_of_peaks=3,
        mean_amplitude=2.0,
        median_amplitude=2.0,
        minimum_amplitude=1.0,
        maximum_amplitude=3.0,
        standard_deviation_amplitude=1.0,
        coefficient_of_variation_amplitude=0.5,
        skewness_amplitude=0.0,
        kurtosis_amplitude=np.nan,
        fitted_normal_mean=2.0,
        fitted_normal_standard_deviation=0.5,
        ks_normal_statistic=np.nan,
        ks_normal_p_value=np.nan,
        amplitudes=np.array([1.0, 2.0, 3.0]),
    )
    amplitude_figure = plot_peak_amplitude_histogram(
        {"standard": amplitude_metrics},
        histogram_color="orange",
        edge_color="red",
        edge_line_width=2.5,
    )
    amplitude_patch = amplitude_figure.axes[0].patches[0]
    assert amplitude_figure.axes[0].get_title() == ""
    assert amplitude_patch.get_facecolor() == pytest.approx(
        (1.0, 0.6470588235294118, 0.0, 0.75)
    )
    assert amplitude_patch.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)
    assert amplitude_patch.get_linewidth() == pytest.approx(2.5)

    width_metrics = PeakWidthDistributionMetrics(
        label="standard",
        x_axis="sample",
        width_unit_label="Samples",
        number_of_peaks=3,
        mean_width=2.0,
        median_width=2.0,
        minimum_width=1.0,
        maximum_width=3.0,
        standard_deviation_width=1.0,
        coefficient_of_variation_width=0.5,
        skewness_width=0.0,
        kurtosis_width=np.nan,
        fitted_lognormal_shape=0.2,
        fitted_lognormal_loc=0.0,
        fitted_lognormal_scale=2.0,
        ks_lognormal_statistic=np.nan,
        ks_lognormal_p_value=np.nan,
        widths=np.array([1.0, 2.0, 3.0]),
    )
    width_figure = plot_peak_width_histogram(
        {"standard": width_metrics},
        histogram_color="cyan",
        edge_color="green",
        edge_line_width=1.75,
    )
    width_patch = width_figure.axes[0].patches[0]
    assert width_figure.axes[0].get_title() == ""
    assert width_patch.get_facecolor() == pytest.approx((0.0, 1.0, 1.0, 0.75))
    assert width_patch.get_edgecolor() == (0.0, 0.5019607843137255, 0.0, 1.0)
    assert width_patch.get_linewidth() == pytest.approx(1.75)

    event_metrics = EventArrivalDistributionMetrics(
        label="standard",
        number_of_events=4,
        observation_start=0.0,
        observation_end=4.0,
        observation_duration=4.0,
        lambda_hat=1.0,
        number_of_inter_arrival_times=3,
        mean_inter_arrival_time=1.0,
        standard_deviation_inter_arrival_time=0.0,
        coefficient_of_variation_inter_arrival_time=0.0,
        ks_exponential_statistic=np.nan,
        ks_exponential_p_value=np.nan,
        ks_rescaled_uniform_statistic=np.nan,
        ks_rescaled_uniform_p_value=np.nan,
        number_of_count_bins=4,
        count_bin_width=1.0,
        mean_count_per_bin=1.0,
        variance_count_per_bin=0.0,
        fano_factor_count_per_bin=0.0,
        chi2_count_statistic=np.nan,
        chi2_count_degrees_of_freedom=0,
        chi2_count_p_value=np.nan,
        event_times=np.array([0.5, 1.5, 2.5, 3.5]),
        inter_arrival_times=np.array([1.0, 1.0, 1.0]),
        counts_per_bin=np.array([1, 0, 2, 1]),
    )
    inter_arrival_figure = plot_inter_arrival_histogram(
        {"standard": event_metrics},
        histogram_color="yellow",
        edge_color="blue",
        edge_line_width=1.5,
    )
    inter_arrival_patch = inter_arrival_figure.axes[0].patches[0]
    assert inter_arrival_figure.axes[0].get_title() == ""
    assert inter_arrival_patch.get_facecolor() == pytest.approx((1.0, 1.0, 0.0, 0.75))
    assert inter_arrival_patch.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)
    assert inter_arrival_patch.get_linewidth() == pytest.approx(1.5)

    count_figure = plot_count_distribution(
        {"standard": event_metrics},
        histogram_color="purple",
        edge_color="magenta",
        edge_line_width=3.0,
    )
    count_patch = count_figure.axes[0].patches[0]
    assert count_figure.axes[0].get_title() == ""
    assert count_patch.get_facecolor() == pytest.approx(
        (0.5019607843137255, 0.0, 0.5019607843137255, 0.75)
    )
    assert count_patch.get_edgecolor() == (1.0, 0.0, 1.0, 1.0)
    assert count_patch.get_linewidth() == pytest.approx(3.0)


def test_peak_count_series_accepts_explicit_trace_files(monkeypatch, tmp_path):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = PeakCountSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(PeakCountSeries, "_load_signal", fake_load_signal)

    result = series.run()

    assert result.dilution.tolist() == [100.0, 10.0]
    assert result.concentration.tolist() == [1.0, 10.0]


def test_expected_particle_flow_scales_from_reference_trace(monkeypatch, tmp_path):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = PeakCountSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        if filename == filenames[0]:
            return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0]), 0.25
        return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 3.0]), 0.25

    monkeypatch.setattr(PeakCountSeries, "_load_signal", fake_load_signal)

    series.run()

    expected_flow = DilutionSeries.get_expected_particle_flow_for_result(
        series, index=1, base_index=0
    )

    assert expected_flow == 5.0
    assert series.get_expected_particle_flow(index=1, base_index=0) == expected_flow


def test_plot_expected_poisson_inter_arrival_histogram_overlays_expected_curve(
    monkeypatch, tmp_path
):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = PeakCountSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        if filename == filenames[0]:
            return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]), 0.25
        return np.array([0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0]), 0.25

    monkeypatch.setattr(PeakCountSeries, "_load_signal", fake_load_signal)

    series.run()

    figure = series.plot_expected_poisson_inter_arrival_histogram(
        index=1,
        base_index=0,
        detector="standard",
        x_axis="time",
        label="standard",
    )

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2


def test_plot_measured_vs_expected_particle_flows_adds_ideal_line(
    monkeypatch, tmp_path
):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = PeakCountSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        if filename == filenames[0]:
            return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0]), 0.25
        return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 3.0]), 0.25

    monkeypatch.setattr(PeakCountSeries, "_load_signal", fake_load_signal)

    series.run()

    figure = series.plot_measured_vs_expected_particle_flows(base_index=0)

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 3
    ideal_line = figure.axes[0].lines[2]
    assert ideal_line.get_xdata().tolist() == ideal_line.get_ydata().tolist()


def test_dilution_series_poisson_accessor_wraps_arrival_api(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    metrics = series.poisson.diagnose(index=0, detector="standard", x_axis="time")
    figure = series.poisson.plot_expected_inter_arrival_histogram(
        index=0,
        base_index=0,
        detector="standard",
        x_axis="time",
        label="standard",
    )

    assert set(metrics) == {"standard"}
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2


def test_dilution_series_amplitude_and_width_accessors_wrap_distribution_api(
    monkeypatch, tmp_path
):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    amplitude_metrics = series.amplitude.diagnose(index=0, detector="standard")
    amplitude_figure = series.amplitude.plot_histogram(
        index=0, detector="standard", label="standard"
    )
    width_metrics = series.width.diagnose(index=0, detector="standard", x_axis="time")
    width_figure = series.width.plot_histogram(
        index=0, detector="standard", x_axis="time", label="standard"
    )

    assert set(amplitude_metrics) == {"standard"}
    assert amplitude_metrics["standard"].number_of_peaks == 2
    assert len(amplitude_figure.axes) == 1
    assert set(width_metrics) == {"standard"}
    assert width_metrics["standard"].number_of_peaks == 2
    assert len(width_figure.axes) == 1


def test_dilution_series_trace_accessor_wraps_trace_plot_api(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    standard_figure = series.trace.standard(index=0)
    wavenet_figure = series.trace.wavenet(index=0)

    assert len(standard_figure.axes) == 1
    assert len(wavenet_figure.axes) == 1


def test_record_trace_plot_functions_accept_analyzer_records():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    standard_figure = plot_standard_detection_trace(record, figsize=(12.0, 4.0))
    wavenet_figure = plot_wavenet_detection_trace(record, figsize=(12.0, 4.0))

    assert len(standard_figure.axes) == 1
    assert len(wavenet_figure.axes) == 1
    assert standard_figure.axes[0].get_xlabel() == "Sample index"
    assert wavenet_figure.axes[0].get_ylabel() == "Signal / prediction amplitude"


def test_trace_record_exposes_plot_methods():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    standard_figure = record.plot_standard_detection(figsize=(12.0, 4.0))
    wavenet_figure = record.plot_wavenet_detection(figsize=(12.0, 4.0))

    assert len(standard_figure.axes) == 1
    assert len(wavenet_figure.axes) == 1
    assert standard_figure.axes[0].get_ylabel() == "Signal amplitude"
    assert wavenet_figure.axes[0].get_xlabel() == "Sample index"


def test_trace_record_plot_methods_raise_clear_error_when_record_is_passed_again():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    with pytest.raises(TypeError, match="do not pass the record again"):
        record.plot_standard_detection(record)

    with pytest.raises(TypeError, match="do not pass the record again"):
        record.plot_wavenet_detection(record)


def test_trace_record_plots_have_no_title_by_default():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    standard_figure = record.plot_standard_detection()
    wavenet_figure = record.plot_wavenet_detection()

    assert standard_figure.axes[0].get_title() == ""
    assert wavenet_figure.axes[0].get_title() == ""


def test_accessor_single_axis_plots_accept_existing_axes(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.0, 4.0))
    left_axis, right_axis = axes

    amplitude_figure = series.amplitude.plot_qq(
        index=0, detector="standard", label="standard", ax=left_axis
    )
    trace_figure = series.trace.wavenet(index=0, ax=right_axis)

    assert amplitude_figure is figure
    assert trace_figure is figure
    assert len(figure.axes) == 2


def test_analyzer_accepts_peak_trigger_instances():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=PeakTrigger(height=1.5),
        cnn_trigger=PeakTrigger(height=0.6, low_pass=40_000_000),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    assert record.standard.peaks.tolist() == [3]
    assert record.cnn.peaks.tolist() == [3]
    assert analyzer.cnn_trigger.low_pass == 40_000_000


def test_sigma_trigger_resolves_hysteresis_in_sigma_units():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=PeakTrigger(sigma=1.0, hysteresis=1.0),
        cnn_trigger=PeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.1, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    result = analyzer.detect_standard_peaks(signal)

    assert result.threshold is not None
    assert result.detection_kwargs["height"] == pytest.approx(result.threshold)
    assert result.detection_kwargs["hysteresis"] == pytest.approx(result.threshold)
    assert result.peaks.tolist() == [3]


def test_peak_trigger_rejects_hysteresis_above_height_early():
    with pytest.raises(ValueError, match="hysteresis must be <= height"):
        PeakTrigger(height=0.15, hysteresis=0.2)


def test_analyzer_rejects_hysteresis_above_sigma_resolved_threshold():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=PeakTrigger(sigma=1.0, hysteresis=10.0),
        cnn_trigger=PeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    with pytest.raises(ValueError, match="resolved detection threshold"):
        analyzer.detect_standard_peaks(np.array([0.0, 0.1, 0.2, 0.1, 0.0]))
