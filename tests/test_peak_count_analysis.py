from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from DeepPeak.analysis import (
    BasePeakTrigger,
    CNNTraceAnalyzer,
    CsvTrace,
    DilutionSeries,
    HeightPeakTrigger,
    PeakCountSeries,
    PeakCountSeriesResult,
    SigmaPeakTrigger,
    StandardTraceAnalyzer,
    WaveNetTraceAnalyzer,
    compute_peak_amplitude_distribution_metrics,
    compute_peak_width_distribution_metrics,
    metrics as analysis_metrics,
)


class DummyWaveNet:
    sequence_length = 8

    def predict(self, signal):
        return np.asarray(signal, dtype=float)


def test_wavenet_trace_analyzer_analyzes_processed_signal():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    assert record.signal.shape == (2, 8)
    assert record.standard.peaks.tolist() == [2, 11]
    assert record.cnn.peaks.tolist() == [2, 11]


def test_standard_trace_analyzer_only_populates_standard_detection():
    analyzer = StandardTraceAnalyzer(
        std_trigger=HeightPeakTrigger(height=1.5),
        sequence_length=8,
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    assert record.standard.peaks.tolist() == [3]
    assert record.prediction.size == 0
    assert record.cnn.peak_count == 0


def test_cnn_trace_analyzer_only_populates_cnn_detection():
    analyzer = CNNTraceAnalyzer(
        wavenet=DummyWaveNet(),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    assert record.standard.peak_count == 0
    assert record.cnn.peaks.tolist() == [3]
    assert record.prediction.shape == (8,)


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
    assert StandardTraceAnalyzer.__name__ == "StandardTraceAnalyzer"
    assert CNNTraceAnalyzer.__name__ == "CNNTraceAnalyzer"
    assert DilutionSeries.__name__ == "DilutionSeries"
    assert callable(DilutionSeries.get_expected_particle_flow_for_result)
    assert callable(DilutionSeries.PlotAccessor.peak_counts)
    assert callable(DilutionSeries.PlotAccessor.particle_flows)
    assert callable(DilutionSeries.PlotAccessor.measured_particle_flows)
    assert callable(DilutionSeries.PlotAccessor.measured_vs_expected_particle_flows)
    assert callable(
        DilutionSeries.PlotAccessor.detected_peaks_per_second_vs_expected_throughput
    )
    assert callable(analysis_metrics.TraceRecord.plot_standard_detection)
    assert callable(analysis_metrics.TraceRecord.plot_wavenet_detection)
    assert callable(analysis_metrics.TraceRecord.plot_standard_detection_with_histogram)
    assert callable(analysis_metrics.TraceRecord.plot_wavenet_detection_with_histogram)
    assert callable(compute_peak_amplitude_distribution_metrics)
    assert callable(compute_peak_width_distribution_metrics)
    assert callable(DilutionSeries.PoissonAnalysisAccessor.PlotAccessor.histogram)
    assert callable(
        DilutionSeries.PoissonAnalysisAccessor.PlotAccessor.expected_histogram
    )
    assert callable(DilutionSeries.PoissonAnalysisAccessor.PlotAccessor.qq)
    assert callable(
        DilutionSeries.PoissonAnalysisAccessor.PlotAccessor.count_distribution
    )
    assert callable(DilutionSeries.AmplitudeAnalysisAccessor.PlotAccessor.histogram)
    assert callable(DilutionSeries.AmplitudeAnalysisAccessor.PlotAccessor.qq)
    assert callable(DilutionSeries.AmplitudeAnalysisAccessor.PlotAccessor.ecdf)
    assert callable(
        DilutionSeries.AmplitudeAnalysisAccessor.PlotAccessor.local_crowding
    )
    assert callable(DilutionSeries.WidthAnalysisAccessor.PlotAccessor.histogram)
    assert callable(DilutionSeries.WidthAnalysisAccessor.PlotAccessor.qq)
    assert callable(DilutionSeries.WidthAnalysisAccessor.PlotAccessor.ecdf)
    assert callable(DilutionSeries.WidthAnalysisAccessor.PlotAccessor.compare)


def test_series_plot_accessors_support_width_comparison_and_local_crowding():
    def make_detection(peaks, amplitudes, widths_pixels):
        peaks = np.asarray(peaks, dtype=int)
        amplitudes = np.asarray(amplitudes, dtype=float)
        return analysis_metrics.PeakDetectionResult(
            peaks=peaks,
            properties={"widths_pixels": np.asarray(widths_pixels, dtype=float)},
            peak_count=int(peaks.size),
            detection_kwargs={},
            amplitudes=amplitudes,
        )

    record = analysis_metrics.TraceRecord(
        filename=Path("trace.csv"),
        dilution=10.0,
        concentration=1.0,
        dx=0.1,
        signal=np.array([0.0, 0.2, 0.6, 0.1, 0.4, 0.8, 0.2, 0.5, 0.0], dtype=float),
        standard=make_detection(
            peaks=[2, 5, 7], amplitudes=[0.6, 0.8, 0.5], widths_pixels=[3.0, 4.0, 3.5]
        ),
        prediction=np.zeros(9, dtype=float),
        cnn=make_detection(
            peaks=[2, 4, 5, 7],
            amplitudes=[0.55, 0.35, 0.65, 0.45],
            widths_pixels=[2.0, 2.5, 2.0, 2.5],
        ),
    )

    result = PeakCountSeriesResult(
        dilution=np.array([10.0]),
        concentration=np.array([1.0]),
        standard_particle_count=np.array([3.0]),
        standard_particle_flow=np.array([record.standard_particle_flow]),
        cnn_particle_count=np.array([4.0]),
        cnn_particle_flow=np.array([record.cnn_particle_flow]),
        water_record=None,
        records=[record],
    )

    series = object.__new__(DilutionSeries)
    series._last_result = result

    width_figure = DilutionSeries.WidthAnalysisAccessor(series).plot.compare(
        index=0,
        x_axis="time",
        plot="histogram",
    )
    crowding_figure = DilutionSeries.AmplitudeAnalysisAccessor(
        series
    ).plot.local_crowding(
        index=0,
        detector="cnn",
        x_axis="time",
    )
    throughput_figure = DilutionSeries.PlotAccessor(
        series
    ).detected_peaks_per_second_vs_expected_throughput(
        show_ideal_line=False,
    )

    assert len(width_figure.axes) == 2
    assert len(crowding_figure.axes) == 1
    assert len(crowding_figure.axes[0].collections) == 1
    assert len(throughput_figure.axes) == 1

    plt.close(width_figure)
    plt.close(crowding_figure)
    plt.close(throughput_figure)


def test_analysis_package_exports_amplitude_metrics_type():
    amplitude_metrics = analysis_metrics.PeakAmplitudeDistribution(
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

    assert amplitude_metrics.number_of_peaks == 2


def test_analysis_package_exports_width_metrics_type():
    width_metrics = analysis_metrics.PeakWidthDistribution(
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

    assert width_metrics.number_of_peaks == 2


def test_histogram_plots_accept_edge_styling_and_have_no_title_by_default():
    amplitude_metrics = analysis_metrics.PeakAmplitudeDistribution(
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
    amplitude_figure = amplitude_metrics.plot.histogram(
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

    width_metrics = analysis_metrics.PeakWidthDistribution(
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
    width_figure = width_metrics.plot.histogram(
        histogram_color="cyan",
        edge_color="green",
        edge_line_width=1.75,
    )
    width_patch = width_figure.axes[0].patches[0]
    assert width_figure.axes[0].get_title() == ""
    assert width_patch.get_facecolor() == pytest.approx((0.0, 1.0, 1.0, 0.75))
    assert width_patch.get_edgecolor() == (0.0, 0.5019607843137255, 0.0, 1.0)
    assert width_patch.get_linewidth() == pytest.approx(1.75)

    event_metrics = analysis_metrics.EventArrivalDistribution(
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
    inter_arrival_figure = event_metrics.plot.histogram(
        histogram_color="yellow",
        edge_color="blue",
        edge_line_width=1.5,
    )
    inter_arrival_patch = inter_arrival_figure.axes[0].patches[0]
    assert inter_arrival_figure.axes[0].get_title() == ""
    assert inter_arrival_patch.get_facecolor() == pytest.approx((1.0, 1.0, 0.0, 0.75))
    assert inter_arrival_patch.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)
    assert inter_arrival_patch.get_linewidth() == pytest.approx(1.5)

    count_figure = event_metrics.plot.count_distribution(
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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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


def test_expected_particle_flow_can_fit_background_from_multiple_references():
    def make_detection(count):
        return analysis_metrics.PeakDetectionResult(
            peaks=np.arange(count, dtype=int),
            properties={},
            peak_count=count,
            detection_kwargs={},
            amplitudes=np.arange(count, dtype=float),
        )

    def make_record(dilution, flow):
        return analysis_metrics.TraceRecord(
            filename=Path(f"trace_{dilution}.csv"),
            dilution=float(dilution),
            concentration=0.0,
            dx=1.0,
            signal=np.zeros(1, dtype=float),
            standard=make_detection(flow),
            prediction=np.zeros(1, dtype=float),
            cnn=make_detection(0),
        )

    result = PeakCountSeriesResult(
        dilution=np.array([10.0, 20.0, 40.0]),
        concentration=np.zeros(3, dtype=float),
        standard_particle_count=np.array([9.0, 5.0, 3.0]),
        standard_particle_flow=np.array([9.0, 5.0, 3.0]),
        cnn_particle_count=np.zeros(3, dtype=float),
        cnn_particle_flow=np.zeros(3, dtype=float),
        water_record=None,
        records=[make_record(10.0, 9), make_record(20.0, 5), make_record(40.0, 3)],
    )

    expected_flow = DilutionSeries.get_expected_particle_flow_for_result(
        result,
        index=1,
        reference_indices=[0, 2],
    )

    assert expected_flow == pytest.approx(5.0)


def test_expected_particle_flow_uses_water_record_as_blank_baseline():
    def make_detection(count):
        return analysis_metrics.PeakDetectionResult(
            peaks=np.arange(count, dtype=int),
            properties={},
            peak_count=count,
            detection_kwargs={},
            amplitudes=np.arange(count, dtype=float),
        )

    def make_record(name, dilution, flow):
        return analysis_metrics.TraceRecord(
            filename=Path(name),
            dilution=float(dilution),
            concentration=0.0,
            dx=1.0,
            signal=np.zeros(1, dtype=float),
            standard=make_detection(flow),
            prediction=np.zeros(1, dtype=float),
            cnn=make_detection(0),
        )

    result = PeakCountSeriesResult(
        dilution=np.array([10.0, 20.0]),
        concentration=np.zeros(2, dtype=float),
        standard_particle_count=np.array([12.0, 7.0]),
        standard_particle_flow=np.array([12.0, 7.0]),
        cnn_particle_count=np.zeros(2, dtype=float),
        cnn_particle_flow=np.zeros(2, dtype=float),
        water_record=make_record("water.csv", float("nan"), 2),
        records=[
            make_record("trace_10.csv", 10.0, 12),
            make_record("trace_20.csv", 20.0, 7),
        ],
    )

    expected_flow = DilutionSeries.get_expected_particle_flow_for_result(
        result,
        index=1,
        base_index=0,
    )

    assert expected_flow == pytest.approx(7.0)


def test_expected_particle_flow_can_ignore_water_record_when_requested():
    def make_detection(count):
        return analysis_metrics.PeakDetectionResult(
            peaks=np.arange(count, dtype=int),
            properties={},
            peak_count=count,
            detection_kwargs={},
            amplitudes=np.arange(count, dtype=float),
        )

    def make_record(name, dilution, flow):
        return analysis_metrics.TraceRecord(
            filename=Path(name),
            dilution=float(dilution),
            concentration=0.0,
            dx=1.0,
            signal=np.zeros(1, dtype=float),
            standard=make_detection(flow),
            prediction=np.zeros(1, dtype=float),
            cnn=make_detection(0),
        )

    result = PeakCountSeriesResult(
        dilution=np.array([10.0, 20.0]),
        concentration=np.zeros(2, dtype=float),
        standard_particle_count=np.array([12.0, 7.0]),
        standard_particle_flow=np.array([12.0, 7.0]),
        cnn_particle_count=np.zeros(2, dtype=float),
        cnn_particle_flow=np.zeros(2, dtype=float),
        water_record=make_record("water.csv", float("nan"), 2),
        records=[
            make_record("trace_10.csv", 10.0, 12),
            make_record("trace_20.csv", 20.0, 7),
        ],
    )

    expected_flow = DilutionSeries.get_expected_particle_flow_for_result(
        result,
        index=1,
        base_index=0,
        use_water_baseline=False,
    )

    assert expected_flow == pytest.approx(6.0)


def test_expected_particle_flow_excludes_target_from_multipoint_fit():
    def make_detection(count):
        return analysis_metrics.PeakDetectionResult(
            peaks=np.arange(count, dtype=int),
            properties={},
            peak_count=count,
            detection_kwargs={},
            amplitudes=np.arange(count, dtype=float),
        )

    def make_record(dilution, flow):
        return analysis_metrics.TraceRecord(
            filename=Path(f"trace_{dilution}.csv"),
            dilution=float(dilution),
            concentration=0.0,
            dx=1.0,
            signal=np.zeros(1, dtype=float),
            standard=make_detection(flow),
            prediction=np.zeros(1, dtype=float),
            cnn=make_detection(0),
        )

    result = PeakCountSeriesResult(
        dilution=np.array([10.0, 20.0, 40.0]),
        concentration=np.zeros(3, dtype=float),
        standard_particle_count=np.array([9.0, 5.0, 6.0]),
        standard_particle_flow=np.array([9.0, 5.0, 6.0]),
        cnn_particle_count=np.zeros(3, dtype=float),
        cnn_particle_flow=np.zeros(3, dtype=float),
        water_record=None,
        records=[make_record(10.0, 9), make_record(20.0, 5), make_record(40.0, 6)],
    )

    expected_flow = DilutionSeries.get_expected_particle_flow_for_result(
        result,
        index=2,
        reference_indices=[0, 1, 2],
    )

    assert expected_flow == pytest.approx(3.0)


def test_expected_particle_flow_requires_non_empty_reference_indices():
    result = PeakCountSeriesResult(
        dilution=np.array([10.0]),
        concentration=np.array([0.0]),
        standard_particle_count=np.array([2.0]),
        standard_particle_flow=np.array([2.0]),
        cnn_particle_count=np.array([0.0]),
        cnn_particle_flow=np.array([0.0]),
        water_record=None,
        records=[
            analysis_metrics.TraceRecord(
                filename=Path("trace.csv"),
                dilution=10.0,
                concentration=0.0,
                dx=1.0,
                signal=np.zeros(1, dtype=float),
                standard=analysis_metrics.PeakDetectionResult(
                    peaks=np.array([0, 1], dtype=int),
                    properties={},
                    peak_count=2,
                    detection_kwargs={},
                    amplitudes=np.array([0.0, 1.0], dtype=float),
                ),
                prediction=np.zeros(1, dtype=float),
                cnn=analysis_metrics.PeakDetectionResult(
                    peaks=np.asarray([], dtype=int),
                    properties={},
                    peak_count=0,
                    detection_kwargs={},
                    amplitudes=np.asarray([], dtype=float),
                ),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="reference_indices must contain at least one usable reference index",
    ):
        DilutionSeries.get_expected_particle_flow_for_result(
            result,
            index=0,
            reference_indices=[],
        )


def test_expected_particle_flow_requires_reference_other_than_target():
    result = PeakCountSeriesResult(
        dilution=np.array([10.0]),
        concentration=np.array([0.0]),
        standard_particle_count=np.array([2.0]),
        standard_particle_flow=np.array([2.0]),
        cnn_particle_count=np.array([0.0]),
        cnn_particle_flow=np.array([0.0]),
        water_record=None,
        records=[
            analysis_metrics.TraceRecord(
                filename=Path("trace.csv"),
                dilution=10.0,
                concentration=0.0,
                dx=1.0,
                signal=np.zeros(1, dtype=float),
                standard=analysis_metrics.PeakDetectionResult(
                    peaks=np.array([0, 1], dtype=int),
                    properties={},
                    peak_count=2,
                    detection_kwargs={},
                    amplitudes=np.array([0.0, 1.0], dtype=float),
                ),
                prediction=np.zeros(1, dtype=float),
                cnn=analysis_metrics.PeakDetectionResult(
                    peaks=np.asarray([], dtype=int),
                    properties={},
                    peak_count=0,
                    detection_kwargs={},
                    amplitudes=np.asarray([], dtype=float),
                ),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="reference_indices must contain at least one usable reference index",
    ):
        DilutionSeries.get_expected_particle_flow_for_result(
            result,
            index=0,
            reference_indices=[0, 0],
        )


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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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

    figure = series.poisson.plot.expected_histogram(
        index=1,
        base_index=0,
        detector="standard",
        x_axis="time",
        label="standard",
    )

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2


def test_plot_inter_arrival_histogram_can_limit_xlim_by_quantile():
    event_metrics = analysis_metrics.EventArrivalDistribution(
        label="standard",
        number_of_events=5,
        observation_start=0.0,
        observation_end=14.0,
        observation_duration=14.0,
        lambda_hat=0.5,
        number_of_inter_arrival_times=4,
        mean_inter_arrival_time=3.5,
        standard_deviation_inter_arrival_time=4.123105625617661,
        coefficient_of_variation_inter_arrival_time=1.178030178747903,
        ks_exponential_statistic=np.nan,
        ks_exponential_p_value=np.nan,
        ks_rescaled_uniform_statistic=np.nan,
        ks_rescaled_uniform_p_value=np.nan,
        number_of_count_bins=4,
        count_bin_width=3.5,
        mean_count_per_bin=1.25,
        variance_count_per_bin=0.25,
        fano_factor_count_per_bin=0.2,
        chi2_count_statistic=np.nan,
        chi2_count_degrees_of_freedom=0,
        chi2_count_p_value=np.nan,
        event_times=np.array([1.0, 2.0, 4.0, 8.0, 14.0]),
        inter_arrival_times=np.array([1.0, 2.0, 4.0, 6.0]),
        counts_per_bin=np.array([1, 1, 1, 2]),
    )

    figure = event_metrics.plot.histogram(xlim_quantile=0.5)

    assert figure.axes[0].get_xlim() == pytest.approx((0.0, 3.0))


def test_peak_histograms_can_limit_xlim_by_quantile():
    amplitude_metrics = analysis_metrics.PeakAmplitudeDistribution(
        label="standard",
        number_of_peaks=4,
        mean_amplitude=3.25,
        median_amplitude=2.5,
        minimum_amplitude=1.0,
        maximum_amplitude=8.0,
        standard_deviation_amplitude=3.095695936834452,
        coefficient_of_variation_amplitude=0.9525218267182929,
        skewness_amplitude=np.nan,
        kurtosis_amplitude=np.nan,
        fitted_normal_mean=3.25,
        fitted_normal_standard_deviation=3.095695936834452,
        ks_normal_statistic=np.nan,
        ks_normal_p_value=np.nan,
        amplitudes=np.array([1.0, 2.0, 2.0, 8.0]),
    )

    amplitude_figure = amplitude_metrics.plot.histogram(xlim_quantile=0.5)

    assert amplitude_figure.axes[0].get_xlim() == pytest.approx((0.0, 2.0))

    width_metrics = analysis_metrics.PeakWidthDistribution(
        label="standard",
        x_axis="time",
        width_unit_label="Seconds",
        number_of_peaks=4,
        mean_width=3.5,
        median_width=3.0,
        minimum_width=1.0,
        maximum_width=8.0,
        standard_deviation_width=2.886751345948129,
        coefficient_of_variation_width=0.8247860988423226,
        skewness_width=np.nan,
        kurtosis_width=np.nan,
        fitted_lognormal_shape=0.4,
        fitted_lognormal_loc=0.0,
        fitted_lognormal_scale=3.0,
        ks_lognormal_statistic=np.nan,
        ks_lognormal_p_value=np.nan,
        widths=np.array([1.0, 2.0, 4.0, 7.0]),
    )

    width_figure = width_metrics.plot.histogram(xlim_quantile=0.5)

    assert width_figure.axes[0].get_xlim() == pytest.approx((0.0, 3.0))


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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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

    figure = series.plot.measured_vs_expected_particle_flows(base_index=0)

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 3
    ideal_line = figure.axes[0].lines[2]
    assert ideal_line.get_xdata().tolist() == ideal_line.get_ydata().tolist()


def test_plot_measured_particle_flows_uses_dilution_axis(monkeypatch, tmp_path):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = PeakCountSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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

    figure = series.plot.measured_particle_flows(x_axis="dilution")

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2
    assert figure.axes[0].get_xlabel() == "Dilution"
    assert figure.axes[0].get_ylabel() == "Measured particle flow"


def test_dilution_series_poisson_accessor_wraps_arrival_api(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    metrics = series.poisson.diagnose(index=0, detector="standard", x_axis="time")
    figure = series.poisson.plot.expected_histogram(
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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    amplitude_metrics = series.amplitude.diagnose(index=0, detector="standard")
    amplitude_figure = series.amplitude.plot.histogram(
        index=0, detector="standard", label="standard"
    )
    width_metrics = series.width.diagnose(index=0, detector="standard", x_axis="time")
    width_figure = series.width.plot.histogram(
        index=0, detector="standard", x_axis="time", label="standard"
    )

    assert set(amplitude_metrics) == {"standard"}
    assert amplitude_metrics["standard"].number_of_peaks == 2
    assert len(amplitude_figure.axes) == 1
    assert set(width_metrics) == {"standard"}
    assert width_metrics["standard"].number_of_peaks == 2
    assert len(width_figure.axes) == 1


def test_dilution_series_records_expose_trace_plot_methods(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()
    record = series.get_record(index=0)

    standard_figure = record.plot_standard_detection()
    wavenet_figure = record.plot_wavenet_detection()

    assert len(standard_figure.axes) == 1
    assert len(wavenet_figure.axes) == 1


def test_dilution_series_plot_accessor_exposes_series_level_plots(
    monkeypatch, tmp_path
):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    particle_flow_figure = series.plot.particle_flows()
    measured_flow_figure = series.plot.measured_particle_flows(x_axis="dilution")

    assert len(particle_flow_figure.axes) == 1
    assert len(measured_flow_figure.axes) == 1


def test_trace_record_wavenet_detection_histogram_adds_inset_and_throughput(
    monkeypatch, tmp_path
):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        if filename == filenames[0]:
            return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0]), 0.25
        return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 3.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()
    record = series.get_record(index=0)
    expected_particle_flow = series.get_expected_particle_flow(index=0)

    figure = record.plot_wavenet_detection_with_histogram(
        x_axis="time",
        bins=np.linspace(0.0, 3.0, 10),
        show_prediction=False,
        show_cnn_prediction_peaks=False,
        show_cnn_signal_peaks=False,
        show_legend=False,
        show_cnn_recovered_signal_peaks=True,
        show_cnn_threshold=False,
        show_cnn_reconstructed_trace=False,
        expected_particle_flow=expected_particle_flow,
        show_throughput=True,
        show_inset=True,
        inset_xlim=(0.0, 0.5),
        inset_ylim=(0.0, 3.0),
        histogram_reference_lines=[0.15],
        histogram_xlim=(0.0, 3.0),
        histogram_title="Histogram",
    )

    assert len(figure.axes) == 3
    assert figure.axes[1].get_title() == "Histogram"
    assert list(figure.axes[1].lines[0].get_ydata()) == [0.15, 0.15]
    assert figure._suptitle is not None
    assert "Throughput:" in figure._suptitle.get_text()


def test_trace_record_standard_detection_histogram_adds_inset_and_throughput(
    monkeypatch, tmp_path
):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        if filename == filenames[0]:
            return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0]), 0.25
        return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 3.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()
    record, expected_particle_flow = series.get_record_with_expected_particle_flow(
        index=0
    )

    figure = record.plot_standard_detection_with_histogram(
        x_axis="time",
        bins=np.linspace(0.0, 3.0, 10),
        show_threshold=False,
        show_peaks=True,
        show_legend=False,
        expected_particle_flow=expected_particle_flow,
        show_throughput=True,
        show_inset=True,
        inset_xlim=(0.0, 0.5),
        inset_ylim=(0.0, 3.0),
        histogram_reference_lines=[0.15],
        histogram_xlim=(0.0, 3.0),
        histogram_title="Histogram",
    )

    assert len(figure.axes) == 3
    assert figure.axes[1].get_title() == "Histogram"
    assert list(figure.axes[1].lines[0].get_ydata()) == [0.15, 0.15]
    assert figure._suptitle is not None
    assert "Throughput:" in figure._suptitle.get_text()


def test_dilution_series_can_return_record_with_expected_particle_flow(
    monkeypatch, tmp_path
):
    filenames = [tmp_path / "replicate_1.csv", tmp_path / "replicate_2.csv"]
    for filename in filenames:
        filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[
            (filenames[0], 10.0),
            (filenames[1], 100.0),
        ],
    )

    def fake_load_signal(self, filename):
        if filename == filenames[0]:
            return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0]), 0.25
        return np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 3.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    record, expected_particle_flow = series.get_record_with_expected_particle_flow(
        index=0
    )

    assert record is series.get_record(index=0)
    assert expected_particle_flow == series.get_expected_particle_flow(index=0)


def test_dilution_series_can_run_standard_only(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    result = series.run_standard(std_trigger=HeightPeakTrigger(height=1.5))

    assert result.standard_particle_count.tolist() == [2.0]
    assert np.isnan(result.cnn_particle_count).all()
    assert result.records[0].standard.peaks.tolist() == [2, 5]
    assert result.records[0].cnn.peaks.tolist() == []


def test_dilution_series_can_run_cnn_only(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    result = series.run_cnn(cnn_trigger=HeightPeakTrigger(height=0.6))

    assert np.isnan(result.standard_particle_count).all()
    assert result.cnn_particle_count.tolist() == [2.0]
    assert result.records[0].standard.peaks.tolist() == []
    assert result.records[0].cnn.peaks.tolist() == [2, 5]


def test_dilution_series_run_requires_both_detector_configurations(tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        trace_files=[(filename, 10.0)],
    )

    with pytest.raises(
        ValueError,
        match=r"run\(\) requires both detector configurations",
    ):
        series.run()


def test_distribution_plot_accessors_expose_namespaced_plot_api(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    poisson_histogram = series.poisson.plot.histogram(
        index=0, detector="standard", x_axis="time", label="standard"
    )
    poisson_qq = series.poisson.plot.qq(
        index=0, detector="standard", x_axis="time", label="standard"
    )
    poisson_counts = series.poisson.plot.count_distribution(
        index=0, detector="standard", x_axis="time", label="standard"
    )
    amplitude_histogram = series.amplitude.plot.histogram(
        index=0, detector="standard", label="standard"
    )
    amplitude_qq = series.amplitude.plot.qq(
        index=0, detector="standard", label="standard"
    )
    amplitude_ecdf = series.amplitude.plot.ecdf(
        index=0, detector="standard", label="standard"
    )
    width_histogram = series.width.plot.histogram(
        index=0, detector="standard", x_axis="time", label="standard"
    )
    width_qq = series.width.plot.qq(
        index=0, detector="standard", x_axis="time", label="standard"
    )
    width_ecdf = series.width.plot.ecdf(
        index=0, detector="standard", x_axis="time", label="standard"
    )

    assert len(poisson_histogram.axes) == 1
    assert len(poisson_qq.axes) == 1
    assert len(poisson_counts.axes) == 1
    assert len(amplitude_histogram.axes) == 1
    assert len(amplitude_qq.axes) == 1
    assert len(amplitude_ecdf.axes) == 1
    assert len(width_histogram.axes) == 1
    assert len(width_qq.axes) == 1
    assert len(width_ecdf.axes) == 1


def test_trace_record_plot_methods_accept_analyzer_records():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    standard_figure = record.plot_standard_detection(figsize=(12.0, 4.0))
    wavenet_figure = record.plot_wavenet_detection(figsize=(12.0, 4.0))

    assert len(standard_figure.axes) == 1
    assert len(wavenet_figure.axes) == 1
    assert standard_figure.axes[0].get_xlabel() == "Sample index"
    assert wavenet_figure.axes[0].get_ylabel() == "Signal / prediction amplitude"


def test_trace_record_exposes_plot_methods():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    standard_figure = record.plot_standard_detection()
    wavenet_figure = record.plot_wavenet_detection()

    assert standard_figure.axes[0].get_title() == ""
    assert wavenet_figure.axes[0].get_title() == ""


def test_trace_record_legends_render_above_peak_markers():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    standard_figure = record.plot_standard_detection(show_legend=True)
    wavenet_figure = record.plot_wavenet_detection(show_legend=True)

    standard_legend = standard_figure.axes[0].get_legend()
    wavenet_legend = wavenet_figure.axes[0].get_legend()

    assert standard_legend is not None
    assert wavenet_legend is not None
    assert (
        standard_legend.get_zorder()
        > record.plot_standard_detection(show_legend=False)
        .axes[0]
        .collections[0]
        .get_zorder()
    )
    assert (
        wavenet_legend.get_zorder()
        > record.plot_wavenet_detection(show_legend=False)
        .axes[0]
        .collections[0]
        .get_zorder()
    )


def test_trace_record_can_show_cnn_recovered_signal_peaks():
    record = analysis_metrics.TraceRecord(
        filename="trace.csv",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.25,
        signal=np.array([[0.0, 1.0, 0.5, 0.0]]),
        standard=analysis_metrics.PeakDetectionResult(
            peaks=np.array([1]),
            properties={},
            peak_count=1,
            detection_kwargs={"height": 0.8},
            threshold=0.8,
        ),
        prediction=np.array([0.0, 0.2, 1.0, 0.1]),
        cnn=analysis_metrics.PeakDetectionResult(
            peaks=np.array([2]),
            properties={},
            peak_count=1,
            detection_kwargs={"height": 0.6},
            threshold=0.6,
            amplitudes=np.array([3.5]),
        ),
    )

    figure = record.plot_wavenet_detection(
        show_signal=False,
        show_prediction=False,
        show_cnn_recovered_signal_peaks=True,
        show_cnn_prediction_peaks=False,
        show_cnn_threshold=False,
        show_legend=False,
    )

    offsets = figure.axes[0].collections[0].get_offsets()

    assert offsets.shape == (1, 2)
    assert offsets[0, 0] == pytest.approx(2.0)
    assert offsets[0, 1] == pytest.approx(3.5)
    assert figure.axes[0].collections[0].get_label() == (
        "CNN recovered amplitudes on signal (1)"
    )


def test_trace_record_can_show_cnn_reconstructed_trace():
    record = analysis_metrics.TraceRecord(
        filename="trace.csv",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.25,
        signal=np.array([[0.0, 1.0, 0.5, 0.0]]),
        standard=analysis_metrics.PeakDetectionResult(
            peaks=np.array([1]),
            properties={},
            peak_count=1,
            detection_kwargs={"height": 0.8},
            threshold=0.8,
        ),
        prediction=np.array([0.0, 0.2, 1.0, 0.1]),
        cnn=analysis_metrics.PeakDetectionResult(
            peaks=np.array([2]),
            properties={"recovered_sigma_samples": 1.0},
            peak_count=1,
            detection_kwargs={"height": 0.6},
            threshold=0.6,
            amplitudes=np.array([3.5]),
        ),
    )

    figure = record.plot_wavenet_detection(
        show_signal=False,
        show_prediction=False,
        show_cnn_reconstructed_trace=True,
        show_cnn_prediction_peaks=False,
        show_cnn_threshold=False,
        show_legend=False,
    )

    line = figure.axes[0].lines[0]
    expected = 3.5 * np.exp(-0.5 * ((np.arange(4, dtype=float) - 2.0) / 1.0) ** 2)

    assert line.get_label() == "CNN reconstructed trace"
    assert np.allclose(line.get_ydata(), expected)


def test_trace_record_can_show_cnn_reconstructed_trace_with_baseline():
    record = analysis_metrics.TraceRecord(
        filename="trace.csv",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.25,
        signal=np.array([[0.5, 1.0, 4.0, 1.0]]),
        standard=analysis_metrics.PeakDetectionResult(
            peaks=np.array([2]),
            properties={},
            peak_count=1,
            detection_kwargs={"height": 0.8},
            threshold=0.8,
        ),
        prediction=np.array([0.0, 0.2, 1.0, 0.1]),
        cnn=analysis_metrics.PeakDetectionResult(
            peaks=np.array([2]),
            properties={"recovered_sigma_samples": 1.0, "recovered_baseline": 0.5},
            peak_count=1,
            detection_kwargs={"height": 0.6},
            threshold=0.6,
            amplitudes=np.array([3.5]),
        ),
    )

    figure = record.plot_wavenet_detection(
        show_signal=False,
        show_prediction=False,
        show_cnn_recovered_signal_peaks=True,
        show_cnn_reconstructed_trace=True,
        show_cnn_prediction_peaks=False,
        show_cnn_threshold=False,
        show_legend=False,
    )

    scatter_offsets = figure.axes[0].collections[0].get_offsets()
    line = figure.axes[0].lines[0]
    expected = 0.5 + 3.5 * np.exp(-0.5 * ((np.arange(4, dtype=float) - 2.0) / 1.0) ** 2)

    assert scatter_offsets[0, 1] == pytest.approx(4.0)
    assert np.allclose(line.get_ydata(), expected)


def test_accessor_single_axis_plots_accept_existing_axes(monkeypatch, tmp_path):
    filename = tmp_path / "replicate_1.csv"
    filename.write_text("placeholder")

    series = DilutionSeries(
        folder=tmp_path,
        wavenet=DummyWaveNet(),
        initial_concentration=100.0,
        nrows=1,
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        trace_files=[(filename, 10.0)],
    )

    def fake_load_signal(self, filename):
        return np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 0.25

    monkeypatch.setattr(DilutionSeries, "_load_signal", fake_load_signal)

    series.run()

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.0, 4.0))
    left_axis, right_axis = axes

    amplitude_figure = series.amplitude.plot.qq(
        index=0, detector="standard", label="standard", ax=left_axis
    )
    trace_figure = series.get_record(index=0).plot_wavenet_detection(ax=right_axis)

    assert amplitude_figure is figure
    assert trace_figure is figure
    assert len(figure.axes) == 2


def test_analyzer_accepts_peak_trigger_instances():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        cnn_low_pass=40_000_000,
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    assert record.standard.peaks.tolist() == [3]
    assert record.cnn.peaks.tolist() == [3]
    assert analyzer.cnn_low_pass == 40_000_000
    assert isinstance(analyzer.std_trigger, BasePeakTrigger)


def test_analyzer_accepts_explicit_cnn_low_pass():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        cnn_low_pass=40_000_000,
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 0.4, 2.5, 0.2, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.25, filename="trace.csv")

    assert record.cnn.peaks.tolist() == [3]
    assert analyzer.cnn_low_pass == 40_000_000


def test_sigma_trigger_resolves_hysteresis_in_sigma_units():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=SigmaPeakTrigger(sigma=1.0, hysteresis=1.0),
        cnn_trigger=HeightPeakTrigger(height=0.6),
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
        HeightPeakTrigger(height=0.15, hysteresis=0.2)


def test_analyzer_rejects_hysteresis_above_sigma_resolved_threshold():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=SigmaPeakTrigger(sigma=1.0, hysteresis=10.0),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    with pytest.raises(ValueError, match="resolved detection threshold"):
        analyzer.detect_standard_peaks(np.array([0.0, 0.1, 0.2, 0.1, 0.0]))
