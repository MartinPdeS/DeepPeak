import numpy as np
import pytest
from matplotlib.figure import Figure

from DeepPeak.analysis import (
    DilutionSeries,
    HeightPeakTrigger,
    WaveNetTraceAnalyzer,
    compute_peak_amplitude_distribution_metrics,
    metrics as analysis_metrics,
)


class DummyWaveNet:
    sequence_length = 8

    def predict(self, signal):
        return np.asarray(signal, dtype=float)


def test_compute_peak_amplitude_distribution_metrics_returns_expected_statistics():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="standard",
    )

    standard_metrics = metrics["standard"]
    assert isinstance(standard_metrics, analysis_metrics.PeakAmplitudeDistribution)
    assert standard_metrics.number_of_peaks == 3
    assert standard_metrics.amplitudes.tolist() == [2.0, 3.0, 4.0]
    assert standard_metrics.mean_amplitude == 3.0
    assert standard_metrics.minimum_amplitude == 2.0
    assert standard_metrics.maximum_amplitude == 4.0


def test_cnn_amplitude_metrics_require_recovered_amplitudes():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=1.5),
        cnn_trigger=HeightPeakTrigger(height=0.6),
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    with pytest.raises(ValueError, match="recovered amplitudes are unavailable"):
        compute_peak_amplitude_distribution_metrics(
            series_or_result=type("Result", (), {"records": [record]})(),
            index=0,
            detector="cnn",
        )


def test_cnn_amplitude_metrics_prefer_recovered_amplitudes_for_overlapping_peaks():
    class WideDummyWaveNet:
        sequence_length = 32

        def predict(self, signal):
            prediction = np.zeros_like(np.asarray(signal, dtype=float))
            prediction[..., 10] = 1.0
            prediction[..., 15] = 1.0
            return prediction

    analyzer = WaveNetTraceAnalyzer(
        wavenet=WideDummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=0.4),
        cnn_trigger=HeightPeakTrigger(height=0.5),
        signal_normalization="minmax",
        cnn_amplitude_sigma_samples=2.0,
    )

    x_values = np.arange(32, dtype=float)
    signal = 2.0 * np.exp(-0.5 * ((x_values - 10.0) / 2.0) ** 2)
    signal += 3.0 * np.exp(-0.5 * ((x_values - 15.0) / 2.0) ** 2)
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="cnn",
    )

    cnn_metrics = metrics["cnn"]
    raw_samples = np.asarray(record.signal, dtype=float).ravel()[record.cnn.peaks]

    assert isinstance(cnn_metrics, analysis_metrics.PeakAmplitudeDistribution)
    assert record.cnn.peak_count == 2
    assert np.allclose(cnn_metrics.amplitudes, np.array([2.0, 3.0]), atol=0.2)
    assert not np.allclose(cnn_metrics.amplitudes, raw_samples, atol=0.05)


def test_cnn_amplitude_recovery_matches_analytical_solution_for_three_overlapping_peaks():
    class WideDummyWaveNet:
        sequence_length = 48

        def predict(self, signal):
            prediction = np.zeros_like(np.asarray(signal, dtype=float))
            prediction[..., 10] = 1.0
            prediction[..., 15] = 1.0
            prediction[..., 19] = 1.0
            return prediction

    analyzer = WaveNetTraceAnalyzer(
        wavenet=WideDummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=0.4),
        cnn_trigger=HeightPeakTrigger(height=0.5),
        signal_normalization="minmax",
        cnn_amplitude_sigma_samples=2.0,
    )

    x_values = np.arange(48, dtype=float)
    signal = 2.0 * np.exp(-0.5 * ((x_values - 10.0) / 2.0) ** 2)
    signal += 3.0 * np.exp(-0.5 * ((x_values - 15.0) / 2.0) ** 2)
    signal += 1.5 * np.exp(-0.5 * ((x_values - 19.0) / 2.0) ** 2)
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="cnn",
    )

    cnn_metrics = metrics["cnn"]

    assert record.cnn.peak_count == 3
    assert np.allclose(cnn_metrics.amplitudes, np.array([2.0, 3.0, 1.5]), atol=0.2)


def test_cnn_amplitude_recovery_can_subtract_constant_baseline():
    class WideDummyWaveNet:
        sequence_length = 48

        def predict(self, signal):
            prediction = np.zeros_like(np.asarray(signal, dtype=float))
            prediction[..., 10] = 1.0
            prediction[..., 15] = 1.0
            return prediction

    analyzer = WaveNetTraceAnalyzer(
        wavenet=WideDummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=0.4),
        cnn_trigger=HeightPeakTrigger(height=0.5),
        signal_normalization="minmax",
        cnn_amplitude_sigma_samples=2.0,
        cnn_amplitude_baseline=0.75,
    )

    x_values = np.arange(48, dtype=float)
    signal = 0.75 * np.ones_like(x_values)
    signal += 2.0 * np.exp(-0.5 * ((x_values - 10.0) / 2.0) ** 2)
    signal += 3.0 * np.exp(-0.5 * ((x_values - 15.0) / 2.0) ** 2)
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="cnn",
    )

    assert np.allclose(metrics["cnn"].amplitudes, np.array([2.0, 3.0]), atol=0.2)
    assert record.cnn.properties["recovered_baseline"] == pytest.approx(0.75)


def test_standard_amplitude_metrics_ignore_recovered_amplitudes_field():
    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=float)
    standard_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 1.5},
        threshold=1.5,
        amplitudes=np.array([20.0, 30.0], dtype=float),
    )
    cnn_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 0.6},
        threshold=0.6,
        amplitudes=np.array([7.0, 8.0], dtype=float),
    )
    record = analysis_metrics.TraceRecord(
        filename="<memory>",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.1,
        signal=signal,
        standard=standard_detection,
        prediction=signal,
        cnn=cnn_detection,
    )

    standard_metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="standard",
    )["standard"]
    cnn_metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="cnn",
    )["cnn"]

    assert standard_metrics.amplitudes.tolist() == [2.0, 3.0]
    assert cnn_metrics.amplitudes.tolist() == [7.0, 8.0]


def test_amplitude_accessor_can_compare_standard_and_cnn_sources():
    class WideDummyWaveNet:
        sequence_length = 32

        def predict(self, signal):
            prediction = np.zeros_like(np.asarray(signal, dtype=float))
            prediction[..., 10] = 1.0
            prediction[..., 15] = 1.0
            return prediction

    analyzer = WaveNetTraceAnalyzer(
        wavenet=WideDummyWaveNet(),
        std_trigger=HeightPeakTrigger(height=0.4),
        cnn_trigger=HeightPeakTrigger(height=0.5),
        signal_normalization="minmax",
        cnn_amplitude_sigma_samples=2.0,
    )

    x_values = np.arange(32, dtype=float)
    signal = 2.0 * np.exp(-0.5 * ((x_values - 10.0) / 2.0) ** 2)
    signal += 3.0 * np.exp(-0.5 * ((x_values - 15.0) / 2.0) ** 2)
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    class DummySeries:
        def __init__(self, record):
            self._record = record

        def get_record(self, index):
            assert index == 0
            return self._record

    accessor = DilutionSeries.AmplitudeAnalysisAccessor(DummySeries(record))
    comparison = accessor.compare_sources(index=0)

    assert comparison["standard_amplitudes"].tolist() == pytest.approx(
        np.asarray(record.signal, dtype=float).ravel()[record.standard.peaks]
    )
    assert comparison["cnn_raw_signal_amplitudes"].tolist() == pytest.approx(
        np.asarray(record.signal, dtype=float).ravel()[record.cnn.peaks]
    )
    assert comparison["cnn_recovered_amplitudes"].tolist() == pytest.approx(
        [2.0, 3.0], abs=0.2
    )


def test_plot_standard_detection_with_histogram_uses_standard_peak_amplitudes():
    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=float)
    standard_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 1.5},
        threshold=1.5,
        amplitudes=np.array([20.0, 30.0], dtype=float),
    )
    cnn_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([1], dtype=int),
        properties={},
        peak_count=1,
        detection_kwargs={"height": 0.6},
        threshold=0.6,
        amplitudes=np.array([7.0], dtype=float),
    )
    record = analysis_metrics.TraceRecord(
        filename="<memory>",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.1,
        signal=signal,
        standard=standard_detection,
        prediction=signal,
        cnn=cnn_detection,
    )

    figure = record.plot_standard_detection_with_histogram(
        show_legend=False,
        histogram_bins=2,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    histogram_axis = figure.axes[1]
    assert sum(patch.get_width() for patch in histogram_axis.patches) == pytest.approx(
        2.0
    )
    assert histogram_axis.get_xlabel() == "Peak count"
    figure.clf()


def test_plot_standard_detection_with_histogram_accepts_bins_alias():
    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=float)
    standard_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 1.5},
        threshold=1.5,
        amplitudes=np.array([20.0, 30.0], dtype=float),
    )
    cnn_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([1], dtype=int),
        properties={},
        peak_count=1,
        detection_kwargs={"height": 0.6},
        threshold=0.6,
        amplitudes=np.array([7.0], dtype=float),
    )
    record = analysis_metrics.TraceRecord(
        filename="<memory>",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.1,
        signal=signal,
        standard=standard_detection,
        prediction=signal,
        cnn=cnn_detection,
    )

    figure = record.plot_standard_detection_with_histogram(
        show_legend=False,
        bins=2,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes[1].patches) == 2
    figure.clf()


def test_plot_wavenet_detection_with_histogram_prefers_recovered_amplitudes():
    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=float)
    standard_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 1.5},
        threshold=1.5,
        amplitudes=np.array([20.0, 30.0], dtype=float),
    )
    cnn_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 0.6},
        threshold=0.6,
        amplitudes=np.array([7.0, 8.0], dtype=float),
    )
    record = analysis_metrics.TraceRecord(
        filename="<memory>",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.1,
        signal=signal,
        standard=standard_detection,
        prediction=signal,
        cnn=cnn_detection,
    )

    figure = record.plot_wavenet_detection_with_histogram(
        show_legend=False,
        histogram_bins=2,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    histogram_axis = figure.axes[1]
    histogram_patch_positions = sorted(
        round(patch.get_y() + patch.get_height() / 2.0, 6)
        for patch in histogram_axis.patches
        if patch.get_width() > 0.0
    )
    assert histogram_patch_positions == [7.25, 7.75]
    figure.clf()


def test_plot_wavenet_detection_with_histogram_restores_recovered_baseline():
    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=float)
    standard_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 1.5},
        threshold=1.5,
        amplitudes=np.array([20.0, 30.0], dtype=float),
    )
    cnn_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={"recovered_baseline": 0.5},
        peak_count=2,
        detection_kwargs={"height": 0.6},
        threshold=0.6,
        amplitudes=np.array([7.0, 8.0], dtype=float),
    )
    record = analysis_metrics.TraceRecord(
        filename="<memory>",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.1,
        signal=signal,
        standard=standard_detection,
        prediction=signal,
        cnn=cnn_detection,
    )

    figure = record.plot_wavenet_detection_with_histogram(
        show_legend=False,
        histogram_bins=2,
        show_prediction=False,
        show_cnn_prediction_peaks=False,
        show_cnn_signal_peaks=False,
        show_cnn_recovered_signal_peaks=True,
    )

    assert isinstance(figure, Figure)
    histogram_axis = figure.axes[1]
    histogram_patch_positions = sorted(
        round(patch.get_y() + patch.get_height() / 2.0, 6)
        for patch in histogram_axis.patches
        if patch.get_width() > 0.0
    )
    assert histogram_patch_positions == [7.75, 8.25]
    figure.clf()


def test_plot_wavenet_detection_with_histogram_falls_back_to_signal_amplitudes():
    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=float)
    standard_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 1.5},
        threshold=1.5,
        amplitudes=np.array([20.0, 30.0], dtype=float),
    )
    cnn_detection = analysis_metrics.PeakDetectionResult(
        peaks=np.array([2, 5], dtype=int),
        properties={},
        peak_count=2,
        detection_kwargs={"height": 0.6},
        threshold=0.6,
        amplitudes=None,
    )
    record = analysis_metrics.TraceRecord(
        filename="<memory>",
        dilution=np.nan,
        concentration=np.nan,
        dx=0.1,
        signal=signal,
        standard=standard_detection,
        prediction=signal,
        cnn=cnn_detection,
    )

    figure = record.plot_wavenet_detection_with_histogram(
        show_legend=False,
        histogram_bins=2,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    histogram_axis = figure.axes[1]
    histogram_patch_positions = sorted(
        round(patch.get_y() + patch.get_height() / 2.0, 6)
        for patch in histogram_axis.patches
        if patch.get_width() > 0.0
    )
    assert histogram_patch_positions == [2.25, 2.75]
    figure.clf()


def test_amplitude_analysis_plot_helpers_return_figures():
    metrics = {
        "standard": analysis_metrics.PeakAmplitudeDistribution(
            label="standard",
            number_of_peaks=3,
            mean_amplitude=3.0,
            median_amplitude=3.0,
            minimum_amplitude=2.0,
            maximum_amplitude=4.0,
            standard_deviation_amplitude=1.0,
            coefficient_of_variation_amplitude=1.0 / 3.0,
            skewness_amplitude=0.0,
            kurtosis_amplitude=np.nan,
            fitted_normal_mean=3.0,
            fitted_normal_standard_deviation=1.0,
            ks_normal_statistic=np.nan,
            ks_normal_p_value=np.nan,
            amplitudes=np.array([2.0, 3.0, 4.0]),
        )
    }

    assert metrics["standard"].plot.histogram().__class__.__name__ == "Figure"
    assert metrics["standard"].plot.qq().__class__.__name__ == "Figure"
    assert metrics["standard"].plot.ecdf().__class__.__name__ == "Figure"
