import numpy as np

from DeepPeak.analysis import (
    PeakAmplitudeDistributionMetrics,
    WaveNetTraceAnalyzer,
    compute_peak_amplitude_distribution_metrics,
    plot_peak_amplitude_ecdf,
    plot_peak_amplitude_histogram,
    plot_peak_amplitude_qq,
)


class DummyWaveNet:
    sequence_length = 8

    def predict(self, signal):
        return np.asarray(signal, dtype=float)


def test_compute_peak_amplitude_distribution_metrics_returns_expected_statistics():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
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
    assert isinstance(standard_metrics, PeakAmplitudeDistributionMetrics)
    assert standard_metrics.number_of_peaks == 3
    assert standard_metrics.amplitudes.tolist() == [2.0, 3.0, 4.0]
    assert standard_metrics.mean_amplitude == 3.0
    assert standard_metrics.minimum_amplitude == 2.0
    assert standard_metrics.maximum_amplitude == 4.0


def test_cnn_amplitude_metrics_use_trace_amplitudes_at_detected_positions():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5},
        cnn_kwargs={"height": 0.6},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.1)

    metrics = compute_peak_amplitude_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="cnn",
    )

    cnn_metrics = metrics["cnn"]
    assert isinstance(cnn_metrics, PeakAmplitudeDistributionMetrics)
    assert cnn_metrics.amplitudes.tolist() == [2.0, 3.0]
    assert cnn_metrics.maximum_amplitude == 3.0


def test_amplitude_analysis_plot_helpers_return_figures():
    metrics = {
        "standard": PeakAmplitudeDistributionMetrics(
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

    assert plot_peak_amplitude_histogram(metrics).__class__.__name__ == "Figure"
    assert plot_peak_amplitude_qq(metrics).__class__.__name__ == "Figure"
    assert plot_peak_amplitude_ecdf(metrics).__class__.__name__ == "Figure"
