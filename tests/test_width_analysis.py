import numpy as np

from DeepPeak.analysis import (
    PeakWidthDistributionMetrics,
    WaveNetTraceAnalyzer,
    compute_peak_width_distribution_metrics,
    plot_peak_width_ecdf,
    plot_peak_width_histogram,
    plot_peak_width_qq,
)


class DummyWaveNet:
    sequence_length = 8

    def predict(self, signal):
        return np.asarray(signal, dtype=float)


def test_compute_peak_width_distribution_metrics_returns_expected_statistics():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_kwargs={"height": 1.5, "required_samples_below_hysteresis": 2},
        cnn_kwargs={"height": 0.6, "required_samples_below_hysteresis": 2},
        signal_normalization="minmax",
    )

    signal = np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    record = analyzer.analyze_processed_signal(signal, dx=0.5)

    metrics = compute_peak_width_distribution_metrics(
        series_or_result=type("Result", (), {"records": [record]})(),
        index=0,
        detector="standard",
        x_axis="sample",
    )

    standard_metrics = metrics["standard"]
    assert isinstance(standard_metrics, PeakWidthDistributionMetrics)
    assert standard_metrics.number_of_peaks == 2
    assert standard_metrics.widths.tolist() == [3.0, 4.0]
    assert standard_metrics.mean_width == 3.5
    assert standard_metrics.minimum_width == 3.0
    assert standard_metrics.maximum_width == 4.0


def test_width_analysis_plot_helpers_return_figures():
    metrics = {
        "standard": PeakWidthDistributionMetrics(
            label="standard",
            x_axis="sample",
            width_unit_label="Samples",
            number_of_peaks=3,
            mean_width=3.0,
            median_width=3.0,
            minimum_width=2.0,
            maximum_width=4.0,
            standard_deviation_width=1.0,
            coefficient_of_variation_width=1.0 / 3.0,
            skewness_width=0.0,
            kurtosis_width=np.nan,
            fitted_lognormal_shape=0.2,
            fitted_lognormal_loc=0.0,
            fitted_lognormal_scale=3.0,
            ks_lognormal_statistic=np.nan,
            ks_lognormal_p_value=np.nan,
            widths=np.array([2.0, 3.0, 4.0]),
        )
    }

    assert plot_peak_width_histogram(metrics).__class__.__name__ == "Figure"
    assert plot_peak_width_qq(metrics).__class__.__name__ == "Figure"
    assert plot_peak_width_ecdf(metrics).__class__.__name__ == "Figure"
