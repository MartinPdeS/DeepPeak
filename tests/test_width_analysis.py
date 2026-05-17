import numpy as np

from DeepPeak.analysis import (
    DilutionSeries,
    HeightPeakTrigger,
    WaveNetTraceAnalyzer,
    compute_peak_width_distribution_metrics,
    metrics as analysis_metrics,
)


class DummyWaveNet:
    sequence_length = 8

    def predict(self, signal):
        return np.asarray(signal, dtype=float)


def test_compute_peak_width_distribution_metrics_returns_expected_statistics():
    analyzer = WaveNetTraceAnalyzer(
        wavenet=DummyWaveNet(),
        std_trigger=HeightPeakTrigger(
            height=1.5,
            required_samples_below_hysteresis=2,
        ),
        cnn_trigger=HeightPeakTrigger(
            height=0.6,
            required_samples_below_hysteresis=2,
        ),
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
    assert isinstance(standard_metrics, analysis_metrics.PeakWidthDistribution)
    assert standard_metrics.number_of_peaks == 2
    assert standard_metrics.widths.tolist() == [3.0, 4.0]
    assert standard_metrics.mean_width == 3.5
    assert standard_metrics.minimum_width == 3.0
    assert standard_metrics.maximum_width == 4.0


def test_width_analysis_plot_helpers_return_figures():
    metrics = {
        "standard": analysis_metrics.PeakWidthDistribution(
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

    assert metrics["standard"].plot.histogram().__class__.__name__ == "Figure"
    assert metrics["standard"].plot.qq().__class__.__name__ == "Figure"
    assert metrics["standard"].plot.ecdf().__class__.__name__ == "Figure"
