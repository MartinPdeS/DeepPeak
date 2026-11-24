import numpy as np
import pytest
from matplotlib.figure import Figure

from DeepPeak.analysis import NoiseAnalyzer


def _build_signal_with_noise_and_peaks() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    time = np.arange(500, dtype=float) * 0.1
    signal = 0.05 + rng.normal(0.0, 0.01, size=time.size)

    pulse_template = np.array(
        [0.0, 0.04, 0.12, 0.35, 0.60, 0.35, 0.12, 0.04, 0.0],
        dtype=float,
    )
    signal[96:105] += pulse_template
    signal[296:305] += 0.8 * pulse_template
    return time, signal


def test_noise_analyzer_masks_peak_neighborhoods():
    _, signal = _build_signal_with_noise_and_peaks()

    analyzer = NoiseAnalyzer(signal=signal).detect_noise(
        use_peaks=[100, 300],
        left_guard=8,
        right_guard=10,
    )

    mask = analyzer.noise_mask
    assert not np.any(mask[92:111])
    assert not np.any(mask[292:311])
    assert mask.sum() == signal.size - (19 + 19)


def test_noise_analyzer_statistics_and_summary_are_intuitive():
    time, signal = _build_signal_with_noise_and_peaks()

    analyzer = NoiseAnalyzer(signal=signal, time=time).detect_noise(
        height=0.15,
        distance=80,
        left_guard=10,
        right_guard=12,
    )

    statistics = analyzer.noise_statistics()
    summary = analyzer.summary()

    expected_keys = {
        "count",
        "mean",
        "median",
        "std",
        "robust_sigma",
        "skewness",
        "kurtosis",
        "min",
        "max",
    }
    assert expected_keys.issubset(statistics.index)
    assert statistics["count"] > 0
    assert statistics["mean"] == pytest.approx(0.05, abs=0.01)
    assert "mean" in summary.columns
    assert "robust_sigma" in summary.columns
    assert "noise" in summary.index


def test_noise_analyzer_plot_methods_return_figures():
    _, signal = _build_signal_with_noise_and_peaks()

    analyzer = NoiseAnalyzer(signal=signal).detect_noise(
        height=0.15,
        distance=80,
        left_guard=10,
        right_guard=12,
    )

    trace = analyzer.plot_trace()
    distribution = analyzer.plot_distribution()
    autocorrelation = analyzer.plot_autocorrelation(max_lag=40)
    psd = analyzer.plot_power_spectral_density(nperseg=64)

    assert isinstance(trace, Figure)
    assert isinstance(distribution, Figure)
    assert isinstance(autocorrelation, Figure)
    assert isinstance(psd, Figure)


def test_noise_analyzer_plot_trace_shows_threshold_when_detected():
    _, signal = _build_signal_with_noise_and_peaks()

    analyzer = NoiseAnalyzer(signal=signal).detect_noise(
        height=0.15,
        distance=80,
        left_guard=10,
        right_guard=12,
    )

    figure = analyzer.plot_trace()
    labels = [line.get_label() for line in figure.axes[0].lines]

    assert any("threshold" in label for label in labels)


def test_noise_analyzer_uses_sample_units_by_default():
    _, signal = _build_signal_with_noise_and_peaks()

    analyzer = NoiseAnalyzer(signal=signal).detect_noise(
        use_peaks=[100],
        left_guard=5,
        right_guard=5,
    )
    autocorrelation = analyzer.autocorrelation(max_lag=3)

    assert analyzer.unit_label == "sample"
    assert np.allclose(autocorrelation["lag"], [0.0, 1.0, 2.0, 3.0])


def test_noise_analyzer_uses_sampling_rate_for_second_units():
    _, signal = _build_signal_with_noise_and_peaks()

    analyzer = NoiseAnalyzer(signal=signal, sampling_rate=10.0).detect_noise(
        use_peaks=[100],
        left_guard=5,
        right_guard=5,
    )
    autocorrelation = analyzer.autocorrelation(max_lag=3)
    psd = analyzer.power_spectral_density(nperseg=64)

    assert analyzer.unit_label == "s"
    assert np.allclose(autocorrelation["lag"], [0.0, 0.1, 0.2, 0.3])
    assert psd["frequency"].iloc[-1] == pytest.approx(5.0)


def test_noise_analyzer_requires_detection_first():
    _, signal = _build_signal_with_noise_and_peaks()
    analyzer = NoiseAnalyzer(signal=signal)

    with pytest.raises(RuntimeError, match="detect_noise"):
        analyzer.noise_statistics()
