import numpy as np
import pytest
from matplotlib.figure import Figure

from DeepPeak.analysis import PulseShapeAnalyzer


def _build_asymmetric_pulse_signal() -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(400, dtype=float) * 0.1
    signal = np.zeros_like(time)

    pulse_template = np.array(
        [0.0, 0.08, 0.35, 0.9, 1.4, 1.05, 0.78, 0.55, 0.38, 0.24, 0.12, 0.04, 0.0],
        dtype=float,
    )

    signal[84:97] += pulse_template
    signal[224:237] += 0.85 * pulse_template
    signal += 0.02
    return time, signal


def test_pulse_shape_analyzer_extracts_windows_and_statistics():
    time, signal = _build_asymmetric_pulse_signal()

    analyzer = (
        PulseShapeAnalyzer(signal=signal, time=time)
        .detect_peaks(height=0.2, distance=40)
        .extract_windows(left_samples=8, right_samples=12)
    )

    statistics = analyzer.shape_statistics()

    assert len(statistics) == 2
    assert np.all(statistics["peak_height"] > 0.8)
    assert np.all(statistics["asymmetry_ratio"] > 1.0)
    assert np.all(statistics["centroid_shift"] > 0.0)
    assert np.all(np.isfinite(statistics["gaussian_rmse"]))
    assert np.all(statistics["fwhm"] > 0.0)


def test_pulse_shape_analyzer_peak_selection_and_summary_are_intuitive():
    time, signal = _build_asymmetric_pulse_signal()

    analyzer = PulseShapeAnalyzer(signal=signal, time=time).detect_peaks(
        height=0.2, distance=40
    )
    assert analyzer.peak_indices.size == 2

    analyzer.select_peaks(min_amplitude=1.25).extract_windows(
        left_samples=8, right_samples=12
    )

    assert analyzer.peak_indices.size == 1
    assert analyzer.pulse_windows.shape == (1, 21)
    assert analyzer.normalized_windows().shape == (1, 21)
    assert analyzer.mean_pulse().shape == (21,)
    assert analyzer.median_pulse(normalize=False).shape == (21,)

    summary = analyzer.shape_summary()
    assert "mean" in summary.columns
    assert "peak_height" in summary.index


def test_pulse_shape_analyzer_plot_methods_return_figures():
    time, signal = _build_asymmetric_pulse_signal()

    analyzer = (
        PulseShapeAnalyzer(signal=signal, time=time)
        .detect_peaks(height=0.2, distance=40)
        .extract_windows(left_samples=8, right_samples=12)
    )

    overlay = analyzer.plot_pulses(
        normalize=True, show_mean=True, show_gaussian_reference=True
    )
    scatter = analyzer.plot_amplitude_vs_width()

    assert isinstance(overlay, Figure)
    assert isinstance(scatter, Figure)


def test_pulse_shape_analyzer_uses_sample_units_by_default():
    _, signal = _build_asymmetric_pulse_signal()

    analyzer = (
        PulseShapeAnalyzer(signal=signal)
        .detect_peaks(height=0.2, distance=40)
        .extract_windows(left_samples=8, right_samples=12)
    )

    statistics = analyzer.shape_statistics()

    assert analyzer.unit_label == "sample"
    assert analyzer.local_time[0] == pytest.approx(-8.0)
    assert np.allclose(analyzer.peak_widths, analyzer.peak_widths_samples)
    assert np.allclose(statistics["peak_width"], statistics["peak_width_samples"])


def test_pulse_shape_analyzer_extracts_kernel():
    _, signal = _build_asymmetric_pulse_signal()

    analyzer = (
        PulseShapeAnalyzer(signal=signal)
        .detect_peaks(height=0.2, distance=40)
        .extract_windows(left_samples=8, right_samples=12)
    )

    kernel = analyzer.extract_kernel(aggregation="median", noise_floor=0.05)

    assert kernel.ndim == 1
    assert kernel.size > 0
    assert kernel.size <= 21
    assert float(np.max(kernel)) == pytest.approx(1.0)
    assert float(np.min(kernel)) >= 0.05 - 1e-9

    kernel_mean = analyzer.extract_kernel(aggregation="mean", noise_floor=0.0)
    assert kernel_mean.size == 21


def test_extract_kernel_library_can_recenter_to_maximum():
    base_kernel = np.array([0.0, 0.1, 0.6, 1.0, 0.4, 0.1, 0.0], dtype=float)
    shifted_left = np.roll(base_kernel, -2)
    shifted_right = np.roll(base_kernel, 1)

    analyzer = PulseShapeAnalyzer(signal=np.zeros(32, dtype=float))
    analyzer._pulse_windows = np.vstack([shifted_left, shifted_right])
    analyzer._local_time = np.arange(base_kernel.size, dtype=float) - 3.0
    analyzer._window_peak_indices = np.array([5, 12], dtype=int)
    analyzer._window_peak_widths_samples = np.array([3.0, 3.0], dtype=float)
    analyzer._window_baselines = np.array([0.0, 0.0], dtype=float)

    library = analyzer.extract_kernel_library(
        smooth_sigma=0.0,
        baseline_samples=1,
        max_kernels=2,
        reject_saturated=False,
        random_selection=False,
        recenter="max",
        taper_fraction=0.0,
    )

    assert library.shape == (2, base_kernel.size)
    assert np.all(np.argmax(library, axis=1) == base_kernel.size // 2)


def test_extract_kernel_library_rejects_unknown_recenter_mode():
    analyzer = PulseShapeAnalyzer(signal=np.zeros(16, dtype=float))
    analyzer._pulse_windows = np.array([[0.0, 1.0, 0.0]], dtype=float)
    analyzer._local_time = np.array([-1.0, 0.0, 1.0], dtype=float)
    analyzer._window_peak_indices = np.array([4], dtype=int)
    analyzer._window_peak_widths_samples = np.array([1.0], dtype=float)
    analyzer._window_baselines = np.array([0.0], dtype=float)

    with pytest.raises(ValueError, match="recenter"):
        analyzer.extract_kernel_library(
            smooth_sigma=0.0,
            baseline_samples=1,
            max_kernels=1,
            reject_saturated=False,
            recenter="foobar",
        )


def test_pulse_shape_analyzer_uses_sampling_rate_for_second_units():
    _, signal = _build_asymmetric_pulse_signal()

    analyzer = (
        PulseShapeAnalyzer(signal=signal, sampling_rate=10.0)
        .detect_peaks(height=0.2, distance=40)
        .extract_windows(left_samples=8, right_samples=12)
    )

    statistics = analyzer.shape_statistics()

    assert analyzer.unit_label == "s"
    assert analyzer.local_time[0] == pytest.approx(-0.8)
    assert np.allclose(analyzer.peak_widths, analyzer.peak_widths_samples / 10.0)
    assert np.allclose(
        statistics["peak_width"], statistics["peak_width_samples"] / 10.0
    )
