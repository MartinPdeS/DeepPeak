from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from DeepPeak.dataset import DataSet
from DeepPeak.kernels import Gaussian, TwoLobeGaussian
from DeepPeak.peak_count import NegativeBinomialCount, PoissonCount
from DeepPeak.signal_generator import SignalGenerator


def _make_dataset(
    *,
    x_values: np.ndarray,
    positions: np.ndarray,
    amplitudes: np.ndarray,
    signals: np.ndarray | None = None,
) -> DataSet:
    if signals is None:
        signals = np.zeros((positions.shape[0], x_values.size), dtype=float)

    return DataSet(
        signals=np.asarray(signals, dtype=float),
        x_values=np.asarray(x_values, dtype=float),
        positions=np.asarray(positions, dtype=float),
        amplitudes=np.asarray(amplitudes, dtype=float),
    )


def _gaussian_reference(
    x_values: np.ndarray,
    center: float,
    amplitude: float,
    width: float,
) -> np.ndarray:
    return amplitude * np.exp(-4.0 * np.log(2.0) * ((x_values - center) / width) ** 2)


def test_reference_pulse_suppresses_peaks_that_are_too_close() -> None:
    x_values = np.arange(11, dtype=float)
    dataset = _make_dataset(
        x_values=x_values,
        positions=np.array([[2.0, 3.0, 8.0]]),
        amplitudes=np.array([[5.0, 4.0, 3.0]]),
    )

    reference = dataset.get_reference_pulse_trace(width=0.5, min_peak_distance=1.5)

    expected = _gaussian_reference(x_values, center=8.0, amplitude=3.0, width=0.5)
    np.testing.assert_allclose(reference[0], expected)


def test_reference_pulse_suppresses_small_peaks_with_absolute_threshold() -> None:
    x_values = np.arange(11, dtype=float)
    dataset = _make_dataset(
        x_values=x_values,
        positions=np.array([[2.0, 8.0]]),
        amplitudes=np.array([[5.0, 1.5]]),
    )

    reference = dataset.get_reference_pulse_trace(width=0.5, min_peak_amplitude=2.0)

    expected = _gaussian_reference(x_values, center=2.0, amplitude=5.0, width=0.5)
    np.testing.assert_allclose(reference[0], expected)


def test_reference_pulse_supports_relative_amplitude_threshold() -> None:
    x_values = np.arange(11, dtype=float)
    dataset = _make_dataset(
        x_values=x_values,
        positions=np.array([[2.0, 8.0]]),
        amplitudes=np.array([[5.0, 2.0]]),
        signals=np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]),
    )

    reference = dataset.get_reference_pulse_trace(
        width=0.5,
        min_peak_amplitude=0.5,
        amplitude_threshold_reference="sample_max_amplitude",
    )

    expected = _gaussian_reference(x_values, center=2.0, amplitude=5.0, width=0.5)
    np.testing.assert_allclose(reference[0], expected)


def test_reference_pulse_suppresses_peaks_with_excess_local_overlap() -> None:
    x_values = np.arange(21, dtype=float)
    dataset = _make_dataset(
        x_values=x_values,
        positions=np.array([[5.0, 5.5, 15.0]]),
        amplitudes=np.array([[5.0, 1.0, 2.0]]),
    )

    reference = dataset.get_reference_pulse_trace(width=2.0, max_peak_overlap=0.5)

    expected = _gaussian_reference(x_values, center=5.0, amplitude=5.0, width=2.0)
    expected += _gaussian_reference(x_values, center=15.0, amplitude=2.0, width=2.0)
    np.testing.assert_allclose(reference[0], expected)


def test_dataset_constructor_tracks_sequence_length_and_n_samples() -> None:
    dataset = DataSet(
        signals=np.zeros((3, 11), dtype=float),
        x_values=np.arange(11, dtype=float),
        positions=np.zeros((3, 2), dtype=float),
        amplitudes=np.ones((3, 2), dtype=float),
    )

    assert dataset.n_samples == 3
    assert dataset.sequence_length == 11


def test_dataset_shuffle_keeps_sample_aligned_arrays_in_sync() -> None:
    dataset = DataSet(
        signals=np.array(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            dtype=float,
        ),
        labels=np.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=float,
        ),
        positions=np.array(
            [[100.0], [200.0], [300.0]],
            dtype=float,
        ),
        amplitudes=np.array(
            [[1000.0], [2000.0], [3000.0]],
            dtype=float,
        ),
        x_values=np.arange(2, dtype=float),
    )

    shuffled = dataset.shuffle(seed=7)

    original_rows = {
        tuple(row)
        for row in np.column_stack(
            [
                dataset.signals[:, 0],
                dataset.labels[:, 0],
                dataset.positions[:, 0],
                dataset.amplitudes[:, 0],
            ]
        )
    }
    shuffled_rows = {
        tuple(row)
        for row in np.column_stack(
            [
                shuffled.signals[:, 0],
                shuffled.labels[:, 0],
                shuffled.positions[:, 0],
                shuffled.amplitudes[:, 0],
            ]
        )
    }

    assert shuffled is not dataset
    assert shuffled.sequence_length == dataset.sequence_length
    assert shuffled.n_samples == dataset.n_samples
    assert original_rows == shuffled_rows
    assert np.array_equal(dataset.x_values, shuffled.x_values)


def test_signal_generator_accepts_poisson_peak_count_spec() -> None:
    generator = SignalGenerator(sequence_length=64)
    kernel = Gaussian(
        amplitude=(1.0, 2.0),
        position=(5.0, 59.0),
        width=(1.0, 3.0),
    )

    dataset = generator.generate(
        n_samples=20,
        kernel=kernel,
        peak_count=PoissonCount(bounds=(1, 5), rate=(1.0, 3.0)),
        seed=7,
    )

    assert dataset.signals.shape == (20, 64)
    assert dataset.num_peaks.shape == (20,)
    assert np.all(dataset.num_peaks >= 1)
    assert np.all(dataset.num_peaks <= 5)


def test_signal_generator_accepts_negative_binomial_peak_count_spec() -> None:
    generator = SignalGenerator(sequence_length=64)
    kernel = Gaussian(
        amplitude=(1.0, 2.0),
        position=(5.0, 59.0),
        width=(1.0, 3.0),
    )

    dataset = generator.generate(
        n_samples=24,
        kernel=kernel,
        peak_count=NegativeBinomialCount(
            bounds=(0, 6),
            mean=(1.5, 4.0),
            dispersion=2.0,
        ),
        seed=11,
    )

    assert dataset.signals.shape == (24, 64)
    assert dataset.num_peaks.shape == (24,)
    assert np.all(dataset.num_peaks >= 0)
    assert np.all(dataset.num_peaks <= 6)


def test_two_lobe_gaussian_can_disable_or_enable_secondary_lobe() -> None:
    x_values = np.arange(0, 401, dtype=float)

    no_lobe_kernel = TwoLobeGaussian(
        amplitude=1.0,
        position=200.0,
        width=25.0,
        secondary_ratio=0.3,
        secondary_offset=90.0,
        secondary_width=20.0,
        secondary_presence=0.0,
    )
    no_lobe_components = no_lobe_kernel.evaluate(
        x_values=x_values,
        n_samples=1,
        n_peaks=(1, 1),
    )

    with_lobe_kernel = TwoLobeGaussian(
        amplitude=1.0,
        position=200.0,
        width=25.0,
        secondary_ratio=0.3,
        secondary_offset=90.0,
        secondary_width=20.0,
        secondary_presence=1.0,
    )
    with_lobe_components = with_lobe_kernel.evaluate(
        x_values=x_values,
        n_samples=1,
        n_peaks=(1, 1),
    )

    no_lobe_signal = np.nansum(no_lobe_components, axis=1)[0]
    with_lobe_signal = np.nansum(with_lobe_components, axis=1)[0]

    shoulder_index = 290

    assert no_lobe_signal[shoulder_index] < 0.02
    assert with_lobe_signal[shoulder_index] > 0.1
    assert with_lobe_signal[shoulder_index] > no_lobe_signal[shoulder_index]


def test_two_lobe_gaussian_plot_extends_to_secondary_lobe_support() -> None:
    kernel = TwoLobeGaussian(
        amplitude=1.0,
        position=(80.0, 320.0),
        width=(18.0, 28.0),
        secondary_ratio=1.0,
        secondary_offset=(55.0, 105.0),
        secondary_width=(12.0, 130.0),
        secondary_presence=1.0,
    )

    ax = kernel.plot()
    line = ax.lines[0]
    plt.close(ax.figure)

    x_values = np.asarray(line.get_xdata(), dtype=float)

    assert x_values[-1] >= 320.0 + 105.0 + 4.0 * 130.0


def test_kernel_plot_draws_on_provided_axis() -> None:
    kernel = Gaussian(
        amplitude=1.0,
        position=0.0,
        width=1.0,
    )
    x_values = np.linspace(-5.0, 5.0, 101)

    fig, ax = plt.subplots()
    try:
        returned_ax = kernel.plot(x_values=x_values, ax=ax, color="black")
    finally:
        plt.close(fig)

    assert returned_ax is ax
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), x_values)


def test_signal_generator_rejects_legacy_n_peaks_keyword() -> None:
    generator = SignalGenerator(sequence_length=64)
    kernel = Gaussian(
        amplitude=(1.0, 2.0),
        position=(5.0, 59.0),
        width=(1.0, 3.0),
    )

    with np.testing.assert_raises(TypeError):
        generator.generate(
            n_samples=8,
            kernel=kernel,
            n_peaks=(1, 4),
        )
