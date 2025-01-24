import pytest
import numpy as np
from DeepPeak.signals import generate_square_dataset


def test_generate_square_peak_dataset_shapes():
    sample_count = 10
    sequence_length = 128
    peak_count = (2, 5)

    signals, amplitudes, peak_counts, positions, widths, x_values = generate_square_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=(1, 5),
        center_range=(0, 127),
        width_range=(5, 20),
        noise_std=0.1,
        normalize=True,
        normalize_x=True,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=True
    )

    assert signals.shape == (sample_count, sequence_length, 1), "Signals shape is incorrect."
    assert amplitudes.shape == (sample_count, peak_count[1]), "Amplitudes shape is incorrect."
    assert peak_counts.shape == (sample_count, peak_count[1] + 1), "Peak counts shape is incorrect."
    assert positions.shape == (sample_count, peak_count[1]), "Positions shape is incorrect."
    assert widths.shape == (sample_count, peak_count[1]), "Widths shape is incorrect."
    assert x_values.shape == (sequence_length,), "X-values shape is incorrect."

def test_generate_square_peak_dataset_peak_properties():
    sample_count = 5
    sequence_length = 64
    peak_count = (3, 3)

    signals, amplitudes, peak_counts, positions, widths, x_values = generate_square_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=(1, 5),
        center_range=(0, 63),
        width_range=(5, 15),
        noise_std=0.0,
        normalize=False,
        normalize_x=False,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    for i in range(sample_count):
        assert np.all(positions[i] >= 0) and np.all(positions[i] < sequence_length), "Peak positions are out of range."
        assert np.all(widths[i] > 0) and np.all(widths[i] <= sequence_length), "Widths are out of range."
        assert np.all(amplitudes[i] >= 1) and np.all(amplitudes[i] <= 5), "Amplitudes are out of range."

def test_generate_square_peak_dataset_normalization():
    sample_count = 3
    sequence_length = 32

    signals, _, _, _, _, _ = generate_square_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=3,
        amplitude_range=(1, 5),
        center_range=(0, 31),
        width_range=(5, 10),
        noise_std=0.1,
        normalize=True,
        normalize_x=False,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    for i in range(sample_count):
        assert np.isclose(np.mean(signals[i]), 0, atol=1e-6), "Signal mean is not zero after normalization."
        assert np.isclose(np.std(signals[i]), 1, atol=1e-6), "Signal standard deviation is not one after normalization."

def test_generate_square_peak_dataset_sorting():
    sample_count = 4
    sequence_length = 50
    peak_count = 4

    signals, amplitudes, _, positions, widths, _ = generate_square_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=(1, 5),
        center_range=(0, 49),
        width_range=(5, 15),
        noise_std=0.0,
        normalize=False,
        normalize_x=False,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    for i in range(sample_count):
        assert np.all(np.diff(positions[i]) >= 0), "Positions are not sorted."


if __name__ == "__main__":
    pytest.main(["-W error", __file__])
