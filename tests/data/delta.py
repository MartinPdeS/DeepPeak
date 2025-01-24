import pytest
import numpy as np
from DeepPeak.data.data_generation import generate_gaussian_dataset, generate_point_peak_dataset, generate_square_peak_dataset

def test_get_gaussian_peaks():
    # Test parameters
    num_samples = 10
    input_length = 128
    num_gaussian = (4, 4)
    amplitude_range = (1, 5)
    center_range = (32, 96)
    width_range = (5, 20)
    noise_amplitude = 0.1

    # Call the function
    X_train, amplitudes, num_peaks, peak_positions, peak_widths, _ = generate_gaussian_dataset(
        sample_count=num_samples,
        sequence_length=input_length,
        peak_count=num_gaussian,
        amplitude_range=amplitude_range,
        center_range=center_range,
        width_range=width_range,
        noise_std=noise_amplitude,
        categorical_peak_count=False,
        normalize=True
    )

    # Check shapes of outputs
    assert X_train.shape == (num_samples, input_length, 1), "X_train shape mismatch"
    assert amplitudes.shape == (num_samples, max(num_gaussian)), "Amplitudes shape mismatch"
    assert num_peaks.shape == (num_samples,), "Num peaks shape mismatch"
    assert peak_positions.shape == (num_samples, max(num_gaussian)), "Peak positions shape mismatch"
    assert peak_widths.shape == (num_samples, max(num_gaussian)), "Peak widths shape mismatch"

    # Check that peak counts are within the specified range
    assert np.all(num_peaks >= num_gaussian[0]) and np.all(num_peaks <= num_gaussian[1]), "Num peaks out of range"

    # Check that peak positions are within the specified range
    assert np.all(peak_positions >= center_range[0]) and np.all(peak_positions <= center_range[1]), "Peak positions out of range"

    # Check that amplitudes are within the specified range
    assert np.all(amplitudes >= amplitude_range[0]) and np.all(amplitudes <= amplitude_range[1]), "Amplitudes out of range"

    # Check that widths are within the specified range
    assert np.all(peak_widths >= width_range[0]) and np.all(peak_widths <= width_range[1]), "Widths out of range"

    # Verify normalization (mean close to 0 and std close to 1)
    assert np.allclose(np.mean(X_train, axis=1), 0, atol=1e-1), "Normalization mean mismatch"
    assert np.allclose(np.std(X_train, axis=1), 1, atol=1e-1), "Normalization std mismatch"

    # Verify peaks are sorted by positions
    for i in range(num_samples):
        sorted_positions = np.sort(peak_positions[i, :num_peaks[i]])
        assert np.allclose(peak_positions[i, :num_peaks[i]], sorted_positions), "Peak positions not sorted"

def test_generate_point_peak_dataset_sorting():
    sample_count = 3
    sequence_length = 50
    peak_count = 5

    signals, amplitudes, _, positions, _, _ = generate_point_peak_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=(1, 5),
        center_range=(0, 49),
        noise_std=0.0,
        normalize=False,
        normalize_x=False,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    # Test that positions are sorted
    for i in range(sample_count):
        assert np.all(np.diff(positions[i]) >= 0), "Positions are not sorted."

def test_generate_point_peak_dataset_shapes():
    sample_count = 10
    sequence_length = 128
    peak_count = (1, 5)

    signals, amplitudes, peak_counts, positions, widths, x_values = generate_point_peak_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=(1, 5),
        center_range=(0, 127),
        noise_std=0.1,
        normalize=True,
        normalize_x=True,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=True
    )

    # Test shapes
    assert signals.shape == (sample_count, sequence_length, 1), "Signals shape is incorrect."
    assert amplitudes.shape == (sample_count, peak_count[1]), "Amplitudes shape is incorrect."
    assert peak_counts.shape == (sample_count, peak_count[1] + 1), "Peak counts shape is incorrect."
    assert positions.shape == (sample_count, peak_count[1]), "Positions shape is incorrect."
    assert widths.shape == (sample_count, peak_count[1]), "Widths shape is incorrect."
    assert x_values.shape == (sequence_length,), "X-values shape is incorrect."

def test_generate_point_peak_dataset_peak_properties():
    sample_count = 5
    sequence_length = 64
    peak_count = (4, 4)

    signals, amplitudes, peak_counts, positions, widths, x_values = generate_point_peak_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=(1, 5),
        center_range=(0, 63),
        noise_std=0.0,
        normalize=False,
        normalize_x=False,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    # Test that peak positions are valid
    for i in range(sample_count):
        assert np.all(positions[i] >= 0) and np.all(positions[i] < sequence_length), "Peak positions are out of range."

    # Test that amplitudes are within the specified range
    assert np.all(amplitudes >= 1) and np.all(amplitudes <= 5), "Amplitudes are out of range."

def test_generate_point_peak_dataset_normalization():
    sample_count = 3
    sequence_length = 32

    signals, _, _, _, _, _ = generate_point_peak_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=3,
        amplitude_range=(1, 5),
        center_range=(0, 31),
        noise_std=0.1,
        normalize=True,
        normalize_x=False,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    # Test that signals are normalized
    for i in range(sample_count):
        assert np.isclose(np.mean(signals[i]), 0, atol=1e-6), "Signal mean is not zero after normalization."
        assert np.isclose(np.std(signals[i]), 1, atol=1e-6), "Signal standard deviation is not one after normalization."

if __name__ == "__main__":
    pytest.main(["-W error", __file__])
