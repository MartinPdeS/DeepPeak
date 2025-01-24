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

if __name__ == "__main__":
    pytest.main(["-W error", __file__])

