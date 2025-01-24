import pytest
import numpy as np
from DeepPeak.utils import plot_dataset  # Replace 'your_module' with the actual module name
import matplotlib.pyplot as plt
from unittest.mock import patch

def test_plot_dataset_valid_input():
    sample_count = 10
    sequence_length = 100
    max_peaks = 3

    signals = np.random.rand(sample_count, sequence_length, 1)
    amplitudes = np.random.rand(sample_count, max_peaks)
    positions = np.random.randint(0, sequence_length, size=(sample_count, max_peaks))
    widths = np.random.randint(5, 20, size=(sample_count, max_peaks))
    x_values = np.linspace(0, 1, sequence_length)

    with patch("matplotlib.pyplot.show"):
        try:
            plot_dataset(signals, amplitudes, positions, widths, x_values, num_samples=5, title="Test Plot")
        except Exception as e:
            pytest.fail(f"plot_dataset raised an exception with valid input: {e}")

def test_plot_dataset_missing_x_values():
    sample_count = 5
    sequence_length = 50

    signals = np.random.rand(sample_count, sequence_length, 1)

    with patch("matplotlib.pyplot.show"):
        try:
            plot_dataset(signals, num_samples=3, title="Test Plot Without X Values")
        except Exception as e:
            pytest.fail(f"plot_dataset raised an exception when x_values was missing: {e}")

def test_plot_dataset_invalid_signals_shape():
    invalid_signals = np.random.rand(10, 100)  # Missing the third dimension

    with patch("matplotlib.pyplot.show"):
        with pytest.raises(ValueError, match="Signals must be a 3D array with shape \(sample_count, sequence_length, 1\)."):
            plot_dataset(invalid_signals)

def test_plot_dataset_handles_nan_positions():
    sample_count = 5
    sequence_length = 50
    max_peaks = 3

    signals = np.random.rand(sample_count, sequence_length, 1)
    amplitudes = np.random.rand(sample_count, max_peaks)
    positions = np.random.randint(0, sequence_length, size=(sample_count, max_peaks)).astype(float)
    positions[0, 0] = np.nan  # Introduce a NaN value

    with patch("matplotlib.pyplot.show"):
        try:
            plot_dataset(signals, amplitudes, positions, num_samples=3, title="Test Plot with NaN Positions")
        except Exception as e:
            pytest.fail(f"plot_dataset raised an exception with NaN positions: {e}")

if __name__ == "__main__":
    pytest.main(["-W error", __file__])

