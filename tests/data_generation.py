import pytest
import numpy as np
from DeepPeak.signals import generate_signal_dataset, Kernel  # Replace with actual module name

@pytest.fixture
def default_params():
    """Fixture providing a set of default parameters for testing."""
    return {
        "n_samples": 10,
        "sequence_length": 100,
        "n_peaks": (1, 3),
        "signal_type": Kernel.GAUSSIAN,
        "amplitude": (1.0, 2.0),
        "position": (0.1, 0.9),
        "width": (0.03, 0.05),
        "seed": 42,
        "noise_std": 0.01,
        "categorical_peak_count": False,
        "extra_kwargs": None
    }

def test_output_shapes(default_params):
    """Test if function returns correctly shaped arrays."""
    dataset = generate_signal_dataset(**default_params)

    assert dataset.signals.shape == (default_params["n_samples"], default_params["sequence_length"])
    assert dataset.amplitudes.shape == (default_params["n_samples"], default_params["n_peaks"][1])
    assert dataset.positions.shape == dataset.amplitudes.shape
    assert dataset.widths.shape == dataset.amplitudes.shape
    assert dataset.x_values.shape == (default_params["sequence_length"],)
    assert dataset.num_peaks.shape[0] == default_params["n_samples"]

def test_signal_types(default_params):
    """Test different signal types to ensure they are generated correctly."""
    for signal_type in [Kernel.GAUSSIAN, Kernel.LORENTZIAN, Kernel.BESSEL, Kernel.SQUARE, Kernel.ASYMMETRIC_GAUSSIAN, Kernel.DIRAC]:
        default_params["signal_type"] = signal_type
        dataset = generate_signal_dataset(**default_params)
        assert dataset.signals.shape == (default_params["n_samples"], default_params["sequence_length"])

def test_random_seed_consistency(default_params):
    """Ensure setting a seed results in reproducible outputs."""
    dataset_0 = generate_signal_dataset(**default_params)
    dataset_1 = generate_signal_dataset(**default_params)

    np.testing.assert_array_equal(dataset_0.signals, dataset_1.signals)
    np.testing.assert_array_equal(dataset_0.amplitudes, dataset_1.amplitudes)
    np.testing.assert_array_equal(dataset_0.positions, dataset_1.positions)

def test_categorical_peak_count(default_params):
    """Check if categorical encoding of peak count works correctly."""
    default_params["categorical_peak_count"] = True
    dataset = generate_signal_dataset(**default_params)

    assert dataset.num_peaks.shape == (default_params["n_samples"], default_params["n_peaks"][1] + 1)
    assert np.all(np.sum(dataset.num_peaks, axis=1) == 1), "Each sample should have a one-hot encoded peak count"

def test_zero_noise(default_params):
    """Ensure setting noise_std to 0 results in noise-free signals."""
    default_params["noise_std"] = 0
    dataset_1 = generate_signal_dataset(**default_params)
    dataset_2 = generate_signal_dataset(**default_params)

    np.testing.assert_array_equal(dataset_1.signals, dataset_2.signals)

def test_edge_case_zero_peaks(default_params):
    """Ensure the function handles n_peaks=(0, 0) correctly."""
    default_params["n_peaks"] = (0, 0)
    default_params["noise_std"] = 0
    dataset = generate_signal_dataset(**default_params)

    assert np.all(dataset.amplitudes == 0), "No peaks should result in zero amplitudes"
    assert np.all(dataset.signals == 0), "No peaks should result in zero signals"

def test_edge_case_extreme_values(default_params):
    """Test function with extreme values for peak parameters."""
    default_params["amplitude"] = (100, 200)
    default_params["width"] = (1.0, 2.0)
    default_params['n_peaks'] = (1, 1)
    dataset = generate_signal_dataset(**default_params)

    assert np.all(dataset.amplitudes >= 100) and np.all(dataset.amplitudes <= 200), "Amplitudes should be within the given range"
    assert np.all(dataset.widths >= 1.0) and np.all(dataset.widths <= 2.0), "Widths should be within the given range"

def test_dirac_signal_requires_kernel(default_params):
    """Ensure 'dirac' signal type works with and without kernel."""
    default_params["signal_type"] = Kernel.DIRAC
    default_params["kernel"] = None
    dataset = generate_signal_dataset(**default_params)
    assert dataset.signals.shape == (default_params["n_samples"], default_params["sequence_length"])

    kernel = np.array([1, -1, 1])  # Example convolution kernel
    default_params["kernel"] = kernel
    dataset = generate_signal_dataset(**default_params)
    assert dataset.signals.shape == (default_params["n_samples"], default_params["sequence_length"])

def test_width_position_amplitude_scalar_interpretation(default_params):
    """Ensure the interpret_input decorator correctly handles scalar values."""
    default_params["width"] = 0.05
    default_params["position"] = 0.5
    default_params["amplitude"] = 5.0
    dataset = generate_signal_dataset(**default_params)

    assert np.all(dataset.widths[~np.isnan(dataset.widths)] == 0.05), "Width scalar should be converted to tuple"
    assert np.all(dataset.positions[~np.isnan(dataset.positions)] == 0.5), "Position scalar should be converted to tuple"
    assert np.all(dataset.amplitudes[~np.isnan(dataset.amplitudes)] == 5.0), "Amplitude scalar should be converted to tuple"

if __name__ == "__main__":
    pytest.main(["-W error", __file__])
