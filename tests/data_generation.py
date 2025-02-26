import pytest
import numpy as np
from tensorflow.keras.utils import to_categorical
from DeepPeak.signals import generate_signal_dataset  # Replace with actual module name

@pytest.fixture
def default_params():
    """Fixture providing a set of default parameters for testing."""
    return {
        "n_samples": 10,
        "sequence_length": 100,
        "n_peaks": (1, 3),
        "signal_type": "gaussian",
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
    signals, _, amplitudes, positions, widths, x_values, num_peaks = generate_signal_dataset(**default_params)

    assert signals.shape == (default_params["n_samples"], default_params["sequence_length"])
    assert amplitudes.shape == (default_params["n_samples"], default_params["n_peaks"][1])
    assert positions.shape == amplitudes.shape
    assert widths.shape == amplitudes.shape
    assert x_values.shape == (default_params["sequence_length"],)
    assert num_peaks.shape[0] == default_params["n_samples"]

def test_signal_types(default_params):
    """Test different signal types to ensure they are generated correctly."""
    for signal_type in ["gaussian", "lorentzian", "bessel", "square", "asym_gaussian", "dirac"]:
        default_params["signal_type"] = signal_type
        signals, _, _, _, _, _, _ = generate_signal_dataset(**default_params)
        assert signals.shape == (default_params["n_samples"], default_params["sequence_length"])

def test_invalid_signal_type(default_params):
    """Ensure an invalid signal type raises a ValueError."""
    default_params["signal_type"] = "invalid_type"
    with pytest.raises(ValueError, match="Invalid signal type"):
        generate_signal_dataset(**default_params)

def test_random_seed_consistency(default_params):
    """Ensure setting a seed results in reproducible outputs."""
    signals_1, _, amplitudes_1, positions_1, _, _, _ = generate_signal_dataset(**default_params)
    signals_2, _, amplitudes_2, positions_2, _, _, _ = generate_signal_dataset(**default_params)

    np.testing.assert_array_equal(signals_1, signals_2)
    np.testing.assert_array_equal(amplitudes_1, amplitudes_2)
    np.testing.assert_array_equal(positions_1, positions_2)

def test_categorical_peak_count(default_params):
    """Check if categorical encoding of peak count works correctly."""
    default_params["categorical_peak_count"] = True
    signals,_,  _, _, _, _, num_peaks = generate_signal_dataset(**default_params)

    assert num_peaks.shape == (default_params["n_samples"], default_params["n_peaks"][1] + 1)
    assert np.all(np.sum(num_peaks, axis=1) == 1), "Each sample should have a one-hot encoded peak count"

def test_zero_noise(default_params):
    """Ensure setting noise_std to 0 results in noise-free signals."""
    default_params["noise_std"] = 0
    signals_1, _, _, _, _, _, _ = generate_signal_dataset(**default_params)
    signals_2, _, _, _, _, _, _ = generate_signal_dataset(**default_params)

    np.testing.assert_array_equal(signals_1, signals_2)

def test_edge_case_zero_peaks(default_params):
    """Ensure the function handles n_peaks=(0, 0) correctly."""
    default_params["n_peaks"] = (0, 0)
    default_params["noise_std"] = 0
    signals, _, amplitudes, _, _, _, _ = generate_signal_dataset(**default_params)

    assert np.all(amplitudes == 0), "No peaks should result in zero amplitudes"
    assert np.all(signals == 0), "No peaks should result in zero signals"

def test_edge_case_extreme_values(default_params):
    """Test function with extreme values for peak parameters."""
    default_params["amplitude"] = (100, 200)
    default_params["width"] = (1.0, 2.0)
    default_params['n_peaks'] = (1, 1)
    _, _, amplitudes, _, widths, _, _ = generate_signal_dataset(**default_params)

    assert np.all(amplitudes >= 100) and np.all(amplitudes <= 200), "Amplitudes should be within the given range"
    assert np.all(widths >= 1.0) and np.all(widths <= 2.0), "Widths should be within the given range"

def test_dirac_signal_requires_kernel(default_params):
    """Ensure 'dirac' signal type works with and without kernel."""
    default_params["signal_type"] = "dirac"
    default_params["kernel"] = None
    signals, _, _, _, _, _, _ = generate_signal_dataset(**default_params)
    assert signals.shape == (default_params["n_samples"], default_params["sequence_length"])

    kernel = np.array([1, -1, 1])  # Example convolution kernel
    default_params["kernel"] = kernel
    signals, _, _, _, _, _, _ = generate_signal_dataset(**default_params)
    assert signals.shape == (default_params["n_samples"], default_params["sequence_length"])

def test_width_position_amplitude_scalar_interpretation(default_params):
    """Ensure the interpret_input decorator correctly handles scalar values."""
    default_params["width"] = 0.05
    default_params["position"] = 0.5
    default_params["amplitude"] = 5.0
    _, _, _, positions, widths, _, _ = generate_signal_dataset(**default_params)

    assert np.all(widths == 0.05), "Width scalar should be converted to tuple"
    assert np.all(positions == 0.5), "Position scalar should be converted to tuple"

if __name__ == "__main__":
    pytest.main(["-W error", __file__])
