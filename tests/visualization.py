# test_signal_plotter.py

import pytest
import numpy as np
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')  # To avoid opening windows during tests
import matplotlib.pyplot as plt

from DeepPeak.utils.visualization import SignalPlotter

@pytest.fixture
def sample_plotter():
    """Returns a fresh instance of SignalPlotter for each test."""
    return SignalPlotter()

def test_init(sample_plotter):
    """Test basic initialization."""
    assert sample_plotter.signals is None
    assert sample_plotter._custom_curves == []
    # Check defaults
    assert sample_plotter.show_peaks is True
    assert sample_plotter.show_amplitudes is True
    assert sample_plotter.show_roi is True

def test_add_signals_shape_ok(sample_plotter):
    """Add valid signals => no error; check stored shape."""
    signals = np.random.rand(5, 100)  # shape (n_samples=5, seq_length=100)
    sample_plotter.add_signals(signals)
    assert sample_plotter.signals.shape == (5, 100)
    # x_values should be auto -> shape (100,)
    assert sample_plotter.x_values.shape == (100,)

def test_add_signals_shape_bad(sample_plotter):
    """Add invalid signals => ValueError."""
    signals = np.random.rand(5, 100, 2)  # shape => 3D instead of 2D
    with pytest.raises(ValueError, match="signals must be 2D"):
        sample_plotter.add_signals(signals)

def test_add_signals_x_values_mismatch(sample_plotter):
    """Check x_values shape mismatch => ValueError."""
    signals = np.random.rand(5, 100)
    x_values = np.linspace(0, 1, 50)  # mismatch length
    with pytest.raises(ValueError, match="x_values length must match"):
        sample_plotter.add_signals(signals, x_values=x_values)

def test_add_roi_mismatch(sample_plotter):
    """ROI shape mismatch => ValueError."""
    signals = np.random.rand(5, 100)
    sample_plotter.add_signals(signals)  # valid

    # Try ROI with shape mismatch
    roi = np.random.randint(0, 2, size=(5, 50))  # shape (5, 50) instead of (5, 100)
    with pytest.raises(ValueError, match="ROI array shape must match signals shape"):
        sample_plotter.add_roi(roi, show_roi=True)

def test_add_roi_valid(sample_plotter):
    """ROI shape matches signals => should work."""
    signals = np.random.rand(5, 100)
    sample_plotter.add_signals(signals)
    roi = np.random.randint(0, 2, size=(5, 100))
    sample_plotter.add_roi(roi, show_roi=True)
    assert sample_plotter.roi.shape == (5, 100)
    assert sample_plotter.show_roi is True

def test_add_positions_amplitudes(sample_plotter):
    """Check positions & amplitudes storing."""
    signals = np.random.rand(5, 100)
    sample_plotter.add_signals(signals)
    positions = np.random.rand(5, 3)   # (n_samples=5, n_peaks=3)
    amplitudes = np.random.rand(5, 3)
    sample_plotter.add_positions(positions)
    sample_plotter.add_amplitudes(amplitudes)
    assert sample_plotter.positions.shape == (5, 3)
    assert sample_plotter.amplitudes.shape == (5, 3)

def test_add_custom_curves_shape_good(sample_plotter):
    """Check adding custom curves with matching shapes."""
    signals = np.random.rand(5, 50)
    sample_plotter.add_signals(signals)

    # We'll define a small function
    def linear_func(x, slope, intercept):
        return slope*x + intercept

    # shape => (5, 2): 5 samples, 2 curves each
    slopes = np.random.rand(5, 2)
    intercepts = np.random.rand(5, 2)

    sample_plotter.add_custom_curves(
        curve_function=linear_func,
        label="Lines",
        slope=slopes,
        intercept=intercepts
    )

    assert len(sample_plotter._custom_curves) == 1
    info = sample_plotter._custom_curves[0]
    assert info["curve_func"] == linear_func
    assert info["kwargs_arrays"]["slope"].shape == (5, 2)

def test_add_custom_curves_shape_mismatch(sample_plotter):
    """Check that mismatch shape among kwargs => ValueError."""
    signals = np.random.rand(5, 50)
    sample_plotter.add_signals(signals)

    def dummy_func(x, a, b):
        return a*x + b

    arr1 = np.random.rand(5, 3)
    arr2 = np.random.rand(5, 4)  # mismatch second dim

    with pytest.raises(ValueError, match="All keyword arrays must have the same shape"):
        sample_plotter.add_custom_curves(
            curve_function=dummy_func,
            arr1=arr1,
            arr2=arr2
        )

def test_add_custom_curves_n_samples_mismatch(sample_plotter):
    """Check that if signals is known, arrays must match n_samples dimension."""
    signals = np.random.rand(5, 50)
    sample_plotter.add_signals(signals)

    def dummy_func(x, a):
        return a*x

    arr_wrong = np.random.rand(6, 2)  # 6 != n_samples(=5)

    with pytest.raises(ValueError, match="shape.*!= n_samples"):
        sample_plotter.add_custom_curves(curve_function=dummy_func, a=arr_wrong)

def test_plot_no_signals(sample_plotter):
    """Plotting without signals => ValueError."""
    with pytest.raises(ValueError, match="No signals to plot"):
        sample_plotter.plot()

@patch("matplotlib.pyplot.show")
def test_plot_ok(mock_show, sample_plotter):
    """
    Basic plot test with correct usage. We mock `plt.show`
    so we don't actually open a window in test.
    """
    signals = np.random.rand(5, 50)
    sample_plotter.add_signals(signals)
    sample_plotter.plot(n_examples=2, n_columns=1, random_select=False)
    # If we reached here, it means it didn't raise any error.
    mock_show.assert_called_once()

def test_fluent_interface(sample_plotter):
    """Check that add_* methods return self for chaining."""
    out = (
        sample_plotter
        .add_signals(np.random.rand(5, 10))
        .add_positions(np.random.rand(5, 2))
        .add_amplitudes(np.random.rand(5, 2))
        .configure_display(show_peaks=False, show_amplitudes=False, show_roi=False)
        .set_title("Test Plot")
    )
    assert out is sample_plotter

if __name__ == "__main__":
    pytest.main(["-W error", __file__])
