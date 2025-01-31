import pytest
import numpy as np
from DeepPeak.signals import generate_gaussian_dataset


@pytest.mark.parametrize("n_peaks", [(0, 3), (1, 5)])
def test_basic_shapes(n_peaks):
    n_samples = 10
    sequence_length = 50
    signals, amps, pos, widths, x_vals, num_peaks = generate_gaussian_dataset(
        n_samples=n_samples,
        sequence_length=sequence_length,
        n_peaks=n_peaks,
        seed=42,
        noise_std=0.0  # No noise for easier checking
    )

    min_peaks, max_peaks = n_peaks

    # Shape checks
    assert signals.shape == (n_samples, sequence_length)
    assert amps.shape == (n_samples, max_peaks)
    assert pos.shape == (n_samples, max_peaks)
    assert widths.shape == (n_samples, max_peaks)
    assert x_vals.shape == (sequence_length,)
    assert num_peaks.shape == (n_samples,)

    # Values are in expected range
    assert np.all((num_peaks >= min_peaks) & (num_peaks <= max_peaks))

@pytest.mark.parametrize("n_peaks, amplitude, position, width", [
    ((0, 3), (1.0, 2.0), (10.0, 20.0), (0.01, 0.05)),
    ((1, 1), (5.0, 5.0), (0.0, 50.0), (0.03, 0.03))
])
def test_parameter_ranges(n_peaks, amplitude, position, width):
    """Check if generated amplitudes, positions, and widths fall within expected ranges."""
    n_samples = 5
    sequence_length = 10
    signals, amps, pos, wid, x_vals, num_pk = generate_gaussian_dataset(
        n_samples=n_samples,
        sequence_length=sequence_length,
        n_peaks=n_peaks,
        amplitude=amplitude,
        position=position,
        width=width,
        noise_std=0.0,
        seed=123
    )

    min_peaks, max_peaks = n_peaks
    min_amp, max_amp = amplitude
    min_pos, max_pos = position
    min_wid, max_wid = width

    # Check amplitude range (non-zero only for valid peaks)
    valid_mask = np.arange(max_peaks) < num_pk[:, None]
    valid_amps = amps[valid_mask]
    assert np.all(valid_amps >= min_amp)
    assert np.all(valid_amps <= max_amp)

    # Check position and width range
    valid_pos = pos[valid_mask]
    valid_wid = wid[valid_mask]
    assert np.all(valid_pos >= min_pos)
    assert np.all(valid_pos <= max_pos)
    assert np.all(valid_wid >= min_wid)
    assert np.all(valid_wid <= max_wid)

def test_zero_peaks():
    """Verify that signals with zero peaks are actually zero."""
    n_samples = 3
    sequence_length = 10
    # n_peaks range starts with 0
    signals, amps, pos, wid, x_vals, num_pk = generate_gaussian_dataset(
        n_samples=n_samples,
        sequence_length=sequence_length,
        n_peaks=(0, 0),
        noise_std=0.0,
        seed=42
    )
    # All signals should be zero
    assert np.allclose(signals, 0.0)
    # Check no peaks
    assert np.all(num_pk == 0)

if __name__ == "__main__":
    pytest.main(["-W error", __file__])

