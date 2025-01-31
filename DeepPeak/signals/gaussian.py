from typing import Tuple
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

import numpy as np
from typing import Tuple

def interpret_input(*inputs):
    """
    Decorator to interpret certain function parameters as tuples if given as scalars.
    If a parameter in `inputs` is passed as an int or float, it is converted
    to a tuple of the form (value, value).

    Parameters
    ----------
    inputs : str
        One or more parameter names that should be interpreted this way.

    Returns
    -------
    Callable
        A decorator function that can be applied to another function.
    """
    def _interpret_input(function):
        def wrapper(**kwargs):
            kwargs = {
                k: (v, v) if k in inputs and isinstance(v, (float, int)) else v
                for k, v in kwargs.items()
            }
            return function(**kwargs)
        return wrapper
    return _interpret_input

@interpret_input('width', 'position', 'amplitude')
def generate_gaussian_dataset(
    n_samples: int,
    sequence_length: int,
    n_peaks: Tuple[int, int],
    amplitude: Tuple[float, float] = (1.0, 2.0),
    position: Tuple[float, float] = (0.0, 50.0),
    width: Tuple[float, float] = (0.03, 0.03),
    seed: int = None,
    noise_std: float = 0.01,
    categorical_peak_count: bool = False
):
    """
    Generate a set of 1D signals with a specified range of possible Gaussian peaks.

    Each generated signal can contain between `n_peaks[0]` and `n_peaks[1]`
    Gaussian peaks (inclusive). Parameters controlling each peak (amplitude,
    position, width) are drawn from their respective ranges. Additionally,
    random Gaussian noise can be added if desired.

    Parameters
    ----------
    n_samples : int
        The number of signals to generate.
    sequence_length : int
        The length (number of points) of each generated signal.
    n_peaks : tuple of int
        A tuple (min_peaks, max_peaks) specifying the range of possible peaks
        in any generated signal.
    amplitude : tuple of float, optional
        A tuple (min_amplitude, max_amplitude) from which peak amplitudes are sampled.
        Defaults to (1.0, 2.0).
    position : tuple of float, optional
        A tuple (min_position, max_position) specifying where peak centers can lie
        in the domain [0..sequence_length]. Defaults to (0.0, 50.0).
    width : tuple of float, optional
        A tuple (min_width, max_width) defining the standard deviation of the Gaussian peaks.
        Defaults to (0.03, 0.03), meaning all peaks use the same width.
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    noise_std : float, optional
        The standard deviation of the added Gaussian noise. Defaults to 0.01.
    categorical_peak_count : bool, optional
        If True, convert the count of peaks for each sample into a categorical format.
        Defaults to False.

    Returns
    -------
    signals : numpy.ndarray of shape (n_samples, sequence_length)
        The generated signals, each being a sum of zero or more Gaussian peaks.
    amplitudes : numpy.ndarray of shape (n_samples, max_peaks)
        The amplitude for each potential peak in a signal. Peaks beyond the actual
        count are zero-padded.
    positions : numpy.ndarray of shape (n_samples, max_peaks)
        The center positions for each potential peak in a signal. Unused positions
        (for signals with fewer peaks) remain valid but their corresponding amplitudes
        are zeroed out.
    widths : numpy.ndarray of shape (n_samples, max_peaks)
        The standard deviations of each Gaussian peak. Unused peaks in a signal have
        corresponding amplitudes of zero, but this array remains fully populated.
    x_values : numpy.ndarray of shape (sequence_length,)
        The x-axis positions spanning from 0 to 1 (for plotting or further analysis).
    num_peaks : numpy.ndarray of shape (n_samples,)
        The actual number of peaks used in each signal. Each entry lies between
        n_peaks[0] and n_peaks[1], inclusive.

    Notes
    -----
    - If `noise_std` is greater than 0, random Gaussian noise is added to each signal.
    - If `categorical_peak_count` is set to True, the returned `num_peaks` may be converted
      to a one-hot or similar categorical representation (not fully implemented in this snippet).

    Examples
    --------
    >>> signals, amps, pos, widths, x_vals, num_pk = generate_gaussian_signals(
    ...     n_samples=10, sequence_length=50, n_peaks=(1, 3), seed=42
    ... )
    >>> signals.shape
    (10, 50)
    >>> amps.shape
    (10, 3)
    >>> x_vals.shape
    (50,)

    """
    if seed is not None:
        np.random.seed(seed)

    min_peaks, max_peaks = n_peaks

    # Randomly decide how many peaks each sample actually has
    num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)

    # Generate random amplitudes, positions, and widths
    amplitudes = np.random.uniform(*amplitude, size=(n_samples, max_peaks))
    positions = np.random.uniform(*position, size=(n_samples, max_peaks))
    widths = np.random.uniform(*width, size=(n_samples, max_peaks))

    # Mask out "non-existing" peaks by zeroing out amplitudes
    peak_indices = np.arange(max_peaks)  # shape (max_peaks,)
    mask = peak_indices < num_peaks[:, None]  # shape (n_samples, max_peaks)
    amplitudes *= mask

    # Create x-values from 0 to 1
    x_values = np.linspace(0, 1, sequence_length)

    # Build signals by summing Gaussian contributions
    x_ = x_values.reshape(1, 1, -1)   # shape: (1, 1, sequence_length)
    pos_ = positions[..., np.newaxis] # shape: (n_samples, max_peaks, 1)
    wid_ = widths[...,   np.newaxis]  # shape: (n_samples, max_peaks, 1)
    amp_ = amplitudes[..., np.newaxis] # shape: (n_samples, max_peaks, 1)

    gaussians = amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_)**2)
    signals = np.sum(gaussians, axis=1)  # shape: (n_samples, sequence_length)

    # Add optional noise
    if noise_std > 0:
        signals += np.random.normal(0, noise_std, signals.shape)

    # Convert num_peaks to a categorical representation if needed (not fully implemented)
    if categorical_peak_count:
        peak_counts = to_categorical(peak_counts, n_peaks[1] + 1)

    return signals, amplitudes, positions, widths, x_values, num_peaks
