from typing import Tuple, Callable, Optional
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

def interpret_input(*inputs):
    """
    Decorator to interpret certain function parameters as tuples if given as scalars.
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
def generate_signal_dataset(
    n_samples: int,
    sequence_length: int,
    n_peaks: Tuple[int, int],
    signal_type: str = 'gaussian',
    amplitude: Tuple[float, float] = (1.0, 2.0),
    position: Tuple[float, float] = (0.0, 50.0),
    width: Tuple[float, float] = (0.03, 0.03),
    seed: Optional[int] = None,
    noise_std: float = 0.01,
    categorical_peak_count: bool = False,
    kernel: Optional[np.ndarray] = None
):
    """
    Generate 1D signals with different types of peaks: Gaussian, Lorentzian, Bessel, or Dirac pulses.

    Parameters
    ----------
    n_samples : int
        Number of signals to generate.
    sequence_length : int
        Length of each generated signal.
    n_peaks : tuple of int
        Min and max number of peaks per signal.
    signal_type : str
        Type of signal to generate: 'gaussian', 'lorentzian', 'bessel', 'dirac'.
    amplitude : tuple of float, optional
        Range for peak amplitudes.
    position : tuple of float, optional
        Range for peak positions.
    width : tuple of float, optional
        Range for peak widths.
    seed : int, optional
        Random seed for reproducibility.
    noise_std : float, optional
        Standard deviation of added noise.
    categorical_peak_count : bool, optional
        If True, convert peak count to a categorical format.
    kernel : np.ndarray, optional
        User-defined kernel for Dirac pulse convolution.

    Returns
    -------
    signals : numpy.ndarray
        Generated signals.
    amplitudes, positions, widths : numpy.ndarray
        Parameters of the generated peaks.
    x_values : numpy.ndarray
        X-axis values.
    num_peaks : numpy.ndarray
        Number of peaks per sample.
    """
    if seed is not None:
        np.random.seed(seed)

    min_peaks, max_peaks = n_peaks
    num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)
    amplitudes = np.random.uniform(*amplitude, size=(n_samples, max_peaks))
    positions = np.random.uniform(*position, size=(n_samples, max_peaks))
    widths = np.random.uniform(*width, size=(n_samples, max_peaks))

    peak_indices = np.arange(max_peaks)
    mask = peak_indices < num_peaks[:, None]
    amplitudes *= mask

    x_values = np.linspace(0, 1, sequence_length)
    x_ = x_values.reshape(1, 1, -1)
    pos_ = positions[..., np.newaxis]
    wid_ = widths[..., np.newaxis]
    amp_ = amplitudes[..., np.newaxis]

    if signal_type == 'gaussian':
        peaks = amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_)**2)
    elif signal_type == 'lorentzian':
        peaks = amp_ / (1 + ((x_ - pos_) / wid_)**2)
    elif signal_type == 'bessel':
        peaks = amp_ * np.abs(np.sin((x_ - pos_) / wid_)) / ((x_ - pos_) / wid_ + 1e-6)
    elif signal_type == 'dirac':
        signals = np.zeros((n_samples, sequence_length))
        for i in range(n_samples):
            signal = np.zeros(sequence_length)
            peak_pos = (positions[i] * (sequence_length - 1)).astype(int)
            signal[peak_pos] = amplitudes[i]
            if kernel is not None:
                signal = np.convolve(signal, kernel, mode='same')
            signals[i] = signal
    else:
        raise ValueError("Invalid signal type. Choose from 'gaussian', 'lorentzian', 'bessel', 'dirac'.")

    if signal_type != 'dirac':
        signals = np.sum(peaks, axis=1)

    if noise_std > 0:
        signals += np.random.normal(0, noise_std, signals.shape)

    if categorical_peak_count:
        num_peaks = to_categorical(num_peaks, max_peaks + 1)

    return signals, amplitudes, positions, widths, x_values, num_peaks