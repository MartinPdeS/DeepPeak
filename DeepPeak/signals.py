from typing import Tuple, Optional, Dict
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

class DataSet:
    """
    A simple container class for datasets.

    This class dynamically sets attributes based on the provided keyword arguments,
    allowing for flexible storage of various dataset components.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to be set as attributes of the instance.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def interpret_input(*inputs):
    """
    Decorator that ensures specified function parameters are interpreted as tuples.

    If a parameter listed in `inputs` is provided as a scalar (int or float),
    it will be converted into a tuple of the form (value, value).

    Parameters
    ----------
    *inputs : str
        Names of the parameters to be interpreted as tuples.

    Returns
    -------
    function
        A wrapped function with specified parameters guaranteed to be tuples.
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
    extra_kwargs: Optional[Dict] = None,
    amplitude: Tuple[float, float] = (1.0, 2.0),
    position: Tuple[float, float] = (0.0, 1.0),  # Normalized position (0-1)
    width: Tuple[float, float] = (0.03, 0.03),
    seed: Optional[int] = None,
    noise_std: float = 0.01,
    categorical_peak_count: bool = False,
    kernel: Optional[np.ndarray] = None
):
    """
    Generate 1D signals with varying numbers of peaks, and produce associated labels and peak parameters.

    The generated signals can be of different types, including Gaussian, Lorentzian, Bessel,
    square, asymmetric Gaussian, and dirac delta signals. For each sample, a random number
    of peaks is generated between a minimum and maximum value. For samples that do not have the
    maximum number of peaks, the amplitudes and positions for the inactive peaks are set to np.nan.

    Parameters
    ----------
    n_samples : int
        Number of signals to generate.
    sequence_length : int
        Length of each generated signal (number of data points).
    n_peaks : tuple of int
        Tuple (min_peaks, max_peaks) specifying the range for the number of peaks per signal.
    signal_type : str, optional
        Type of signal to generate. Options include:
        'gaussian', 'lorentzian', 'bessel', 'square', 'asym_gaussian', 'dirac'. Default is 'gaussian'.
    extra_kwargs : dict, optional
        Additional keyword arguments used for certain signal types (e.g., separation and second_peak_ratio for 'asym_gaussian').
    amplitude : tuple of float, optional
        Range (min, max) for generating peak amplitudes. Default is (1.0, 2.0).
    position : tuple of float, optional
        Range (min, max) for generating peak positions (normalized between 0 and 1). Default is (0.0, 1.0).
    width : tuple of float, optional
        Range (min, max) for generating peak widths. Default is (0.03, 0.03).
    seed : int, optional
        Seed for the random number generator. If provided, results are reproducible.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the signal. Default is 0.01.
    categorical_peak_count : bool, optional
        If True, the number of peaks is returned as a categorical (one-hot encoded) array.
        Otherwise, it is returned as integer counts.
    kernel : np.ndarray, optional
        Convolution kernel to be applied to the signal when signal_type is 'dirac'.

    Returns
    -------
    DataSet
        An instance of DataSet containing the following attributes:
            - signals : np.ndarray
                Array of generated signals.
            - labels : np.ndarray
                Binary label array where 1 indicates the presence of a peak.
            - amplitudes : np.ndarray
                Array of peak amplitudes. Inactive peak entries (when a sample has fewer than the maximum peaks) are set to np.nan.
            - positions : np.ndarray
                Array of peak positions (normalized). Inactive peak entries are set to np.nan.
            - widths : np.ndarray
                Array of peak widths.
            - x_values : np.ndarray
                Array of x-axis values for the signal.
            - num_peaks : np.ndarray or one-hot encoded array
                Number of peaks per sample (or one-hot encoded representation if categorical_peak_count is True).
    """
    if seed is not None:
        np.random.seed(seed)

    if extra_kwargs is None:
        extra_kwargs = {}

    min_peaks, max_peaks = n_peaks
    num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)

    amplitudes = np.random.uniform(*amplitude, size=(n_samples, max_peaks))
    positions = np.random.uniform(*position, size=(n_samples, max_peaks))
    widths = np.random.uniform(*width, size=(n_samples, max_peaks))

    # Keep a copy of the original positions for label computation
    positions_for_labels = positions.copy()

    # Create a mask for active peaks (True for indices < num_peaks for each sample)
    peak_indices = np.arange(max_peaks)
    mask = peak_indices < num_peaks[:, None]

    # Set inactive peak amplitudes and positions to np.nan
    amplitudes[~mask] = np.nan
    positions[~mask] = np.nan
    widths[~mask] = np.nan

    x_values = np.linspace(0, 1, sequence_length)
    x_ = x_values.reshape(1, 1, -1)
    # Use original positions for signal generation to ensure valid computation
    pos_ = positions_for_labels[..., np.newaxis]
    wid_ = widths[..., np.newaxis]
    amp_ = amplitudes[..., np.newaxis]

    # Initialize labels array (all zeros)
    labels = np.zeros((n_samples, sequence_length))
    match signal_type.lower():
        case 'gaussian':
            peaks = amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_)**2)
        case 'lorentzian':
            peaks = amp_ / (1 + ((x_ - pos_) / wid_)**2)
        case 'bessel':
            peaks = amp_ * np.abs(np.sin((x_ - pos_) / wid_)) / ((x_ - pos_) / wid_ + 1e-6)
        case 'square':
            peaks = amp_ * ((np.abs(x_ - pos_) < wid_) * 1.0)
        case 'asym_gaussian':
            separation = extra_kwargs.get('separation', 0.1)
            second_peak_ratio = extra_kwargs.get('second_peak_ratio', 0.5)
            peaks = amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_)**2) + (amp_ * second_peak_ratio * np.exp(-0.5 * ((x_ - (pos_ + separation)) / (wid_ * 0.5))**2))
        case 'dirac':
            signals = np.zeros((n_samples, sequence_length))
            for i in range(n_samples):
                signal = np.zeros(sequence_length)
                # Use original positions to determine the peak index
                peak_pos = (positions_for_labels[i, :num_peaks[i]] * (sequence_length - 1)).astype(int)
                signal[peak_pos] = amplitudes[i, :num_peaks[i]]
                if kernel is not None:
                    signal = np.convolve(signal, kernel, mode='same')
                signals[i] = signal
        case _:
            raise ValueError("Invalid signal type. Choose from 'gaussian', 'lorentzian', 'bessel', 'square', 'asym_gaussian', 'dirac'.")

    # Compute label array: Mark the positions where peaks occur using original positions
    peak_positions = (positions_for_labels * (sequence_length - 1)).astype(int)
    for i in range(n_samples):
        labels[i, peak_positions[i, :num_peaks[i]]] = 1

    if signal_type != 'dirac':
        signals = np.nansum(peaks, axis=1)

    if noise_std > 0:
        signals += np.random.normal(0, noise_std, signals.shape)

    if categorical_peak_count:
        num_peaks = to_categorical(num_peaks, max_peaks + 1)

    return DataSet(
        signals=signals,
        labels=labels,
        amplitudes=amplitudes,
        positions=positions,
        widths=widths,
        x_values=x_values,
        num_peaks=num_peaks
    )
