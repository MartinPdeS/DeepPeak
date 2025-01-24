from typing import Tuple
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

def generate_square_dataset(
    sample_count: int,
    sequence_length: int,
    peak_count: tuple | int = (1, 5),
    amplitude_range: tuple | float = (1, 5),
    center_range: tuple | float = (0, 1),
    width_range: tuple | float = (0.1, 0.1),
    noise_std: float = 0.0,
    normalize: bool = True,
    normalize_x: bool = True,
    nan_values: float = 0,
    sort_peak: str = 'position',
    categorical_peak_count: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dataset of square peaks with optional Gaussian noise.

    Each "peak" is a square pulse centered at a specific index with a specified width.

    Parameters
    ----------
    sample_count : int
        Number of sequences to generate.
    sequence_length : int
        Length of each sequence.
    peak_count : int or tuple of int
        Number of square peaks per sequence (fixed or random range).
    amplitude_range : tuple of float or float
        Range or fixed value for amplitudes of the peaks.
    center_range : tuple of int or int
        Range or fixed value for center positions of the peaks (integer indices).
    width_range : tuple of float or float
        Range or fixed value for the widths of the square peaks.
    noise_std : float
        Standard deviation of Gaussian noise added to each sequence.
    normalize : bool
        Whether to normalize each sequence after noise addition.
    normalize_x : bool
        Whether to use a normalized x-axis (0 to 1) or integer indices (0 to sequence_length-1)
        in the returned x_values array. If True, widths are also normalized.
    nan_values : float
        Fill value for unused array entries (e.g., if you have max_peaks but actual < max_peaks).
    sort_peak : {'position', 'amplitude', 'width'}
        Sort criteria for ordering peaks along axis=1.
    categorical_peak_count : bool
        Whether to convert the peak count to a one-hot encoded vector.

    Returns
    -------
    signals : np.ndarray
        Array of shape (sample_count, sequence_length, 1) containing sequences with square peaks.
    amplitudes : np.ndarray
        Array of shape (sample_count, max_peaks) containing the amplitudes of each peak.
    peak_counts : np.ndarray
        If `categorical_peak_count=True`, returns one-hot encoded counts of shape (sample_count, max_peaks+1).
        Otherwise, returns an integer array of shape (sample_count,) with the peak counts.
    positions : np.ndarray
        Positions of peaks (integer indices), shape (sample_count, max_peaks).
    widths : np.ndarray
        Array of shape (sample_count, max_peaks) containing the widths of the peaks.
    x_values : np.ndarray
        The x-axis values, either 0..1 (if normalize_x=True) or 0..(sequence_length-1).
    """

    if isinstance(peak_count, tuple):
        min_peaks, max_peaks = peak_count
    else:
        min_peaks = max_peaks = peak_count
    max_peaks = max(max_peaks, 1)

    if isinstance(amplitude_range, (int, float)):
        amplitude_range = (amplitude_range, amplitude_range)
    if isinstance(center_range, (int, float)):
        center_range = (center_range, center_range)
    if isinstance(width_range, (int, float)):
        width_range = (width_range, width_range)

    if normalize_x:
        x_values = np.linspace(0, 1, sequence_length)
        width_range = tuple(w / sequence_length for w in width_range)  # Normalize widths
    else:
        x_values = np.arange(sequence_length)

    signals = np.zeros((sample_count, sequence_length, 1))
    amplitudes = np.zeros((sample_count, max_peaks)) + nan_values
    positions = np.zeros((sample_count, max_peaks)) + nan_values
    widths = np.zeros((sample_count, max_peaks)) + nan_values
    peak_counts = np.zeros((sample_count,), dtype=int)

    for i in range(sample_count):
        current_peak_count = np.random.randint(min_peaks, max_peaks + 1)

        current_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=current_peak_count)
        low_position = max(0, center_range[0])
        high_position = min(1 if normalize_x else sequence_length - 1, center_range[1])
        current_centers = np.random.uniform(low_position, high_position, size=current_peak_count)
        current_widths = np.random.uniform(width_range[0], width_range[1], size=current_peak_count)

        if normalize_x:
            _current_widths = current_widths * sequence_length
        else:
            _current_widths = current_widths

        signal = np.zeros(sequence_length)
        for amp, pos, width in zip(current_amplitudes, current_centers, _current_widths):
            if normalize_x:
                start_idx = int(max(0, (pos - width / 2) * sequence_length))
                end_idx = int(min(sequence_length, (pos + width / 2) * sequence_length))
            else:
                start_idx = int(max(0, pos - width / 2))
                end_idx = int(min(sequence_length, pos + width / 2))

            signal[start_idx:end_idx] += amp

        if noise_std > 0:
            signal += np.random.normal(0, noise_std, sequence_length)

        if normalize:
            std = np.std(signal)
            if std != 0:
                signal = (signal - np.mean(signal)) / std

        signals[i, :, 0] = signal
        amplitudes[i, :current_peak_count] = current_amplitudes
        positions[i, :current_peak_count] = current_centers
        widths[i, :current_peak_count] = current_widths
        peak_counts[i] = current_peak_count

    if sort_peak in ['position', 'amplitude', 'width']:
        match sort_peak:
            case 'position':
                sorted_indices = np.argsort(positions, axis=1)
            case 'amplitude':
                sorted_indices = np.argsort(amplitudes, axis=1)
            case 'width':
                sorted_indices = np.argsort(widths, axis=1)

        amplitudes = np.take_along_axis(amplitudes, sorted_indices, axis=1)
        positions = np.take_along_axis(positions, sorted_indices, axis=1)
        widths = np.take_along_axis(widths, sorted_indices, axis=1)

    if categorical_peak_count:
        peak_counts = to_categorical(peak_counts, max_peaks + 1)

    return signals, amplitudes, peak_counts, positions, widths, x_values
