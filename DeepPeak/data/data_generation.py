from typing import Tuple
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

def generate_gaussian_dataset(
    sample_count: int,
    sequence_length: int,
    peak_count: tuple | int = (1, 5),
    amplitude_range: tuple | float=(1, 5),
    center_range: tuple | float = (0, 1),
    width_range: tuple | float = (5, 20),
    noise_std: float = 0.0,
    normalize: bool = True,
    normalize_x: bool = True,
    nan_values: float = 0,
    sort_peak: str = 'position',
    categorical_peak_count: bool = True) -> Tuple:
    """
    Generate a dataset of Gaussian curves with optional Gaussian noise.

    Parameters
    ----------
    sample_count : int
        Number of sequences to generate.
    sequence_length : int
        Length of each sequence.
    peak_count : int or tuple of int
        Number of Gaussian peaks per sequence (fixed or random range).
    amplitude_range : tuple of float or float
        Range or fixed value for amplitudes of Gaussian peaks.
    center_range : tuple of int or int
        Range or fixed value for center positions of Gaussian peaks.
    width_range : tuple of float or float
        Range or fixed value for standard deviations (widths) of Gaussian peaks.
    noise_std : float
        Standard deviation of Gaussian noise added to each sequence.
    normalize : bool
        Whether to normalize each sequence.

    Returns
    -------
    signals : numpy.ndarray
        Array of sequences with Gaussian peaks and added noise, shape (sample_count, sequence_length, 1).
    amplitudes : numpy.ndarray
        Array of amplitudes for each peak, shape (sample_count, max_peaks).
    peak_counts : numpy.ndarray
        Number of peaks per sequence, shape (sample_count,).
    peak_positions : numpy.ndarray
        Positions of peaks, shape (sample_count, max_peaks).
    peak_widths : numpy.ndarray
        Widths of peaks, shape (sample_count, max_peaks).
    """
    if isinstance(peak_count, tuple):
        min_peaks, max_peaks = peak_count
    else:
        min_peaks = max_peaks = peak_count

    if isinstance(amplitude_range, (int, float)):
        amplitude_range = (amplitude_range, amplitude_range)

    if isinstance(center_range, (int, float)):
        center_range = (center_range, center_range)

    if isinstance(width_range, (int, float)):
        width_range = (width_range, width_range)

    max_peaks = max(max_peaks, 1)  # Ensure at least one peak
    if normalize_x:
        x_values = np.linspace(0, 1, sequence_length)
    else:
        x_values = np.arange(sequence_length)  # Shared x-axis for all sequences

    # Preallocate arrays
    signals = np.zeros((sample_count, sequence_length, 1))
    amplitudes = np.zeros((sample_count, max_peaks)) * nan_values
    positions = np.zeros((sample_count, max_peaks)) * nan_values
    widths = np.zeros((sample_count, max_peaks)) * nan_values
    peak_counts = np.zeros((sample_count), dtype=int)

    # Generate Gaussian parameters
    for i in range(sample_count):
        current_peak_count = np.random.randint(min_peaks, max_peaks + 1)

        current_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=current_peak_count)
        current_centers = np.random.uniform(center_range[0], center_range[1], size=current_peak_count)
        current_widths = np.random.uniform(width_range[0], width_range[1], size=current_peak_count)

        # Compute Gaussian curves in vectorized form
        gaussians = current_amplitudes[:, None] * np.exp(-((x_values - current_centers[:, None])**2) / (2 * current_widths[:, None]**2))
        combined_curve = np.sum(gaussians, axis=0)

        # Add Gaussian noise if applicable
        if noise_std > 0:
            combined_curve += np.random.normal(0, noise_std, sequence_length)

        if normalize:
            combined_curve = (combined_curve - np.mean(combined_curve)) / np.std(combined_curve)

        # Store results
        signals[i, :, 0] = combined_curve
        amplitudes[i, :current_peak_count] = current_amplitudes
        positions[i, :current_peak_count] = current_centers
        widths[i, :current_peak_count] = current_widths
        peak_counts[i] = current_peak_count

    # Sort peaks by positions
    match sort_peak:
        case 'position':
            sorted_indices = np.argsort(positions, axis=1)
        case 'amplitude':
            sorted_indices = np.argsort(amplitudes, axis=1)
        case 'width':
            sorted_indices = np.argsort(widths, axis=1)

    amplitudes = np.take_along_axis(amplitudes, sorted_indices, axis=1)
    widths = np.take_along_axis(widths, sorted_indices, axis=1)
    positions = np.take_along_axis(positions, sorted_indices, axis=1)

    if categorical_peak_count:
        peak_counts = to_categorical(peak_counts, max_peaks + 1)

    return signals, amplitudes, peak_counts, positions, widths, x_values

def generate_point_peak_dataset(
    sample_count: int,
    sequence_length: int,
    peak_count: tuple | int = (1, 5),
    amplitude_range: tuple | float = (1, 5),
    center_range: tuple | float = (0, 1),
    noise_std: float = 0.0,
    normalize: bool = True,
    normalize_x: bool = True,
    nan_values: float = 0,
    sort_peak: str = 'position',
    categorical_peak_count: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dataset of discrete point peaks with optional Gaussian noise.

    In this version, each "peak" is a discrete impulse at a particular index.

    Parameters
    ----------
    sample_count : int
        Number of sequences to generate.
    sequence_length : int
        Length of each sequence.
    peak_count : int or tuple of int
        Number of point peaks per sequence (fixed or random range).
    amplitude_range : tuple of float or float
        Range or fixed value for amplitudes of the peaks.
    center_range : tuple of int or int
        Range or fixed value for center positions of the peaks (integer indices).
    width_range : tuple of float or float
        Unused for point peaks (retained for signature consistency).
    noise_std : float
        Standard deviation of Gaussian noise added to each sequence.
    normalize : bool
        Whether to normalize each sequence after noise addition.
    normalize_x : bool
        Whether to use a normalized x-axis (0 to 1) or integer indices (0 to sequence_length-1)
        in the returned x_values array. Note that for point peaks, x-axis scaling only affects the
        returned x_values; it does not affect how the peaks themselves are placed in the sequence.
    nan_values : float
        Fill value for unused array entries (e.g., if you have max_peaks but actual < max_peaks).
    sort_peak : {'position', 'amplitude', 'width'}
        Sort criteria for ordering peaks along axis=1.
    categorical_peak_count : bool
        Whether to convert the peak count to a one-hot encoded vector.

    Returns
    -------
    signals : np.ndarray
        Array of shape (sample_count, sequence_length, 1) containing sequences with discrete peaks.
    amplitudes : np.ndarray
        Array of shape (sample_count, max_peaks) containing the amplitudes of each peak.
    peak_counts : np.ndarray
        If `categorical_peak_count=True`, returns one-hot encoded counts of shape (sample_count, max_peaks+1).
        Otherwise, returns an integer array of shape (sample_count,) with the peak counts.
    positions : np.ndarray
        Positions of peaks (integer indices), shape (sample_count, max_peaks).
    widths : np.ndarray
        Array of shape (sample_count, max_peaks). For point peaks, this is not meaningful,
        but returned for consistency (filled with `nan_values` or zeros).
    x_values : np.ndarray
        The x-axis values, either 0..1 (if normalize_x=True) or 0..(sequence_length-1).
    """

    # Handle peak_count parameter
    if isinstance(peak_count, tuple):
        min_peaks, max_peaks = peak_count
    else:
        min_peaks = max_peaks = peak_count
    max_peaks = max(max_peaks, 1)  # Ensure at least one peak

    # Convert amplitude, center, and width parameters to tuples if needed
    if isinstance(amplitude_range, (int, float)):
        amplitude_range = (amplitude_range, amplitude_range)
    if isinstance(center_range, (int, float)):
        center_range = (center_range, center_range)
    # width_range is unused for the actual signal but kept in the signature
    # to remain consistent with the original function.

    # Create x-values for reference
    if normalize_x:
        x_values = np.linspace(0, 1, sequence_length)
    else:
        x_values = np.arange(sequence_length)

    # Preallocate arrays
    signals = np.zeros((sample_count, sequence_length, 1))
    amplitudes = np.zeros((sample_count, max_peaks)) + nan_values
    positions = np.zeros((sample_count, max_peaks)) + nan_values
    widths = np.zeros((sample_count, max_peaks)) + nan_values  # Not used for point peaks
    peak_counts = np.zeros((sample_count,), dtype=int)

    # Generate point-peak parameters
    for i in range(sample_count):
        current_peak_count = np.random.randint(min_peaks, max_peaks + 1)

        current_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=current_peak_count)
        # Positions must be integer indices within [center_range[0], center_range[1]] but also within [0, sequence_length-1]
        # We'll clamp them to valid integer indices
        low_position = max(0, int(center_range[0]))
        high_position = min(sequence_length - 1, int(center_range[1]))
        current_centers = np.random.randint(low_position, high_position + 1, size=current_peak_count)

        # Construct the signal with discrete peaks
        signal = np.zeros(sequence_length)
        for amp, pos in zip(current_amplitudes, current_centers):
            signal[pos] += amp  # If multiple peaks land on the same position, they sum

        # Add Gaussian noise if applicable
        if noise_std > 0:
            signal += np.random.normal(0, noise_std, sequence_length)

        # Optional normalization (mean=0, std=1)
        if normalize:
            std = np.std(signal)
            if std != 0:
                signal = (signal - np.mean(signal)) / std

        # Store results
        signals[i, :, 0] = signal
        amplitudes[i, :current_peak_count] = current_amplitudes
        positions[i, :current_peak_count] = current_centers
        # For point peaks, width is not meaningful, but you could store zeros if needed
        widths[i, :current_peak_count] = 0
        peak_counts[i] = current_peak_count

    # Sort peaks if requested
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

    # Convert peak_counts to one-hot encoding if requested
    if categorical_peak_count:
        peak_counts = to_categorical(peak_counts, max_peaks + 1)

    return signals, amplitudes, peak_counts, positions, widths, x_values

def generate_square_peak_dataset(
    sample_count: int,
    sequence_length: int,
    peak_count: tuple | int = (1, 5),
    amplitude_range: tuple | float = (1, 5),
    center_range: tuple | float = (0, 1),
    width_range: tuple | float = (5, 20),
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
        in the returned x_values array.
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
        low_position = max(0, int(center_range[0]))
        high_position = min(sequence_length - 1, int(center_range[1]))
        current_centers = np.random.randint(low_position, high_position + 1, size=current_peak_count)
        current_widths = np.random.uniform(width_range[0], width_range[1], size=current_peak_count).astype(int)

        signal = np.zeros(sequence_length)
        for amp, pos, width in zip(current_amplitudes, current_centers, current_widths):
            start = max(0, pos - width // 2)
            end = min(sequence_length, pos + width // 2)
            signal[start:end] += amp

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
