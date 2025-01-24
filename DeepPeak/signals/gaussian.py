from typing import Tuple
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

def generate_gaussian_dataset(
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
    probability_range: tuple = (1.0, 1.0)) -> Tuple:
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
    probability_range : tuple of float
        Defines the range multiplier for acceptable widths (e.g., (0.5, 1.5)).

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
    labels : numpy.ndarray
        Binary mask labels for peak regions, shape (sample_count, sequence_length, 1).
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
    labels = np.zeros_like(signals)

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

        # Generate labels based on probability_range
        for pos, width in zip(current_centers, current_widths):
            if not np.isnan(pos):
                adjusted_width = width * np.random.uniform(probability_range[0], probability_range[1])
                start = max(0, int((pos - adjusted_width / 2) * sequence_length))
                end = min(sequence_length, int((pos + adjusted_width / 2) * sequence_length))
                labels[i, start:end, 0] = 1

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

    return signals, amplitudes, peak_counts, positions, widths, x_values, labels
