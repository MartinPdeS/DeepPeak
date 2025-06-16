from tensorflow.keras.utils import to_categorical  # type: ignore
from itertools import islice
import sklearn.model_selection as sk
import numpy as np
import pywt


def batched(iterable, n: int):  # Function is present in itertools for python 3.12+
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def dataset_split(test_size: float, random_state: float, **kwargs) -> dict:
    values = list(kwargs.values())

    splitted = sk.train_test_split(*values, test_size=test_size, random_state=random_state)

    output = {
        'train': dict(), 'test': dict()
    }

    for (k, v), (train_data, test_data) in zip(kwargs.items(), batched(splitted, 2)):
        output['train'][k] = train_data
        output['test'][k] = test_data

    return output


def filter_with_wavelet_transform(signals: np.ndarray, low_boundary: int, high_boundary: int, kernel: str = 'mexh') -> tuple[np.ndarray, np.ndarray]:
    """
    Efficient filtering of multiple signals using CWT, with minimal looping.

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_signals, n_samples).
    low_boundary : int
        Lower bound of the scale range.
    high_boundary : int
        Upper bound of the scale range.
    kernel : str, optional
        Name of the wavelet kernel. Default is 'mexh'.

    Returns
    -------
    filtered_signals : np.ndarray
        Filtered signals of shape (n_signals, n_samples).
    all_coeffs : np.ndarray
        All CWT coefficients of shape (n_signals, n_scales, n_samples).
    """
    signals = np.atleast_2d(signals)
    n_signals, n_samples = signals.shape

    # Define scales and mask
    all_scales = np.arange(1, 100)
    scale_mask = (all_scales >= low_boundary) & (all_scales <= high_boundary)
    n_selected_scales = np.sum(scale_mask)

    # Perform CWT individually (CWT must be done per-signal due to PyWavelets)
    coeffs_list = [pywt.cwt(signals[i], all_scales, kernel)[0] for i in range(n_signals)]
    coeffs = np.stack(coeffs_list, axis=0)  # Shape: (n_signals, n_scales, n_samples)

    # Apply mask and reconstruct using broadcasting
    filtered_coeffs = coeffs * scale_mask[:, np.newaxis]  # scale_mask: (n_scales, 1)
    filtered_signals = np.sum(filtered_coeffs, axis=1) / np.sqrt(n_selected_scales) / abs(high_boundary - low_boundary)

    return filtered_signals, coeffs