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


def filter_with_wavelet_transform(signal: np.ndarray, low_boundary: int, high_boundary: int, kernel: str = 'mexh') -> np.ndarray:
    """
    Filters a signal using continuous wavelet transform (CWT) with a Mexican hat kernel.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to be filtered.
    low_boundary : int
        The lower boundary of the scale range to keep.
    high_boundary : int
        The upper boundary of the scale range to keep.
    kernel : str, optional
        The wavelet kernel to use for the CWT. Default is 'mexh' (Mexican hat).

    Returns
    -------
    np.ndarray
        The filtered signal reconstructed from the specified scale range.

    Notes
    -----
    This function performs a continuous wavelet transform on the input signal using the specified kernel,
    filters the coefficients to keep only those within the specified scale range, and reconstructs the signal
    from the filtered coefficients. The Mexican hat wavelet is commonly used for this purpose.
    The function assumes that the input signal is a 1D numpy array.
    """

    # Define all scales and the desired scale range for filtering
    all_scales = np.arange(1, 100)

    # Perform continuous wavelet transform (CWT) with Mexican hat
    coefficients, _ = pywt.cwt(signal, all_scales, 'mexh')

    # Zero-out unwanted scales
    filtered_coeffs = np.zeros_like(coefficients)
    scale_mask = (all_scales >= low_boundary) & (all_scales <= high_boundary)
    filtered_coeffs[scale_mask, :] = coefficients[scale_mask, :]

    # Approximate signal reconstruction from selected scales
    reconstructed_signal = np.sum(filtered_coeffs, axis=0) / np.sqrt(np.sum(scale_mask)) / abs(high_boundary - low_boundary)

    return reconstructed_signal