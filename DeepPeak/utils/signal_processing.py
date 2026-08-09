from typing import Literal

import numpy as np

from DeepPeak.processing import low_pass_filter


def filter_with_wavelet_transform(
    signals: np.ndarray,
    low_boundary: int,
    high_boundary: int,
    kernel: str = "mexh",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter signals by masking CWT scales in a selected range.
    """
    try:
        import pywt  # type: ignore
    except ImportError as error:
        raise ImportError(
            "filter_with_wavelet_transform requires the optional 'PyWavelets' dependency."
        ) from error

    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    if low_boundary < 1 or high_boundary < low_boundary:
        raise ValueError("Expected 1 <= low_boundary <= high_boundary.")

    all_scales = np.arange(1, 100)
    scale_mask = (all_scales >= low_boundary) & (all_scales <= high_boundary)
    number_of_selected_scales = int(np.sum(scale_mask))
    if number_of_selected_scales == 0:
        raise ValueError(
            "The requested scale range does not overlap the available scales."
        )

    coefficients = np.stack(
        [pywt.cwt(signal, all_scales, kernel)[0] for signal in signals],
        axis=0,
    )
    filtered_coefficients = coefficients * scale_mask[None, :, None]
    filtered_signals = np.sum(filtered_coefficients, axis=1) / np.sqrt(
        number_of_selected_scales
    )

    return filtered_signals, coefficients


def robust_sigma_from_diff(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")

    diff_signal = np.diff(signal)
    median_absolute_deviation = np.median(np.abs(diff_signal - np.median(diff_signal)))
    sigma_diff = 1.4826 * median_absolute_deviation
    return float(sigma_diff / np.sqrt(2.0))


def segment_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Segment a 1D signal into fixed-length windows.

    When ``stride`` is ``None`` (default) the signal is split into
    **non-overlapping** windows and the last window is zero-padded to fill
    ``window_size``.  The return value is a 2-D array of shape
    ``(n_windows, window_size)`` — identical to the original behaviour.

    When ``stride`` is provided the windows **overlap** and no zero-padding is
    applied; only complete windows are returned.  The return value is a tuple
    ``(windows, start_indices)`` where ``start_indices`` is a 1-D integer array
    containing the sample index at which each window begins.  Use
    ``stride = window_size // 2`` for 50 % overlap, which ensures every sample
    (except the very first and last ``window_size // 2`` samples) is covered by
    at least two windows.

    Parameters
    ----------
    signal : array-like
        1-D input signal.
    window_size : int
        Number of samples in each window.  Must be strictly positive.
    stride : int, optional
        Step between consecutive window starts.  Must satisfy
        ``1 <= stride <= window_size``.  When *None* the non-overlapping
        (stride = window_size) mode is used and the original return type is
        preserved.

    Returns
    -------
    windows : ndarray, shape (n_windows, window_size)
        Stacked windows.
    start_indices : ndarray of int, shape (n_windows,)
        Only returned when *stride* is not ``None``.  The sample offset of each
        window within the original signal.
    """
    signal = np.asarray(signal)
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be a strictly positive integer.")

    flattened = signal.reshape(-1)

    if stride is None:
        n_windows = int(np.ceil(flattened.size / window_size))
        padded = np.zeros(n_windows * window_size, dtype=flattened.dtype)
        padded[: flattened.size] = flattened
        return padded.reshape(n_windows, window_size)

    stride = int(stride)
    if not (1 <= stride <= window_size):
        raise ValueError(
            f"stride must satisfy 1 <= stride <= window_size, got stride={stride}."
        )

    starts = np.arange(0, flattened.size - window_size + 1, stride)
    windows = np.stack([flattened[s : s + window_size] for s in starts])
    return windows, starts.astype(np.intp)


def get_normalized_signal(
    signals: np.ndarray,
    normalization: Literal["l1", "l2", "min-max"] = "l1",
) -> np.ndarray:
    """
    Normalize a batch of signals along axis 1.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    eps = 1e-8
    normalization = normalization.lower()

    if normalization == "l1":
        scale = np.sum(np.abs(signals), axis=1, keepdims=True)
        return signals / (scale + eps)

    if normalization == "l2":
        scale = np.linalg.norm(signals, axis=1, keepdims=True)
        return signals / (scale + eps)

    if normalization == "min-max":
        min_values = np.min(signals, axis=1, keepdims=True)
        max_values = np.max(signals, axis=1, keepdims=True)
        return (signals - min_values) / (max_values - min_values + eps)

    raise ValueError(f"Unknown normalization method: {normalization}")


def process_signal(data: object, sequence_length: int) -> np.ndarray:
    """
    Normalize `data.y_processed` and segment it into fixed-length windows.
    """
    signal = np.asarray(data.y_processed, dtype=float).reshape(1, -1)
    normalized_signal = get_normalized_signal(signal, normalization="min-max")
    return segment_signal(normalized_signal.ravel(), sequence_length)
