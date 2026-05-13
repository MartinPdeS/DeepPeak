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


def segment_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Segment a 1D signal into non-overlapping windows, zero-padding the tail if needed.
    """
    signal = np.asarray(signal)
    if int(window_size) <= 0:
        raise ValueError("window_size must be a strictly positive integer.")

    flattened_signal = signal.reshape(-1)
    window_size = int(window_size)
    number_of_windows = int(np.ceil(flattened_signal.size / window_size))
    padded_size = number_of_windows * window_size

    padded_signal = np.zeros(padded_size, dtype=flattened_signal.dtype)
    padded_signal[: flattened_signal.size] = flattened_signal

    return padded_signal.reshape(number_of_windows, window_size)


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
