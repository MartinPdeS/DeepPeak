import numpy as np

def compute_rois_from_signals(
    signals: np.ndarray,
    positions: np.ndarray,
    amplitudes: np.ndarray,
    width_in_pixels: int) -> np.ndarray:
    """
    Compute a binary ROI mask of the same shape as the input signals, using
    the given peak positions and a width specified in pixels.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape (n_samples, sequence_length). Each row represents a 1D signal.
    positions : np.ndarray
        Array of shape (n_samples, n_peaks). Each entry is a floating-point
        position of a peak in [0,1] range, or 0 if no peak exists.
    width_in_pixels : int
        The width around each peak (in pixels) to mark as ROI. The function will
        mark ±(width_in_pixels // 2) around each peak center.

    Returns
    -------
    rois : np.ndarray
        Binary array of shape (n_samples, sequence_length), with the same shape
        as `signals`. Entries are 1 where the signal is considered within the
        region of interest for a peak, and 0 otherwise.
    """
    n_samples, sequence_length = signals.shape
    _, n_peaks = positions.shape

    # Prepare the output array
    rois = np.zeros_like(signals, dtype=np.int32)  # (n_samples, sequence_length)

    # Convert normalized peak positions [0, 1] to pixel indices [0, sequence_length - 1]
    pixel_positions = (positions * (sequence_length - 1)).astype(int)

    # Half-width in pixels around each peak
    half_w = width_in_pixels // 2

    for i in range(n_samples):
        for j in range(n_peaks):
            center_idx = pixel_positions[i, j]
            amp = amplitudes[i, j]
            if amp == 0:
                continue

            # Skip if no peak (e.g., amplitude was zero -> pos might be 0 or outside range)
            # If your data uses 0 for "no peak," you may need additional checks.
            if center_idx < 0 or center_idx > (sequence_length - 1):
                continue

            # Compute the start and end indices for the region of interest
            start_idx = max(0, center_idx - half_w)
            end_idx   = min(sequence_length, center_idx + half_w + 1)

            rois[i, start_idx:end_idx] = 1

    return rois

def get_positions_amplitudes(signals, ROIs):
    """
    Compute positions and amplitudes for ROIs in a fully vectorized way.

    Parameters
    ----------
    signals : np.ndarray
        Array of signals, shape (n_signals, sequence_length).
    ROIs : np.ndarray
        Array of binary ROIs, shape (n_signals, sequence_length, 1).

    Returns
    -------
    positions : np.ndarray
        Padded array containing middle indices of segments for each signal, shape (n_signals, max_segments).
    amplitudes : np.ndarray
        Padded array containing amplitudes at the middle indices for each signal, shape (n_signals, max_segments).
    """
    # Squeeze ROIs and signals to remove unnecessary dimensions
    signals = signals.squeeze(-1) if signals.ndim == 3 else signals

    # Compute where segments start and end
    changes = np.diff(ROIs, prepend=0, append=0, axis=1)
    start_indices = np.where(changes == 1)
    end_indices = np.where(changes == -1)

    # Compute middle indices for all segments
    middle_indices = (start_indices[1] + end_indices[1] - 1) // 2

    # Map middle indices back to their corresponding signal indices
    signal_indices = start_indices[0]

    # Explicitly limit max_segments to 3
    n_signals = signals.shape[0]
    max_segments = min(3, np.max(np.bincount(signal_indices)))  # Clamp max_segments to 3
    positions = np.full((n_signals, max_segments), 0)
    amplitudes = np.full((n_signals, max_segments), 0)

    # Populate positions and amplitudes
    for i in range(n_signals):
        mask = signal_indices == i
        segment_indices = middle_indices[mask][:max_segments]  # Limit to first 3 segments
        positions[i, :len(segment_indices)] = segment_indices
        amplitudes[i, :len(segment_indices)] = signals[i, segment_indices]  # Correct broadcasting here

    return positions, amplitudes