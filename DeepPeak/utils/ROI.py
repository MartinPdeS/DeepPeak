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

def compute_segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """
    Compute segmentation metrics between the predicted and ground truth ROI masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary ROI mask. Expected shape is (n_samples, sequence_length) (or any shape,
        as long as it matches true_mask).
    true_mask : np.ndarray
        Ground truth binary ROI mask. Must have the same shape as pred_mask.

    Returns
    -------
    metrics : dict
        Dictionary with the following keys:
          - "precision": The precision (positive predictive value).
          - "recall": The recall (sensitivity).
          - "f1_score": The harmonic mean of precision and recall.
          - "iou": Intersection-over-Union metric.
          - "dice": Dice coefficient.
    """
    # Flatten the arrays so that each pixel is a sample
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    # Compute precision, recall, and F1 score.
    precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='binary')

    # Compute Intersection over Union (IoU)
    intersection = np.logical_and(true_flat, pred_flat).sum()
    union = np.logical_or(true_flat, pred_flat).sum()
    iou = intersection / union if union > 0 else 0.0

    # Compute Dice coefficient: (2 * Intersection) / (Total area of both masks)
    dice = (2 * intersection) / (true_flat.sum() + pred_flat.sum()) if (true_flat.sum() + pred_flat.sum()) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "dice": dice
    }