import numpy as np
from numpy.typing import NDArray
from typing import Optional


def _find_local_maxima(signal: np.ndarray) -> NDArray[np.int64]:
    """Return indices of strict local maxima (signal[i] > both neighbours)."""
    signal = np.asarray(signal, dtype=float)
    if signal.size < 3:
        return np.array([], dtype=np.int64)
    mask = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    return (np.where(mask)[0] + 1).astype(np.int64)


def _compute_peak_prominence(
    signal: np.ndarray,
    peaks: np.ndarray,
    wlen: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute prominence of each peak. Pure NumPy — no external dependencies.

    For each peak the prominence is the peak height minus the highest
    saddle (key col) connecting it to any strictly higher neighbour.
    When no higher neighbour exists within the search window, the signal
    boundary is used as the reference.

    Parameters
    ----------
    signal : array-like
        1D signal (already oriented so peaks are maxima).
    peaks : array-like of int
        Sorted peak indices into *signal*.
    wlen : int, optional
        Half-window (in samples) used to bound the search for each peak.
        When ``None`` the entire signal is searched.

    Returns
    -------
    numpy.ndarray
        Prominence value for every peak, in the same order as *peaks*.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    peaks = np.asarray(peaks, dtype=int)

    if peaks.size == 0:
        return np.array([], dtype=float)

    n = signal.size
    prominences = np.empty(peaks.size, dtype=float)

    for i, peak in enumerate(peaks):
        peak_height = float(signal[peak])

        if wlen is not None:
            half = max(1, int(wlen) // 2)
            left_bound = max(0, peak - half)
            right_bound = min(n - 1, peak + half)
        else:
            left_bound = 0
            right_bound = n - 1

        # Left base: scan left until a strictly higher sample or the boundary.
        left_stop = left_bound
        for j in range(peak - 1, left_bound - 1, -1):
            if signal[j] > peak_height:
                left_stop = j + 1
                break
        left_base = (
            float(np.min(signal[left_stop:peak])) if left_stop < peak else peak_height
        )

        # Right base: scan right until a strictly higher sample or the boundary.
        right_stop = right_bound
        for j in range(peak + 1, right_bound + 1):
            if signal[j] > peak_height:
                right_stop = j - 1
                break
        right_base = (
            float(np.min(signal[peak + 1 : right_stop + 1]))
            if peak + 1 <= right_stop
            else peak_height
        )

        prominences[i] = peak_height - max(left_base, right_base)

    return prominences


def find_peaks_prominence(
    signal_values: NDArray[np.floating],
    min_prominence: float,
    wlen: Optional[int] = None,
    *,
    pulse_polarity: str = "positive",
    holdoff_samples: int = 0,
) -> tuple[NDArray[np.int64], dict]:
    """Detect peaks by prominence threshold. No absolute height threshold needed.

    All strict local maxima are found first, then filtered to those whose
    prominence is at least *min_prominence*. An optional *holdoff_samples*
    guard ensures a minimum spacing between consecutive accepted peaks
    (sequentially: the first accepted peak blocks the next
    *holdoff_samples* samples, mirroring the behaviour of
    :func:`find_peaks_standard`).

    Parameters
    ----------
    signal_values : array-like
        1D signal to search.
    min_prominence : float
        Minimum prominence required to accept a peak.
    wlen : int, optional
        Window half-length for local prominence computation. When ``None``
        the full signal is used as the reference for every peak.
    pulse_polarity : {"positive", "negative"}, default="positive"
        Flip the signal before detection when ``"negative"``.
    holdoff_samples : int, default=0
        Minimum samples between two consecutive accepted peaks.

    Returns
    -------
    peak_indices : numpy.ndarray of int
    properties : dict
        Same keys as :func:`find_peaks_standard`:
        ``start_indices``, ``end_indices``, ``peak_values``,
        ``widths_pixels``.
    """
    signal_values = np.asarray(signal_values, dtype=float).ravel()

    empty: tuple[NDArray[np.int64], dict] = (
        np.array([], dtype=np.int64),
        {
            "start_indices": np.array([], dtype=np.int64),
            "end_indices": np.array([], dtype=np.int64),
            "peak_values": np.array([], dtype=float),
            "widths_pixels": np.array([], dtype=float),
        },
    )

    if signal_values.size < 3:
        return empty

    if pulse_polarity == "positive":
        working = signal_values
    elif pulse_polarity == "negative":
        working = -signal_values
    else:
        raise ValueError("pulse_polarity must be 'positive' or 'negative'.")

    candidates = _find_local_maxima(working)
    if candidates.size == 0:
        return empty

    prominences = _compute_peak_prominence(working, candidates, wlen=wlen)
    mask = prominences >= float(min_prominence)
    accepted = candidates[mask]

    if accepted.size == 0:
        return empty

    # Sequential holdoff: scan left-to-right, block the next holdoff_samples
    # after each accepted peak (same semantics as find_peaks_standard).
    if holdoff_samples > 0:
        kept: list[int] = []
        ignore_until = 0
        for pk in accepted:
            if int(pk) < ignore_until:
                continue
            kept.append(int(pk))
            ignore_until = int(pk) + 1 + holdoff_samples
        accepted = np.asarray(kept, dtype=np.int64)

    if accepted.size == 0:
        return empty

    # Build start / end indices as the nearest trough on each side of the peak.
    n = signal_values.size
    start_indices = np.empty(accepted.size, dtype=np.int64)
    end_indices = np.empty(accepted.size, dtype=np.int64)

    for k, p in enumerate(accepted):
        # Walk left while the signal is still descending from the peak.
        s = p
        for j in range(p - 1, -1, -1):
            if working[j] >= working[s]:
                break
            s = j
        start_indices[k] = s

        # Walk right while the signal is still descending from the peak.
        e = p
        for j in range(p + 1, n):
            if working[j] >= working[e]:
                break
            e = j
        end_indices[k] = e

    return accepted.astype(np.int64), {
        "start_indices": start_indices,
        "end_indices": end_indices,
        "peak_values": signal_values[accepted],
        "widths_pixels": (end_indices - start_indices + 1).astype(float),
    }


def find_peaks_standard(
    signal_values: NDArray[np.floating],
    height: float,
    hysteresis: float | None = None,
    *,
    pulse_polarity: str = "positive",
    holdoff_samples: int = 0,
    required_samples_above_threshold: int = 1,
    required_samples_below_hysteresis: int = 1,
) -> tuple[NDArray[np.int64], dict]:
    """
    Simple flow cytometer style trigger with optional hysteresis plus debounce.

    If hysteresis is None, the end level is taken to be threshold (no amplitude hysteresis).
    In that case, the event ends when the signal is below threshold for
    required_samples_below_hysteresis samples.

    Returns
    -------
    peak_indices, properties

    properties keys
        start_indices
        end_indices
        peak_values
        widths_pixels
    """
    signal_values = np.asarray(signal_values, dtype=float).ravel()
    if signal_values.size == 0:
        return np.array([], dtype=np.int64), {
            "start_indices": np.array([], dtype=np.int64),
            "end_indices": np.array([], dtype=np.int64),
            "peak_values": np.array([], dtype=float),
            "widths_pixels": np.array([], dtype=float),
        }

    threshold = float(height)
    end_level = threshold if hysteresis is None else float(hysteresis)

    holdoff_samples = int(holdoff_samples)
    required_samples_above_threshold = int(required_samples_above_threshold)
    required_samples_below_hysteresis = int(required_samples_below_hysteresis)

    if required_samples_above_threshold < 1:
        raise ValueError("required_samples_above_threshold must be >= 1")
    if required_samples_below_hysteresis < 1:
        raise ValueError("required_samples_below_hysteresis must be >= 1")
    if holdoff_samples < 0:
        raise ValueError("holdoff_samples must be >= 0")

    if end_level > threshold:
        raise ValueError("hysteresis must be <= threshold (or None)")

    if pulse_polarity == "positive":
        working_signal = signal_values
    elif pulse_polarity == "negative":
        working_signal = -signal_values
    else:
        raise ValueError("pulse_polarity must be 'positive' or 'negative'")

    start_indices: list[int] = []
    end_indices: list[int] = []
    peak_indices: list[int] = []
    peak_values: list[float] = []
    widths_pixels: list[float] = []

    in_event = False
    ignore_until_index = 0

    consecutive_above_threshold = 0
    consecutive_below_end_level = 0

    event_start_index = 0
    event_peak_index = 0
    event_peak_value_working = -np.inf

    number_of_samples = int(working_signal.size)

    for sample_index in range(number_of_samples):
        if sample_index < ignore_until_index:
            continue

        current_value_working = float(working_signal[sample_index])

        if not in_event:
            if current_value_working >= threshold:
                consecutive_above_threshold += 1
            else:
                consecutive_above_threshold = 0

            if consecutive_above_threshold >= required_samples_above_threshold:
                start_candidate_index = (
                    sample_index - required_samples_above_threshold + 1
                )
                in_event = True
                consecutive_below_end_level = 0

                event_start_index = int(start_candidate_index)
                event_peak_index = int(sample_index)
                event_peak_value_working = float(current_value_working)

            continue

        if current_value_working > event_peak_value_working:
            event_peak_value_working = float(current_value_working)
            event_peak_index = int(sample_index)

        if current_value_working <= end_level:
            consecutive_below_end_level += 1
        else:
            consecutive_below_end_level = 0

        if consecutive_below_end_level >= required_samples_below_hysteresis:
            end_candidate_index = sample_index - required_samples_below_hysteresis + 1
            event_end_index = int(end_candidate_index)

            start_indices.append(event_start_index)
            end_indices.append(event_end_index)
            peak_indices.append(event_peak_index)
            peak_values.append(float(signal_values[event_peak_index]))
            widths_pixels.append(float(event_end_index - event_start_index + 1))

            in_event = False
            consecutive_above_threshold = 0
            consecutive_below_end_level = 0

            ignore_until_index = event_end_index + 1 + holdoff_samples

    return np.asarray(peak_indices, dtype=np.int64), {
        "start_indices": np.asarray(start_indices, dtype=np.int64),
        "end_indices": np.asarray(end_indices, dtype=np.int64),
        "peak_values": np.asarray(peak_values, dtype=float),
        "widths_pixels": np.asarray(widths_pixels, dtype=float),
    }
