import numpy as np
from numpy.typing import NDArray


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
