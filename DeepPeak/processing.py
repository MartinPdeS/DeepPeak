from typing import Literal, Optional
import numpy as np


def _as_float_array(adata: np.ndarray) -> np.ndarray:
    adata = np.asarray(adata)
    if adata.size == 0:
        raise ValueError("adata must be non empty.")
    if not np.issubdtype(adata.dtype, np.number):
        raise TypeError("adata must be a numeric array.")
    return adata


def _validate_sampling_rate(sampling_rate: float) -> float:
    sampling_rate = float(sampling_rate)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be a finite positive number.")
    return sampling_rate


def _validate_cutoff_hz(cutoff_hz: float, sampling_rate: float) -> float:
    cutoff_hz = float(cutoff_hz)
    nyquist_hz = 0.5 * sampling_rate
    if not np.isfinite(cutoff_hz) or cutoff_hz < 0.0:
        raise ValueError("bandlimit must be a finite number >= 0.")
    if cutoff_hz > nyquist_hz:
        raise ValueError(
            f"bandlimit must be <= Nyquist ({nyquist_hz:g} Hz) for sampling_rate={sampling_rate:g}."
        )
    return cutoff_hz


def _validate_transition_width_hz(transition_width_hz: float) -> float:
    transition_width_hz = float(transition_width_hz)
    if not np.isfinite(transition_width_hz) or transition_width_hz < 0.0:
        raise ValueError("transition_width_hz must be a finite number >= 0.")
    return transition_width_hz


def _cosine_taper_low_pass(
    frequencies_hz: np.ndarray,
    cutoff_hz: float,
    transition_width_hz: float,
) -> np.ndarray:
    """
    Magnitude response for a cosine tapered low pass filter.

    Response is:
      1 for f <= cutoff - w/2
      0 for f >= cutoff + w/2
      cosine taper in between
    """
    if transition_width_hz == 0.0:
        return (frequencies_hz <= cutoff_hz).astype(float)

    half_width_hz = 0.5 * transition_width_hz
    pass_edge_hz = max(0.0, cutoff_hz - half_width_hz)
    stop_edge_hz = cutoff_hz + half_width_hz

    response = np.zeros_like(frequencies_hz, dtype=float)

    pass_mask = frequencies_hz <= pass_edge_hz
    response[pass_mask] = 1.0

    transition_mask = (frequencies_hz > pass_edge_hz) & (frequencies_hz < stop_edge_hz)
    if np.any(transition_mask):
        x = (frequencies_hz[transition_mask] - pass_edge_hz) / (
            stop_edge_hz - pass_edge_hz
        )
        response[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * x))

    return response


def _cosine_taper_high_pass(
    frequencies_hz: np.ndarray,
    cutoff_hz: float,
    transition_width_hz: float,
) -> np.ndarray:
    """
    Magnitude response for a cosine tapered high pass filter.

    This is 1 minus the corresponding low pass response.
    """
    return 1.0 - _cosine_taper_low_pass(frequencies_hz, cutoff_hz, transition_width_hz)


def low_pass_filter(
    adata: np.ndarray,
    bandlimit: float = 1000.0,
    sampling_rate: float = 44100.0,
    *,
    axis: int = -1,
    transition_width_hz: float = 0.0,
    response_shape: Literal["brickwall", "cosine"] = "cosine",
    pad_to_length: Optional[int] = None,
    pad_mode: str = "constant",
    pad_constant_values: float = 0.0,
    return_complex: bool = False,
) -> np.ndarray:
    """
    Low pass filter a signal using frequency domain masking.

    This function performs an FFT along `axis`, applies a low pass magnitude response,
    and transforms back. For real valued input, it uses rFFT for efficiency.

    Important note on artifacts:
    A brick wall response in frequency corresponds to ringing in time (Gibbs effects).
    If you want less ringing, use a nonzero `transition_width_hz` with `response_shape="cosine"`.

    Parameters
    ----------
    adata
        Input signal array. Can be real or complex, and can be multidimensional.
    bandlimit
        Cutoff frequency in Hz. Frequencies above this cutoff are attenuated.
        Must be in [0, sampling_rate / 2].
    sampling_rate
        Sampling rate in Hz.
    axis
        Axis along which the signal is filtered.
    transition_width_hz
        Width of the transition region in Hz. If 0, the response can be brick wall.
        If > 0 and response_shape is "cosine", a cosine taper is used.
    response_shape
        "brickwall" applies an ideal step response (may ring strongly).
        "cosine" applies a cosine taper when transition_width_hz > 0, otherwise brick wall.
    pad_to_length
        If provided, pad the signal along `axis` to this length before filtering,
        then crop back to the original length after inverse transform.
        This can change frequency bin alignment and sometimes improves behavior.
    pad_mode
        Padding mode passed to np.pad (for example "constant", "reflect", "edge").
    pad_constant_values
        Constant padding value when pad_mode is "constant".
    return_complex
        If True, return the complex inverse FFT result.
        If False, return a real array when the input is real.

    Returns
    -------
    np.ndarray
        Filtered signal array with the same shape as `adata`.

    Raises
    ------
    ValueError
        If parameters are invalid (empty input, invalid sampling rate, cutoff out of range, etc.).
    """
    adata = _as_float_array(adata)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    cutoff_hz = _validate_cutoff_hz(bandlimit, sampling_rate)
    transition_width_hz = _validate_transition_width_hz(transition_width_hz)

    original_length = adata.shape[axis]
    if pad_to_length is not None:
        pad_to_length = int(pad_to_length)
        if pad_to_length < original_length:
            raise ValueError("pad_to_length must be >= the length of adata along axis.")
        pad_width = [(0, 0)] * adata.ndim
        pad_width[axis] = (0, pad_to_length - original_length)
        if pad_mode == "constant":
            adata_padded = np.pad(
                adata,
                pad_width=pad_width,
                mode=pad_mode,
                constant_values=pad_constant_values,
            )
        else:
            adata_padded = np.pad(adata, pad_width=pad_width, mode=pad_mode)
    else:
        adata_padded = adata

    filtered_length = adata_padded.shape[axis]
    is_real_input = np.isrealobj(adata_padded)

    if is_real_input:
        spectrum = np.fft.rfft(adata_padded, axis=axis)
        frequencies_hz = np.fft.rfftfreq(filtered_length, d=1.0 / sampling_rate)
    else:
        spectrum = np.fft.fft(adata_padded, axis=axis)
        frequencies_hz = np.fft.fftfreq(filtered_length, d=1.0 / sampling_rate)
        frequencies_hz = np.abs(frequencies_hz)

    if response_shape == "brickwall" or transition_width_hz == 0.0:
        magnitude_response = (frequencies_hz <= cutoff_hz).astype(float)
    elif response_shape == "cosine":
        magnitude_response = _cosine_taper_low_pass(
            frequencies_hz, cutoff_hz, transition_width_hz
        )
    else:
        raise ValueError('response_shape must be either "brickwall" or "cosine".')

    reshape_shape = [1] * spectrum.ndim
    reshape_shape[axis] = magnitude_response.shape[0]
    magnitude_response = magnitude_response.reshape(reshape_shape)

    spectrum_filtered = spectrum * magnitude_response

    if is_real_input:
        filtered = np.fft.irfft(spectrum_filtered, n=filtered_length, axis=axis)
        if pad_to_length is not None:
            slicer = [slice(None)] * filtered.ndim
            slicer[axis] = slice(0, original_length)
            filtered = filtered[tuple(slicer)]
        return (
            filtered.astype(adata.dtype, copy=False)
            if not return_complex
            else filtered.astype(np.complex128, copy=False)
        )

    filtered = np.fft.ifft(spectrum_filtered, axis=axis)
    if pad_to_length is not None:
        slicer = [slice(None)] * filtered.ndim
        slicer[axis] = slice(0, original_length)
        filtered = filtered[tuple(slicer)]
    if return_complex:
        return filtered
    return np.real(filtered)


def high_pass_filter(
    adata: np.ndarray,
    bandlimit: float = 1000.0,
    sampling_rate: float = 44100.0,
    *,
    axis: int = -1,
    transition_width_hz: float = 0.0,
    response_shape: Literal["brickwall", "cosine"] = "cosine",
    pad_to_length: Optional[int] = None,
    pad_mode: str = "constant",
    pad_constant_values: float = 0.0,
    return_complex: bool = False,
) -> np.ndarray:
    """
    High pass filter a signal using frequency domain masking.

    This function performs an FFT along `axis`, applies a high pass magnitude response,
    and transforms back. For real valued input, it uses rFFT for efficiency.

    Parameters
    ----------
    adata
        Input signal array. Can be real or complex, and can be multidimensional.
    bandlimit
        Cutoff frequency in Hz. Frequencies below this cutoff are attenuated.
        Must be in [0, sampling_rate / 2].
    sampling_rate
        Sampling rate in Hz.
    axis
        Axis along which the signal is filtered.
    transition_width_hz
        Width of the transition region in Hz. If 0, the response can be brick wall.
        If > 0 and response_shape is "cosine", a cosine taper is used.
    response_shape
        "brickwall" applies an ideal step response (may ring strongly).
        "cosine" applies a cosine taper when transition_width_hz > 0, otherwise brick wall.
    pad_to_length
        If provided, pad the signal along `axis` to this length before filtering,
        then crop back to the original length after inverse transform.
    pad_mode
        Padding mode passed to np.pad (for example "constant", "reflect", "edge").
    pad_constant_values
        Constant padding value when pad_mode is "constant".
    return_complex
        If True, return the complex inverse FFT result.
        If False, return a real array when the input is real.

    Returns
    -------
    np.ndarray
        Filtered signal array with the same shape as `adata`.
    """
    adata = _as_float_array(adata)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    cutoff_hz = _validate_cutoff_hz(bandlimit, sampling_rate)
    transition_width_hz = _validate_transition_width_hz(transition_width_hz)

    original_length = adata.shape[axis]
    if pad_to_length is not None:
        pad_to_length = int(pad_to_length)
        if pad_to_length < original_length:
            raise ValueError("pad_to_length must be >= the length of adata along axis.")
        pad_width = [(0, 0)] * adata.ndim
        pad_width[axis] = (0, pad_to_length - original_length)
        if pad_mode == "constant":
            adata_padded = np.pad(
                adata,
                pad_width=pad_width,
                mode=pad_mode,
                constant_values=pad_constant_values,
            )
        else:
            adata_padded = np.pad(adata, pad_width=pad_width, mode=pad_mode)
    else:
        adata_padded = adata

    filtered_length = adata_padded.shape[axis]
    is_real_input = np.isrealobj(adata_padded)

    if is_real_input:
        spectrum = np.fft.rfft(adata_padded, axis=axis)
        frequencies_hz = np.fft.rfftfreq(filtered_length, d=1.0 / sampling_rate)
    else:
        spectrum = np.fft.fft(adata_padded, axis=axis)
        frequencies_hz = np.fft.fftfreq(filtered_length, d=1.0 / sampling_rate)
        frequencies_hz = np.abs(frequencies_hz)

    if response_shape == "brickwall" or transition_width_hz == 0.0:
        magnitude_response = (frequencies_hz >= cutoff_hz).astype(float)
    elif response_shape == "cosine":
        magnitude_response = _cosine_taper_high_pass(
            frequencies_hz, cutoff_hz, transition_width_hz
        )
    else:
        raise ValueError('response_shape must be either "brickwall" or "cosine".')

    reshape_shape = [1] * spectrum.ndim
    reshape_shape[axis] = magnitude_response.shape[0]
    magnitude_response = magnitude_response.reshape(reshape_shape)

    spectrum_filtered = spectrum * magnitude_response

    if is_real_input:
        filtered = np.fft.irfft(spectrum_filtered, n=filtered_length, axis=axis)
        if pad_to_length is not None:
            slicer = [slice(None)] * filtered.ndim
            slicer[axis] = slice(0, original_length)
            filtered = filtered[tuple(slicer)]
        return (
            filtered.astype(adata.dtype, copy=False)
            if not return_complex
            else filtered.astype(np.complex128, copy=False)
        )

    filtered = np.fft.ifft(spectrum_filtered, axis=axis)
    if pad_to_length is not None:
        slicer = [slice(None)] * filtered.ndim
        slicer[axis] = slice(0, original_length)
        filtered = filtered[tuple(slicer)]
    if return_complex:
        return filtered
    return np.real(filtered)


def normalize_signal(
    signals, normalization: str = "zscore", axis: int = 1
) -> np.ndarray:
    """
    Normalize dataset signals on given axis.

    Supported normalization modes
    -----------------------------
    "none"
        Return a float copy of the signals.
    "l1"
        Divide each signal by its L1 norm (sum of absolute values).
    "l2"
        Divide each signal by its L2 norm.
    "minmax"
        Map each signal to [0, 1] using per signal min and max.
    "zscore"
        Per signal standardization: (x - mean) / std.
    "robust_zscore"
        Per signal robust standardization: (x - median) / (1.4826 * MAD).
    "maxabs"
        Divide each signal by its max absolute value.

    Notes
    -----
    - "minmax" guarantees an output in [0, 1] (per signal).
    - "zscore" and "robust_zscore" are usually better for neural network training.
    """
    signals = np.asarray(signals, dtype=np.float32)
    epsilon = 1e-8

    mode = normalization.lower().strip()

    if mode in {"none", "raw"}:
        return signals

    if mode == "l1":
        l1_norm = np.sum(np.abs(signals), axis=1, keepdims=True)
        return signals / (l1_norm + epsilon)

    if mode == "l2":
        l2_norm = np.linalg.norm(signals, axis=1, keepdims=True)
        return signals / (l2_norm + epsilon)

    if mode in {"minmax", "min-max"}:
        min_vals = np.min(signals, axis=1, keepdims=True)
        max_vals = np.max(signals, axis=1, keepdims=True)
        return (signals - min_vals) / (max_vals - min_vals + epsilon)

    if mode in {"zscore", "z-score", "standard", "standardize"}:
        mean_vals = np.mean(signals, axis=1, keepdims=True)
        std_vals = np.std(signals, axis=1, keepdims=True)
        return (signals - mean_vals) / (std_vals + epsilon)

    if mode in {"robust_zscore", "robust-zscore", "robust"}:
        median_vals = np.median(signals, axis=1, keepdims=True)
        mad_vals = np.median(np.abs(signals - median_vals), axis=1, keepdims=True)
        robust_std = 1.4826 * mad_vals
        return (signals - median_vals) / (robust_std + epsilon)

    if mode in {"maxabs", "max_abs", "max-abs"}:
        max_abs_vals = np.max(np.abs(signals), axis=1, keepdims=True)
        return signals / (max_abs_vals + epsilon)

    raise ValueError(
        f"Unknown normalization method: {normalization}. "
        "Use one of: none, l1, l2, minmax, zscore, robust_zscore, maxabs."
    )
