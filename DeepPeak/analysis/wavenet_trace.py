"""Single-trace helpers for processed-signal and WaveNet output analysis.

The functions and classes in this module turn one processed trace into a
standardized :class:`TraceRecord` that can later be aggregated across a dilution
series or visualized with the plotting helpers.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from DeepPeak import processing, utils
from DeepPeak.algorithms.peak_locator import find_peaks_standard
from DeepPeak.analysis.trace_io import CsvTrace

from .results import PeakDetectionResult, TraceRecord, WaveNetAnalyzerConfig
from .triggers import PeakTrigger, TriggerLike, coerce_peak_trigger


def _infer_sequence_length(
    wavenet: Any,
    sequence_length: Optional[int] = None,
) -> int:
    """Resolve the expected model window length from explicit input or the model.

    Parameters
    ----------
    wavenet : object
        Model wrapper or model-like object.
    sequence_length : int, optional
        Explicit sequence length override.

    Returns
    -------
    int
        Positive window length expected by the model.
    """

    if sequence_length is not None:
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a strictly positive integer.")
        return sequence_length

    candidate = getattr(wavenet, "sequence_length", None)
    if candidate is not None:
        candidate = int(candidate)
        if candidate > 0:
            return candidate

    model = getattr(wavenet, "model", None)
    input_shape = getattr(model, "input_shape", None)
    if input_shape is not None and len(input_shape) >= 2 and input_shape[1] is not None:
        candidate = int(input_shape[1])
        if candidate > 0:
            return candidate

    raise ValueError(
        "Unable to infer the WaveNet sequence length. Provide sequence_length explicitly "
        "or expose `wavenet.sequence_length`."
    )


def _coerce_signal_batch(
    signal: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Convert a trace into a 2D batch of windows expected by the model.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal as either one 1D trace or a 2D batch of windows.
    sequence_length : int
        Expected width of each model input window.

    Returns
    -------
    numpy.ndarray
        Two-dimensional batch of windows with shape ``(n_windows, sequence_length)``.
    """

    signal = np.asarray(signal, dtype=float)

    if signal.ndim == 1:
        return utils.segment_signal(signal, window_size=sequence_length)

    if signal.ndim != 2:
        raise ValueError("signal must be either a 1D trace or a 2D batch of windows.")

    if signal.shape[1] != int(sequence_length):
        raise ValueError(
            f"Expected signal windows of length {sequence_length}, got {signal.shape[1]}."
        )

    return signal


class WaveNetTraceAnalyzer:
    """
    Analyze one processed trace and its WaveNet prediction.

    The analyzer owns the shared preprocessing, model invocation, and peak
    detection configuration used for one experiment. Its public methods provide
    small reusable steps for notebooks as well as an end-to-end
    ``analyze_processed_signal`` convenience method.

    Parameters
    ----------
    wavenet : object
        Trained model exposing a ``predict(signal=...)`` interface.
    std_trigger, cnn_trigger : PeakTrigger or mapping, optional
        Preferred trigger configurations for the standard and WaveNet-based detectors.
    std_kwargs, cnn_kwargs : mapping, optional
        Legacy dictionary-based trigger configurations kept for compatibility.
    sequence_length : int, optional
        Explicit model sequence length override.
    signal_normalization : str, default="zscore"
        Normalization strategy passed to :func:`DeepPeak.processing.normalize_signal`.
    prediction_sampling_rate_hz : float, default=125_000_000.0
        Sampling rate used when low-pass filtering the prediction.
    """

    def __init__(
        self,
        wavenet: Any,
        std_kwargs: Optional[Mapping[str, Any]] = None,
        cnn_kwargs: Optional[Mapping[str, Any]] = None,
        *,
        std_trigger: Optional[TriggerLike] = None,
        cnn_trigger: Optional[TriggerLike] = None,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
    ) -> None:
        self.wavenet = wavenet
        self.std_trigger = coerce_peak_trigger(
            std_trigger, std_kwargs, name="std_trigger"
        )
        self.cnn_trigger = coerce_peak_trigger(
            cnn_trigger, cnn_kwargs, name="cnn_trigger"
        )
        self.std_kwargs = self.std_trigger.to_kwargs()
        self.cnn_kwargs = self.cnn_trigger.to_kwargs()
        self.config = WaveNetAnalyzerConfig(
            sequence_length=_infer_sequence_length(
                wavenet, sequence_length=sequence_length
            ),
            signal_normalization=str(signal_normalization),
            prediction_sampling_rate_hz=float(prediction_sampling_rate_hz),
        )

    def load_processed_signal(
        self,
        filename: Path,
        *,
        nrows: int,
        low_pass: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Load one CSV trace and convert it to segmented processed windows.

        Returns the processed signal batch together with the original sampling
        interval ``dx``.

        Parameters
        ----------
        filename : pathlib.Path
            Trace file to load.
        nrows : int
            Maximum number of rows to load from the trace file.
        low_pass : float, optional
            Optional low-pass frequency applied before signal processing.

        Returns
        -------
        signal : numpy.ndarray
            Processed signal windows.
        dx : float
            Sampling interval of the source trace.
        """

        data = CsvTrace(filename=filename, n_rows=nrows)

        if low_pass is not None:
            data.low_pass_filter(low_pass)

        signal = utils.process_signal(data, sequence_length=self.config.sequence_length)
        return np.asarray(signal, dtype=float), float(data.dx)

    def prepare_model_input(self, signal: np.ndarray) -> np.ndarray:
        """Normalize processed windows into the format expected by the WaveNet.

        Parameters
        ----------
        signal : numpy.ndarray
            Processed signal as a 1D trace or 2D batch.

        Returns
        -------
        numpy.ndarray
            Normalized 2D batch ready for model inference.
        """

        signal_batch = _coerce_signal_batch(signal, self.config.sequence_length)
        return processing.normalize_signal(
            signals=signal_batch,
            normalization=self.config.signal_normalization,
            axis=1,
        )

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run the WaveNet model on a normalized batch and return its prediction.

        Parameters
        ----------
        signal : numpy.ndarray
            Normalized model input batch.

        Returns
        -------
        numpy.ndarray
            Raw model prediction converted to a NumPy array.
        """

        prediction = self.wavenet.predict(signal=signal)
        return np.asarray(prediction, dtype=float)

    def _resolve_detection_kwargs(
        self,
        values: np.ndarray,
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        """Resolve threshold-related detection settings into explicit kwargs.

        Parameters
        ----------
        values : numpy.ndarray
            Signal values used to compute sigma-derived thresholds.
        kwargs : dict
            Detector keyword arguments, possibly containing ``sigma``.

        Returns
        -------
        resolved_kwargs : dict
            Detector kwargs with explicit absolute thresholds when they can be
            inferred.
        threshold : float or None
            Effective scalar detection threshold.
        """

        resolved_kwargs = dict(kwargs)
        threshold = None
        sigma = resolved_kwargs.pop("sigma", None)
        hysteresis = resolved_kwargs.get("hysteresis", None)

        if sigma is not None:
            flattened_values = np.asarray(values, dtype=float).ravel()
            sigma_noise = utils.robust_sigma_from_diff(flattened_values)
            signal_median = float(np.median(flattened_values))
            threshold = float(float(sigma) * sigma_noise + signal_median)
            resolved_kwargs["height"] = threshold
            if hysteresis is not None:
                resolved_kwargs["hysteresis"] = float(
                    float(hysteresis) * sigma_noise + signal_median
                )
        elif "height" in resolved_kwargs and resolved_kwargs["height"] is not None:
            threshold = float(resolved_kwargs["height"])

        hysteresis = resolved_kwargs.get("hysteresis", None)
        if (
            threshold is not None
            and hysteresis is not None
            and float(hysteresis) > float(threshold)
        ):
            raise ValueError(
                "hysteresis must be <= the resolved detection threshold (or None). "
                f"Got hysteresis={hysteresis} and threshold={threshold}."
            )

        return resolved_kwargs, threshold

    def detect_standard_peaks(self, signal: np.ndarray) -> PeakDetectionResult:
        """Run the standard peak detector on the processed signal.

        Parameters
        ----------
        signal : numpy.ndarray
            Processed signal as a 1D trace or 2D batch.

        Returns
        -------
        PeakDetectionResult
            Standard detector result on the flattened processed signal.
        """

        flattened_signal = np.asarray(signal, dtype=float).ravel()
        kwargs, threshold = self._resolve_detection_kwargs(
            flattened_signal, self.std_kwargs
        )
        peaks, properties = find_peaks_standard(flattened_signal, **kwargs)

        return PeakDetectionResult(
            peaks=np.asarray(peaks, dtype=int),
            properties=properties,
            peak_count=int(np.asarray(peaks).size),
            detection_kwargs=kwargs,
            threshold=threshold,
        )

    def postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Apply optional low-pass filtering to the WaveNet prediction.

        Parameters
        ----------
        prediction : numpy.ndarray
            Raw model prediction.

        Returns
        -------
        numpy.ndarray
            One-dimensional postprocessed prediction.
        """

        prediction = np.asarray(prediction, dtype=float).ravel()
        low_pass = self.cnn_kwargs.get("low_pass", None)

        if low_pass is None:
            return prediction

        filtered_prediction = processing.low_pass_filter(
            prediction,
            sampling_rate=self.config.prediction_sampling_rate_hz,
            bandlimit=low_pass,
        )
        return np.asarray(filtered_prediction, dtype=float).ravel() - np.median(
            filtered_prediction
        )

    def detect_cnn_peaks(
        self, prediction: np.ndarray
    ) -> Tuple[np.ndarray, PeakDetectionResult]:
        """Postprocess a prediction and detect peaks on the resulting 1D signal.

        Parameters
        ----------
        prediction : numpy.ndarray
            Raw model prediction.

        Returns
        -------
        postprocessed_prediction : numpy.ndarray
            One-dimensional prediction after optional filtering.
        detection : PeakDetectionResult
            Detector result computed on the postprocessed prediction.
        """

        postprocessed_prediction = self.postprocess_prediction(prediction)
        kwargs = dict(self.cnn_kwargs)
        kwargs.pop("low_pass", None)
        kwargs, threshold = self._resolve_detection_kwargs(
            postprocessed_prediction, kwargs
        )
        peaks, properties = find_peaks_standard(postprocessed_prediction, **kwargs)

        detection = PeakDetectionResult(
            peaks=np.asarray(peaks, dtype=int),
            properties=properties,
            peak_count=int(np.asarray(peaks).size),
            detection_kwargs=kwargs,
            threshold=threshold,
        )
        return postprocessed_prediction, detection

    def analyze_processed_signal(
        self,
        signal: np.ndarray,
        *,
        dx: float,
        filename: Union[str, Path] = "<memory>",
        dilution: float = np.nan,
        concentration: float = np.nan,
    ) -> TraceRecord:
        """Run the full single-trace analysis pipeline and return a record.

        The input signal may already be segmented or may be a single 1D trace;
        in the latter case it is segmented automatically using the analyzer's
        configured sequence length.

        Parameters
        ----------
        signal : numpy.ndarray
            Processed signal as a 1D trace or 2D batch.
        dx : float
            Sampling interval of the source trace.
        filename : str or pathlib.Path, default="<memory>"
            Label associated with the analyzed trace.
        dilution : float, default=numpy.nan
            Dilution factor associated with the trace.
        concentration : float, default=numpy.nan
            Concentration associated with the trace.

        Returns
        -------
        TraceRecord
            Complete analysis record for the trace.
        """

        signal_batch = _coerce_signal_batch(signal, self.config.sequence_length)
        standard = self.detect_standard_peaks(signal_batch)
        normalized_signal = self.prepare_model_input(signal_batch)
        raw_prediction = self.predict(normalized_signal)
        prediction, cnn = self.detect_cnn_peaks(raw_prediction)

        return TraceRecord(
            filename=Path(filename),
            dilution=float(dilution),
            concentration=float(concentration),
            dx=float(dx),
            signal=signal_batch,
            standard=standard,
            prediction=prediction,
            cnn=cnn,
        )
