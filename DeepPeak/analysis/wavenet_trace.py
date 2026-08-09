"""Single-trace helpers for processed-signal and WaveNet output analysis.

The functions and classes in this module turn one processed trace into a
standardized :class:`TraceRecord` that can later be aggregated across a dilution
series or visualized with the plotting helpers.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from DeepPeak import processing, utils
from DeepPeak.detection.base import BaseAmplitudeSolver
from DeepPeak.detection.peak_locator import find_peaks_prominence, find_peaks_standard
from DeepPeak.io.trace_io import CsvTrace
from DeepPeak.core.types import DetectionResult
from DeepPeak.core.config import DetectionConfig
from DeepPeak.core.types import Trace
from DeepPeak.core.exceptions import (
    AnalysisStateError,
    InvalidConfigurationError,
    InvalidDetectorError,
    MissingDetectorError,
)

from . import metrics
from DeepPeak.detection.triggers import BasePeakTrigger


class _BaseTraceAnalyzer:
    """Shared processed-trace utilities used by standard and CNN analyzers."""

    @staticmethod
    def _infer_sequence_length(
        wavenet: Optional[Any],
        sequence_length: Optional[int] = None,
    ) -> int:
        """Resolve the expected model window length from explicit input or the model."""

        if sequence_length is not None:
            sequence_length = int(sequence_length)
            if sequence_length <= 0:
                raise InvalidConfigurationError(
                    "sequence_length must be a strictly positive integer."
                )
            return sequence_length

        if wavenet is not None:
            candidate = getattr(wavenet, "sequence_length", None)
            if candidate is not None:
                candidate = int(candidate)
                if candidate > 0:
                    return candidate

            model = getattr(wavenet, "model", None)
            input_shape = getattr(model, "input_shape", None)
            if (
                input_shape is not None
                and len(input_shape) >= 2
                and input_shape[1] is not None
            ):
                candidate = int(input_shape[1])
                if candidate > 0:
                    return candidate

        raise InvalidConfigurationError(
            "Unable to infer the sequence length. Provide sequence_length explicitly "
            "or expose `wavenet.sequence_length`."
        )

    @staticmethod
    def _coerce_signal_batch(
        signal: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        """Convert a trace into a 2D batch of windows expected by the model."""

        signal = np.asarray(signal, dtype=float)

        if signal.ndim == 1:
            return utils.segment_signal(signal, window_size=sequence_length)

        if signal.ndim != 2:
            raise InvalidConfigurationError(
                "signal must be either a 1D trace or a 2D batch of windows."
            )

        if signal.shape[1] != int(sequence_length):
            raise InvalidConfigurationError(
                f"Expected signal windows of length {sequence_length}, got {signal.shape[1]}."
            )

        return signal

    @staticmethod
    def _validate_optional_sigma_samples(
        sigma_samples: Optional[float],
    ) -> Optional[float]:
        """Validate an optional Gaussian width expressed in samples."""

        if sigma_samples is None:
            return None

        sigma_samples = float(sigma_samples)
        if not np.isfinite(sigma_samples) or sigma_samples <= 0.0:
            raise InvalidConfigurationError(
                "cnn_amplitude_sigma_samples must be a finite positive number when provided."
            )
        return sigma_samples

    @staticmethod
    def _validate_cluster_radius_sigma(cluster_radius_sigma: float) -> float:
        """Validate the interaction radius used to group overlapping peaks."""

        cluster_radius_sigma = float(cluster_radius_sigma)
        if not np.isfinite(cluster_radius_sigma) or cluster_radius_sigma <= 0.0:
            raise InvalidConfigurationError(
                "cnn_amplitude_cluster_radius_sigma must be a finite positive number."
            )
        return cluster_radius_sigma

    @staticmethod
    def _validate_optional_amplitude_baseline(
        baseline: Optional[Union[float, str]],
    ) -> Optional[Union[float, str]]:
        """Validate an optional constant baseline setting for CNN amplitude recovery."""

        if baseline is None:
            return None

        if isinstance(baseline, str):
            normalized = baseline.strip().lower()
            if normalized != "median":
                raise InvalidConfigurationError(
                    'cnn_amplitude_baseline must be None, a finite float, or "median".'
                )
            return normalized

        baseline = float(baseline)
        if not np.isfinite(baseline):
            raise InvalidConfigurationError(
                'cnn_amplitude_baseline must be None, a finite float, or "median".'
            )
        return baseline

    @staticmethod
    def _resolve_amplitude_baseline(
        baseline: Optional[Union[float, str]],
        signal: np.ndarray,
    ) -> float:
        """Resolve a configured CNN amplitude baseline into a concrete scalar."""

        baseline = _BaseTraceAnalyzer._validate_optional_amplitude_baseline(baseline)
        if baseline is None:
            return 0.0
        if baseline == "median":
            return float(np.median(np.asarray(signal, dtype=float).ravel()))
        return float(baseline)

    @staticmethod
    def _validate_peak_trigger(
        trigger: Optional[BasePeakTrigger],
        *,
        name: str,
    ) -> Optional[BasePeakTrigger]:
        """Validate that a configured detector trigger is typed explicitly."""

        if trigger is None:
            return None
        if not isinstance(trigger, BasePeakTrigger):
            raise InvalidConfigurationError(
                f"{name} must be a BasePeakTrigger instance or None."
            )
        return trigger

    @staticmethod
    def _cluster_peak_indices(
        peak_indices: np.ndarray,
        max_gap: float,
    ) -> list[np.ndarray]:
        """Group sorted peak indices into local interaction clusters."""

        peak_indices = np.asarray(peak_indices, dtype=int)
        if peak_indices.size == 0:
            return []

        sorted_peaks = np.sort(peak_indices)
        clusters: list[np.ndarray] = []
        cluster_start = 0

        for index in range(1, sorted_peaks.size):
            if float(sorted_peaks[index] - sorted_peaks[index - 1]) > float(max_gap):
                clusters.append(sorted_peaks[cluster_start:index])
                cluster_start = index

        clusters.append(sorted_peaks[cluster_start:])
        return clusters

    @staticmethod
    def _recover_clustered_amplitudes(
        signal: np.ndarray,
        peak_indices: np.ndarray,
        *,
        sigma_samples: Optional[float],
        cluster_radius_sigma: float = 4.0,
    ) -> Optional[np.ndarray]:
        r"""Recover amplitudes from local Gaussian interaction clusters analytically."""

        sigma_samples = _BaseTraceAnalyzer._validate_optional_sigma_samples(
            sigma_samples
        )
        if sigma_samples is None:
            return None

        signal = np.asarray(signal, dtype=float).ravel()
        peak_indices = np.asarray(peak_indices, dtype=int)
        if peak_indices.size == 0:
            return np.asarray([], dtype=float)

        amplitudes = np.full(peak_indices.shape, np.nan, dtype=float)
        cluster_gap = max(1.0, float(cluster_radius_sigma) * sigma_samples)

        for cluster in _BaseTraceAnalyzer._cluster_peak_indices(
            peak_indices, max_gap=cluster_gap
        ):
            if cluster.size == 0:
                continue

            valid_cluster = cluster[(cluster >= 0) & (cluster < signal.size)]
            if valid_cluster.size == 0:
                continue

            cluster_centers = np.asarray(valid_cluster, dtype=float)
            fit_radius = int(np.ceil(float(cluster_radius_sigma) * sigma_samples))
            fit_start = max(0, int(valid_cluster.min()) - fit_radius)
            fit_stop = min(signal.size, int(valid_cluster.max()) + fit_radius + 1)
            fit_indices = np.arange(fit_start, fit_stop, dtype=float)
            fit_signal = signal[fit_start:fit_stop]
            design_matrix = np.exp(
                -0.5
                * ((fit_indices[:, None] - cluster_centers[None, :]) / sigma_samples)
                ** 2
            )

            try:
                cluster_amplitudes, *_ = np.linalg.lstsq(
                    design_matrix,
                    fit_signal,
                    rcond=None,
                )
            except np.linalg.LinAlgError:
                cluster_amplitudes = np.linalg.pinv(design_matrix) @ fit_signal

            for peak_index, amplitude in zip(valid_cluster, cluster_amplitudes):
                amplitudes[np.where(peak_indices == peak_index)[0][0]] = float(
                    amplitude
                )

        return amplitudes

    def __init__(
        self,
        *,
        wavenet: Optional[Any] = None,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        if config is not None:
            sequence_length = config.sequence_length or sequence_length
            signal_normalization = config.normalization
            if config.sampling_rate_hz is not None:
                prediction_sampling_rate_hz = config.sampling_rate_hz
        self.wavenet = wavenet
        self.detection_config = config
        self.config = metrics.WaveNetAnalyzerConfig(
            sequence_length=self._infer_sequence_length(
                wavenet, sequence_length=sequence_length
            ),
            signal_normalization=str(signal_normalization),
            prediction_sampling_rate_hz=float(prediction_sampling_rate_hz),
        )
        self.std_trigger: Optional[BasePeakTrigger] = None
        self.cnn_trigger: Optional[BasePeakTrigger] = None
        self.std_kwargs: Optional[Dict[str, Any]] = None
        self.cnn_kwargs: Optional[Dict[str, Any]] = None

    def _build_trace_record(
        self,
        *,
        signal_batch: np.ndarray,
        dx: float,
        filename: Union[str, Path],
        dilution: float,
        concentration: float,
        standard: Optional[DetectionResult] = None,
        prediction: Optional[np.ndarray] = None,
        cnn: Optional[DetectionResult] = None,
    ) -> metrics.TraceRecord:
        """Assemble the canonical trace record from one or both detector outputs."""

        return metrics.TraceRecord(
            filename=Path(filename),
            dilution=float(dilution),
            concentration=float(concentration),
            dx=float(dx),
            signal=np.asarray(signal_batch, dtype=float),
            standard=self._empty_detection_result() if standard is None else standard,
            prediction=(
                np.asarray([], dtype=float)
                if prediction is None
                else np.asarray(prediction, dtype=float)
            ),
            cnn=self._empty_detection_result() if cnn is None else cnn,
        )

    def load_processed_signal(
        self,
        filename: Path,
        *,
        nrows: int,
        low_pass: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Load one CSV trace and convert it to segmented processed windows."""

        data = CsvTrace(filename=filename, n_rows=nrows)

        if low_pass is not None:
            data.low_pass_filter(low_pass)

        signal = utils.process_signal(data, sequence_length=self.config.sequence_length)
        return np.asarray(signal, dtype=float), float(data.dx)

    @staticmethod
    def _empty_detection_result() -> DetectionResult:
        """Return an empty detection result for disabled detector paths."""

        return DetectionResult(
            peaks=np.asarray([], dtype=int),
            properties={},
            peak_count=0,
            detection_kwargs={},
            threshold=None,
            amplitudes=np.asarray([], dtype=float),
        )

    def _resolve_detection_kwargs(
        self,
        values: np.ndarray,
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        """Resolve threshold-related detection settings into explicit kwargs."""

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
            raise InvalidConfigurationError(
                "hysteresis must be <= the resolved detection threshold (or None). "
                f"Got hysteresis={hysteresis} and threshold={threshold}."
            )

        return resolved_kwargs, threshold


class StandardTraceAnalyzer(_BaseTraceAnalyzer):
    """Analyze one processed trace with the standard peak detector only."""

    def __init__(
        self,
        *,
        std_trigger: Optional[BasePeakTrigger] = None,
        sequence_length: Optional[int] = None,
        wavenet: Optional[Any] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        if std_trigger is None and config is not None:
            std_trigger = config.trigger
        if std_trigger is None:
            raise MissingDetectorError(
                "std_trigger or a DetectionConfig with trigger is required."
            )
        super().__init__(
            wavenet=wavenet,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
            config=config,
        )
        self.std_trigger = self._validate_peak_trigger(
            std_trigger,
            name="std_trigger",
        )
        self.std_kwargs = self.std_trigger.to_kwargs()

    def detect(self, trace: Trace, *, detector: str = "standard") -> DetectionResult:
        """Detect standard peaks from a canonical :class:`~DeepPeak.core.Trace`."""

        if detector != "standard":
            raise InvalidDetectorError(
                'StandardTraceAnalyzer only supports detector="standard".'
            )
        return self.detect_standard_peaks(trace.signal)

    def detect_standard_peaks(self, signal: np.ndarray) -> DetectionResult:
        """Run the standard peak detector on the processed signal."""

        flattened_signal = np.asarray(signal, dtype=float).ravel()
        working_kwargs = dict(self.std_kwargs)
        min_prominence = working_kwargs.pop("prominence", None)

        if min_prominence is not None:
            wlen = working_kwargs.pop("wlen", None)
            peaks, properties = find_peaks_prominence(
                flattened_signal,
                min_prominence=min_prominence,
                wlen=wlen,
                pulse_polarity=working_kwargs.get("pulse_polarity", "positive"),
                holdoff_samples=int(working_kwargs.get("holdoff_samples", 0)),
            )
            threshold = None
            detection_kwargs = working_kwargs
        else:
            working_kwargs.pop("wlen", None)
            detection_kwargs, threshold = self._resolve_detection_kwargs(
                flattened_signal, working_kwargs
            )
            peaks, properties = find_peaks_standard(
                flattened_signal, **detection_kwargs
            )

        return DetectionResult(
            peaks=np.asarray(peaks, dtype=int),
            properties=properties,
            peak_count=int(np.asarray(peaks).size),
            detection_kwargs=detection_kwargs,
            threshold=threshold,
        )

    def analyze_processed_signal(
        self,
        signal: np.ndarray,
        *,
        dx: float,
        filename: Union[str, Path] = "<memory>",
        dilution: float = np.nan,
        concentration: float = np.nan,
    ) -> metrics.TraceRecord:
        """Run the standard single-trace analysis pipeline and return a record."""

        signal_batch = self._coerce_signal_batch(signal, self.config.sequence_length)
        standard = self.detect_standard_peaks(signal_batch)
        return self._build_trace_record(
            signal_batch=signal_batch,
            dx=dx,
            filename=filename,
            dilution=dilution,
            concentration=concentration,
            standard=standard,
        )


class CNNTraceAnalyzer(_BaseTraceAnalyzer):
    """Analyze one processed trace with WaveNet prediction and CNN peak detection."""

    def __init__(
        self,
        wavenet: Any,
        *,
        cnn_trigger: BasePeakTrigger,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        cnn_low_pass: Optional[float] = None,
        cnn_amplitude_sigma_samples: Optional[float] = None,
        cnn_amplitude_cluster_radius_sigma: float = 4.0,
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        if config is not None:
            cnn_trigger = cnn_trigger or config.trigger
            if cnn_low_pass is None:
                cnn_low_pass = config.low_pass
            if cnn_amplitude_sigma_samples is None:
                cnn_amplitude_sigma_samples = config.amplitude_sigma_samples
            if config.amplitude_cluster_radius_sigma != 4.0:
                cnn_amplitude_cluster_radius_sigma = (
                    config.amplitude_cluster_radius_sigma
                )
            if cnn_amplitude_baseline is None:
                cnn_amplitude_baseline = config.amplitude_baseline
        if cnn_trigger is None:
            raise MissingDetectorError(
                "cnn_trigger or a DetectionConfig with trigger is required."
            )
        super().__init__(
            wavenet=wavenet,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
            config=config,
        )
        self.cnn_trigger = self._validate_peak_trigger(
            cnn_trigger,
            name="cnn_trigger",
        )
        self.cnn_kwargs = self.cnn_trigger.to_kwargs()
        self.cnn_low_pass = None if cnn_low_pass is None else float(cnn_low_pass)
        self.cnn_amplitude_sigma_samples = self._validate_optional_sigma_samples(
            cnn_amplitude_sigma_samples
        )
        self.cnn_amplitude_cluster_radius_sigma = self._validate_cluster_radius_sigma(
            cnn_amplitude_cluster_radius_sigma
        )
        self.cnn_amplitude_baseline = self._validate_optional_amplitude_baseline(
            cnn_amplitude_baseline
        )

    def detect(self, trace: Trace, *, detector: str = "cnn") -> DetectionResult:
        """Detect CNN peaks from a canonical :class:`~DeepPeak.core.Trace`."""

        if detector != "cnn":
            raise InvalidDetectorError('CNNTraceAnalyzer only supports detector="cnn".')
        record = self.analyze_processed_signal(
            trace.signal,
            dx=trace.dx,
            filename=trace.filename or "<memory>",
        )
        return record.cnn

    def prepare_model_input(self, signal: np.ndarray) -> np.ndarray:
        """Normalize processed windows into the format expected by the WaveNet."""

        signal_batch = self._coerce_signal_batch(signal, self.config.sequence_length)
        return processing.normalize_signal(
            signals=signal_batch,
            normalization=self.config.signal_normalization,
            axis=1,
        )

    def normalize_flat_signal(self, flat_signal: np.ndarray) -> np.ndarray:
        """Normalize a 1-D signal globally (over all samples at once).

        This is the correct pre-processing for inference: the entire trace
        shares one mean/std so that baseline-only windows are not artificially
        amplified relative to windows that contain peaks.
        """
        mode = self.config.signal_normalization.lower().strip()
        flat = np.asarray(flat_signal, dtype=np.float32)
        if mode in {"zscore"}:
            return (flat - flat.mean()) / (flat.std() + 1e-8)
        if mode in {"robust_zscore"}:
            median = np.median(flat)
            mad = np.median(np.abs(flat - median))
            return (flat - median) / (1.4826 * mad + 1e-8)
        if mode in {"minmax", "min-max"}:
            lo, hi = flat.min(), flat.max()
            return (flat - lo) / (hi - lo + 1e-8)
        if mode == "maxabs":
            return flat / (np.abs(flat).max() + 1e-8)
        return flat

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run the WaveNet model on a normalized batch and return its prediction."""

        prediction = self.wavenet.predict(signal=signal)
        return np.asarray(prediction, dtype=float)

    def postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Apply optional low-pass filtering to the WaveNet prediction."""

        prediction = np.asarray(prediction, dtype=float).ravel()
        low_pass = self.cnn_low_pass

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
        self,
        prediction: np.ndarray,
        *,
        signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, DetectionResult]:
        """Postprocess a prediction and detect peaks on the resulting 1D signal."""

        postprocessed_prediction = self.postprocess_prediction(prediction)
        working_kwargs = dict(self.cnn_kwargs)
        min_prominence = working_kwargs.pop("prominence", None)

        if min_prominence is not None:
            wlen = working_kwargs.pop("wlen", None)
            peaks, properties = find_peaks_prominence(
                postprocessed_prediction,
                min_prominence=min_prominence,
                wlen=wlen,
                pulse_polarity=working_kwargs.get("pulse_polarity", "positive"),
                holdoff_samples=int(working_kwargs.get("holdoff_samples", 0)),
            )
            threshold = None
        else:
            working_kwargs.pop("wlen", None)
            working_kwargs, threshold = self._resolve_detection_kwargs(
                postprocessed_prediction, working_kwargs
            )
            peaks, properties = find_peaks_standard(
                postprocessed_prediction, **working_kwargs
            )
        amplitudes = None
        amplitude_baseline = 0.0
        if signal is not None:
            flattened_signal = np.asarray(signal, dtype=float).ravel()
            amplitude_baseline = self._resolve_amplitude_baseline(
                self.cnn_amplitude_baseline,
                flattened_signal,
            )
            amplitudes = self._recover_clustered_amplitudes(
                signal=flattened_signal - amplitude_baseline,
                peak_indices=np.asarray(peaks, dtype=int),
                sigma_samples=self.cnn_amplitude_sigma_samples,
                cluster_radius_sigma=self.cnn_amplitude_cluster_radius_sigma,
            )
        detection_properties = dict(properties)
        if self.cnn_amplitude_sigma_samples is not None:
            detection_properties["recovered_sigma_samples"] = float(
                self.cnn_amplitude_sigma_samples
            )
            detection_properties["recovered_cluster_radius_sigma"] = float(
                self.cnn_amplitude_cluster_radius_sigma
            )
            detection_properties["recovered_baseline"] = float(amplitude_baseline)

        detection = DetectionResult(
            peaks=np.asarray(peaks, dtype=int),
            properties=detection_properties,
            peak_count=int(np.asarray(peaks).size),
            detection_kwargs=working_kwargs,
            threshold=threshold,
            amplitudes=amplitudes,
        )
        return postprocessed_prediction, detection

    def _predict_with_stride(
        self,
        flat_signal: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """Run prediction with overlapping windows and stitch via max-merge.

        Each sample is covered by multiple windows when ``stride < window_size``.
        The final prediction at each sample is the maximum over all windows that
        overlap it, which ensures that a peak clipped at one window boundary is
        still captured cleanly by the adjacent window.

        Parameters
        ----------
        flat_signal : ndarray, shape (n_samples,)
            The 1-D processed signal.
        stride : int
            Step size between consecutive windows.  Typically
            ``sequence_length // 2`` for 50 % overlap.

        Returns
        -------
        ndarray, shape (n_samples,)
            Per-sample prediction values on the original timeline.
        """

        window_size = self.config.sequence_length
        norm_signal = self.normalize_flat_signal(flat_signal)
        windows, starts = utils.segment_signal(
            norm_signal, window_size=window_size, stride=stride
        )

        normalized = windows[..., np.newaxis].astype(np.float32)
        raw_pred = self.predict(normalized)
        flat_pred_windows = np.asarray(raw_pred, dtype=float).reshape(
            len(windows), window_size
        )

        merged = np.zeros(flat_signal.size, dtype=float)
        for win_pred, start in zip(flat_pred_windows, starts):
            end = start + window_size
            merged[start:end] = np.maximum(merged[start:end], win_pred)

        return merged

    def analyze_processed_signal(
        self,
        signal: np.ndarray,
        *,
        dx: float,
        filename: Union[str, Path] = "<memory>",
        dilution: float = np.nan,
        concentration: float = np.nan,
        stride: Optional[int] = None,
    ) -> metrics.TraceRecord:
        """Run the CNN single-trace analysis pipeline and return a record.

        Parameters
        ----------
        signal : array-like
            Either a 1-D trace or a 2-D pre-segmented batch of windows.
        dx : float
            Sampling interval in seconds.
        filename : str or Path, optional
            Source filename stored in the returned record.
        dilution : float, optional
            Dilution factor stored in the returned record.
        concentration : float, optional
            Concentration stored in the returned record.
        stride : int, optional
            When set, the signal is segmented with overlapping windows of this
            step size and per-sample predictions are stitched together with a
            max-merge.  Use ``stride = sequence_length // 2`` for 50 % overlap,
            which ensures every sample (except the very edges) is seen by at
            least two windows — eliminating boundary-split artefacts.
            When *None* (default) the original non-overlapping segmentation is
            used.
        """

        if stride is not None:
            flat = np.asarray(signal, dtype=float).ravel()
            merged_pred = self._predict_with_stride(flat, stride=int(stride))
            # Re-segment into non-overlapping windows so the rest of the
            # pipeline (amplitude recovery, record building) is unchanged.
            signal_batch = utils.segment_signal(
                flat, window_size=self.config.sequence_length
            )
            padded_pred = utils.segment_signal(
                merged_pred, window_size=self.config.sequence_length
            )
            prediction, cnn = self.detect_cnn_peaks(padded_pred, signal=signal_batch)
        else:
            flat = np.asarray(signal, dtype=float).ravel()
            norm_flat = self.normalize_flat_signal(flat)
            signal_batch = utils.segment_signal(
                flat, window_size=self.config.sequence_length
            )
            norm_batch = utils.segment_signal(
                norm_flat, window_size=self.config.sequence_length
            )
            norm_batch = norm_batch[..., np.newaxis].astype(np.float32)
            raw_prediction = self.predict(norm_batch)
            prediction, cnn = self.detect_cnn_peaks(raw_prediction, signal=signal_batch)

        return self._build_trace_record(
            signal_batch=signal_batch,
            dx=dx,
            filename=filename,
            dilution=dilution,
            concentration=concentration,
            prediction=prediction,
            cnn=cnn,
        )


class WaveNetTraceAnalyzer(_BaseTraceAnalyzer):
    """Backward-compatible analyzer exposing both standard and CNN detectors.

    New code should prefer composing :class:`StandardTraceAnalyzer` and
    :class:`CNNTraceAnalyzer` explicitly.
    """

    def __init__(
        self,
        wavenet: Any,
        *,
        std_trigger: Optional[BasePeakTrigger] = None,
        cnn_trigger: Optional[BasePeakTrigger] = None,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        cnn_low_pass: Optional[float] = None,
        cnn_amplitude_sigma_samples: Optional[float] = None,
        cnn_amplitude_cluster_radius_sigma: float = 4.0,
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        if config is not None:
            std_trigger = std_trigger or config.trigger
            cnn_trigger = cnn_trigger or config.trigger
        super().__init__(
            wavenet=wavenet,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
            config=config,
        )
        self.standard_analyzer = (
            None
            if std_trigger is None
            else StandardTraceAnalyzer(
                std_trigger=std_trigger,
                wavenet=wavenet,
                sequence_length=self.config.sequence_length,
                signal_normalization=signal_normalization,
                prediction_sampling_rate_hz=prediction_sampling_rate_hz,
            )
        )
        self.cnn_analyzer = (
            None
            if cnn_trigger is None
            else CNNTraceAnalyzer(
                wavenet=wavenet,
                cnn_trigger=cnn_trigger,
                sequence_length=self.config.sequence_length,
                signal_normalization=signal_normalization,
                prediction_sampling_rate_hz=prediction_sampling_rate_hz,
                cnn_low_pass=cnn_low_pass,
                cnn_amplitude_sigma_samples=cnn_amplitude_sigma_samples,
                cnn_amplitude_cluster_radius_sigma=cnn_amplitude_cluster_radius_sigma,
                cnn_amplitude_baseline=cnn_amplitude_baseline,
            )
        )
        self.std_trigger = (
            None
            if self.standard_analyzer is None
            else self.standard_analyzer.std_trigger
        )
        self.cnn_trigger = (
            None if self.cnn_analyzer is None else self.cnn_analyzer.cnn_trigger
        )
        self.std_kwargs = (
            None
            if self.standard_analyzer is None
            else dict(self.standard_analyzer.std_kwargs)
        )
        self.cnn_kwargs = (
            None if self.cnn_analyzer is None else dict(self.cnn_analyzer.cnn_kwargs)
        )
        self.cnn_low_pass = (
            None if self.cnn_analyzer is None else self.cnn_analyzer.cnn_low_pass
        )
        self.cnn_amplitude_sigma_samples = (
            None
            if self.cnn_analyzer is None
            else self.cnn_analyzer.cnn_amplitude_sigma_samples
        )
        self.cnn_amplitude_cluster_radius_sigma = (
            None
            if self.cnn_analyzer is None
            else self.cnn_analyzer.cnn_amplitude_cluster_radius_sigma
        )
        self.cnn_amplitude_baseline = (
            None
            if self.cnn_analyzer is None
            else self.cnn_analyzer.cnn_amplitude_baseline
        )

    def detect(self, trace: Trace, *, detector: str = "standard") -> DetectionResult:
        """Detect peaks from a canonical trace using the selected detector."""

        if detector not in {"standard", "cnn"}:
            raise InvalidDetectorError('detector must be either "standard" or "cnn".')
        if detector == "standard":
            return self.detect_standard_peaks(trace.signal)
        record = self.analyze_processed_signal(
            trace.signal,
            dx=trace.dx,
            filename=trace.filename or "<memory>",
            include_standard=False,
            include_cnn=True,
        )
        return record.cnn

    def detect_standard_peaks(self, signal: np.ndarray) -> DetectionResult:
        """Run the standard peak detector on the processed signal."""

        if self.standard_analyzer is None:
            raise AnalysisStateError(
                "Standard peak detection is not configured for this analyzer."
            )
        return self.standard_analyzer.detect_standard_peaks(signal)

    def prepare_model_input(self, signal: np.ndarray) -> np.ndarray:
        """Normalize processed windows into the format expected by the WaveNet."""

        if self.cnn_analyzer is None:
            raise AnalysisStateError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.prepare_model_input(signal)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run the WaveNet model on a normalized batch and return its prediction."""

        if self.cnn_analyzer is None:
            raise AnalysisStateError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.predict(signal)

    def postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Apply optional low-pass filtering to the WaveNet prediction."""

        if self.cnn_analyzer is None:
            raise AnalysisStateError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.postprocess_prediction(prediction)

    def detect_cnn_peaks(
        self,
        prediction: np.ndarray,
        *,
        signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, DetectionResult]:
        """Postprocess a prediction and detect peaks on the resulting 1D signal."""

        if self.cnn_analyzer is None:
            raise AnalysisStateError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.detect_cnn_peaks(prediction, signal=signal)

    def analyze_processed_signal(
        self,
        signal: np.ndarray,
        *,
        dx: float,
        filename: Union[str, Path] = "<memory>",
        dilution: float = np.nan,
        concentration: float = np.nan,
        include_standard: bool = True,
        include_cnn: bool = True,
    ) -> metrics.TraceRecord:
        """Run the full single-trace analysis pipeline and return a record."""

        signal_batch = self._coerce_signal_batch(signal, self.config.sequence_length)

        if include_standard:
            standard = self.detect_standard_peaks(signal_batch)
        else:
            standard = self._empty_detection_result()

        if include_cnn:
            normalized_signal = self.prepare_model_input(signal_batch)
            raw_prediction = self.predict(normalized_signal)
            prediction, cnn = self.detect_cnn_peaks(raw_prediction, signal=signal_batch)
        else:
            prediction = np.asarray([], dtype=float)
            cnn = self._empty_detection_result()

        return self._build_trace_record(
            signal_batch=signal_batch,
            dx=dx,
            filename=filename,
            dilution=dilution,
            concentration=concentration,
            standard=standard,
            prediction=prediction,
            cnn=cnn,
        )
