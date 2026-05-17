"""Single-trace helpers for processed-signal and WaveNet output analysis.

The functions and classes in this module turn one processed trace into a
standardized :class:`TraceRecord` that can later be aggregated across a dilution
series or visualized with the plotting helpers.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from DeepPeak import processing, utils
from DeepPeak.algorithms.base import BaseAmplitudeSolver
from DeepPeak.algorithms.peak_locator import find_peaks_standard
from DeepPeak.analysis.trace_io import CsvTrace

from . import metrics
from .triggers import BasePeakTrigger


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
                raise ValueError("sequence_length must be a strictly positive integer.")
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

        raise ValueError(
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
            raise ValueError(
                "signal must be either a 1D trace or a 2D batch of windows."
            )

        if signal.shape[1] != int(sequence_length):
            raise ValueError(
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
            raise ValueError(
                "cnn_amplitude_sigma_samples must be a finite positive number when provided."
            )
        return sigma_samples

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
                raise ValueError(
                    'cnn_amplitude_baseline must be None, a finite float, or "median".'
                )
            return normalized

        baseline = float(baseline)
        if not np.isfinite(baseline):
            raise ValueError(
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
            raise TypeError(f"{name} must be a BasePeakTrigger instance or None.")
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

            cluster_centers = np.asarray(valid_cluster, dtype=float)[None, :]
            center_samples = signal[valid_cluster][None, :]
            response_matrix = BaseAmplitudeSolver._response_matrix_from_centers(
                cluster_centers,
                sigma_samples,
            )
            try:
                cluster_amplitudes = np.linalg.solve(
                    response_matrix[0], center_samples[0]
                )
            except np.linalg.LinAlgError:
                cluster_amplitudes = (
                    np.linalg.pinv(response_matrix[0]) @ center_samples[0]
                )

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
    ) -> None:
        self.wavenet = wavenet
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
        standard: Optional[metrics.PeakDetectionResult] = None,
        prediction: Optional[np.ndarray] = None,
        cnn: Optional[metrics.PeakDetectionResult] = None,
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
    def _empty_detection_result() -> metrics.PeakDetectionResult:
        """Return an empty detection result for disabled detector paths."""

        return metrics.PeakDetectionResult(
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
            raise ValueError(
                "hysteresis must be <= the resolved detection threshold (or None). "
                f"Got hysteresis={hysteresis} and threshold={threshold}."
            )

        return resolved_kwargs, threshold


class StandardTraceAnalyzer(_BaseTraceAnalyzer):
    """Analyze one processed trace with the standard peak detector only."""

    def __init__(
        self,
        *,
        std_trigger: BasePeakTrigger,
        sequence_length: Optional[int] = None,
        wavenet: Optional[Any] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
    ) -> None:
        super().__init__(
            wavenet=wavenet,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
        )
        self.std_trigger = self._validate_peak_trigger(
            std_trigger,
            name="std_trigger",
        )
        self.std_kwargs = self.std_trigger.to_kwargs()

    def detect_standard_peaks(self, signal: np.ndarray) -> metrics.PeakDetectionResult:
        """Run the standard peak detector on the processed signal."""

        flattened_signal = np.asarray(signal, dtype=float).ravel()
        kwargs, threshold = self._resolve_detection_kwargs(
            flattened_signal, self.std_kwargs
        )
        peaks, properties = find_peaks_standard(flattened_signal, **kwargs)

        return metrics.PeakDetectionResult(
            peaks=np.asarray(peaks, dtype=int),
            properties=properties,
            peak_count=int(np.asarray(peaks).size),
            detection_kwargs=kwargs,
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
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
    ) -> None:
        super().__init__(
            wavenet=wavenet,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
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
        self.cnn_amplitude_baseline = self._validate_optional_amplitude_baseline(
            cnn_amplitude_baseline
        )

    def prepare_model_input(self, signal: np.ndarray) -> np.ndarray:
        """Normalize processed windows into the format expected by the WaveNet."""

        signal_batch = self._coerce_signal_batch(signal, self.config.sequence_length)
        return processing.normalize_signal(
            signals=signal_batch,
            normalization=self.config.signal_normalization,
            axis=1,
        )

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
    ) -> Tuple[np.ndarray, metrics.PeakDetectionResult]:
        """Postprocess a prediction and detect peaks on the resulting 1D signal."""

        postprocessed_prediction = self.postprocess_prediction(prediction)
        kwargs = dict(self.cnn_kwargs)
        kwargs, threshold = self._resolve_detection_kwargs(
            postprocessed_prediction, kwargs
        )
        peaks, properties = find_peaks_standard(postprocessed_prediction, **kwargs)
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
            )
        detection_properties = dict(properties)
        if self.cnn_amplitude_sigma_samples is not None:
            detection_properties["recovered_sigma_samples"] = float(
                self.cnn_amplitude_sigma_samples
            )
            detection_properties["recovered_baseline"] = float(amplitude_baseline)

        detection = metrics.PeakDetectionResult(
            peaks=np.asarray(peaks, dtype=int),
            properties=detection_properties,
            peak_count=int(np.asarray(peaks).size),
            detection_kwargs=kwargs,
            threshold=threshold,
            amplitudes=amplitudes,
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
    ) -> metrics.TraceRecord:
        """Run the CNN single-trace analysis pipeline and return a record."""

        signal_batch = self._coerce_signal_batch(signal, self.config.sequence_length)
        normalized_signal = self.prepare_model_input(signal_batch)
        raw_prediction = self.predict(normalized_signal)
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
        cnn_amplitude_baseline: Optional[Union[float, str]] = None,
    ) -> None:
        super().__init__(
            wavenet=wavenet,
            sequence_length=sequence_length,
            signal_normalization=signal_normalization,
            prediction_sampling_rate_hz=prediction_sampling_rate_hz,
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
        self.cnn_amplitude_baseline = (
            None
            if self.cnn_analyzer is None
            else self.cnn_analyzer.cnn_amplitude_baseline
        )

    def detect_standard_peaks(self, signal: np.ndarray) -> metrics.PeakDetectionResult:
        """Run the standard peak detector on the processed signal."""

        if self.standard_analyzer is None:
            raise RuntimeError(
                "Standard peak detection is not configured for this analyzer."
            )
        return self.standard_analyzer.detect_standard_peaks(signal)

    def prepare_model_input(self, signal: np.ndarray) -> np.ndarray:
        """Normalize processed windows into the format expected by the WaveNet."""

        if self.cnn_analyzer is None:
            raise RuntimeError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.prepare_model_input(signal)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run the WaveNet model on a normalized batch and return its prediction."""

        if self.cnn_analyzer is None:
            raise RuntimeError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.predict(signal)

    def postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Apply optional low-pass filtering to the WaveNet prediction."""

        if self.cnn_analyzer is None:
            raise RuntimeError(
                "CNN peak detection is not configured for this analyzer."
            )
        return self.cnn_analyzer.postprocess_prediction(prediction)

    def detect_cnn_peaks(
        self,
        prediction: np.ndarray,
        *,
        signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, metrics.PeakDetectionResult]:
        """Postprocess a prediction and detect peaks on the resulting 1D signal."""

        if self.cnn_analyzer is None:
            raise RuntimeError(
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
