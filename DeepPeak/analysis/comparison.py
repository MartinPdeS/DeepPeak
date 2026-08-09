"""Paired comparison of direct and neural-deconvolved peak detection.

The comparison API deliberately keeps the neural stage optional.  A trace can
therefore be analyzed directly, or passed through a CNN/WaveNet/U-Net model
before the same style of peak detector is applied.  Both branches expose the
same arrival-time, amplitude, and width distributions.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

import numpy as np
import scipy.stats as stats

from ..core.types import DetectionResult, Trace
from ..detection.triggers import BasePeakTrigger
from .wavenet_trace import CNNTraceAnalyzer, StandardTraceAnalyzer

Branch = Literal["standard", "deconvolved"]
DistributionName = Literal["arrival", "amplitude", "width"]


def _finite(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _peak_amplitudes(
    signal: np.ndarray,
    detection: DetectionResult,
    *,
    use_detection_amplitudes: bool,
) -> np.ndarray:
    if (
        use_detection_amplitudes
        and detection.amplitudes is not None
        and detection.amplitudes.size
    ):
        amplitudes = _finite(detection.amplitudes)
        if amplitudes.size:
            return amplitudes

    peaks = np.asarray(detection.peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < signal.size)]
    return _finite(signal[peaks])


def _peak_widths(detection: DetectionResult, dx: float) -> np.ndarray:
    properties = detection.properties
    for key in ("widths_pixels", "widths_samples", "widths"):
        if key in properties:
            return _finite(properties[key]) * float(dx)
    return np.asarray([], dtype=float)


@dataclass
class TraceComparisonResult:
    """Direct and optional deconvolved detection results for one trace."""

    trace: Trace
    standard: DetectionResult
    deconvolved: Optional[DetectionResult] = None
    deconvolved_signal: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.deconvolved_signal is not None:
            self.deconvolved_signal = np.asarray(
                self.deconvolved_signal, dtype=float
            ).ravel()
        self.metadata = dict(self.metadata)

    @property
    def has_deconvolved_branch(self) -> bool:
        return self.deconvolved is not None

    def detection(self, branch: Branch) -> DetectionResult:
        if branch == "standard":
            return self.standard
        if branch == "deconvolved" and self.deconvolved is not None:
            return self.deconvolved
        raise ValueError("The deconvolved branch is not available for this result.")

    def signal(self, branch: Branch) -> np.ndarray:
        if branch == "standard":
            return np.asarray(self.trace.signal, dtype=float).ravel()
        if branch == "deconvolved" and self.deconvolved_signal is not None:
            return self.deconvolved_signal
        raise ValueError("The deconvolved branch is not available for this result.")

    def distribution(self, name: DistributionName, branch: Branch) -> np.ndarray:
        detection = self.detection(branch)
        if name == "arrival":
            return _finite(detection.peaks * float(self.trace.dx))
        if name == "amplitude":
            return _peak_amplitudes(
                self.signal(branch),
                detection,
                # A deconvolution comparison measures amplitudes on the
                # reconstructed branch signal. Explicit amplitude recovery
                # remains available on the underlying DetectionResult.
                use_detection_amplitudes=(branch == "standard"),
            )
        if name == "width":
            return _peak_widths(detection, self.trace.dx)
        raise ValueError("name must be 'arrival', 'amplitude', or 'width'.")

    def summary(self, branch: Branch) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in ("arrival", "amplitude", "width"):
            values = self.distribution(name, branch)
            result[name] = _summary(values)
        return result

    def compare_distribution(self, name: DistributionName) -> dict[str, float]:
        """Compare one distribution between branches on this trace."""

        if not self.has_deconvolved_branch:
            raise ValueError("A deconvolved branch is required for comparison.")
        standard = self.distribution(name, "standard")
        deconvolved = self.distribution(name, "deconvolved")
        return _compare_values(standard, deconvolved)

    def compare(self) -> dict[str, dict[str, float]]:
        return {
            name: self.compare_distribution(name)
            for name in ("arrival", "amplitude", "width")
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "standard": self.standard.to_dict(),
            "deconvolved": (
                None if self.deconvolved is None else self.deconvolved.to_dict()
            ),
            "deconvolved_signal": (
                None
                if self.deconvolved_signal is None
                else self.deconvolved_signal.tolist()
            ),
            "metadata": self.metadata,
        }


@dataclass
class SeriesComparisonResult:
    """Aggregate distribution comparisons over an ordered trace collection."""

    records: list[TraceComparisonResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def has_deconvolved_branch(self) -> bool:
        return any(record.has_deconvolved_branch for record in self.records)

    def append(self, record: TraceComparisonResult) -> None:
        self.records.append(record)

    def distribution(self, name: DistributionName, branch: Branch) -> np.ndarray:
        values = [record.distribution(name, branch) for record in self.records]
        if not values:
            return np.asarray([], dtype=float)
        return _finite(np.concatenate(values))

    def summary(self, branch: Branch) -> dict[str, Any]:
        return {
            name: _summary(self.distribution(name, branch))
            for name in ("arrival", "amplitude", "width")
        }

    def compare_distribution(self, name: DistributionName) -> dict[str, float]:
        if not self.has_deconvolved_branch:
            raise ValueError("A deconvolved branch is required for comparison.")
        return _compare_values(
            self.distribution(name, "standard"),
            self.distribution(name, "deconvolved"),
        )

    def compare(self) -> dict[str, dict[str, float]]:
        return {
            name: self.compare_distribution(name)
            for name in ("arrival", "amplitude", "width")
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "metadata": self.metadata,
        }


class TraceComparisonAnalyzer:
    """Run direct detection and optional neural deconvolution side by side.

    Parameters
    ----------
    standard_trigger : BasePeakTrigger
        Trigger used on the original signal.
    deconvolver : object, optional
        Model exposing ``predict(signal=...)``.  CNN, WaveNet, and U-Net model
        wrappers are supported.  If omitted, only the standard branch runs.
    deconvolved_trigger : BasePeakTrigger, optional
        Trigger applied to the deconvolved signal. Defaults to the standard
        trigger.
    """

    def __init__(
        self,
        standard_trigger: BasePeakTrigger,
        *,
        deconvolver: Optional[Any] = None,
        deconvolved_trigger: Optional[BasePeakTrigger] = None,
        sequence_length: Optional[int] = None,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
        deconvolved_low_pass: Optional[float] = None,
        deconvolved_amplitude_sigma_samples: Optional[float] = None,
        deconvolved_amplitude_cluster_radius_sigma: float = 4.0,
        deconvolved_amplitude_baseline: Optional[float | str] = None,
    ) -> None:
        self.standard_trigger = standard_trigger
        self.deconvolver = deconvolver
        self.deconvolved_trigger = deconvolved_trigger or standard_trigger
        self.sequence_length = sequence_length
        self.signal_normalization = signal_normalization
        self.prediction_sampling_rate_hz = prediction_sampling_rate_hz
        self.deconvolved_low_pass = deconvolved_low_pass
        self.deconvolved_amplitude_sigma_samples = deconvolved_amplitude_sigma_samples
        self.deconvolved_amplitude_cluster_radius_sigma = (
            deconvolved_amplitude_cluster_radius_sigma
        )
        self.deconvolved_amplitude_baseline = deconvolved_amplitude_baseline

    def compare(self, trace: Trace) -> TraceComparisonResult:
        """Analyze one canonical trace through the configured branches."""

        sequence_length = self.sequence_length or trace.n_samples
        standard_analyzer = StandardTraceAnalyzer(
            std_trigger=self.standard_trigger,
            sequence_length=sequence_length,
        )
        standard = standard_analyzer.detect(trace)

        if self.deconvolver is None:
            return TraceComparisonResult(trace=trace, standard=standard)

        deconvolved_analyzer = CNNTraceAnalyzer(
            wavenet=self.deconvolver,
            cnn_trigger=self.deconvolved_trigger,
            sequence_length=sequence_length,
            signal_normalization=self.signal_normalization,
            prediction_sampling_rate_hz=self.prediction_sampling_rate_hz,
            cnn_low_pass=self.deconvolved_low_pass,
            cnn_amplitude_sigma_samples=self.deconvolved_amplitude_sigma_samples,
            cnn_amplitude_cluster_radius_sigma=(
                self.deconvolved_amplitude_cluster_radius_sigma
            ),
            cnn_amplitude_baseline=self.deconvolved_amplitude_baseline,
        )
        record = deconvolved_analyzer.analyze_processed_signal(
            trace.signal,
            dx=trace.dx,
            filename=trace.filename or "<memory>",
        )
        return TraceComparisonResult(
            trace=trace,
            standard=standard,
            deconvolved=record.cnn,
            deconvolved_signal=record.prediction,
            metadata={"deconvolution_model": type(self.deconvolver).__name__},
        )

    def compare_many(self, traces: Iterable[Trace]) -> SeriesComparisonResult:
        return SeriesComparisonResult(records=[self.compare(trace) for trace in traces])


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = _finite(values)
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "standard_deviation": np.nan,
            "quantiles": {},
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": (
            float(np.std(values, ddof=1)) if values.size > 1 else np.nan
        ),
        "quantiles": {
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
        },
    }


def _compare_values(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = _finite(first)
    second = _finite(second)
    result = {
        "count_difference": float(second.size - first.size),
        "mean_difference": np.nan,
        "median_difference": np.nan,
        "standard_deviation_difference": np.nan,
        "wasserstein_distance": np.nan,
        "ks_statistic": np.nan,
        "ks_p_value": np.nan,
    }
    if first.size:
        result["mean_difference"] = (
            float(np.mean(second) - np.mean(first)) if second.size else np.nan
        )
        result["median_difference"] = (
            float(np.median(second) - np.median(first)) if second.size else np.nan
        )
    if first.size > 1 and second.size > 1:
        result["standard_deviation_difference"] = float(
            np.std(second, ddof=1) - np.std(first, ddof=1)
        )
        result["wasserstein_distance"] = float(
            stats.wasserstein_distance(first, second)
        )
        ks = stats.ks_2samp(first, second)
        result["ks_statistic"] = float(ks.statistic)
        result["ks_p_value"] = float(ks.pvalue)
    return result


__all__ = [
    "SeriesComparisonResult",
    "TraceComparisonAnalyzer",
    "TraceComparisonResult",
]
