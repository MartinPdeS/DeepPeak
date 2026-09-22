"""Reproducible, model-agnostic benchmarks for one-dimensional peak detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

Array = NDArray[np.float64]
Detector = Callable[[Array], Iterable[int] | NDArray[np.integer]]


@dataclass(frozen=True)
class BenchmarkCase:
    """One trace and its sample-aligned peak ground truth."""

    scenario: str
    level: float
    signal: Array
    peak_indices: NDArray[np.int64]
    peak_amplitudes: Array


@dataclass(frozen=True)
class DetectionMetrics:
    """Aggregate metrics for one method and benchmark condition."""

    method: str
    scenario: str
    level: float
    precision: float
    recall: float
    timing_error: float
    count_error: float
    amplitude_error: float
    throughput: float
    calibration_error: float = float("nan")
    uncertainty_coverage: float = float("nan")

    def to_dict(self) -> dict[str, str | float]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def make_benchmark_dataset(
    *,
    scenarios: Mapping[str, Iterable[float]] | None = None,
    traces_per_level: int = 24,
    sequence_length: int = 512,
    seed: int = 1729,
) -> list[BenchmarkCase]:
    """Generate compact traces with exact, reproducible ground truth.

    The scenarios independently vary overlap, noise, baseline drift, and pulse
    shape (domain shift). Severity levels must lie in the interval [0, 1].
    """

    if traces_per_level < 1 or sequence_length < 64:
        raise ValueError("traces_per_level must be positive and length must be >= 64")
    scenarios = scenarios or {
        "overlap": (0.0, 0.5, 1.0),
        "noise": (0.0, 0.5, 1.0),
        "baseline_drift": (0.0, 0.5, 1.0),
        "domain_shift": (0.0, 0.5, 1.0),
    }
    allowed = {"overlap", "noise", "baseline_drift", "domain_shift"}
    if unknown := set(scenarios) - allowed:
        raise ValueError(f"Unknown benchmark scenarios: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    x = np.arange(sequence_length, dtype=float)
    cases: list[BenchmarkCase] = []
    for scenario, levels in scenarios.items():
        for raw_level in levels:
            level = float(raw_level)
            if not 0.0 <= level <= 1.0:
                raise ValueError("scenario levels must lie in [0, 1]")
            for _ in range(traces_per_level):
                count = int(rng.integers(3, 7))
                if scenario == "overlap":
                    spacing = int(round(36 - 25 * level))
                    count = min(count, max(1, (sequence_length - 50) // spacing))
                    maximum_start = sequence_length - spacing * count - 10
                    start = int(rng.integers(30, maximum_start))
                    centers = start + spacing * np.arange(count)
                else:
                    centers = np.sort(
                        rng.choice(
                            np.arange(30, sequence_length - 30), count, replace=False
                        )
                    )
                amplitudes = rng.uniform(0.75, 1.25, count)
                signal = np.zeros_like(x)
                for center, amplitude in zip(centers, amplitudes):
                    distance = (x - center) / 5.0
                    gaussian = np.exp(-0.5 * distance**2)
                    if scenario == "domain_shift":
                        tail = np.exp(-(x - center) / (8.0 + 10.0 * level)) * (
                            x >= center
                        )
                        pulse = (1.0 - level) * gaussian + level * tail
                    else:
                        pulse = gaussian
                    signal += amplitude * pulse
                noise_scale = 0.03 + (0.22 * level if scenario == "noise" else 0.0)
                signal += rng.normal(0.0, noise_scale, sequence_length)
                if scenario == "baseline_drift":
                    phase = rng.uniform(0, 2 * np.pi)
                    signal += (
                        0.55 * level * np.sin(2 * np.pi * x / sequence_length + phase)
                    )
                cases.append(
                    BenchmarkCase(
                        scenario,
                        level,
                        signal.astype(float),
                        np.asarray(centers, dtype=np.int64),
                        np.asarray(amplitudes, dtype=float),
                    )
                )
    return cases


def classical_detector(
    *, threshold: float = 0.25, minimum_distance: int = 7
) -> Detector:
    """Create a robust SciPy-prominence classical baseline."""

    def detect(signal: Array) -> NDArray[np.int64]:
        differences = np.diff(signal)
        noise = (
            1.4826
            * np.median(np.abs(differences - np.median(differences)))
            / np.sqrt(2)
        )
        prominence = max(float(threshold), 4.0 * float(noise))
        peaks, _ = find_peaks(signal, prominence=prominence, distance=minimum_distance)
        return peaks.astype(np.int64)

    return detect


def _match_peaks(
    truth: NDArray[np.int64], predicted: NDArray[np.int64], tolerance: int
):
    if truth.size == 0 or predicted.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    distances = np.abs(truth[:, None] - predicted[None, :])
    truth_rows, predicted_columns = linear_sum_assignment(distances)
    accepted = distances[truth_rows, predicted_columns] <= tolerance
    return truth_rows[accepted], predicted_columns[accepted]


def expected_calibration_error(
    probabilities: Array, outcomes: Array, *, bins: int = 10
) -> float:
    """Compute weighted expected calibration error for binary predictions."""

    probabilities = np.asarray(probabilities, dtype=float).ravel()
    outcomes = np.asarray(outcomes, dtype=float).ravel()
    if probabilities.shape != outcomes.shape or probabilities.size == 0:
        raise ValueError("probabilities and outcomes must be non-empty and aligned")
    if bins < 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("bins must be positive and probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if np.any(selected):
            error += np.mean(selected) * abs(
                np.mean(probabilities[selected]) - np.mean(outcomes[selected])
            )
    return float(error)


def evaluate_detector(
    method: str,
    detector: Detector,
    cases: Iterable[BenchmarkCase],
    *,
    tolerance: int = 5,
) -> list[DetectionMetrics]:
    """Evaluate a detector and aggregate scores by scenario severity."""

    grouped: dict[tuple[str, float], list[BenchmarkCase]] = {}
    for case in cases:
        grouped.setdefault((case.scenario, case.level), []).append(case)
    output = []
    for (scenario, level), group in grouped.items():
        true_positives = false_positives = false_negatives = 0
        timing_errors: list[float] = []
        amplitude_errors: list[float] = []
        count_errors: list[float] = []
        start = perf_counter()
        for case in group:
            predicted = np.asarray(list(detector(case.signal)), dtype=np.int64)
            truth_rows, predicted_columns = _match_peaks(
                case.peak_indices, predicted, tolerance
            )
            matches = truth_rows.size
            true_positives += matches
            false_positives += predicted.size - matches
            false_negatives += case.peak_indices.size - matches
            count_errors.append(abs(predicted.size - case.peak_indices.size))
            if matches:
                timing_errors.extend(
                    abs(case.peak_indices[truth_rows] - predicted[predicted_columns])
                )
                amplitude_errors.extend(
                    abs(
                        case.signal[predicted[predicted_columns]]
                        - case.peak_amplitudes[truth_rows]
                    )
                )
        elapsed = max(perf_counter() - start, np.finfo(float).eps)
        output.append(
            DetectionMetrics(
                method,
                scenario,
                level,
                true_positives / max(true_positives + false_positives, 1),
                true_positives / max(true_positives + false_negatives, 1),
                float(np.mean(timing_errors)) if timing_errors else float("nan"),
                float(np.mean(count_errors)),
                float(np.mean(amplitude_errors)) if amplitude_errors else float("nan"),
                len(group) / elapsed,
            )
        )
    return output


def bootstrap_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 1729,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for the mean."""

    values = np.asarray(list(values), dtype=float)
    if values.size == 0 or samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError(
            "values/samples must be non-empty and confidence must lie in (0, 1)"
        )
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(values, size=(samples, values.size), replace=True), axis=1
    )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)
