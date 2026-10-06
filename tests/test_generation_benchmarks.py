"""Reproducibility and scientific smoke benchmarks for synthetic signals."""

import numpy as np
import pytest

from DeepPeak.detection import find_peaks_prominence, find_peaks_standard
from DeepPeak.generation import Gaussian, SignalGenerator, UniformCount


def _benchmark_trace(
    rng: np.random.Generator, noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.arange(256, dtype=float)
    centers = np.array([40, 90, 140, 200], dtype=int)
    signal = np.zeros_like(x_values)
    for center in centers:
        signal += np.exp(-0.5 * ((x_values - center) / 2.5) ** 2)
    signal += rng.normal(0.0, noise_std, size=x_values.size)
    return signal, centers


def _score_detection(
    predicted: np.ndarray,
    expected: np.ndarray,
    tolerance: int = 3,
) -> tuple[int, int, int, list[float]]:
    matched: set[int] = set()
    errors: list[float] = []
    true_positive = 0
    false_positive = 0

    for peak in predicted:
        distances = np.abs(expected - int(peak))
        candidate = int(np.argmin(distances))
        if distances[candidate] <= tolerance and candidate not in matched:
            matched.add(candidate)
            true_positive += 1
            errors.append(float(distances[candidate]))
        else:
            false_positive += 1

    false_negative = int(expected.size - len(matched))
    return true_positive, false_positive, false_negative, errors


@pytest.mark.parametrize(
    ("detector", "minimum_precision"),
    [
        ("standard", 0.95),
        ("prominence", 0.95),
    ],
)
def test_peak_detectors_recover_clean_gaussian_events(detector, minimum_precision):
    """High-SNR synthetic events should be detected accurately and completely."""

    rng = np.random.default_rng(2026)
    true_positive = false_positive = false_negative = 0
    localization_errors: list[float] = []

    for _ in range(100):
        signal, expected = _benchmark_trace(rng, noise_std=0.03)
        if detector == "standard":
            predicted, _ = find_peaks_standard(signal, height=0.25, hysteresis=0.15)
        else:
            predicted, _ = find_peaks_prominence(signal, min_prominence=0.25)

        scores = _score_detection(predicted, expected)
        true_positive += scores[0]
        false_positive += scores[1]
        false_negative += scores[2]
        localization_errors.extend(scores[3])

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    assert precision >= minimum_precision
    assert recall >= 0.95
    assert np.median(localization_errors) <= 1.0


def test_signal_generation_is_seeded_without_mutating_global_numpy_state():
    """A generator seed controls all generated components independently."""

    generator = SignalGenerator(sequence_length=128)
    kernel = Gaussian(amplitude=(0.8, 1.2), position=(20.0, 100.0), width=(2.0, 4.0))
    peak_count = UniformCount(bounds=(1, 3))

    np.random.seed(12345)
    before = np.random.get_state()
    first = generator.generate(
        n_samples=16,
        kernel=kernel,
        peak_count=peak_count,
        seed=7,
        noise_std=0.03,
        drift=(0.0, 0.1),
    )
    after = np.random.get_state()

    np.random.seed(98765)
    second = generator.generate(
        n_samples=16,
        kernel=kernel,
        peak_count=peak_count,
        seed=7,
        noise_std=0.03,
        drift=(0.0, 0.1),
    )

    assert np.array_equal(before[1], after[1])
    assert np.array_equal(first.signals, second.signals)
    assert np.array_equal(first.labels, second.labels)
