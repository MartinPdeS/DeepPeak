import numpy as np
import pytest

from DeepPeak.benchmarking import (
    BenchmarkCase,
    bootstrap_interval,
    classical_detector,
    evaluate_detector,
    expected_calibration_error,
    make_benchmark_dataset,
)


def test_benchmark_dataset_is_reproducible_and_covers_conditions():
    first = make_benchmark_dataset(traces_per_level=1, sequence_length=128, seed=4)
    second = make_benchmark_dataset(traces_per_level=1, sequence_length=128, seed=4)
    assert len(first) == 12
    assert {case.scenario for case in first} == {
        "overlap",
        "noise",
        "baseline_drift",
        "domain_shift",
    }
    np.testing.assert_allclose(first[0].signal, second[0].signal)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"traces_per_level": 0},
        {"sequence_length": 10},
        {"scenarios": {"unknown": [0.0]}},
        {"scenarios": {"noise": [2.0]}},
    ],
)
def test_benchmark_dataset_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_benchmark_dataset(**kwargs)


def test_evaluate_detector_reports_exact_detection_metrics():
    case = BenchmarkCase(
        "noise",
        0.0,
        np.array([0.0, 1.0, 0.0, 2.0, 0.0]),
        np.array([1, 3]),
        np.array([1.0, 2.0]),
    )
    [result] = evaluate_detector("perfect", lambda signal: [1, 3], [case], tolerance=0)
    assert result.precision == result.recall == 1.0
    assert result.timing_error == result.count_error == result.amplitude_error == 0.0
    assert result.throughput > 0
    assert result.to_dict()["method"] == "perfect"


def test_classical_detector_finds_clear_pulses():
    signal = np.zeros(64)
    signal[[12, 42]] = 1.0
    np.testing.assert_array_equal(classical_detector(threshold=0.2)(signal), [12, 42])


def test_calibration_and_bootstrap_are_deterministic():
    assert expected_calibration_error(
        np.array([0.1, 0.9]), np.array([0, 1]), bins=2
    ) == pytest.approx(0.1)
    first = bootstrap_interval([1, 2, 3], samples=100, seed=8)
    assert first == bootstrap_interval([1, 2, 3], samples=100, seed=8)
    assert first[0] <= 2 <= first[1]


@pytest.mark.parametrize(
    "call",
    [
        lambda: expected_calibration_error(np.array([]), np.array([])),
        lambda: expected_calibration_error(np.array([1.2]), np.array([1.0])),
        lambda: bootstrap_interval([]),
        lambda: bootstrap_interval([1], confidence=1.0),
    ],
)
def test_uncertainty_helpers_reject_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
