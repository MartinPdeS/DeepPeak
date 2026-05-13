from types import SimpleNamespace

import numpy as np
import pytest

utils = pytest.importorskip("DeepPeak.utils")


def test_utils_package_exports_lightweight_symbols():
    assert utils.__name__ == "DeepPeak.utils"
    assert callable(utils.batched)
    assert callable(utils.robust_sigma_from_diff)


def test_batched_groups_values_without_dropping_tail():
    batches = list(utils.batched(range(7), 3))
    assert batches == [(0, 1, 2), (3, 4, 5), (6,)]


def test_segment_signal_zero_pads_last_window():
    segmented = utils.segment_signal(np.array([1, 2, 3, 4, 5]), window_size=3)
    np.testing.assert_array_equal(segmented, np.array([[1, 2, 3], [4, 5, 0]]))


def test_get_normalized_signal_min_max_scales_each_row():
    signals = np.array([[2.0, 4.0, 6.0], [3.0, 3.0, 9.0]])
    normalized = utils.get_normalized_signal(signals, normalization="min-max")
    np.testing.assert_allclose(
        normalized,
        np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]]),
    )


def test_process_signal_normalizes_and_segments_data_object():
    data = SimpleNamespace(y_processed=np.array([0.0, 2.0, 4.0, 6.0, 8.0]))
    processed = utils.process_signal(data, sequence_length=2)
    np.testing.assert_allclose(
        processed,
        np.array([[0.0, 0.25], [0.5, 0.75], [1.0, 0.0]]),
    )


def test_robust_sigma_from_diff_is_zero_for_constant_step_signal():
    sigma = utils.robust_sigma_from_diff(np.array([1.0, 3.0, 5.0, 7.0]))
    assert sigma == pytest.approx(0.0)
