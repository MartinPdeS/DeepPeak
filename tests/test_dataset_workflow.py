import numpy as np
import pytest

from DeepPeak.generation import DataSet


def test_dataset_model_contract_and_reproducible_split():
    dataset = DataSet(
        signals=np.arange(40, dtype=float).reshape(5, 8),
        labels=np.zeros((5, 8)),
        clean_signals=np.ones((5, 8)),
        x_values=np.arange(8),
        seed=17,
    )

    assert dataset.to_model_inputs().shape == (5, 8, 1)
    assert dataset.targets().shape == (5, 8, 1)
    train_a, test_a = dataset.train_test_split(0.4, seed=3)
    train_b, test_b = dataset.train_test_split(0.4, seed=3)
    np.testing.assert_array_equal(train_a.signals, train_b.signals)
    np.testing.assert_array_equal(test_a.signals, test_b.signals)
    assert train_a.seed == test_a.seed == 17


def test_dataset_rejects_misaligned_signal_arrays():
    with pytest.raises(ValueError, match="signals must have shape"):
        DataSet(signals=np.ones((2, 3, 1)))


def test_dataset_targets_can_build_shaped_reference_targets():
    dataset = DataSet(
        signals=np.zeros((1, 9)),
        positions=np.array([[4.0]]),
        amplitudes=np.array([[2.0]]),
        x_values=np.arange(9, dtype=float),
    )

    target = dataset.targets(
        target="reference",
        width=2.0,
        normalize_peak_to_one=True,
    )

    assert target.shape == (1, 9, 1)
    assert target[0, 4, 0] == 2.0
