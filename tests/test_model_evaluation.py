import numpy as np
import pytest

pytest.importorskip("tensorflow")

from DeepPeak.generation import DataSet
from DeepPeak.models.base import BaseDeconvolver
from DeepPeak.models.evaluation import ModelEvaluationResult


class FakeDeconvolver(BaseDeconvolver):
    def _ensure_built(self):
        pass

    def evaluate(self, x, y, *, batch_size=32, verbose=0):
        return {"loss": 0.25, "mae": 0.5}

    def predict(self, signal, *, batch_size=32, verbose=0):
        return signal * 0.5


def test_evaluate_dataset_returns_metrics_and_residuals():
    dataset = DataSet(
        signals=np.ones((3, 4)),
        clean_signals=np.zeros((3, 4)),
    )
    result = FakeDeconvolver().evaluate_dataset(dataset)

    assert isinstance(result, ModelEvaluationResult)
    assert result["loss"] == 0.25
    np.testing.assert_allclose(result.residuals, 0.5)
    assert result.to_dict()["metrics"] == {"loss": 0.25, "mae": 0.5}


def test_evaluate_dataset_accepts_precomputed_targets():
    dataset = DataSet(signals=np.ones((3, 4)))
    targets = np.zeros((3, 4))
    result = FakeDeconvolver().evaluate_dataset(dataset, target=targets)

    np.testing.assert_allclose(result.targets, 0.0)
