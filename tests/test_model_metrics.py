import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from DeepPeak.models.metrics import BinaryIoU  # noqa: E402

pytestmark = pytest.mark.ml


def test_binary_iou_updates_resets_and_serializes():
    metric = BinaryIoU(threshold=0.5)
    metric.update_state(
        tf.constant([[1.0, 1.0, 0.0]]),
        tf.constant([[0.9, 0.2, 0.8]]),
    )
    assert float(metric.result()) == pytest.approx(1 / 3)
    assert metric.get_config()["threshold"] == 0.5
    metric.reset_state()
    assert float(metric.result()) == 0.0


def test_binary_iou_accepts_sample_weight():
    metric = BinaryIoU()
    metric.update_state(np.array([1.0]), np.array([1.0]), sample_weight=0.5)
    assert float(metric.result()) == 1.0
