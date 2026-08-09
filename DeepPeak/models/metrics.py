"""Typed metrics used by the optional neural-network models."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="DeepPeak")
class BinaryIoU(tf.keras.metrics.Metric):
    """Intersection-over-union for binary sequence predictions."""

    def __init__(self, name: str = "binary_iou", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ) -> None:
        y_true = tf.cast(y_true >= self.threshold, self.dtype)
        y_pred = tf.cast(y_pred >= self.threshold, self.dtype)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            intersection *= sample_weight
            union *= sample_weight
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.intersection, self.union)

    def reset_state(self) -> None:
        self.intersection.assign(0.0)
        self.union.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
