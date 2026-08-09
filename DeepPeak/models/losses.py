"""Regression losses for one-dimensional pulse-trace deconvolution."""

import tensorflow as tf


def _prepare(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank is not None and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if y_pred.shape.rank is not None and y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    return y_true, y_pred


@tf.keras.utils.register_keras_serializable(package="DeepPeak")
class WeightedHuber(tf.keras.losses.Loss):
    """Huber reconstruction loss weighted by clean pulse amplitude."""

    def __init__(self, delta: float = 1.0, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = float(delta)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        y_true, y_pred = _prepare(y_true, y_pred)
        error = y_true - y_pred
        absolute = tf.abs(error)
        quadratic = tf.minimum(absolute, self.delta)
        linear = absolute - quadratic
        huber = 0.5 * tf.square(quadratic) + self.delta * linear
        weights = 1.0 + self.alpha * tf.abs(y_true)
        return tf.reduce_mean(weights * huber)

    def get_config(self):
        return {**super().get_config(), "delta": self.delta, "alpha": self.alpha}


@tf.keras.utils.register_keras_serializable(package="DeepPeak")
class ShapeAwarePulseLoss(tf.keras.losses.Loss):
    """Reconstruction loss combining amplitude, shape, and smoothness terms."""

    def __init__(
        self,
        amplitude_weight: float = 1.0,
        shape_weight: float = 0.25,
        smoothness_weight: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude_weight = float(amplitude_weight)
        self.shape_weight = float(shape_weight)
        self.smoothness_weight = float(smoothness_weight)

    def call(self, y_true, y_pred):
        y_true, y_pred = _prepare(y_true, y_pred)
        amplitude_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        shape_loss = tf.reduce_mean(tf.square(true_norm - pred_norm))
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        smoothness_loss = tf.reduce_mean(tf.square(true_diff - pred_diff))
        return (
            self.amplitude_weight * amplitude_loss
            + self.shape_weight * shape_loss
            + self.smoothness_weight * smoothness_loss
        )

    def get_config(self):
        return {
            **super().get_config(),
            "amplitude_weight": self.amplitude_weight,
            "shape_weight": self.shape_weight,
            "smoothness_weight": self.smoothness_weight,
        }


def weighted_huber(
    y_true=None,
    y_pred=None,
    *,
    delta: float = 1.0,
    alpha: float = 1.0,
):
    """Function-style or configured weighted Huber reconstruction loss."""

    loss = WeightedHuber(delta=delta, alpha=alpha)
    if y_true is None or y_pred is None:
        return loss
    return loss(y_true, y_pred)


def shape_aware_pulse_loss(
    y_true=None,
    y_pred=None,
    *,
    amplitude_weight: float = 1.0,
    shape_weight: float = 0.25,
    smoothness_weight: float = 0.05,
):
    """Function-style or configured shape-aware reconstruction loss."""

    loss = ShapeAwarePulseLoss(
        amplitude_weight=amplitude_weight,
        shape_weight=shape_weight,
        smoothness_weight=smoothness_weight,
    )
    if y_true is None or y_pred is None:
        return loss
    return loss(y_true, y_pred)


__all__ = [
    "ShapeAwarePulseLoss",
    "WeightedHuber",
    "shape_aware_pulse_loss",
    "weighted_huber",
]
