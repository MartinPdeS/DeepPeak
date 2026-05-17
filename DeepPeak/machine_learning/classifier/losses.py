import tensorflow as tf


def _maybe_squeeze_trailing_unit_axis(tensor: tf.Tensor) -> tf.Tensor:
    if tensor.shape.rank is not None and tensor.shape[-1] == 1:
        return tf.squeeze(tensor, axis=-1)
    return tensor


def _prepare_loss_tensors(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = _maybe_squeeze_trailing_unit_axis(y_true)
    y_pred = _maybe_squeeze_trailing_unit_axis(y_pred)
    return y_true, y_pred


def _target_weights(y_true: tf.Tensor, alpha: float) -> tf.Tensor:
    return 1.0 + float(alpha) * tf.abs(y_true)


@tf.keras.utils.register_keras_serializable(package="DeepPeak")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary crossentropy with positive-class upweighting."""

    def __init__(
        self,
        alpha: float = 1.0,
        from_logits: bool = False,
        reduction: str = "sum_over_batch_size",
        name: str = "weighted_bce",
    ) -> None:
        super().__init__(name=name, reduction=reduction)
        self.alpha = float(alpha)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true, y_pred = _prepare_loss_tensors(y_true, y_pred)

        bce = tf.keras.backend.binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
        )
        weights = _target_weights(y_true, self.alpha)
        return tf.reduce_mean(weights * bce)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "alpha": self.alpha,
            "from_logits": self.from_logits,
        }


def weighted_bce(
    y_true=None,
    y_pred=None,
    alpha: float = 1.0,
    *,
    from_logits: bool = False,
    name: str = "weighted_bce",
):
    """Serializable weighted BCE that supports both Keras call styles.

    Use either:
    - ``loss=weighted_bce`` for the default ``alpha=1.0`` function form
    - ``loss=weighted_bce(alpha=4.0)`` for a configured serializable loss object
    """

    if y_true is not None and y_pred is not None:
        return WeightedBinaryCrossentropy(
            alpha=alpha,
            from_logits=from_logits,
            name=name,
        )(y_true, y_pred)

    return WeightedBinaryCrossentropy(
        alpha=alpha,
        from_logits=from_logits,
        name=name,
    )


@tf.keras.utils.register_keras_serializable(package="DeepPeak")
class WeightedHuber(tf.keras.losses.Loss):
    """Huber regression with stronger emphasis near true pulses."""

    def __init__(
        self,
        alpha: float = 1.0,
        delta: float = 1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "weighted_huber",
    ) -> None:
        super().__init__(name=name, reduction=reduction)
        self.alpha = float(alpha)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        y_true, y_pred = _prepare_loss_tensors(y_true, y_pred)
        error = y_pred - y_true
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + self.delta * linear
        weights = _target_weights(y_true, self.alpha)
        return tf.reduce_mean(weights * huber)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "alpha": self.alpha,
            "delta": self.delta,
        }


@tf.keras.utils.register_keras_serializable(package="DeepPeak")
class ShapeAwarePulseLoss(tf.keras.losses.Loss):
    """Weighted Huber with an additional derivative-matching term."""

    def __init__(
        self,
        alpha: float = 1.0,
        delta: float = 1.0,
        derivative_weight: float = 0.25,
        derivative_delta: float = 1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "shape_aware_pulse_loss",
    ) -> None:
        super().__init__(name=name, reduction=reduction)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.derivative_weight = float(derivative_weight)
        self.derivative_delta = float(derivative_delta)

    def _weighted_huber(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        *,
        delta: float,
        alpha: float,
    ) -> tf.Tensor:
        error = y_pred - y_true
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + delta * linear
        return tf.reduce_mean(_target_weights(y_true, alpha) * huber)

    def call(self, y_true, y_pred):
        y_true, y_pred = _prepare_loss_tensors(y_true, y_pred)

        value_loss = self._weighted_huber(
            y_true,
            y_pred,
            delta=self.delta,
            alpha=self.alpha,
        )

        true_grad = y_true[:, 1:] - y_true[:, :-1]
        pred_grad = y_pred[:, 1:] - y_pred[:, :-1]
        grad_loss = self._weighted_huber(
            true_grad,
            pred_grad,
            delta=self.derivative_delta,
            alpha=self.alpha,
        )

        return value_loss + self.derivative_weight * grad_loss

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "alpha": self.alpha,
            "delta": self.delta,
            "derivative_weight": self.derivative_weight,
            "derivative_delta": self.derivative_delta,
        }


def weighted_huber(
    y_true=None,
    y_pred=None,
    alpha: float = 1.0,
    *,
    delta: float = 1.0,
    name: str = "weighted_huber",
):
    """Serializable weighted Huber supporting direct and configured use."""

    if y_true is not None and y_pred is not None:
        return WeightedHuber(
            alpha=alpha,
            delta=delta,
            name=name,
        )(y_true, y_pred)

    return WeightedHuber(
        alpha=alpha,
        delta=delta,
        name=name,
    )


def shape_aware_pulse_loss(
    y_true=None,
    y_pred=None,
    alpha: float = 1.0,
    *,
    delta: float = 1.0,
    derivative_weight: float = 0.25,
    derivative_delta: float = 1.0,
    name: str = "shape_aware_pulse_loss",
):
    """Serializable pulse loss with value and derivative matching terms."""

    if y_true is not None and y_pred is not None:
        return ShapeAwarePulseLoss(
            alpha=alpha,
            delta=delta,
            derivative_weight=derivative_weight,
            derivative_delta=derivative_delta,
            name=name,
        )(y_true, y_pred)

    return ShapeAwarePulseLoss(
        alpha=alpha,
        delta=delta,
        derivative_weight=derivative_weight,
        derivative_delta=derivative_delta,
        name=name,
    )
