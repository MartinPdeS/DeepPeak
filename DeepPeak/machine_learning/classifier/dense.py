from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from dataclasses import dataclass, field

from DeepPeak.machine_learning.classifier.base import BaseClassifier


@dataclass
class DenseNet(BaseClassifier):
    """
    Compact 1D ConvNet for per-timestep peak classification.

    Architecture
    ------------
    - Three 1D Conv layers with ReLU activations and exponentially increasing dilation
      (default: 1, 2, 4), padding='same'
    - Final 1x1 Conv with sigmoid -> per-step probability map named 'ROI'

    Output
    ------
    - ROI: shape (batch, sequence_length, 1) with probabilities in [0, 1]
    """

    sequence_length: int
    filters: Tuple[int, int, int] = (32, 64, 128)
    dilation_rates: Tuple[int, int, int] = (1, 2, 4)
    kernel_size: int = 3
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam"
    loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy"
    metrics: Tuple[Union[str, tf.keras.metrics.Metric], ...] = ("accuracy",)
    seed: Optional[int] = None

    # filled after build()
    model: tf.keras.Model = field(init=False, repr=False, default=None)
    history_: Optional[tf.keras.callbacks.History] = field(init=False, repr=False, default=None)

    # --------------------------------------------------------------------- #
    # Construction / compilation
    # --------------------------------------------------------------------- #
    def build(self) -> tf.keras.Model:
        """Build and compile the Dense 1D ConvNet model."""
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        inputs = layers.Input(shape=(self.sequence_length, 1), name="input")

        x = inputs
        for i, (f, d) in enumerate(zip(self.filters, self.dilation_rates)):
            x = layers.Conv1D(
                filters=f,
                kernel_size=self.kernel_size,
                dilation_rate=int(d),
                activation="relu",
                padding="same",
                name=f"conv_{i}",
            )(x)

        roi = layers.Conv1D(1, kernel_size=1, activation="sigmoid", padding="same", name="ROI")(x)

        self.model = models.Model(inputs=inputs, outputs=roi, name="DenseNetDetector")
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=list(self.metrics))
        return self.model
