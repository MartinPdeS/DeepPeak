from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from dataclasses import dataclass, field

from DeepPeak.machine_learning.classifier.base import BaseClassifier



@dataclass
class Autoencoder(BaseClassifier):
    """
    1D convolutional autoencoder for predicting an ROI (Region Of Interest) mask.

    Architecture
    ------------
    Encoder:
      - Conv1D(f[0], K, relu, same) -> Dropout(p) -> MaxPool1D(2)
      - Conv1D(f[1], K, relu, same) -> Dropout(p) -> MaxPool1D(2)
    Bottleneck:
      - Conv1D(f[2], K, relu, same) -> Dropout(p)
    Decoder:
      - UpSampling1D(2) -> Conv1D(f[1], K, relu, same)
      - UpSampling1D(2) -> Conv1D(f[0], K, relu, same)
    Output:
      - Conv1D(1, 1, sigmoid, name="ROI")

    Output shape
    ------------
    (batch, sequence_length, 1)

    Notes
    -----
    - Loss: binary_crossentropy on the 'ROI' head
    - Metrics: configurable (default: accuracy)
    - The pooling/upsampling ladder assumes sequence_length divisible by 4 to
      reconstruct the original length exactly (with padding='same' this typically
      works well; validate with a quick model.summary()).
    """

    sequence_length: int
    dropout_rate: float = 0.30
    filters: Tuple[int, int, int] = (32, 64, 128)
    kernel_size: int = 3
    pool_size: int = 2
    upsample_size: int = 2
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam"
    loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy"
    metrics: Tuple[Union[str, tf.keras.metrics.Metric], ...] = ("accuracy",)
    seed: Optional[int] = None

    # filled after build()
    model: tf.keras.Model = field(init=False, repr=False, default=None)
    history_: Optional[tf.keras.callbacks.History] = field(init=False, repr=False, default=None)

    # --------------------------------------------------------------------- #
    # Build / compile
    # --------------------------------------------------------------------- #
    def build(self) -> tf.keras.Model:
        """Build and compile the autoencoder model."""
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        inputs = layers.Input(shape=(self.sequence_length, 1), name="input")

        # Encoder
        x = layers.Conv1D(self.filters[0], self.kernel_size, activation="relu", padding="same", name="enc_conv0")(inputs)
        x = layers.Dropout(self.dropout_rate, name="enc_drop0")(x)
        x = layers.MaxPooling1D(pool_size=self.pool_size, padding="same", name="enc_pool0")(x)

        x = layers.Conv1D(self.filters[1], self.kernel_size, activation="relu", padding="same", name="enc_conv1")(x)
        x = layers.Dropout(self.dropout_rate, name="enc_drop1")(x)
        x = layers.MaxPooling1D(pool_size=self.pool_size, padding="same", name="enc_pool1")(x)

        # Bottleneck
        x = layers.Conv1D(self.filters[2], self.kernel_size, activation="relu", padding="same", name="bottleneck_conv")(x)
        x = layers.Dropout(self.dropout_rate, name="bottleneck_drop")(x)

        # Decoder
        x = layers.UpSampling1D(size=self.upsample_size, name="dec_up0")(x)
        x = layers.Conv1D(self.filters[1], self.kernel_size, activation="relu", padding="same", name="dec_conv0")(x)
        x = layers.UpSampling1D(size=self.upsample_size, name="dec_up1")(x)
        x = layers.Conv1D(self.filters[0], self.kernel_size, activation="relu", padding="same", name="dec_conv1")(x)

        # Output head
        roi = layers.Conv1D(1, kernel_size=1, activation="sigmoid", name="ROI")(x)

        self.model = models.Model(inputs=inputs, outputs=roi, name="AutoencoderROILocator")
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=list(self.metrics))
        return self.model
