from typing import Iterable, Optional, Tuple, Union
from dataclasses import dataclass, field
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

from .base import BaseDeconvolver


@dataclass
class DenseNet(BaseDeconvolver):
    """
    Compact 1D convolutional deconvolver for pulse-trace reconstruction.

    Parameters
    ----------
    sequence_length: int
        Length of the input sequences.
    filters: Tuple[int, int, int]
        Number of filters in each convolutional layer.
    dilation_rates: Tuple[int, int, int]
        Dilation rates for each convolutional layer.
    kernel_size: int
        Size of the convolutional kernels.
    optimizer: Union[str, tf.keras.optimizers.Optimizer]
        Optimizer for model compilation.
    loss: Union[str, tf.keras.losses.Loss]
        Loss function for model training.
    metrics: Tuple[Union[str, tf.keras.metrics.Metric]]
        Metrics for model evaluation.
    seed: Optional[int]
        Random seed for reproducibility.


    Notes
    -----
    Architecture:

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
    loss: Union[str, tf.keras.losses.Loss] = "huber"
    metrics: Tuple[Union[str, tf.keras.metrics.Metric]] = ("mae",)
    seed: Optional[int] = None

    # filled after build()
    model: tf.keras.Model = field(init=False, repr=False, default=None)
    history_: Optional[tf.keras.callbacks.History] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        if not isinstance(self.metrics, (tuple, list)):
            self.metrics = (self.metrics,)

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

        reconstruction = layers.Conv1D(
            1, kernel_size=1, activation="linear", padding="same", name="reconstruction"
        )(x)

        self.model = models.Model(
            inputs=inputs, outputs=reconstruction, name="DenseNetDeconvolver"
        )
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=list(self.metrics)
        )
        return self.model
