from typing import Optional, Tuple, Union, Iterable
from dataclasses import dataclass, field
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

from DeepPeak.machine_learning.classifier.base import BaseClassifier
from DeepPeak.machine_learning.classifier.metrics import BinaryIoU


@dataclass
class WaveNet(BaseClassifier):
    """
    WaveNet-style 1D detector for per-timestep peak classification.


    Parameters
    ----------
    sequence_length: int
        Length of the input sequences.
    num_filters: int
        Number of filters in the convolutional layers.
    num_dilation_layers: int
        Number of dilated convolutional layers.
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

    Architecture
    ------------
    - Input projection (1x1) to `num_filters` channels
    - Stack of dilated causal Conv1D blocks with exponentially increasing dilation
      (1, 2, 4, ..., 2^(L-1)), residual connections, and skip connections
    - Aggregated skip path -> ReLU -> 1x1 sigmoid for per-step probability

    Notes
    -----
    - Output shape: (batch, sequence_length, 1), sigmoid probabilities
    - Loss: binary_crossentropy (per time-step)
    - This class encapsulates build/fit/evaluate/predict and plotting utilities
    """

    sequence_length: int
    num_filters: int = 64
    num_dilation_layers: int = 6
    kernel_size: int = 3
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam"
    loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy"
    metrics: Tuple[Union[str, tf.keras.metrics.Metric]] = "accuracy"
    seed: Optional[int] = None

    # filled after build()
    model: tf.keras.Model = field(init=False, repr=False, default=None)
    history_: Optional[tf.keras.callbacks.History] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if not isinstance(self.metrics, (tuple, list)):
            self.metrics = (self.metrics,)

    # --------------------------------------------------------------------- #
    # Construction / compilation
    # --------------------------------------------------------------------- #
    def build(self) -> tf.keras.Model:
        """
        Build and compile the WaveNet model.
        """
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        inputs = layers.Input(shape=(self.sequence_length, 1), name="input")

        # Project input to the working channel dimension for residual additions
        x = layers.Conv1D(self.num_filters, 1, padding="same", name="input_projection")(inputs)

        skip_paths = []

        for i in range(self.num_dilation_layers):
            dilation = 2**i

            # Dilated causal conv
            h = layers.Conv1D(
                self.num_filters,
                kernel_size=self.kernel_size,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
                name=f"dilated_conv_{i}",
            )(x)

            # Residual (1x1) and add back to x
            res = layers.Conv1D(self.num_filters, 1, padding="same", name=f"res_{i}")(h)
            x = layers.Add(name=f"residual_add_{i}")([x, res])

            # Skip path (1x1) from the block output
            skip = layers.Conv1D(self.num_filters, 1, padding="same", name=f"skip_{i}")(x)
            skip_paths.append(skip)

        # Aggregate all skip connections
        s = layers.Add(name="skip_add")(skip_paths)
        s = layers.ReLU(name="post_relu")(s)

        # Final per-timestep probability (peak / no-peak)
        outputs = layers.Conv1D(1, 1, activation="sigmoid", name="output")(s)

        self.model = models.Model(inputs=inputs, outputs=outputs, name="WaveNetDetector")
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=list(self.metrics))
        return self.model
