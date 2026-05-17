from typing import Any, Mapping, Optional, Tuple, Union
from dataclasses import dataclass, field
import tempfile
import os


import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore

from DeepPeak.machine_learning.classifier.base import BaseClassifier
from DeepPeak.machine_learning.classifier.losses import (
    ShapeAwarePulseLoss,
    WeightedBinaryCrossentropy,
    WeightedHuber,
    shape_aware_pulse_loss,
    weighted_bce,
    weighted_huber,
)
from DeepPeak.machine_learning.classifier.metrics import BinaryIoU


def _build_custom_object_map(
    custom_objects: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    resolved = {
        "BinaryIoU": BinaryIoU,
        "ShapeAwarePulseLoss": ShapeAwarePulseLoss,
        "WeightedBinaryCrossentropy": WeightedBinaryCrossentropy,
        "WeightedHuber": WeightedHuber,
        "shape_aware_pulse_loss": shape_aware_pulse_loss,
        "weighted_bce": weighted_bce,
        "weighted_huber": weighted_huber,
    }
    if custom_objects:
        resolved.update(custom_objects)
    return resolved


def _serialize_compile_identifier(identifier: Any) -> Any:
    if isinstance(identifier, str):
        return identifier
    return tf.keras.utils.serialize_keras_object(identifier)


def _deserialize_compile_identifier(
    identifier: Any,
    custom_objects: Mapping[str, Any],
) -> Any:
    if not isinstance(identifier, dict):
        return identifier
    try:
        return tf.keras.utils.deserialize_keras_object(
            identifier,
            custom_objects=dict(custom_objects),
        )
    except TypeError:
        return identifier


def _deserialize_optimizer_identifier(
    identifier: Any,
    custom_objects: Mapping[str, Any],
) -> Any:
    identifier = _deserialize_compile_identifier(identifier, custom_objects)

    if not isinstance(identifier, dict):
        return identifier

    return tf.keras.optimizers.deserialize(
        identifier,
        custom_objects=dict(custom_objects),
    )


def _resolve_loss_identifier(identifier: Any, custom_objects: Mapping[str, Any]) -> Any:
    serialized_identifier = identifier

    try:
        identifier = _deserialize_compile_identifier(identifier, custom_objects)
    except (TypeError, ValueError) as exc:
        if isinstance(serialized_identifier, dict):
            loss_name = serialized_identifier.get(
                "config",
                serialized_identifier.get("class_name", serialized_identifier),
            )
            raise ValueError(
                "Unknown loss identifier during WaveNet.load(): "
                f"{loss_name!r}. Pass it via custom_objects, for example "
                f"WaveNet.load(path, custom_objects={{'{loss_name}': your_loss}})."
            ) from exc
        raise

    if not isinstance(identifier, str):
        if isinstance(identifier, dict):
            try:
                return tf.keras.losses.deserialize(
                    identifier,
                    custom_objects=dict(custom_objects),
                )
            except (TypeError, ValueError) as exc:
                loss_name = identifier.get("class_name", identifier)
                raise ValueError(
                    "Unknown loss identifier during WaveNet.load(): "
                    f"{loss_name!r}. Pass it via custom_objects, for example "
                    f"WaveNet.load(path, custom_objects={{'{loss_name}': your_loss}})."
                ) from exc
        return identifier

    if identifier in custom_objects:
        loss = custom_objects[identifier]
        return loss() if isinstance(loss, type) else loss

    try:
        tf.keras.losses.get(identifier)
    except ValueError as exc:
        raise ValueError(
            "Unknown loss identifier during WaveNet.load(): "
            f"{identifier!r}. Pass it via custom_objects, for example "
            f"WaveNet.load(path, custom_objects={{'{identifier}': your_loss}})."
        ) from exc

    return identifier


def _resolve_metric_identifier(
    identifier: Any, custom_objects: Mapping[str, Any]
) -> Any:
    identifier = _deserialize_compile_identifier(identifier, custom_objects)

    if not isinstance(identifier, str):
        if isinstance(identifier, dict):
            return tf.keras.metrics.deserialize(
                identifier,
                custom_objects=dict(custom_objects),
            )
        return identifier

    if identifier in custom_objects:
        metric = custom_objects[identifier]
        return metric() if isinstance(metric, type) else metric

    return identifier


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

    Notes
    -----

    Architecture:

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
    history_: Optional[tf.keras.callbacks.History] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        if not isinstance(self.metrics, (tuple, list)):
            self.metrics = (self.metrics,)

        self.histories = []

    # --------------------------------------------------------------------- #
    # Construction / compilation
    # --------------------------------------------------------------------- #
    def build(self) -> tf.keras.Model:
        """
        Build and compile the WaveNet model.
        """
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        inputs = layers.Input(shape=(None, 1), name="input")

        # Project input to the working channel dimension for residual additions
        x = layers.Conv1D(self.num_filters, 1, padding="same", name="input_projection")(
            inputs
        )

        skip_paths = []

        for i in range(self.num_dilation_layers):
            dilation = 2**i

            # Dilated causal conv
            h = layers.Conv1D(
                self.num_filters,
                kernel_size=self.kernel_size,
                padding="same",
                dilation_rate=dilation,
                activation="relu",
                name=f"dilated_conv_{i}",
            )(x)

            # Residual (1x1) and add back to x
            res = layers.Conv1D(self.num_filters, 1, padding="same", name=f"res_{i}")(h)
            x = layers.Add(name=f"residual_add_{i}")([x, res])

            # Skip path (1x1) from the block output
            skip = layers.Conv1D(self.num_filters, 1, padding="same", name=f"skip_{i}")(
                h
            )
            skip_paths.append(skip)

        # Aggregate all skip connections
        s = layers.Add(name="skip_add")(skip_paths)
        s = layers.ReLU(name="post_relu")(s)

        # Final per-timestep probability (peak / no-peak)
        outputs = layers.Conv1D(1, 1, activation="sigmoid", name="output")(s)
        # outputs = layers.Conv1D(1, 1, activation=None, name="output")(s)

        self.model = models.Model(
            inputs=inputs, outputs=outputs, name="WaveNetDetector"
        )
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=list(self.metrics)
        )
        return self.model

    def save(self, path: str):
        """
        Save the WaveNet model, its weights, and training history to a directory.

        Parameters
        ----------
        path : str
            Directory where to save model components.
        """
        import os, json

        os.makedirs(path, exist_ok=True)

        # --- Build config dict ---
        config = {
            "sequence_length": self.sequence_length,
            "num_filters": self.num_filters,
            "num_dilation_layers": self.num_dilation_layers,
            "kernel_size": self.kernel_size,
            "optimizer": _serialize_compile_identifier(self.optimizer),
            "loss": _serialize_compile_identifier(self.loss),
            "metrics": [_serialize_compile_identifier(m) for m in self.metrics],
            "seed": self.seed,
        }

        # --- Write config.json ---
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # --- Save weights ---
        self.model.save_weights(os.path.join(path, ".weights.h5"))

        # --- Save training history ---
        history_data = self.history if hasattr(self, "history") else None

        with open(os.path.join(path, "history.json"), "w") as f:
            json.dump(history_data, f, indent=2)

        print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        *,
        custom_objects: Optional[Mapping[str, Any]] = None,
    ) -> "WaveNet":
        """
        Load a complete WaveNet instance (architecture, weights, and metadata)
        from a directory or a single .h5/.keras file.

        Parameters
        ----------
        path : str
            Directory or file path where the model was previously saved.
            - If `.keras` or `.h5`, loads a full Keras model directly.
            - Otherwise, expects a directory with:
                config.json, weights.h5, and (optionally) history.json.

        Returns
        -------
        WaveNet
            Fully reconstructed WaveNet instance.
        """
        import os
        import json
        from tensorflow import keras

        resolved_custom_objects = _build_custom_object_map(custom_objects)

        # Case 1 — direct .h5 or .keras model file
        if os.path.isfile(path) and (path.endswith(".h5") or path.endswith(".keras")):
            model = keras.models.load_model(
                path, custom_objects=resolved_custom_objects
            )
            instance = cls(
                sequence_length=model.input_shape[1],
                num_filters=model.get_layer("input_projection").filters,
                num_dilation_layers=len(
                    [l for l in model.layers if l.name.startswith("dilated_conv_")]
                ),
                kernel_size=model.get_layer("dilated_conv_0").kernel_size[0],
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            )
            instance.model = model
            print(f"Loaded full model from file: {path}")
            return instance

        # Case 2 — directory-based save
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, ".weights.h5")
        history_path = os.path.join(path, "history.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {path}")

        # Load config.json
        with open(config_path, "r") as f:
            config = json.load(f)

        # Map metric names back to instances
        metrics = [
            _resolve_metric_identifier(metric, resolved_custom_objects)
            for metric in config["metrics"]
        ]
        loss = _resolve_loss_identifier(config["loss"], resolved_custom_objects)
        optimizer = _deserialize_optimizer_identifier(
            config["optimizer"],
            resolved_custom_objects,
        )

        # Instantiate model class
        instance = cls(
            sequence_length=config["sequence_length"],
            num_filters=config["num_filters"],
            num_dilation_layers=config["num_dilation_layers"],
            kernel_size=config["kernel_size"],
            optimizer=optimizer,
            loss=loss,
            metrics=tuple(metrics),
            seed=config.get("seed"),
        )

        # Build Keras model architecture
        instance.build()

        # Load weights if available
        if os.path.exists(weights_path):
            instance.model.load_weights(weights_path)
            print(f"Weights loaded from {weights_path}")

        # Load history if available
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                instance.history = json.load(f)
            print(f"Training history loaded from {history_path}")

        print(f"WaveNet instance fully reconstructed from {path}")
        return instance
