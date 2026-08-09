from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

from .base import BaseDeconvolver
from .wavenet import (
    _build_custom_object_map,
    _deserialize_optimizer_identifier,
    _resolve_loss_identifier,
    _resolve_metric_identifier,
    _serialize_compile_identifier,
)


def _match_time_length(
    x: tf.Tensor,
    reference: tf.Tensor,
    *,
    name: str,
) -> tf.Tensor:
    """Crop or right-pad an upsampled tensor to match a skip connection length."""

    def _align(tensors):
        source, target = tensors
        target_length = tf.shape(target)[1]
        source = source[:, :target_length, :]
        current_length = tf.shape(source)[1]
        pad_amount = tf.maximum(target_length - current_length, 0)
        return tf.pad(source, [[0, 0], [0, pad_amount], [0, 0]])

    return layers.Lambda(_align, name=name)([x, reference])


@dataclass
class UNet1D(BaseDeconvolver):
    """1D U-Net for pulse-trace deconvolution."""

    sequence_length: int
    num_filters: int = 32
    num_levels: int = 4
    kernel_size: int = 3
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam"
    loss: Union[str, tf.keras.losses.Loss] = "huber"
    metrics: Tuple[Union[str, tf.keras.metrics.Metric]] = ("mae",)
    seed: Optional[int] = None

    model: tf.keras.Model = field(init=False, repr=False, default=None)
    history_: Optional[tf.keras.callbacks.History] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        if not isinstance(self.metrics, (tuple, list)):
            self.metrics = (self.metrics,)
        self.histories = []

    def _conv_block(
        self,
        x: tf.Tensor,
        filters: int,
        *,
        prefix: str,
    ) -> tf.Tensor:
        x = layers.Conv1D(
            filters,
            self.kernel_size,
            padding="same",
            activation="relu",
            name=f"{prefix}_conv_a",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn_a")(x)
        x = layers.Conv1D(
            filters,
            self.kernel_size,
            padding="same",
            activation="relu",
            name=f"{prefix}_conv_b",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn_b")(x)
        return x

    def build(self) -> tf.keras.Model:
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        inputs = layers.Input(shape=(self.sequence_length, 1), name="input")
        x = inputs
        skips = []

        for level in range(self.num_levels):
            filters = self.num_filters * (2**level)
            x = self._conv_block(x, filters, prefix=f"encoder_{level}")
            skips.append(x)
            x = layers.MaxPooling1D(
                pool_size=2,
                padding="same",
                name=f"encoder_pool_{level}",
            )(x)

        x = self._conv_block(
            x,
            self.num_filters * (2**self.num_levels),
            prefix="bottleneck",
        )

        for level in reversed(range(self.num_levels)):
            filters = self.num_filters * (2**level)
            x = layers.UpSampling1D(size=2, name=f"decoder_up_{level}")(x)
            x = _match_time_length(
                x,
                skips[level],
                name=f"decoder_match_{level}",
            )
            x = layers.Concatenate(axis=-1, name=f"decoder_concat_{level}")(
                [x, skips[level]]
            )
            x = self._conv_block(x, filters, prefix=f"decoder_{level}")

        outputs = layers.Conv1D(1, 1, activation="linear", name="reconstruction")(x)

        self.model = models.Model(
            inputs=inputs, outputs=outputs, name="UNet1DDeconvolver"
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=list(self.metrics),
        )
        return self.model

    def save(self, path: str):
        import json
        import os

        os.makedirs(path, exist_ok=True)
        config = {
            "sequence_length": self.sequence_length,
            "num_filters": self.num_filters,
            "num_levels": self.num_levels,
            "kernel_size": self.kernel_size,
            "optimizer": _serialize_compile_identifier(self.optimizer),
            "loss": _serialize_compile_identifier(self.loss),
            "metrics": [_serialize_compile_identifier(m) for m in self.metrics],
            "seed": self.seed,
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self.model.save_weights(os.path.join(path, ".weights.h5"))

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
    ) -> "UNet1D":
        import json
        import os
        from tensorflow import keras

        resolved_custom_objects = _build_custom_object_map(custom_objects)
        resolved_custom_objects = {
            **resolved_custom_objects,
            "_align": lambda tensors: tensors,
        }

        if os.path.isfile(path) and (path.endswith(".h5") or path.endswith(".keras")):
            model = keras.models.load_model(
                path,
                custom_objects=resolved_custom_objects,
                safe_mode=False,
            )
            instance = cls(
                sequence_length=model.input_shape[1],
                num_filters=model.get_layer("encoder_0_conv_a").filters,
                num_levels=len(
                    [
                        layer
                        for layer in model.layers
                        if layer.name.startswith("encoder_pool_")
                    ]
                ),
                kernel_size=model.get_layer("encoder_0_conv_a").kernel_size[0],
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            )
            instance.model = model
            print(f"Loaded full model from file: {path}")
            return instance

        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, ".weights.h5")
        history_path = os.path.join(path, "history.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        metrics = [
            _resolve_metric_identifier(metric, resolved_custom_objects)
            for metric in config["metrics"]
        ]
        loss = _resolve_loss_identifier(config["loss"], resolved_custom_objects)
        optimizer = _deserialize_optimizer_identifier(
            config["optimizer"],
            resolved_custom_objects,
        )

        instance = cls(
            sequence_length=config["sequence_length"],
            num_filters=config["num_filters"],
            num_levels=config["num_levels"],
            kernel_size=config["kernel_size"],
            optimizer=optimizer,
            loss=loss,
            metrics=tuple(metrics),
            seed=config.get("seed"),
        )
        instance.build()

        if os.path.exists(weights_path):
            instance.model.load_weights(weights_path)
            print(f"Weights loaded from {weights_path}")

        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                instance.history = json.load(f)
            print(f"Training history loaded from {history_path}")

        print(f"UNet1D instance fully reconstructed from {path}")
        return instance
