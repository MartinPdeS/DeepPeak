from unittest.mock import patch
import json

import numpy as np
import pytest
import tensorflow as tf

from DeepPeak.machine_learning.classifier import (
    Autoencoder,
    DenseNet,
    ShapeAwarePulseLoss,
    WaveNet,
    WeightedBinaryCrossentropy,
    WeightedHuber,
    shape_aware_pulse_loss,
    weighted_bce,
    weighted_huber,
)
from DeepPeak.signal_generator import SignalGenerator
from DeepPeak.kernels import Gaussian
from DeepPeak.peak_count import UniformCount

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

architectures = [DenseNet, WaveNet, Autoencoder]


@pytest.fixture
def dataset():
    kernel = Gaussian(
        amplitude=(1, 20),
        position=(0.1, 0.9),
        width=(0.03, 0.05),
    )

    generator = SignalGenerator(sequence_length=SEQUENCE_LENGTH)

    dataset = generator.generate(
        n_samples=600,
        kernel=kernel,
        peak_count=UniformCount(bounds=(1, NUM_PEAKS)),
        noise_std=0.1,
        categorical_peak_count=False,
    )

    return dataset


@pytest.mark.parametrize("architecture", architectures)
@patch("matplotlib.pyplot.show")
def test_architecture(patch, dataset, architecture):
    model = architecture(
        sequence_length=200,
    )
    model.build()
    model.summary()

    roi = dataset.get_region_of_interest(width_in_pixels=5)

    history = model.fit(
        dataset.signals,
        roi,
        validation_split=0.2,
        epochs=4,
        batch_size=64,
    )

    model.plot_model_history()

    model.predict(signal=dataset.signals[0:1, :], threshold=0.4)


def test_wavenet_can_resume_training_with_serializable_weighted_bce(tmp_path):
    x = np.random.rand(16, 32, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(16, 32, 1)).astype(np.float32)

    model = WaveNet(
        sequence_length=32,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=WeightedBinaryCrossentropy(alpha=4.0),
        metrics=("accuracy",),
    )
    model.build()
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    save_path = tmp_path / "serializable-wavenet"
    model.save(str(save_path))

    loaded = WaveNet.load(str(save_path))
    history = loaded.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_accepts_weighted_bce_function_style():
    x = np.random.rand(8, 24, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(8, 24, 1)).astype(np.float32)

    model = WaveNet(
        sequence_length=24,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=weighted_bce,
        metrics=("accuracy",),
    )
    model.build()
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_accepts_weighted_bce_with_2d_targets():
    x = np.random.rand(8, 1500, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(8, 1500)).astype(np.float32)

    model = WaveNet(
        sequence_length=1500,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=weighted_bce(alpha=4.0),
        metrics=("accuracy",),
    )
    model.build()
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_accepts_weighted_huber_with_2d_targets():
    x = np.random.rand(8, 1500, 1).astype(np.float32)
    y = np.random.rand(8, 1500).astype(np.float32)

    model = WaveNet(
        sequence_length=1500,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=weighted_huber(alpha=4.0, delta=0.2),
        metrics=("accuracy",),
    )
    model.build()
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_accepts_shape_aware_pulse_loss_function_style():
    x = np.random.rand(8, 64, 1).astype(np.float32)
    y = np.random.rand(8, 64, 1).astype(np.float32)

    model = WaveNet(
        sequence_length=64,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=shape_aware_pulse_loss,
        metrics=("accuracy",),
    )
    model.build()
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_can_resume_training_with_serializable_weighted_huber(tmp_path):
    x = np.random.rand(16, 32, 1).astype(np.float32)
    y = np.random.rand(16, 32, 1).astype(np.float32)

    model = WaveNet(
        sequence_length=32,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=WeightedHuber(alpha=3.0, delta=0.25),
        metrics=("accuracy",),
    )
    model.build()
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    save_path = tmp_path / "serializable-weighted-huber-wavenet"
    model.save(str(save_path))

    loaded = WaveNet.load(str(save_path))
    history = loaded.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_can_resume_training_with_serializable_shape_aware_loss(tmp_path):
    x = np.random.rand(16, 32, 1).astype(np.float32)
    y = np.random.rand(16, 32, 1).astype(np.float32)

    model = WaveNet(
        sequence_length=32,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=ShapeAwarePulseLoss(alpha=2.0, delta=0.25, derivative_weight=0.5),
        metrics=("accuracy",),
    )
    model.build()
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    save_path = tmp_path / "serializable-shape-aware-wavenet"
    model.save(str(save_path))

    loaded = WaveNet.load(str(save_path))
    history = loaded.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_load_requires_custom_objects_for_unknown_notebook_loss(tmp_path):
    def notebook_weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean((1.0 + 4.0 * y_true) * bce)

    x = np.random.rand(12, 24, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(12, 24, 1)).astype(np.float32)

    model = WaveNet(
        sequence_length=24,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=notebook_weighted_loss,
        metrics=("accuracy",),
    )
    model.build()
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    save_path = tmp_path / "notebook-loss-wavenet"
    model.save(str(save_path))

    loaded_without_custom_objects = WaveNet.load(str(save_path))

    with pytest.raises(
        (ValueError, TypeError),
        match="notebook_weighted_loss|custom_objects|Could not interpret",
    ):
        loaded_without_custom_objects.fit(x, y, epochs=1, batch_size=4, verbose=0)

    loaded = WaveNet.load(
        str(save_path),
        custom_objects={"notebook_weighted_loss": notebook_weighted_loss},
    )
    history = loaded.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_load_accepts_package_weighted_bce_name_without_custom_objects(
    tmp_path,
):
    x = np.random.rand(12, 24, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(12, 24, 1)).astype(np.float32)

    model = WaveNet(
        sequence_length=24,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=weighted_bce,
        metrics=("accuracy",),
    )
    model.build()
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    save_path = tmp_path / "package-weighted-bce-wavenet"
    model.save(str(save_path))

    loaded = WaveNet.load(str(save_path))
    history = loaded.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_wavenet_load_accepts_dict_serialized_compile_config(tmp_path):
    x = np.random.rand(8, 32, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(8, 32, 1)).astype(np.float32)

    model = WaveNet(
        sequence_length=32,
        num_filters=4,
        num_dilation_layers=2,
        kernel_size=3,
        loss=WeightedBinaryCrossentropy(alpha=2.0),
        metrics=(tf.keras.metrics.BinaryAccuracy(name="BinaryAccuracy"),),
    )
    model.build()
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    save_path = tmp_path / "dict-config-wavenet"
    model.save(str(save_path))

    config_path = save_path / "config.json"
    with config_path.open("r") as handle:
        config = json.load(handle)

    config["optimizer"] = tf.keras.optimizers.serialize(tf.keras.optimizers.Adam())
    config["loss"] = tf.keras.losses.serialize(WeightedBinaryCrossentropy(alpha=2.0))
    config["metrics"] = [
        tf.keras.metrics.serialize(
            tf.keras.metrics.BinaryAccuracy(name="BinaryAccuracy")
        )
    ]

    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)

    loaded = WaveNet.load(str(save_path))

    history = loaded.fit(x, y, epochs=1, batch_size=4, verbose=0)
    assert "loss" in history.history


if __name__ == "__main__":
    pytest.main([__file__])
