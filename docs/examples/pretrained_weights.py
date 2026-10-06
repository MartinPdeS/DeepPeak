"""
Train, Save, and Reload Models
==============================

This example trains compact WaveNet and U-Net models on a small synthetic
dataset, saves them using DeepPeak's model-artifact format, reloads them, and
compares both reconstructions with the clean target. The same save directories
can later be moved into ``DeepPeak/weights`` for reuse.
"""

# %%
# Imports and reproducibility
# ---------------------------
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from DeepPeak import Gaussian, SignalGenerator, UniformCount
from DeepPeak.models import TrainingConfig, UNet1D, WaveNet

rng = np.random.default_rng(42)
sequence_length = 256

# %%
# Generate a small synthetic training dataset
# ---------------------------------------------
generator = SignalGenerator(sequence_length=sequence_length)
kernel = Gaussian(
    amplitude=(0.6, 1.0),
    position=(0.1 * sequence_length, 0.9 * sequence_length),
    width=(3.0, 5.0),
)
dataset = generator.generate(
    n_samples=128,
    kernel=kernel,
    peak_count=UniformCount(bounds=(4, 4)),
    noise_std=0.04,
    seed=42,
)

# %%
# Train compact models and reload their saved artifacts
# ------------------------------------------------------
training_config = TrainingConfig(
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    patience=5,
    verbose=0,
    seed=42,
)
inputs = np.asarray(dataset.signals)[..., None]
targets = np.asarray(dataset.clean_signals)[..., None]

wavenet = WaveNet(
    sequence_length=sequence_length,
    num_filters=16,
    num_dilation_layers=3,
    kernel_size=3,
    output_activation="linear",
    loss="huber",
    metrics=("mae",),
)
unet = UNet1D(
    sequence_length=sequence_length,
    num_filters=16,
    num_levels=3,
    kernel_size=3,
    loss="huber",
    metrics=("mae",),
)

with TemporaryDirectory(prefix="deeppeak-example-weights-") as directory:
    directory = Path(directory)
    wavenet_path = directory / "wavenet-small"
    unet_path = directory / "unet-small"

    wavenet.fit(inputs, targets, config=training_config)
    unet.fit(inputs, targets, config=training_config)
    wavenet.save(wavenet_path)
    unet.save(unet_path)

    wavenet = WaveNet.load(wavenet_path)
    unet = UNet1D.load(unet_path)

# %%
# Predict on one held-out example
# -------------------------------
signal = inputs[:1]
clean = targets[0, ..., 0]

# %%
# Predict and compare reconstructions
# ------------------------------------
wavenet_prediction = np.asarray(wavenet.predict(signal, verbose=0))[0, ..., 0]
unet_prediction = np.asarray(unet.predict(signal, verbose=0))[0, ..., 0]

figure, axis = plt.subplots(figsize=(12, 4))
axis.plot(signal[0, :, 0], color="0.65", linewidth=1.0, label="observed")
axis.plot(clean, color="black", linewidth=1.4, label="clean target")
axis.plot(wavenet_prediction, linewidth=1.1, label="WaveNet")
axis.plot(unet_prediction, linewidth=1.1, label="UNet1D")
axis.set(
    title="Reconstructions after training, saving, and reloading",
    xlabel="Sample index",
    ylabel="Signal amplitude",
)
axis.legend()
axis.grid(alpha=0.2)
figure.tight_layout()
plt.show()
