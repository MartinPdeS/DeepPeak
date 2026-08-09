"""
WaveNet Deconvolver: Reconstructing Clean Pulse Traces
======================================================================

This example demonstrates how to train DeepPeak's WaveNet as a deconvolver.

We will:
- Generate a dataset of noisy signals with random Gaussian peaks
- Build and train a WaveNet reconstruction model
- Visualize the training process and reconstructed signals

.. note::
    This example is fully reproducible and suitable for Sphinx-Gallery documentation.

"""

# %%
# Imports and reproducibility
# -----------------------------
import numpy as np

from DeepPeak.models import TrainingConfig, WaveNet
from DeepPeak.generation import SignalGenerator
from DeepPeak import Gaussian, UniformCount

np.random.seed(42)

# %%
# Generate synthetic dataset
# ---------------------------
NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

pulse_kernel = Gaussian(
    amplitude=(10, 20),
    position=(0.1, 0.9),
    width=(5, 10),
)

generator = SignalGenerator(sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    n_samples=1000,
    kernel=pulse_kernel,
    peak_count=UniformCount(bounds=(1, NUM_PEAKS)),
    noise_std=0.03,
    categorical_peak_count=False,
)

# %%
# Visualize observed and clean example signals
# -------------------------------------------------------------
_ = dataset.plot(
    number_of_samples=6,
    number_of_columns=3,
    reference_pulse_trace=dataset.clean_signals,
)

# %%
# Build and summarize the WaveNet deconvolver
# ------------------------------------------
wavenet = WaveNet(
    sequence_length=SEQUENCE_LENGTH,
    num_filters=64,
    num_dilation_layers=3,
    kernel_size=4,
    optimizer="adam",
    output_activation="linear",
    loss="huber",
    metrics=["mae"],
)

wavenet.build()

# %%
# Train against clean pulse traces
# --------------------
history = wavenet.fit(
    dataset.signals,
    dataset.clean_signals[..., None],
    config=TrainingConfig(epochs=40, batch_size=64, validation_split=0.2),
)

# %%
# Plot training history
# ---------------------
_ = wavenet.plot_model_history()
