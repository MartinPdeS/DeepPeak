"""
U-Net Deconvolver: Reconstructing Clean Pulse Traces
====================================================================

This example demonstrates how to train DeepPeak's U-Net as a deconvolver.

We will:
- Generate a dataset of noisy signals with random Gaussian peaks
- Build and train a U-Net reconstruction model
- Visualize the training process and reconstructed signals

.. note::
    This example is fully reproducible and suitable for Sphinx-Gallery documentation.

"""

# %%
# Imports and reproducibility
# -----------------------------
import numpy as np

from DeepPeak.models import TrainingConfig, UNet1D
from DeepPeak.generation import SignalGenerator
from DeepPeak import Lorentzian, UniformCount

np.random.seed(42)

# %%
# Generate synthetic dataset
# ---------------------------
NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

pulse_kernel = Lorentzian(
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
)

generator = SignalGenerator(sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    n_samples=100,
    kernel=pulse_kernel,
    peak_count=UniformCount(bounds=(1, NUM_PEAKS)),
    noise_std=0.1,
    categorical_peak_count=False,
)

# %%
# Visualize observed and clean example signals
# -------------------------------------------------------------
dataset.plot(number_of_samples=3, reference_pulse_trace=dataset.clean_signals)

# %%
# Build and summarize the U-Net deconvolver
# ------------------------------------------
unet = UNet1D(
    sequence_length=SEQUENCE_LENGTH,
    num_filters=32,
    num_levels=3,
    kernel_size=3,
    optimizer="adam",
    loss="huber",
    metrics=["mae"],
)
unet.build()
unet.summary()

# %%
# Train against clean pulse traces
# --------------------
history = unet.fit(
    dataset.signals,
    dataset.clean_signals[..., None],
    config=TrainingConfig(epochs=20, batch_size=64, validation_split=0.2),
)

# %%
# Plot training history
# ---------------------
unet.plot_model_history()
