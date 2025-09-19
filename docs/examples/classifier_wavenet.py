"""
DenseNet Classifier: Detecting Regions of Interest in Synthetic Signals
=======================================================================

This example demonstrates how to use DeepPeak's DenseNet classifier to identify
regions of interest (ROIs) in synthetic 1D signals containing Gaussian peaks.

We will:
- Generate a dataset of noisy signals with random Gaussian peaks
- Build and train a DenseNet classifier to detect ROIs
- Visualize the training process and model predictions

.. note::
    This example is fully reproducible and suitable for Sphinx-Gallery documentation.

"""

# %%
# Imports and reproducibility
# --------------------------
import numpy as np

from DeepPeak.machine_learning.classifier import WaveNet, BinaryIoU
from DeepPeak.signals import SignalDatasetGenerator
from DeepPeak import kernel

np.random.seed(42)

# %%
# Generate synthetic dataset
# -------------------------
NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

kernel = kernel.Lorentzian(
    amplitude=(1, 20),  # Amplitude range
    position=(0.1, 0.9),  # Peak position range
    width=(0.03, 0.05),
)

generator = SignalDatasetGenerator(n_samples=100, sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    kernel=kernel,
    n_peaks=(1, NUM_PEAKS),
    noise_std=0.1,
    categorical_peak_count=False,
    compute_region_of_interest=True,
)

# %%
# Visualize a few example signals and their regions of interest
# ------------------------------------------------------------
dataset.plot(number_of_samples=3)

# %%
# Build and summarize the WaveNet classifier
# ------------------------------------------
dense_net = WaveNet(
    sequence_length=SEQUENCE_LENGTH,
    num_filters=64,
    num_dilation_layers=6,
    kernel_size=3,
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", BinaryIoU(threshold=0.5)],
)
dense_net.build()
dense_net.summary()

# %%
# Train the classifier
# --------------------
history = dense_net.fit(
    dataset.signals,
    dataset.region_of_interest,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
)

# %%
# Plot training history
# ---------------------
dense_net.plot_model_history()

# %%
# Predict and visualize on a test signal
# --------------------------------------
dense_net.plot_prediction(signal=dataset.signals[0:1, :], threshold=0.4)
