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
import DeepPeak

np.random.seed(42)

# %%
# Generate synthetic dataset
# -------------------------
NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

kernel = DeepPeak.kernel.Gaussian(
    amplitude=(10, 20),
    position=(0.1, 0.9),
    width=(5, 10),
)

generator = SignalDatasetGenerator(sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    n_samples=1000,
    kernel=kernel,
    n_peaks=(1, NUM_PEAKS),
    noise_std=0.03,
    categorical_peak_count=False,
)

dataset.compute_region_of_interest(width_in_pixels=5)

# %%
# Visualize a few example signals and their regions of interest
# ------------------------------------------------------------
_ = dataset.plot(number_of_samples=6, number_of_columns=3)

# %%
# Build and summarize the WaveNet classifier
# ------------------------------------------
wavenet = WaveNet(
    sequence_length=SEQUENCE_LENGTH,
    num_filters=64,
    num_dilation_layers=3,
    kernel_size=4,
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", BinaryIoU(threshold=0.5)],
)

wavenet.build()

# %%
# Train the classifier
# --------------------
history = wavenet.fit(
    dataset.signals,
    dataset.region_of_interest,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
)

# %%
# Plot training history
# ---------------------
_ = wavenet.plot_model_history()
