"""
Generating data With a Custom Kernel
====================================

This example demonstrates how to:
  1. Generate synthetic signals with up to 3 Gaussian pulses.
  2. Compute a Region of Interest (ROI) mask based on pulse positions.
  3. Visualize signals with peak positions, amplitudes, and the ROI mask.
"""

# %%
# Imports
# -------
from DeepPeak.signals import SignalDatasetGenerator
from DeepPeak import kernel
import numpy as np

# %%
# Generate Synthetic Signal Dataset
# ---------------------------------
#
# We generate a dataset with `NUM_PEAKS` Gaussian pulses per signal.
# The peak amplitudes, positions, and widths are randomly chosen within
# specified ranges.


x = np.linspace(-1, 1, 600)
_kernel = np.exp(-((x + 0.05) ** 2) / (2 * (0.03**2))) - np.exp(
    -((x - 0.05) ** 2) / (2 * (0.03**2))
)

_kernel = kernel.CustomKernel(kernel=_kernel, amplitude=(10, 300), position=(0.3, 0.7))

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200
sample_count = 12


x_values = np.linspace(0, 4, SEQUENCE_LENGTH)
generator = SignalDatasetGenerator(
    sequence_length=SEQUENCE_LENGTH,
    x_values=x_values,
)


dataset = generator.generate(
    n_samples=sample_count,
    kernel=_kernel,
    n_peaks=(1, 1),
    noise_std=(0, 1),  # Add some noise
    categorical_peak_count=False,
    drift=(0, 10),
)

dataset.compute_region_of_interest(width_in_pixels=5)

dataset.plot(number_of_columns=3, number_of_samples=9)
