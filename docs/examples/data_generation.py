"""
Generating and Visualizing Signal Data
======================================

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

# %%
# Generate Synthetic Signal Dataset
# ---------------------------------
#
# We generate a dataset with `NUM_PEAKS` Gaussian pulses per signal.
# The peak amplitudes, positions, and widths are randomly chosen within
# specified ranges.

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200
sample_count = 12

generator = SignalDatasetGenerator(sequence_length=SEQUENCE_LENGTH)

kernel = kernel.Gaussian(
    amplitude=(10, 300),  # Amplitude range
    position=(10, 190),  # Peak position range
    width=10,
)

dataset = generator.generate(
    n_samples=sample_count,
    kernel=kernel,
    n_peaks=(3, 3),
    noise_std=0,  # Add some noise
    categorical_peak_count=False,
)

dataset.compute_region_of_interest(width_in_pixels=5)

dataset.plot(number_of_columns=3, number_of_samples=9)
