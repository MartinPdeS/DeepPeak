"""
Generating data With a Custom Kernel
====================================

This example demonstrates how to:
  1. Generate synthetic signals with up to 3 Gaussian pulses.
  2. Retain clean pulse traces as reconstruction targets.
  3. Visualize observed and clean pulse signals.
"""

# %%
# Imports
# -------
from DeepPeak.generation import SignalGenerator
from DeepPeak import CustomKernel, UniformCount
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

_kernel = CustomKernel(kernel=_kernel, amplitude=(10, 300), position=(0.3, 0.7))

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200
sample_count = 12


x_values = np.linspace(0, 4, SEQUENCE_LENGTH)
generator = SignalGenerator(
    sequence_length=SEQUENCE_LENGTH,
    x_values=x_values,
)


dataset = generator.generate(
    n_samples=sample_count,
    kernel=_kernel,
    peak_count=UniformCount(bounds=(1, 1)),
    noise_std=(0, 1),  # Add some noise
    categorical_peak_count=False,
    drift=(0, 10),
)

dataset.plot(
    number_of_columns=3,
    number_of_samples=9,
    reference_pulse_trace=dataset.clean_signals,
)
