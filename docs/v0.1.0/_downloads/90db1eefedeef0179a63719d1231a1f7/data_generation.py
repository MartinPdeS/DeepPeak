"""
Generating and Visualizing Signal Data
======================================

This example demonstrates how to:
  1. Generate synthetic signals with up to 3 Gaussian pulses.
  2. Retain clean pulse traces as reconstruction targets.
  3. Visualize observed and clean pulse signals.
"""

# %%
# Imports
# -------
from DeepPeak.generation import SignalGenerator
from DeepPeak import Gaussian, UniformCount

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

generator = SignalGenerator(sequence_length=SEQUENCE_LENGTH)

pulse_kernel = Gaussian(
    amplitude=(10, 300),  # Amplitude range
    position=(10, 190),  # Peak position range
    width=10,
)

dataset = generator.generate(
    n_samples=sample_count,
    kernel=pulse_kernel,
    peak_count=UniformCount(bounds=(3, 3)),
    noise_std=0,  # Add some noise
    categorical_peak_count=False,
)

dataset.plot(
    number_of_columns=3,
    number_of_samples=9,
    reference_pulse_trace=dataset.clean_signals,
)
