"""
Generating and Visualizing Signal Data
======================================

This example demonstrates how to:
  1. Generate synthetic signals with up to 3 Gaussian pulses.
  2. Compute a Region of Interest (ROI) mask based on pulse positions.
  3. Visualize signals with peak positions, amplitudes, and the ROI mask.

We use:
  - ``generate_signal_dataset`` to create the signals.
  - ``compute_rois_from_signals`` to generate the ROI mask.
  - ``SignalPlotter`` to visualize the results.

"""

# %%
# Imports
# -------
from DeepPeak.signals import Kernel, SignalDatasetGenerator

# %%
# Generate Synthetic Signal Dataset
# ---------------------------------
#
# We generate a dataset with `NUM_PEAKS` Gaussian pulses per signal.
# The peak amplitudes, positions, and widths are randomly chosen within
# specified ranges.

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200
sample_count = 3

generator = SignalDatasetGenerator(n_samples=sample_count, sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    signal_type=Kernel.GAUSSIAN,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 100),  # Amplitude range
    position=(0.1, 0.9),  # Peak position range
    width=(0.03, 0.05),  # Width range
    noise_std=0.1,  # Add some noise
    categorical_peak_count=False,
    compute_region_of_interest=True,
)

dataset.plot()
