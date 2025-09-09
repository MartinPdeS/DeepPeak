"""
Non-Maximum Suppression for Gaussian Pulse Detection
====================================================

This example demonstrates the use of the NonMaximumSuppression class to detect
Gaussian pulses in a one-dimensional signal. It generates a synthetic dataset
of Gaussian pulses, applies the non-maximum suppression algorithm, and plots
the results.
"""

from DeepPeak.algorithms import NonMaximumSuppression
from DeepPeak.signals import Kernel, SignalDatasetGenerator

NUM_PEAKS = 3
SEQUENCE_LENGTH = 400

gaussian_width = 0.03

generator = SignalDatasetGenerator(n_samples=200, sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    signal_type=Kernel.GAUSSIAN,
    n_peaks=(3, 3),
    amplitude=(10, 300),  # Amplitude range
    position=(0.3, 0.7),  # Peak position range
    width=gaussian_width,  # Width range
    noise_std=2,  # Add some noise
    categorical_peak_count=False,
)

# dataset.plot()

# %%
# Configure and run the detector
peak_locator = NonMaximumSuppression(
    gaussian_sigma=0.003,
    threshold="auto",
    maximum_number_of_pulses=5,
    kernel_truncation_radius_in_sigmas=3,
)

batch = peak_locator.run_batch(time_samples=dataset.x_values, signal=dataset.signals)

# %%
# Plot the results
batch.plot_histogram_counts()

# %%
# Plot the results
batch.plot(ncols=3, max_plots=6)
