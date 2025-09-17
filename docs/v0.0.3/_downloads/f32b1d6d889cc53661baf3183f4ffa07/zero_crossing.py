"""
Zero Crossing Peak Detection Example
====================================

This example demonstrates the use of the ZeroCrossing class to detect
zero-crossing points in a one-dimensional signal. It generates a synthetic dataset
of Gaussian pulses, applies the zero-crossing detection algorithm, and plots
the results.
"""

from DeepPeak.algorithms import ZeroCrossing
from DeepPeak.signals import Kernel, SignalDatasetGenerator

NUM_PEAKS = 3
SEQUENCE_LENGTH = 400

gaussian_width = 0.02

generator = SignalDatasetGenerator(n_samples=6, sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    signal_type=Kernel.GAUSSIAN,
    n_peaks=(3, 3),
    amplitude=(10, 300),  # Amplitude range
    position=(0.3, 0.7),  # Peak position range
    width=gaussian_width,  # Width range
    noise_std=2,  # Add some noise
    categorical_peak_count=False,
)
print(dataset)

dataset.plot()

# %%
# Configure and run the detector
peak_locator = ZeroCrossing(
    gaussian_sigma=0.005,
    threshold="auto",
    threshold_k=3.0,
)

batch = peak_locator.run_batch(time_samples=dataset.x_values, signal=dataset.signals)

# %%
# Plot the results
# batch.plot_histogram_counts()

# %%
# Plot the results
batch.plot(
    ncols=3,
    max_plots=6,
    # ground_truth=dataset.positions
)
