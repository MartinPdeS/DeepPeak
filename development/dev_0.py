"""
Non-Maximum Suppression for Gaussian Pulse Detection
====================================================

This example demonstrates the use of the NonMaximumSuppression class to detect
Gaussian pulses in a one-dimensional signal. It generates a synthetic dataset
of Gaussian pulses, applies the non-maximum suppression algorithm, and plots
the results.
"""

from DeepPeak.algorithms import NonMaximumSuppression
from DeepPeak.kernel import Gaussian
from DeepPeak.signal_generator import SignalGenerator

NUM_PEAKS = 3
SEQUENCE_LENGTH = 400

gaussian_width = 0.02

generator = SignalGenerator(sequence_length=SEQUENCE_LENGTH)

kernel = Gaussian(
    amplitude=(10, 300),
    position=(0.3, 0.7),
    width=gaussian_width,
)

dataset = generator.generate(
    n_samples=6,
    kernel=kernel,
    n_peaks=(3, 3),
    noise_std=2,  # Add some noise
    categorical_peak_count=False,
)
print(dataset)

dataset.plot()

# %%
# Configure and run the detector
peak_locator = NonMaximumSuppression(
    gaussian_sigma=0.005,
    threshold="auto",
    maximum_number_of_pulses=5,
    kernel_truncation_radius_in_sigmas=3,
)


result = peak_locator.run(time_samples=dataset.x_values, signal=dataset.signals[0])

# result.plot_kernel_vs_average_pulse()

batch = peak_locator.run_batch(time_samples=dataset.x_values, signal=dataset.signals)

# %%
# Plot the results
# batch.plot_histogram_counts()

# %%
# Plot the results
batch.plot(ncols=3, max_plots=6, ground_truth=dataset.positions)
