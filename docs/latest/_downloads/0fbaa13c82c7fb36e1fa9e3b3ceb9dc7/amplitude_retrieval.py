"""
Non-Maximum Suppression for Gaussian Pulse Detection
====================================================

This example demonstrates the use of the NonMaximumSuppression class to detect
Gaussian pulses in a one-dimensional signal. It generates a synthetic dataset
of Gaussian pulses, applies the non-maximum suppression algorithm, and plots
the results.
"""

from DeepPeak.algorithms import NonMaximumSuppression
from DeepPeak.algorithms.amplitude import ClosedFormSolver
from DeepPeak.signals import Kernel, SignalDatasetGenerator

NUM_PEAKS = 3
SEQUENCE_LENGTH = 400

gaussian_width = 0.03

generator = SignalDatasetGenerator(n_samples=1, sequence_length=SEQUENCE_LENGTH)

dataset = generator.generate(
    signal_type=Kernel.GAUSSIAN,
    n_peaks=2,
    amplitude=(50, 100),  # Amplitude range
    position=(0.3, 0.6),  # Peak position range
    width=gaussian_width,  # Width range
    noise_std=0.3,  # Add some noise
    categorical_peak_count=False,
)

dataset.plot()

# %%
# Configure and run the detector
peak_locator = NonMaximumSuppression(
    gaussian_sigma=0.003,
    threshold="auto",
    maximum_number_of_pulses=3,
    kernel_truncation_radius_in_sigmas=3,
)

peak_locator.run(time_samples=dataset.x_values, signal=dataset.signals.squeeze())


peak_amplitudes = peak_locator.results["peak_amplitude"]
peak_centers = peak_locator.results["peak_times"]


solver = ClosedFormSolver(sigma=dataset.widths.mean().squeeze() / 1.5)

result = solver.run(centers=peak_centers, matched_responses=peak_locator.results["peak_amplitude"])


print(
    f"""
    True Centers:
    {dataset.positions}
    Measured Centers:
    {peak_locator.results['peak_times']}
    True Amplitudes:
    {dataset.amplitudes}
    Solved Amplitudes:
    {result}\n
    """
)

solver.plot(true_amplitudes=dataset.amplitudes.squeeze())
