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
from DeepPeak.signals import generate_signal_dataset
from DeepPeak.visualization import SignalPlotter
from DeepPeak.classifier.utils import compute_rois_from_signals

# %%
# Generate Synthetic Signal Dataset
# ---------------------------------
#
# We generate a dataset with `NUM_PEAKS` Gaussian pulses per signal.
# The peak amplitudes, positions, and widths are randomly chosen within
# specified ranges.

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200
sample_count = 6000

signals, labels, amplitudes, positions, widths, x_values, num_peaks = generate_signal_dataset(
    n_samples=sample_count,
    signal_type='gaussian',
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 100),      # Amplitude range
    position=(0.1, 0.9),     # Peak position range
    width=(0.03, 0.05),      # Width range
    noise_std=0.1,           # Add some noise
    categorical_peak_count=False,
)

# %%
# Compute Region of Interest (ROI)
# --------------------------------
#
# We compute a binary mask that marks areas around peak centers as belonging to the ROI.
# The width of the ROI is defined in pixels.

ROI = compute_rois_from_signals(
    signals=signals,
    positions=positions,
    width_in_pixels=3,  # Define ROI width around each peak
    amplitudes=amplitudes
)

# %%
# Visualizing the Signals and ROI
# --------------------------------
#
# We use `SignalPlotter` to visualize the generated signals.
# The plot shows:
#
# - Blue curves: Original signals
# - Red dots: Peak positions
# - Green shading: Region of Interest (ROI)
# - Labels: Amplitude annotations

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_vline(positions)
plotter.add_hline(amplitudes)
plotter.add_roi(ROI)
plotter.set_title("Demo: Signals + Peaks + ROI")

# Display multiple examples in a grid
plotter.plot(n_examples=6, n_columns=3, random_select=False)
