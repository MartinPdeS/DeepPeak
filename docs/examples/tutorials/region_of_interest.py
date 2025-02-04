"""
Region of interest
==================

This example demonstrates how to:
  1. Generate synthetic signals composed of up to 3 Gaussian pulses.
  2. Compute a Region Of Interest (ROI) mask for those signals (where the pulses exist).
  3. Train a small 1D CNN model to predict the ROI from the signal alone.
  4. Visualize both the ground-truth and predicted ROIs, overlaid on the signals.

We use:
  - ``generate_signals_dataset`` to create the signals and peak metadata.
  - ``compute_rois_from_signals`` to convert peak positions into binary ROI masks.
  - ``build_ROI_model`` (a simple autoencoder-like CNN) to learn the ROI detection.
  - ``SignalPlotter`` to visualize signals, peaks, and ROI masks.

Finally, we add a custom curve overlay (the individual Gaussians) to see how
the predicted ROI aligns with the true peaks.
"""

# %%
# Imports
# -------
import numpy as np
from DeepPeak.signals import generate_signal_dataset
from DeepPeak.utils.visualization import plot_training_history, SignalPlotter
from DeepPeak.models import build_ROI_model, filter_predictions
from DeepPeak.utils.ROI import compute_rois_from_signals

# %%
# Generate Synthetic Data
# -----------------------
#
# Here we generate a dataset of 1D signals of length ``SEQUENCE_LENGTH``, each
# containing up to 3 Gaussian pulses. We add a small amount of noise. For each
# sample, the function returns:
#
# - ``signals``: the time-domain waveforms.
# - ``amplitudes``, ``positions``, ``widths``: the parameters of the Gaussians.
# - ``x_values``: the x-axis for reference (0..1).
# - ``num_peaks``: how many peaks are active in each sample.
#

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

signals, amplitudes, positions, widths, x_values, num_peaks = generate_signal_dataset(
    n_samples=6000,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)

# %%
# Create Ground-Truth ROI Masks
# -----------------------------
#
# We define a pulse's ROI by marking a region of ~3 pixels around each
# peak's center. The function ``compute_rois_from_signals`` returns a binary
# mask of shape ``(n_samples, SEQUENCE_LENGTH)`` indicating where pulses lie.

ROI = compute_rois_from_signals(
    signals=signals,
    positions=positions,
    width_in_pixels=3,
    amplitudes=amplitudes
)

# %%
# Visualize Signals and Ground-Truth ROI
# --------------------------------------
#
# We use ``SignalPlotter`` to see a few examples of signals, their peak
# positions, amplitudes, and the ROI mask overlay.

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_positions(positions)
plotter.add_amplitudes(amplitudes)
plotter.add_roi(ROI)
plotter.set_title("Demo: Signals + Peaks + ROI")
_ = plotter.plot(n_examples=6, n_columns=3, random_select=False)

# %%
# Build and Train an ROI Model
# ----------------------------
#
# We train a small 1D CNN autoencoder-like model with dropout. The model attempts
# to reconstruct the ROI mask given only the signal as input.

roi_model = build_ROI_model(SEQUENCE_LENGTH)

history = roi_model.fit(
    signals, ROI,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)

# %%
# Plot Training History
# ---------------------
#
# We can then check how the loss evolves over epochs.

_ = plot_training_history(history, filtering=['*loss*'])

# %%
# ROI Inference
# -------------
#
# We predict on the entire dataset for demonstration. We then threshold at 0.9
# to obtain a binary mask.
signals, amplitudes, positions, _, _, _ = generate_signal_dataset(
    n_samples=100,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)

predictions, uncertainty = filter_predictions(
    signals=signals,
    model=roi_model,
    n_samples=30,
    threshold=0.9
)


# %%
# Compare Predicted ROI with Original Signals
# -------------------------------------------
#
# We overlay the predicted ROI mask on the signals, and also draw the
# individual Gaussians using a custom curve function.

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_positions(positions)
plotter.add_amplitudes(amplitudes)
plotter.add_roi(predictions)

plotter.set_title("Demo: Signals + Peaks + ROI")
_ = plotter.plot(n_examples=6, n_columns=3, random_select=True)
