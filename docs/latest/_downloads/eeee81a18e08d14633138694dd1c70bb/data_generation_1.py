"""
Reference pulse trace labels
============================

This example shows how to generate a smooth label trace aligned with the true
pulse positions and amplitudes. This can be used as a training target to reduce
class imbalance compared to a binary ROI mask.
"""

import numpy as np
import matplotlib.pyplot as plt

from DeepPeak.signals import SignalDatasetGenerator
from DeepPeak import kernel

# If DataSet is part of your package, import it from there instead
# from DeepPeak.signals import DataSet


# %%
# Generate a synthetic dataset
# ----------------------------
SEQUENCE_LENGTH = 200
sample_count = 12
number_of_peaks = 3

generator = SignalDatasetGenerator(sequence_length=SEQUENCE_LENGTH)

pulse_kernel = kernel.Lorentzian(
    amplitude=(10, 300),
    position=(10, 190),
    width=10,
)

dataset = generator.generate(
    n_samples=sample_count,
    kernel=pulse_kernel,
    n_peaks=(number_of_peaks, number_of_peaks),
    noise_std=5.0,
    categorical_peak_count=False,
)

# %%
# ROI mask (binary) around each peak
# ----------------------------------
roi_width_in_pixels = 3
dataset.region_of_interest = dataset.get_region_of_interest(
    width_in_pixels=roi_width_in_pixels
)

# %%
# Build a smooth reference label trace
# ------------------------------------
# This places one pulse per ground truth peak using dataset.positions and dataset.amplitudes.
# The output is a dense target that can be used for regression training or thresholded later.
reference_width = 2  # same scale as the kernel width used to generate
reference = dataset.get_reference_pulse_trace(
    width=reference_width,
    amplitude=None,  # use dataset.amplitudes
    profile="gaussian",  # match the generator
    width_definition="fwhm",
    normalize_peak_to_one=True,
)

# %%
# Visualize a few samples
# -----------------------
number_of_rows = 3
number_of_columns = 3
number_to_plot = min(sample_count, number_of_rows * number_of_columns)

fig, axes = plt.subplots(
    nrows=number_of_rows,
    ncols=number_of_columns,
    figsize=(12, 8),
    squeeze=False,
)

for plot_index, ax in zip(range(number_to_plot), axes.flatten()):
    signal = dataset.signals[plot_index]
    roi = dataset.region_of_interest[plot_index]
    reference_trace = reference[plot_index]

    ax.plot(dataset.x_values, signal, color="black", label="signal")

    roi_patch = ax.fill_between(
        dataset.x_values,
        y1=0,
        y2=1,
        where=(roi != 0),
        color="lightblue",
        alpha=1.0,
        transform=ax.get_xaxis_transform(),
        label="roi mask",
    )

    ax.plot(dataset.x_values, reference_trace, label="reference trace")

    ax.set_title(f"Sample {plot_index}")

handles, labels = axes[0, 0].get_legend_handles_labels()
by_label = {}
for h, l in zip(handles, labels):
    if l and not l.startswith("_") and l not in by_label:
        by_label[l] = h
fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3)

fig.supxlabel("Time step [AU]", y=0.02)
fig.supylabel("Signal [AU]", x=0.02)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# %%
# Check class balance improvement
# -------------------------------
roi_positive_fraction = float(np.mean(dataset.region_of_interest != 0))
reference_positive_fraction = float(np.mean(reference > 0.1))

print(f"ROI positive fraction: {roi_positive_fraction:.5f}")
print(f"Reference trace fraction above 0.1: {reference_positive_fraction:.5f}")
