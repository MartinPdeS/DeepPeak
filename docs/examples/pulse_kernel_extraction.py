"""
Extracting Pulse Kernels from Experimental Traces
==================================================

This example extracts representative pulse shapes and a small kernel library
from experimental CSV traces. The example data are stored in
``data/run_5_ref`` in the repository.
"""

# %%
# Imports and data paths
# ----------------------
from pathlib import Path

import matplotlib.pyplot as plt
from MPSPlots.styles import scientific

from DeepPeak.analysis import PulseShapeAnalyzer
from DeepPeak.io import CsvTrace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "run_5_ref"
KERNEL_ROOT = PROJECT_ROOT / "data" / "kernels"
KERNEL_ROOT.mkdir(parents=True, exist_ok=True)

# %%
# Inspect normalized pulse windows
# --------------------------------
trace = CsvTrace(DATA_ROOT / "dilution_300x_1.csv", n_rows=300_000)
analyzer = (
    PulseShapeAnalyzer.from_trace(trace, use_processed=True)
    .detect_peaks("35sigma", distance=10)
    .select_peaks(min_width_samples=10)
    .extract_windows(left_samples=160, right_samples=190)
)

with plt.style.context(scientific):
    figure = analyzer.plot_pulses(normalize=True)
    figure.axes[0].set_ylim(-0.1, 1.0)
    figure.axes[0].set_ylabel("Normalized amplitude")
    figure

# %%
# Extract and save a kernel library
# ---------------------------------
library_trace = CsvTrace(DATA_ROOT / "dilution_1000x_1.csv", n_rows=300_000)
library_analyzer = (
    PulseShapeAnalyzer.from_trace(library_trace, use_processed=True)
    .detect_peaks("15sigma", distance=10)
    .select_peaks(min_width_samples=10)
    .extract_windows(left_samples=200, right_samples=200)
)

library = library_analyzer.extract_kernel_library(
    smooth_sigma=5.0,
    baseline_samples=20,
    max_kernels=40,
    taper_fraction=0.1,
    reject_saturated=True,
    recenter="max",
    save_path=KERNEL_ROOT / "samples_0.npy",
    plot=True,
)

print(f"Extracted kernel library with shape {library.shape}")
