"""
Noise and Pulse-Shape Analysis
==============================

Use the analysis layer to characterize noise regions and align detected pulses
for shape statistics.
"""

# %%
# Build a synthetic trace
# -----------------------
import matplotlib.pyplot as plt
import numpy as np

from DeepPeak.analysis import NoiseAnalyzer, PulseShapeAnalyzer

rng = np.random.default_rng(7)
x_values = np.arange(600, dtype=float)
signal = 0.08 * rng.normal(size=x_values.size)
for center, amplitude, width in [(150, 1.0, 8), (310, 1.4, 12), (470, 0.8, 7)]:
    signal += amplitude * np.exp(-0.5 * ((x_values - center) / width) ** 2)

# %%
# Estimate noise statistics and align pulse windows
# --------------------------------------------------
noise = NoiseAnalyzer(signal, dx=1.0).detect_noise(
    height=0.4,
    distance=20,
    left_guard=15,
    right_guard=20,
)
pulse = (
    PulseShapeAnalyzer(signal, dx=1.0)
    .detect_peaks(height=0.2, distance=20)
    .extract_windows(left_samples=30, right_samples=40, baseline_samples=10)
)

print(noise.noise_statistics()[["mean", "std", "robust_sigma"]])
print(pulse.shape_summary())

# %%
# Compare the retained noise and normalized pulse shapes
# --------------------------------------------------------
figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(noise.noise_samples, bins=40, color="C0", alpha=0.8)
axes[0].set_title("Retained noise")
axes[0].set_xlabel("Amplitude")
axes[0].set_ylabel("Count")

for window in pulse.normalized_windows():
    axes[1].plot(pulse.local_time, window, alpha=0.35, color="C1")
axes[1].plot(
    pulse.local_time,
    pulse.mean_pulse(),
    color="black",
    linewidth=2,
    label="mean pulse",
)
axes[1].set_title("Aligned pulse shapes")
axes[1].set_xlabel("Local sample offset")
axes[1].set_ylabel("Normalized amplitude")
axes[1].legend()
figure.tight_layout()
figure
