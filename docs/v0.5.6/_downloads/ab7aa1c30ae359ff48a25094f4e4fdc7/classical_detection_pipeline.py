"""
Classical Detection Pipeline with Stable Results
=================================================

Generate a noisy synthetic signal, run the classical detector, wrap its raw
output in a stable ``DetectionResult``, and visualize the result.
"""

# %%
# Generate a trace
# ----------------
import matplotlib.pyplot as plt
import numpy as np

from DeepPeak.core import DetectionResult, Trace
from DeepPeak.detection import find_peaks_standard
from DeepPeak.generation import Gaussian, SignalGenerator, UniformCount

sequence_length = 500
generator = SignalGenerator(sequence_length=sequence_length)
dataset = generator.generate(
    n_samples=1,
    kernel=Gaussian(
        amplitude=(4.0, 8.0),
        position=(80.0, 420.0),
        width=8.0,
    ),
    peak_count=UniformCount(bounds=(3, 3)),
    noise_std=0.15,
    seed=42,
)
trace = Trace(signal=dataset.signals[0], dx=1.0)

# %%
# Detect peaks and create a stable result object
# -----------------------------------------------
peak_indices, properties = find_peaks_standard(
    trace.signal,
    height=1.0,
    hysteresis=0.5,
    holdoff_samples=10,
)
detection = DetectionResult(
    peaks=peak_indices,
    properties=properties,
    detection_kwargs={
        "height": 1.0,
        "hysteresis": 0.5,
        "holdoff_samples": 10,
    },
    threshold=1.0,
)
print(f"Detected {detection.peak_count} peaks at {detection.peaks.tolist()}")

# %%
# Plot the detector output
# ------------------------
figure, axis = plt.subplots(figsize=(10, 3.5))
axis.plot(trace.signal, color="black", linewidth=0.9, label="signal")
axis.axhline(detection.threshold, color="C1", linestyle="--", label="threshold")
axis.scatter(
    detection.peaks,
    trace.signal[detection.peaks],
    color="C3",
    zorder=3,
    label="detected peaks",
)
axis.set_xlabel("Sample index")
axis.set_ylabel("Amplitude")
axis.legend()
figure.tight_layout()
figure
