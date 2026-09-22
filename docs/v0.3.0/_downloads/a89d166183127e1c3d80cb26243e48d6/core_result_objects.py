"""
Core Result Objects: Trace, DetectionResult, and MetricResult
==============================================================

This example introduces the lightweight result objects shared by the new
DeepPeak architecture. They contain numerical data and metadata, but no
TensorFlow or plotting dependencies.
"""

# %%
# Create a trace and a detection result
# -------------------------------------
import json

import matplotlib.pyplot as plt
import numpy as np

from DeepPeak.core import DetectionResult, MetricResult, Trace

rng = np.random.default_rng(42)
x_values = np.arange(400, dtype=float)
signal = 0.05 * rng.normal(size=x_values.size)
signal += np.exp(-0.5 * ((x_values - 120.0) / 7.0) ** 2)
signal += 0.7 * np.exp(-0.5 * ((x_values - 275.0) / 12.0) ** 2)

trace = Trace(signal=signal, dx=1.0, metadata={"source": "synthetic"})
detection = DetectionResult(
    peaks=np.array([120, 275]),
    properties={"peak_values": signal[[120, 275]]},
    detection_kwargs={"method": "known synthetic peaks"},
    threshold=0.25,
)
mean_amplitude = MetricResult(
    name="mean_amplitude",
    values=float(np.mean(signal[detection.peaks])),
    units="arbitrary amplitude",
)

# %%
# Serialize results for a report or experiment record
# ---------------------------------------------------
serialized = {
    "trace": trace.to_dict(),
    "detection": detection.to_dict(),
    "metric": mean_amplitude.to_dict(),
}
print(json.dumps(serialized, indent=2)[:500] + "...")

# %%
# Plot the result without coupling plotting to the result objects
# ----------------------------------------------------------------
figure, axis = plt.subplots(figsize=(9, 3.5))
axis.plot(x_values, trace.signal, color="C0", label="trace")
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
