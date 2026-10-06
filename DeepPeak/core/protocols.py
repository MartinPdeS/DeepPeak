"""Protocols defining the boundaries between DeepPeak subsystems."""

from typing import Literal, Protocol

from .types import DetectionResult, MetricResult, Trace


class Detector(Protocol):
    """Common interface for classical and neural detectors."""

    def detect(
        self,
        trace: Trace,
        *,
        detector: Literal["standard", "cnn"] = "standard",
    ) -> DetectionResult:
        """Detect peaks in a trace."""


class MetricCalculator(Protocol):
    """Interface for pure metric calculators."""

    def compute(self, trace: Trace, detection: DetectionResult) -> MetricResult:
        """Compute metrics without creating plots or modifying the inputs."""
