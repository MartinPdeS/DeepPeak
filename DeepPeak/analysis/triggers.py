"""Typed trigger configuration for standard and WaveNet-based detection."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BasePeakTrigger:
    """Shared trigger configuration independent of threshold semantics."""

    hysteresis: Optional[float] = None
    pulse_polarity: str = "positive"
    holdoff_samples: int = 0
    required_samples_above_threshold: int = 1
    required_samples_below_hysteresis: int = 1

    def __post_init__(self) -> None:
        if self.pulse_polarity not in {"positive", "negative"}:
            raise ValueError("pulse_polarity must be either 'positive' or 'negative'.")

        if int(self.holdoff_samples) < 0:
            raise ValueError("holdoff_samples must be >= 0.")

        if int(self.required_samples_above_threshold) < 1:
            raise ValueError("required_samples_above_threshold must be >= 1.")

        if int(self.required_samples_below_hysteresis) < 1:
            raise ValueError("required_samples_below_hysteresis must be >= 1.")

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert the shared trigger configuration into detector kwargs."""

        return {
            "hysteresis": self.hysteresis,
            "pulse_polarity": self.pulse_polarity,
            "holdoff_samples": int(self.holdoff_samples),
            "required_samples_above_threshold": int(
                self.required_samples_above_threshold
            ),
            "required_samples_below_hysteresis": int(
                self.required_samples_below_hysteresis
            ),
        }


@dataclass(frozen=True)
class HeightPeakTrigger(BasePeakTrigger):
    """Trigger configuration using an absolute detection threshold."""

    height: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(float(self.height)):
            raise ValueError("height must be a finite float.")
        if self.hysteresis is not None and float(self.hysteresis) > float(self.height):
            raise ValueError(
                "hysteresis must be <= height (or None). "
                f"Got hysteresis={self.hysteresis} and height={self.height}."
            )

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs["height"] = float(self.height)
        return kwargs


@dataclass(frozen=True)
class SigmaPeakTrigger(BasePeakTrigger):
    """Trigger configuration using a sigma-derived detection threshold."""

    sigma: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(float(self.sigma)):
            raise ValueError("sigma must be a finite float.")

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs["sigma"] = float(self.sigma)
        return kwargs


__all__ = [
    "BasePeakTrigger",
    "HeightPeakTrigger",
    "SigmaPeakTrigger",
]
