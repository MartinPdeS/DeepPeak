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


@dataclass(frozen=True)
class ProminencePeakTrigger(BasePeakTrigger):
    """Trigger configuration using peak prominence as the detection criterion.

    Unlike :class:`HeightPeakTrigger` and :class:`SigmaPeakTrigger`, which
    require the signal to cross an absolute threshold, this trigger accepts
    every local maximum whose prominence is at least *min_prominence*.

    Prominence measures how much a peak stands above the surrounding baseline:
    it is the peak height minus the highest saddle that connects the peak to
    any strictly higher neighbour.  Peaks that project strongly above the local
    baseline are accepted even when the baseline itself is low, making this
    trigger robust to slow baseline drift.

    Parameters
    ----------
    min_prominence : float
        Minimum prominence (in signal amplitude units) required to accept a
        peak.
    wlen : int, optional
        Window half-length (in samples) used to compute prominence locally.
        When ``None`` (default) the full signal is used as the reference for
        every peak — accurate but sensitive to the global signal range.
        Providing a finite *wlen* restricts the reference to a neighbourhood
        of each peak, which is useful when the baseline varies slowly.

    Notes
    -----
    ``hysteresis``, ``required_samples_above_threshold``, and
    ``required_samples_below_hysteresis`` inherited from
    :class:`BasePeakTrigger` are not used by the prominence detector.
    Only ``pulse_polarity`` and ``holdoff_samples`` are forwarded.
    """

    min_prominence: float = 0.0
    wlen: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(float(self.min_prominence)):
            raise ValueError("min_prominence must be a finite float.")
        if self.min_prominence < 0.0:
            raise ValueError("min_prominence must be >= 0.")
        if self.wlen is not None and int(self.wlen) < 2:
            raise ValueError("wlen must be >= 2 when provided.")

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs["prominence"] = float(self.min_prominence)
        kwargs["wlen"] = None if self.wlen is None else int(self.wlen)
        return kwargs


__all__ = [
    "BasePeakTrigger",
    "HeightPeakTrigger",
    "ProminencePeakTrigger",
    "SigmaPeakTrigger",
]
