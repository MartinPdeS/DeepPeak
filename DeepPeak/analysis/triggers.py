"""Typed trigger configuration for standard and WaveNet-based detection.

This module replaces ad hoc detector kwargs dictionaries with a small typed
object that can validate common trigger settings and convert them back into the
keyword arguments expected by the detection pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class PeakTrigger:
    """Typed trigger configuration for peak detection.

    Parameters
    ----------
    height : float, optional
        Absolute detection threshold.
    sigma : float, optional
        Threshold expressed as a multiple of a robust noise estimate. Exactly
        one of ``height`` or ``sigma`` must be provided.
    hysteresis : float, optional
        End-of-event threshold used by the hysteresis detector. When
        ``height`` is provided, this is interpreted as an absolute signal
        level. When ``sigma`` is provided, this is interpreted as a sigma
        multiple resolved from the same robust noise estimate as the opening
        threshold.
    pulse_polarity : {"positive", "negative"}, default="positive"
        Pulse polarity expected by the detector.
    holdoff_samples : int, default=0
        Number of samples ignored after one event is closed.
    required_samples_above_threshold : int, default=1
        Minimum number of consecutive samples required to open an event.
    required_samples_below_hysteresis : int, default=1
        Minimum number of consecutive samples required to close an event.
    low_pass : float, optional
        Optional low-pass cutoff applied before CNN peak detection.
    """

    height: Optional[float] = None
    sigma: Optional[float] = None
    hysteresis: Optional[float] = None
    pulse_polarity: str = "positive"
    holdoff_samples: int = 0
    required_samples_above_threshold: int = 1
    required_samples_below_hysteresis: int = 1
    low_pass: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.height is None) == (self.sigma is None):
            raise ValueError("Exactly one of `height` or `sigma` must be provided.")

        if self.pulse_polarity not in {"positive", "negative"}:
            raise ValueError("pulse_polarity must be either 'positive' or 'negative'.")

        if int(self.holdoff_samples) < 0:
            raise ValueError("holdoff_samples must be >= 0.")

        if int(self.required_samples_above_threshold) < 1:
            raise ValueError("required_samples_above_threshold must be >= 1.")

        if int(self.required_samples_below_hysteresis) < 1:
            raise ValueError("required_samples_below_hysteresis must be >= 1.")

        if (
            self.height is not None
            and self.hysteresis is not None
            and float(self.hysteresis) > float(self.height)
        ):
            raise ValueError(
                "hysteresis must be <= height (or None). "
                f"Got hysteresis={self.hysteresis} and height={self.height}."
            )

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert the trigger configuration into detector keyword arguments.

        Returns
        -------
        dict
            Keyword arguments compatible with the DeepPeak detection pipeline.
        """

        kwargs: Dict[str, Any] = {
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

        if self.height is not None:
            kwargs["height"] = float(self.height)

        if self.sigma is not None:
            kwargs["sigma"] = float(self.sigma)

        if self.low_pass is not None:
            kwargs["low_pass"] = float(self.low_pass)

        return kwargs

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PeakTrigger":
        """Build a trigger configuration from a mapping.

        Parameters
        ----------
        mapping : mapping
            Dictionary-like trigger configuration.

        Returns
        -------
        PeakTrigger
            Trigger configuration built from the mapping.
        """

        return cls(**dict(mapping))


TriggerLike = Union[PeakTrigger, Mapping[str, Any]]


def coerce_peak_trigger(
    trigger: Optional[TriggerLike],
    legacy_kwargs: Optional[Mapping[str, Any]],
    *,
    name: str,
) -> PeakTrigger:
    """Resolve trigger input supplied as a typed object or legacy kwargs.

    Parameters
    ----------
    trigger : PeakTrigger or mapping, optional
        Preferred trigger input.
    legacy_kwargs : mapping, optional
        Legacy dictionary-based trigger input.
    name : str
        Human-readable parameter name used in error messages.

    Returns
    -------
    PeakTrigger
        Validated trigger configuration.
    """

    if trigger is not None and legacy_kwargs is not None:
        raise ValueError(
            f"Provide either `{name}` or its legacy kwargs alias, not both."
        )

    resolved = trigger if trigger is not None else legacy_kwargs

    if resolved is None:
        raise ValueError(f"A trigger configuration is required for `{name}`.")

    if isinstance(resolved, PeakTrigger):
        return resolved

    if isinstance(resolved, Mapping):
        return PeakTrigger.from_mapping(resolved)

    raise TypeError(f"`{name}` must be a PeakTrigger or a mapping of trigger settings.")


__all__ = ["PeakTrigger", "TriggerLike", "coerce_peak_trigger"]
