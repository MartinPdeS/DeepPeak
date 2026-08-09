"""Validated configuration objects shared by DeepPeak domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .exceptions import InvalidConfigurationError


_NORMALIZATIONS = {"none", "zscore", "robust_zscore", "minmax", "min-max", "maxabs"}


@dataclass(frozen=True)
class TraceConfig:
    """Sampling and preprocessing settings for a trace workflow."""

    sequence_length: Optional[int] = None
    normalization: str = "zscore"
    sampling_rate_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sequence_length is not None and int(self.sequence_length) <= 0:
            raise InvalidConfigurationError(
                "sequence_length must be positive when provided."
            )
        normalized = str(self.normalization).strip().lower()
        if normalized not in _NORMALIZATIONS:
            choices = ", ".join(sorted(_NORMALIZATIONS))
            raise InvalidConfigurationError(f"normalization must be one of {choices}.")
        object.__setattr__(self, "normalization", normalized)
        if self.sampling_rate_hz is not None and float(self.sampling_rate_hz) <= 0:
            raise InvalidConfigurationError(
                "sampling_rate_hz must be positive when provided."
            )


@dataclass(frozen=True)
class DetectionConfig(TraceConfig):
    """Configuration shared by standard and neural detectors."""

    trigger: Any = None
    low_pass: Optional[float] = None
    amplitude_sigma_samples: Optional[float] = None
    amplitude_cluster_radius_sigma: float = 4.0
    amplitude_baseline: Optional[float | str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("low_pass", "amplitude_sigma_samples"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise InvalidConfigurationError(
                    f"{name} must be positive when provided."
                )
        if float(self.amplitude_cluster_radius_sigma) <= 0:
            raise InvalidConfigurationError(
                "amplitude_cluster_radius_sigma must be positive."
            )
        if isinstance(self.amplitude_baseline, str):
            if self.amplitude_baseline.strip().lower() != "median":
                raise InvalidConfigurationError(
                    'amplitude_baseline must be a number, "median", or None.'
                )
        elif self.amplitude_baseline is not None and not isinstance(
            self.amplitude_baseline, (int, float)
        ):
            raise InvalidConfigurationError(
                'amplitude_baseline must be a number, "median", or None.'
            )


@dataclass(frozen=True)
class SeriesConfig(TraceConfig):
    """Configuration shared by dilution-series workflows."""

    initial_concentration: float = 1.0
    nrows: int = 200_000
    low_pass: Optional[float] = None
    cnn_low_pass: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.initial_concentration) <= 0:
            raise InvalidConfigurationError("initial_concentration must be positive.")
        if int(self.nrows) <= 0:
            raise InvalidConfigurationError("nrows must be positive.")
        for name in ("low_pass", "cnn_low_pass"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise InvalidConfigurationError(
                    f"{name} must be positive when provided."
                )


@dataclass(frozen=True)
class PlotConfig:
    """Non-interactive plotting defaults shared by plotting entry points."""

    figsize: tuple[float, float] = (10.0, 4.0)
    show: bool = False
    close: bool = False
    dpi: int = 300
    title: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.figsize) != 2 or any(float(value) <= 0 for value in self.figsize):
            raise InvalidConfigurationError("figsize must contain two positive values.")
        if int(self.dpi) <= 0:
            raise InvalidConfigurationError("dpi must be positive.")


__all__ = ["DetectionConfig", "PlotConfig", "SeriesConfig", "TraceConfig"]
