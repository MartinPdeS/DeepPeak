"""Stable, dependency-light result and input objects.

These types deliberately contain no plotting or TensorFlow dependencies and
are the canonical data contract shared by the package's domain modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _as_1d_array(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    return array


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/container values into JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class Trace:
    """A one-dimensional signal and its sampling metadata.

    Parameters
    ----------
    signal : array-like
        One-dimensional signal samples.
    dx : float, default=1.0
        Distance between adjacent samples, in the caller's units.
    filename : path-like, optional
        Source file when the trace originated from disk.
    metadata : mapping, optional
        Additional acquisition or preprocessing metadata.
    """

    signal: np.ndarray
    dx: float = 1.0
    filename: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = _as_1d_array(self.signal, dtype=float, name="signal")
        self.dx = float(self.dx)
        if not np.isfinite(self.dx) or self.dx <= 0.0:
            raise ValueError("dx must be a finite, strictly positive number.")
        if self.filename is not None:
            self.filename = Path(self.filename)
        self.metadata = dict(self.metadata)

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def duration(self) -> float:
        """Return the duration represented by the samples."""

        return float(self.n_samples * self.dx)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the trace metadata."""

        return {
            "signal": self.signal.tolist(),
            "dx": self.dx,
            "filename": None if self.filename is None else str(self.filename),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trace":
        """Reconstruct a trace from :meth:`to_dict` output."""

        return cls(
            signal=data["signal"],
            dx=data.get("dx", 1.0),
            filename=data.get("filename"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DetectionResult:
    """Peaks detected in a trace and the configuration used to find them."""

    peaks: np.ndarray
    properties: Mapping[str, Any] = field(default_factory=dict)
    peak_count: Optional[int] = None
    detection_kwargs: Mapping[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    amplitudes: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.peaks = _as_1d_array(self.peaks, dtype=int, name="peaks")
        if np.any(self.peaks < 0):
            raise ValueError("peaks cannot contain negative indices.")
        self.properties = dict(self.properties)
        self.detection_kwargs = dict(self.detection_kwargs)
        self.peak_count = int(
            self.peaks.size if self.peak_count is None else self.peak_count
        )
        if self.peak_count != self.peaks.size:
            raise ValueError("peak_count must match the number of peak indices.")
        if self.threshold is not None:
            self.threshold = float(self.threshold)
        if self.amplitudes is not None:
            self.amplitudes = _as_1d_array(
                self.amplitudes,
                dtype=float,
                name="amplitudes",
            )
            if self.amplitudes.size not in {0, self.peaks.size}:
                raise ValueError(
                    "amplitudes must be empty or match the number of peaks."
                )

    @property
    def std_kwargs(self) -> dict[str, Any]:
        return dict(self.detection_kwargs)

    @property
    def cnn_kwargs(self) -> dict[str, Any]:
        return dict(self.detection_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the detection."""

        return {
            "peaks": self.peaks.tolist(),
            "properties": _jsonable(self.properties),
            "peak_count": self.peak_count,
            "detection_kwargs": _jsonable(self.detection_kwargs),
            "threshold": self.threshold,
            "amplitudes": None if self.amplitudes is None else self.amplitudes.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectionResult":
        """Reconstruct a detection result from :meth:`to_dict` output."""

        return cls(
            peaks=data.get("peaks", []),
            properties=data.get("properties", {}),
            peak_count=data.get("peak_count"),
            detection_kwargs=data.get("detection_kwargs", {}),
            threshold=data.get("threshold"),
            amplitudes=data.get("amplitudes"),
        )


@dataclass(frozen=True)
class MetricResult:
    """Named scalar or array-valued metrics with optional units and metadata."""

    name: str
    values: Any
    units: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("name must be a non-empty string.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        values = _jsonable(self.values)
        return {
            "name": self.name,
            "values": values,
            "units": self.units,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricResult":
        """Reconstruct a metric result from :meth:`to_dict` output."""

        return cls(
            name=data["name"],
            values=data.get("values"),
            units=data.get("units"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SeriesResult:
    """Container for ordered trace results and their series metadata."""

    records: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def append(self, record: Any) -> None:
        """Append one record while preserving the ordered-series contract."""

        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        serialized = [
            record.to_dict() if hasattr(record, "to_dict") else _jsonable(record)
            for record in self.records
        ]
        return {"records": serialized, "metadata": _jsonable(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeriesResult":
        """Reconstruct a generic series result from serialized records."""

        return cls(
            records=list(data.get("records", [])),
            metadata=dict(data.get("metadata", {})),
        )
