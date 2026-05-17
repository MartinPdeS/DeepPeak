from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


IntBounds = tuple[int, int]
FloatLike = float | int
FloatRangeLike = FloatLike | tuple[FloatLike, FloatLike]


def _normalize_bounds(bounds: IntBounds) -> IntBounds:
    low = int(bounds[0])
    high = int(bounds[1])
    if low < 0:
        raise ValueError("peak count bounds must be non-negative.")
    if high < low:
        raise ValueError("peak count bounds must satisfy low <= high.")
    return (low, high)


def _sample_scalar_or_range(
    value: FloatRangeLike, n_samples: int, *, name: str
) -> np.ndarray:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} range must contain exactly two values.")
        low = float(value[0])
        high = float(value[1])
        if not np.isfinite(low) or not np.isfinite(high) or low < 0.0 or high < low:
            raise ValueError(
                f"{name} range must contain finite non-negative values with low <= high."
            )
        return np.random.uniform(low, high, size=n_samples)

    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return np.full(n_samples, scalar, dtype=float)


@dataclass(frozen=True)
class PeakCount(ABC):
    bounds: IntBounds

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _normalize_bounds(self.bounds))

    @property
    def min_peaks(self) -> int:
        return self.bounds[0]

    @property
    def max_peaks(self) -> int:
        return self.bounds[1]

    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    def _clip(self, counts: np.ndarray) -> np.ndarray:
        return np.clip(counts, self.min_peaks, self.max_peaks).astype(np.int64)


@dataclass(frozen=True)
class UniformCount(PeakCount):
    def sample(self, n_samples: int) -> np.ndarray:
        return np.random.randint(
            low=self.min_peaks,
            high=self.max_peaks + 1,
            size=int(n_samples),
        ).astype(np.int64)


@dataclass(frozen=True)
class PoissonCount(PeakCount):
    rate: FloatRangeLike

    def sample(self, n_samples: int) -> np.ndarray:
        rate = _sample_scalar_or_range(self.rate, int(n_samples), name="rate")
        return self._clip(np.random.poisson(lam=rate, size=int(n_samples)))


@dataclass(frozen=True)
class NegativeBinomialCount(PeakCount):
    mean: FloatRangeLike
    dispersion: FloatRangeLike

    def sample(self, n_samples: int) -> np.ndarray:
        sample_count = int(n_samples)
        mean = _sample_scalar_or_range(self.mean, sample_count, name="mean")
        dispersion = _sample_scalar_or_range(
            self.dispersion,
            sample_count,
            name="dispersion",
        )
        if np.any(dispersion <= 0.0):
            raise ValueError("dispersion must be strictly positive.")

        probability = dispersion / (dispersion + mean)
        counts = np.random.negative_binomial(dispersion, probability, size=sample_count)
        return self._clip(counts)
