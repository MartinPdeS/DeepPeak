from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

RangeValue = Tuple[float, float] | float | Tuple[int, int] | int
FloatRange = Tuple[float, float]


class BaseNoise(ABC):
    """Base class for synthetic additive noise models."""

    @staticmethod
    def _ensure_tuple(value: RangeValue) -> Tuple[float, float] | Tuple[int, int]:
        if isinstance(value, (int, float)):
            return (value, value)  # type: ignore[return-value]
        if len(value) != 2:
            raise ValueError("Noise parameter ranges must contain exactly two values.")
        return value  # type: ignore[return-value]

    @classmethod
    def _normalize_range(
        cls,
        name: str,
        value: RangeValue,
        *,
        minimum: float | None = None,
        inclusive_minimum: bool = True,
    ) -> FloatRange:
        low_raw, high_raw = cls._ensure_tuple(value)
        low = float(low_raw)
        high = float(high_raw)

        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"{name} must contain only finite values.")
        if high < low:
            raise ValueError(f"{name} must satisfy low <= high.")
        if minimum is not None:
            if inclusive_minimum:
                valid = low >= minimum and high >= minimum
            else:
                valid = low > minimum and high > minimum
            if not valid:
                comparator = ">=" if inclusive_minimum else ">"
                raise ValueError(f"{name} values must be {comparator} {minimum}.")
        return (low, high)

    def _sample_uniform(
        self,
        bounds: FloatRange,
        *,
        size: tuple[int, ...],
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.uniform(bounds[0], bounds[1], size=size)

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, int],
        *,
        x_values: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return additive noise samples with the requested shape.

        Parameters
        ----------
        shape : tuple of int
            Requested ``(n_traces, n_samples)`` output shape.
        x_values : ndarray, optional
            Sample coordinates used by nonstationary noise models.
        rng : numpy.random.Generator, optional
            Random number generator used for sampling.

        Returns
        -------
        ndarray
            Additive noise with shape ``shape``.
        """
