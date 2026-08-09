"""Gaussian noise models whose variance changes over a trace."""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .base import BaseNoise


@dataclass(repr=False)
class NonstationaryGaussianNoise(BaseNoise):
    """Gaussian noise with a linearly varying standard deviation.

    ``std`` is the standard deviation at the start of each trace and
    ``end_scale`` multiplies it at the end. For example, ``end_scale=2``
    produces noise whose standard deviation doubles across the acquisition.

    Parameters
    ----------
    std : float or tuple of float
        Starting standard deviation or sampling range.
    end_scale : float or tuple of float, default=1.0
        End-to-start standard-deviation ratio.
    mean : float or tuple of float, default=0.0
        Mean or sampling range.
    """

    std: Union[float, Tuple[float, float]]
    end_scale: Union[float, Tuple[float, float]] = 1.0
    mean: Union[float, Tuple[float, float]] = 0.0

    def __post_init__(self) -> None:
        self._std = self._normalize_range(
            "std", self.std, minimum=0.0, inclusive_minimum=True
        )
        self._end_scale = self._normalize_range(
            "end_scale", self.end_scale, minimum=0.0, inclusive_minimum=True
        )
        self._mean = self._normalize_range("mean", self.mean)

    def sample(
        self,
        shape: tuple[int, int],
        *,
        x_values: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return independent Gaussian samples with a linear variance trend.

        Parameters
        ----------
        shape : tuple of int
            Requested output shape.
        x_values : ndarray, optional
            Sample coordinates retained for interface compatibility.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        ndarray
            Nonstationary Gaussian noise samples.
        """

        rng = np.random.default_rng() if rng is None else rng
        n_samples, sequence_length = int(shape[0]), int(shape[1])
        starts = self._sample_uniform(self._std, size=(n_samples, 1), rng=rng)
        end_scales = self._sample_uniform(self._end_scale, size=(n_samples, 1), rng=rng)
        means = self._sample_uniform(self._mean, size=(n_samples, 1), rng=rng)
        coordinate = np.linspace(0.0, 1.0, sequence_length)[None, :]
        stds = starts * (1.0 + coordinate * (end_scales - 1.0))
        return means + rng.normal(size=(n_samples, sequence_length)) * stds
