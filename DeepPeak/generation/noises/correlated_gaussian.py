"""Correlated Gaussian noise for realistic acquisition traces."""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .base import BaseNoise


@dataclass(repr=False)
class CorrelatedGaussianNoise(BaseNoise):
    """Gaussian noise with exponentially correlated samples.

    ``correlation_length`` is expressed in samples. A value of zero produces
    independent noise; larger values produce slowly varying baseline noise.

    Parameters
    ----------
    std : float or tuple of float
        Standard deviation or sampling range.
    correlation_length : float, default=3.0
        Exponential correlation length in samples.
    mean : float or tuple of float, default=0.0
        Mean or sampling range.
    """

    std: Union[float, Tuple[float, float]]
    correlation_length: float = 3.0
    mean: Union[float, Tuple[float, float]] = 0.0

    def __post_init__(self) -> None:
        self._std = self._normalize_range(
            "std", self.std, minimum=0.0, inclusive_minimum=True
        )
        self._mean = self._normalize_range("mean", self.mean)
        if not np.isfinite(self.correlation_length) or self.correlation_length < 0:
            raise ValueError("correlation_length must be finite and non-negative.")

    def sample(
        self,
        shape: tuple[int, int],
        *,
        x_values: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return noise samples with an exponential autocorrelation profile.

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
            Correlated Gaussian noise samples.
        """

        rng = np.random.default_rng() if rng is None else rng
        n_samples, sequence_length = int(shape[0]), int(shape[1])
        stds = self._sample_uniform(self._std, size=(n_samples, 1), rng=rng)
        means = self._sample_uniform(self._mean, size=(n_samples, 1), rng=rng)
        white = rng.normal(size=(n_samples, sequence_length))
        if self.correlation_length == 0.0:
            return means + white * stds

        rho = float(np.exp(-1.0 / self.correlation_length))
        correlated = np.empty_like(white)
        correlated[:, 0] = white[:, 0]
        innovation_scale = np.sqrt(1.0 - rho**2)
        for index in range(1, sequence_length):
            correlated[:, index] = (
                rho * correlated[:, index - 1] + innovation_scale * white[:, index]
            )
        return means + correlated * stds
