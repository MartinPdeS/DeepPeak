from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .base import BaseNoise


@dataclass(repr=False)
class GaussianNoise(BaseNoise):
    """Independent Gaussian noise with per-trace sampled standard deviation."""

    std: Union[float, Tuple[float, float]]
    mean: Union[float, Tuple[float, float]] = 0.0

    def __post_init__(self) -> None:
        self._std = self._normalize_range(
            "std", self.std, minimum=0.0, inclusive_minimum=True
        )
        self._mean = self._normalize_range("mean", self.mean)

    def sample(self, shape: tuple[int, int], *, x_values=None, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        n_samples, sequence_length = int(shape[0]), int(shape[1])
        means = self._sample_uniform(self._mean, size=(n_samples, 1), rng=rng)
        stds = self._sample_uniform(self._std, size=(n_samples, 1), rng=rng)
        return means + rng.normal(0.0, 1.0, size=(n_samples, sequence_length)) * stds
