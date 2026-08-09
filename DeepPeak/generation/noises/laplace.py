from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .base import BaseNoise


@dataclass(repr=False)
class LaplaceNoise(BaseNoise):
    """Independent Laplace noise for heavier-tailed augmentation."""

    scale: Union[float, Tuple[float, float]]
    mean: Union[float, Tuple[float, float]] = 0.0

    def __post_init__(self) -> None:
        self._scale = self._normalize_range(
            "scale", self.scale, minimum=0.0, inclusive_minimum=True
        )
        self._mean = self._normalize_range("mean", self.mean)

    def sample(self, shape: tuple[int, int], *, x_values=None, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        n_samples, sequence_length = int(shape[0]), int(shape[1])
        means = self._sample_uniform(self._mean, size=(n_samples, 1), rng=rng)
        scales = self._sample_uniform(self._scale, size=(n_samples, 1), rng=rng)
        return rng.laplace(means, scales, size=(n_samples, sequence_length))
