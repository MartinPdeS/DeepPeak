"""Synthetic signal dataset generation utilities."""

from typing import Optional, Tuple, Union

import numpy as np

from DeepPeak.dataset import DataSet  # type: ignore
from DeepPeak.kernel import BaseKernel


class SignalGenerator:
    """Generate synthetic 1D peak signals from kernel parameter samplers.

    Key features:
    - Supports all kernel classes derived from ``BaseKernel``
    - Returns labels marking discrete peak locations
    - Optional Gaussian noise
    - Optional NumPy-based one-hot encoding for the number of peaks
    - Optional ROI mask computation through the resulting ``DataSet``
    """

    last_rois_: Optional[np.ndarray] = None

    def __init__(
        self, sequence_length: int, x_values: Optional[np.ndarray] = None
    ) -> None:
        self.sequence_length = sequence_length
        self.x_values = (
            x_values if x_values is not None else np.arange(self.sequence_length)
        )

        assert self.x_values.ndim == 1 and len(self.x_values) == self.sequence_length, (
            "x_values must be a 1D array of length sequence_length "
            f"[{self.sequence_length}] got shape {self.x_values.shape}"
        )

    def generate(
        self,
        *,
        n_samples: int,
        n_peaks: Tuple[int, int] | int,
        kernel: BaseKernel,
        seed: Optional[int] = None,
        noise_std: Optional[Union[float, Tuple[float, float]]] = None,
        drift: Optional[Union[float, Tuple[float, float]]] = None,
        categorical_peak_count: bool = False,
    ) -> DataSet:
        """Generate a dataset of parametric peak signals."""
        self.n_samples = n_samples

        n_peaks = self._ensure_tuple(n_peaks)
        noise_std = self._ensure_tuple(noise_std) if noise_std is not None else None
        drift = self._ensure_tuple(drift) if drift is not None else None

        if seed is not None:
            np.random.seed(seed)

        peak_components = kernel.evaluate(
            self.x_values, self.n_samples, n_peaks, categorical_peak_count
        )
        signals = np.nansum(peak_components, axis=1)

        labels = np.zeros((self.n_samples, self.sequence_length))
        true_positions = kernel.positions_for_labels
        diff = np.abs(true_positions[..., None] - self.x_values[None, None, :])
        peak_indices = diff.argmin(axis=-1)
        peak_indices = np.clip(peak_indices, 0, self.sequence_length - 1)

        for i in range(self.n_samples):
            labels[i, peak_indices[i, : kernel.num_peaks[i]]] = 1

        if noise_std is not None:
            noise_levels = np.random.uniform(
                noise_std[0], noise_std[1], size=(self.n_samples, 1)
            )
            noise = np.random.normal(0.0, 1.0, size=signals.shape) * noise_levels
            signals = signals + noise

        if drift is not None:
            drift_levels = np.random.uniform(
                drift[0], drift[1], size=(self.n_samples, 1)
            )
            baseline = drift_levels * np.linspace(0, 1, self.sequence_length)
            signals = signals + baseline

        dataset = DataSet(
            signals=signals,
            **kernel.get_kwargs(),
            labels=labels,
            x_values=self.x_values,
            num_peaks=kernel.num_peaks,
        )

        dataset.n_samples = self.n_samples
        dataset.sequence_length = self.sequence_length
        return dataset

    @staticmethod
    def _ensure_tuple(
        value: Tuple[float, float] | float | Tuple[int, int] | int,
    ) -> Tuple[float, float] | Tuple[int, int]:
        """If value is a scalar, return ``(v, v)``; otherwise return value."""
        if isinstance(value, (int, float)):
            return (value, value)  # type: ignore[return-value]
        return value  # type: ignore[return-value]
