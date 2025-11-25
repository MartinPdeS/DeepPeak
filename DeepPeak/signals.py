from typing import Optional, Tuple, Union
import numpy as np

from DeepPeak.dataset import DataSet  # type: ignore
from DeepPeak.kernel import BaseKernel


class SignalDatasetGenerator:
    """
    Class-based generator for synthetic 1D signals with variable peak counts and shapes.
    Mirrors the behavior of the original `generate_signal_dataset` function, but without
    relying on Keras for one-hot encoding.

    Key features:
    - Supports Gaussian, Lorentzian, Bessel-like, Square, Asymmetric Gaussian, and Dirac kernels
    - Returns labels marking discrete peak locations
    - Optional Gaussian noise
    - Optional NumPy-based one-hot encoding for the number of peaks
    - Optional ROI mask computation (exposed via `last_rois_` attribute)
    """

    # --- public attributes updated on each generate() call ---
    last_rois_: Optional[np.ndarray] = None

    def __init__(
        self, sequence_length: int, x_values: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize the signal dataset generator.

        Parameters
        ----------
        sequence_length : int
            Length of each signal (columns).
        x_values : array-like, optional
            Grid on which signals are evaluated. Defaults to [0, 1].
        """
        self.sequence_length = sequence_length

        # Default x-grid
        self.x_values = (
            x_values if x_values is not None else np.arange(self.sequence_length)
        )

        assert (
            self.x_values.ndim == 1 and len(self.x_values) == self.sequence_length
        ), f"x_values must be a 1D array of length sequence_length [{self.sequence_length}] got shape {self.x_values.shape}"

    # -------------------------- public API --------------------------

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
        """
        Generate a dataset of parametric peak signals.

        Parameters
        ----------
        n_samples : int
            Number of signals (rows) to generate.
        n_peaks : int or (int, int)
            Either a fixed number of peaks or a range (min, max).
        kernel : BaseKernel
            A kernel instance describing peak shape and parameter sampling.
        seed : int, optional
            RNG seed for reproducibility.
        noise_std : float or (float, float), optional
            Gaussian noise standard deviation.
            - scalar: fixed noise level for all samples
            - tuple: random level drawn uniformly per sample
        drift : float or (float, float), optional
            Linear baseline drift slope added per sample.
            - scalar: fixed slope
            - tuple: random slope drawn uniformly per sample
        categorical_peak_count : bool
            If True, return one hot encoded peak-counts from kernel.

        Returns
        -------
        DataSet
            Structured dataset containing:
            - signals
            - labels (peak locations)
            - amplitudes, positions, widths from the kernel
            - x_values
            - num_peaks
            - region_of_interest (if computed)
        """
        self.n_samples = n_samples

        # Resolve peak count boundaries
        n_peaks = self._ensure_tuple(n_peaks)
        noise_std = self._ensure_tuple(noise_std) if noise_std is not None else None
        drift = self._ensure_tuple(drift) if drift is not None else None

        if seed is not None:
            np.random.seed(seed)

        # --------------------------------------------------------------
        # 1. Generate raw peak components
        # --------------------------------------------------------------
        peak_components = kernel.evaluate(
            self.x_values, self.n_samples, n_peaks, categorical_peak_count
        )
        signals = np.nansum(peak_components, axis=1)  # shape: (n_samples, seq_len)

        # --------------------------------------------------------------
        # 2. Generate labels using original (unmasked) peak positions
        # --------------------------------------------------------------
        labels = np.zeros((self.n_samples, self.sequence_length))

        true_positions = kernel.positions_for_labels  # real x-values

        diff = np.abs(true_positions[..., None] - self.x_values[None, None, :])
        peak_indices = diff.argmin(axis=-1)
        peak_indices = np.clip(peak_indices, 0, self.sequence_length - 1)

        for i in range(self.n_samples):
            labels[i, peak_indices[i, : kernel.num_peaks[i]]] = 1

        # --------------------------------------------------------------
        # 3. Add Gaussian noise (optional)
        # --------------------------------------------------------------
        if noise_std is not None:
            noise_levels = np.random.uniform(
                noise_std[0], noise_std[1], size=(self.n_samples, 1)
            )

            noise = np.random.normal(0.0, 1.0, size=signals.shape) * noise_levels
            signals = signals + noise

        # --------------------------------------------------------------
        # 4. Add drift (optional)
        # --------------------------------------------------------------
        if drift is not None:
            drift_levels = np.random.uniform(
                drift[0], drift[1], size=(self.n_samples, 1)
            )

            baseline = drift_levels * np.linspace(0, 1, self.sequence_length)
            signals = signals + baseline

        # --------------------------------------------------------------
        # 5. Wrap into DataSet structure
        # --------------------------------------------------------------
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

    # -------------------------- helpers --------------------------
    @staticmethod
    def _ensure_tuple(
        value: Tuple[float, float] | float | Tuple[int, int] | int,
    ) -> Tuple[float, float] | Tuple[int, int]:
        """If value is a scalar, return (v, v); otherwise return value."""
        if isinstance(value, (int, float)):
            return (value, value)  # type: ignore[return-value]
        return value  # type: ignore[return-value]
