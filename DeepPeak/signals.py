from typing import Optional, Tuple
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

    def __init__(self, n_samples: int, sequence_length: int) -> None:
        """
        Initialize the signal dataset generator.

        Parameters
        ----------
        n_samples : int
            Number of signals (rows) to generate.
        sequence_length : int
            Length of each signal (columns).
        """
        self.n_samples = n_samples
        self.sequence_length = sequence_length

    # -------------------------- public API --------------------------

    def generate(
        self,
        *,
        n_peaks: Tuple[int, int] | int,
        kernel: BaseKernel,
        seed: Optional[int] = None,
        noise_std: Optional[float] = None,
        categorical_peak_count: bool = False,
        compute_region_of_interest: bool = False,
        roi_width_in_pixels: int = 4,
    ) -> DataSet:
        """
        Generate a dataset of 1D signals with varying number of peaks.

        Parameters
        ----------
        n_peaks : (int,int) or int
            (min_peaks, max_peaks) inclusive; if int, uses (v, v).
        kernel : Kernel
            Peak shape type (Kernel).
        seed : int, optional
            RNG seed.
        noise_std : Optional[float]
            Additive Gaussian noise std.
        categorical_peak_count : bool
            If True, return one-hot encoding of peak counts (NumPy-based).
        convolution_kernel : np.ndarray, optional
            Convolution kernel used only for DIRAC.
        compute_region_of_interest : bool
            If True, compute an ROI mask around each discrete peak (stored in `last_rois_`).
        roi_width_in_pixels : int
            ROI full width in samples (integer), used when `compute_region_of_interest=True`.

        Returns
        -------
        DataSet
            Object with fields: signals, labels, amplitudes, positions, widths, x_values, num_peaks.
        """
        # coerce scalars to (v, v)
        n_peaks = self._ensure_tuple(n_peaks)

        if seed is not None:
            np.random.seed(seed)

        x_values = np.linspace(0.0, 1.0, self.sequence_length)

        signals = kernel.evaluate(
            x_values, self.n_samples, n_peaks, categorical_peak_count
        )
        signals = np.nansum(signals, axis=1)

        # Labels: 1 at discrete peak centers using *original* positions
        labels = np.zeros((self.n_samples, self.sequence_length))
        peak_positions = (
            kernel.positions_for_labels * (self.sequence_length - 1)
        ).astype(int)
        for i in range(self.n_samples):
            labels[i, peak_positions[i, : kernel.num_peaks[i]]] = 1

        # Add noise
        if noise_std is not None:
            signals = signals + np.random.normal(0.0, noise_std, signals.shape)

        # Optional ROI
        self.last_rois_ = None
        if compute_region_of_interest:
            self.last_rois_ = self._compute_rois_from_signals(
                signals=signals,
                positions=kernel.positions,
                amplitudes=kernel.amplitudes,
                width_in_pixels=roi_width_in_pixels,
            )

        dataset = DataSet(
            signals=signals,
            **kernel.get_kwargs(),
            labels=labels,
            x_values=x_values,
            num_peaks=kernel.num_peaks,
            region_of_interest=self.last_rois_,
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

    @staticmethod
    def _compute_rois_from_signals(
        signals: np.ndarray,
        positions: np.ndarray,
        amplitudes: np.ndarray,
        width_in_pixels: int,
    ) -> np.ndarray:
        """
        Vectorized ROI builder: marks ±(width_in_pixels//2) around each valid peak center.
        - No Python loops over samples/peaks.
        - Handles NaNs in positions/amplitudes.

        Parameters
        ----------
        signals: np.ndarray
            The input signals.
        positions: np.ndarray
            The positions of the peaks.
        amplitudes: np.ndarray
            The amplitudes of the peaks.
        width_in_pixels: int
            The width of the ROI in pixels.

        Returns
        -------
        np.ndarray
            The computed ROIs.
        """
        n_samples, sequence_length = signals.shape
        assert positions.shape[0] == n_samples and amplitudes.shape == positions.shape

        # Convert normalized positions -> pixel centers (int), keep shape
        tmp = positions * (sequence_length - 1)  # float, may have NaN/inf
        centers = np.full_like(
            tmp, fill_value=-1, dtype=np.int64
        )  # sentinel for invalid
        valid_pos = np.isfinite(tmp)
        centers[valid_pos] = np.rint(tmp[valid_pos]).astype(np.int64)
        np.clip(centers, 0, sequence_length - 1, out=centers)

        # Valid peaks must also have finite, non-zero amplitude
        valid_amp = np.isfinite(amplitudes) & (amplitudes != 0)
        valid = valid_pos & valid_amp

        # Interval [start, end) per peak, clipped to bounds
        w = int(width_in_pixels)
        if w < 0:
            raise ValueError("width_in_pixels must be non-negative")
        half = w // 2
        starts = np.clip(centers - half, 0, sequence_length)
        ends = np.clip(centers + half + 1, 0, sequence_length)  # +1 for inclusive end

        # Difference array per sample: add +1 at start, -1 at end
        diff = np.zeros((n_samples, sequence_length + 1), dtype=np.int32)
        ii, jj = np.nonzero(valid)  # indices of valid (sample, peak) pairs
        if ii.size:
            s = starts[ii, jj]
            e = ends[ii, jj]
            np.add.at(diff, (ii, s), 1)
            np.add.at(diff, (ii, e), -1)

        # Cumulative sum -> coverage counts; binarize
        rois = (np.cumsum(diff[:, :sequence_length], axis=1) > 0).astype(np.int32)
        return rois
