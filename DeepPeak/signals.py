from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from DeepPeak.dataset import DataSet  # type: ignore


class Kernel(Enum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    BESSEL = "bessel"
    SQUARE = "square"
    ASYMMETRIC_GAUSSIAN = "asym_gaussian"
    DIRAC = "dirac"


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
        signal_type: Kernel | str = Kernel.GAUSSIAN,
        extra_kwargs: Optional[Dict] = None,
        amplitude: Tuple[float, float] | float = (1.0, 2.0),
        position: Tuple[float, float] | float = (0.0, 1.0),  # normalized 0..1
        width: Tuple[float, float] | float = (0.03, 0.03),
        seed: Optional[int] = None,
        noise_std: float = 0.01,
        categorical_peak_count: bool = False,
        kernel: Optional[np.ndarray] = None,
        compute_region_of_interest: bool = False,
        roi_width_in_pixels: int = 4,
    ) -> DataSet:
        """
        Generate a dataset of 1D signals with varying number of peaks.

        Parameters
        ----------
        n_peaks : (int,int) or int
            (min_peaks, max_peaks) inclusive; if int, uses (v, v).
        signal_type : Kernel | str
            Peak shape type (Kernel enum or matching string value).
        extra_kwargs : dict, optional
            Additional args for specific kernels (e.g., `separation`, `second_peak_ratio` for ASYMMETRIC_GAUSSIAN).
        amplitude : (float,float) or float
            Amplitude range; if float, uses (v, v).
        position : (float,float) or float
            Position range in [0, 1]; if float, uses (v, v).
        width : (float,float) or float
            Width range; if float, uses (v, v).
        seed : int, optional
            RNG seed.
        noise_std : float
            Additive Gaussian noise std.
        categorical_peak_count : bool
            If True, return one-hot encoding of peak counts (NumPy-based).
        kernel : np.ndarray, optional
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
        # -------------------- sanitize/normalize inputs --------------------
        if isinstance(signal_type, str):
            signal_type = Kernel(signal_type)
        self._assert_kernel(signal_type)

        # coerce scalars to (v, v)
        n_peaks = self._ensure_tuple(n_peaks)
        amplitude = self._ensure_tuple(amplitude)
        position = self._ensure_tuple(position)
        width = self._ensure_tuple(width)

        if seed is not None:
            np.random.seed(seed)
        if extra_kwargs is None:
            extra_kwargs = {}

        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])
        num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=self.n_samples)

        amplitudes = np.random.uniform(*amplitude, size=(self.n_samples, max_peaks))
        positions = np.random.uniform(*position, size=(self.n_samples, max_peaks))
        widths = np.random.uniform(*width, size=(self.n_samples, max_peaks))

        # Keep a copy for label computation prior to NaN-masking
        positions_for_labels = positions.copy()

        # Mask inactive peaks (index >= num_peaks[i]) -> set to NaN
        peak_indices = np.arange(max_peaks)
        mask = peak_indices < num_peaks[:, None]
        amplitudes[~mask] = np.nan
        positions[~mask] = np.nan
        widths[~mask] = np.nan

        x_values = np.linspace(0.0, 1.0, self.sequence_length)
        x_ = x_values.reshape(1, 1, -1)
        pos_ = positions_for_labels[..., np.newaxis]
        wid_ = widths[..., np.newaxis]
        amp_ = amplitudes[..., np.newaxis]

        # Build signals
        if signal_type == Kernel.DIRAC:
            signals = np.zeros((self.n_samples, self.sequence_length))
            for i in range(self.n_samples):
                sig = np.zeros(self.sequence_length)
                # Use original positions (not NaN-masked) to construct impulses
                peak_pos = (positions_for_labels[i, : num_peaks[i]] * (self.sequence_length - 1)).astype(int)
                sig[peak_pos] = amplitudes[i, : num_peaks[i]]
                if kernel is not None:
                    sig = np.convolve(sig, kernel, mode="same")
                signals[i] = sig
        else:
            peaks = self._build_peaks(signal_type, x_, pos_, wid_, amp_, extra_kwargs)
            signals = np.nansum(peaks, axis=1)

        # Labels: 1 at discrete peak centers using *original* positions
        labels = np.zeros((self.n_samples, self.sequence_length))
        peak_positions = (positions_for_labels * (self.sequence_length - 1)).astype(int)
        for i in range(self.n_samples):
            labels[i, peak_positions[i, : num_peaks[i]]] = 1

        # Add noise
        if noise_std > 0:
            signals = signals + np.random.normal(0.0, noise_std, signals.shape)

        # Optional one-hot (no Keras)
        if categorical_peak_count:
            num_peaks_out = self._one_hot_numpy(num_peaks, max_peaks + 1, dtype=np.float32)
        else:
            num_peaks_out = num_peaks

        # Optional ROI
        self.last_rois_ = None
        if compute_region_of_interest:
            self.last_rois_ = self._compute_rois_from_signals(
                signals=signals,
                positions=positions,
                amplitudes=amplitudes,
                width_in_pixels=roi_width_in_pixels,
            )

        return DataSet(
            signals=signals,
            labels=labels,
            amplitudes=amplitudes,
            positions=positions,
            widths=widths,
            x_values=x_values,
            num_peaks=num_peaks_out,
            region_of_interest=self.last_rois_,
        )

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
    def _assert_kernel(signal_type: Kernel) -> None:
        if not isinstance(signal_type, Kernel):
            raise ValueError(f"`signal_type` must be a Kernel enum or matching string, got {signal_type!r}")

    @staticmethod
    def _one_hot_numpy(indices: np.ndarray, num_classes: int, dtype=np.float32) -> np.ndarray:
        """
        Fast, pure-NumPy one-hot encoder.

        Parameters
        ----------
        indices: np.ndarray
            The indices to one-hot encode.
        num_classes: int
            The number of classes for the one-hot encoding.
        dtype: type
            The data type of the output array.
        """
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size == 0:
            return np.zeros((0, num_classes), dtype=dtype)
        if (indices < 0).any() or (indices >= num_classes).any():
            raise ValueError("indices out of range for the specified num_classes")

        out = np.zeros((indices.shape[0], num_classes), dtype=dtype)
        out[np.arange(indices.shape[0]), indices] = 1
        return out

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
        centers = np.full_like(tmp, fill_value=-1, dtype=np.int64)  # sentinel for invalid
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

    @staticmethod
    def _build_peaks(
        signal_type: Kernel,
        x_: np.ndarray,
        pos_: np.ndarray,
        wid_: np.ndarray,
        amp_: np.ndarray,
        extra_kwargs: Dict,
    ) -> np.ndarray:
        """
        Vectorized construction of peaks for non-DIRAC kernels.

        Parameters
        ----------
        signal_type: Kernel
            The type of signal kernel to use.
        x_: np.ndarray
            The x-coordinates at which to evaluate the peaks.
        pos_: np.ndarray
            The positions of the peaks.
        wid_: np.ndarray
            The widths of the peaks.
        amp_: np.ndarray
            The amplitudes of the peaks.
        extra_kwargs: Dict
            Additional keyword arguments for specific kernel types.

        Returns
        -------
        np.ndarray
            The constructed peaks.
        """
        match signal_type:
            case Kernel.GAUSSIAN:
                return amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_) ** 2)
            case Kernel.LORENTZIAN:
                return amp_ / (1.0 + ((x_ - pos_) / wid_) ** 2)
            case Kernel.BESSEL:
                z = (x_ - pos_) / wid_
                return amp_ * np.abs(np.sin(z)) / (z + 1e-6)
            case Kernel.SQUARE:
                return amp_ * ((np.abs(x_ - pos_) < wid_) * 1.0)
            case Kernel.ASYMMETRIC_GAUSSIAN:
                separation = extra_kwargs.get("separation", 0.1)
                second_peak_ratio = extra_kwargs.get("second_peak_ratio", 0.5)
                return amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_) ** 2) + (amp_ * second_peak_ratio) * np.exp(-0.5 * ((x_ - (pos_ + separation)) / (wid_ * 0.5)) ** 2)
            case _:
                raise ValueError(f"Unsupported signal_type: {signal_type}")
