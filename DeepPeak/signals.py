from __future__ import annotations

from typing import Tuple, Optional, Dict
import numpy as np
from enum import Enum
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
            region_of_interest=self.last_rois_
        )

    # -------------------------- helpers --------------------------

    @staticmethod
    def _ensure_tuple(value: Tuple[float, float] | float | Tuple[int, int] | int) -> Tuple[float, float] | Tuple[int, int]:
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
        Compute binary ROI mask with ±(width_in_pixels // 2) around each peak center.
        """
        n_samples, sequence_length = signals.shape
        _, n_peaks = positions.shape

        rois = np.zeros_like(signals, dtype=np.int32)
        pixel_positions = (positions * (sequence_length - 1)).astype(int)
        half_w = width_in_pixels // 2

        for i in range(n_samples):
            for j in range(n_peaks):
                center_idx = pixel_positions[i, j]
                amp = amplitudes[i, j]
                if np.isnan(amp) or amp == 0:
                    continue
                if center_idx < 0 or center_idx > (sequence_length - 1):
                    continue
                start_idx = max(0, center_idx - half_w)
                end_idx = min(sequence_length, center_idx + half_w + 2)
                rois[i, start_idx:end_idx] = 1
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
                return (
                    amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_) ** 2)
                    + (amp_ * second_peak_ratio)
                    * np.exp(-0.5 * ((x_ - (pos_ + separation)) / (wid_ * 0.5)) ** 2)
                )
            case _:
                raise ValueError(f"Unsupported signal_type: {signal_type}")
