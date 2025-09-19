from typing import Tuple
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray


class BaseKernel:
    @staticmethod
    def _ensure_tuple(
        value: Tuple[float, float] | float | Tuple[int, int] | int,
    ) -> Tuple[float, float] | Tuple[int, int]:
        """If value is a scalar, return (v, v); otherwise return value."""
        if isinstance(value, (int, float)):
            return (value, value)  # type: ignore[return-value]
        return value  # type: ignore[return-value]

    @staticmethod
    def _one_hot_numpy(indices: NDArray, num_classes: int, dtype=np.float32) -> NDArray:
        """
        Fast, pure-NumPy one-hot encoder.

        Parameters
        ----------
        indices: NDArray
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


@dataclass
class Gaussian(BaseKernel):
    """
    Simple Gaussian pulse model.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the Gaussian.
    position : float
        Mean (center) of the Gaussian.
    width : float
        Standard deviation (width) of the Gaussian.
    """

    amplitude: float
    position: float
    width: float

    def __post_init__(self):
        self.amplitude = self._ensure_tuple(self.amplitude)
        self.position = self._ensure_tuple(self.position)
        self.width = self._ensure_tuple(self.width)

    def get_kwargs(self) -> dict:
        return {
            "amplitudes": self.amplitudes,
            "positions": self.positions,
            "widths": self.widths,
        }

    def evaluate(self, x_values: NDArray, n_samples: int, n_peaks: tuple, categorical_peak_count: bool = False) -> np.ndarray:
        """
        Evaluate a batch of Gaussian pulses.
        Parameters
        ----------
        x_values : NDArray
            1D array of x-values where the Gaussian pulses are evaluated.
        n_samples : int
            Number of samples (signals) to generate.
        n_peaks : tuple
            Tuple (min_peaks, max_peaks) specifying the range of number of peaks per signal.
        categorical_peak_count : bool, optional
            If True, the number of peaks is returned as a one-hot encoded vector. Default is False.

        Returns
        -------
        NDArray
            Array of shape (n_samples, max_peaks, len(x_values)) containing the evaluated Gaussian pulses.
        """
        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])

        self.num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)
        self.amplitudes = np.random.uniform(*self.amplitude, size=(n_samples, max_peaks))
        self.positions = np.random.uniform(*self.position, size=(n_samples, max_peaks))
        self.widths = np.random.uniform(*self.width, size=(n_samples, max_peaks))

        # Keep a copy for label computation prior to NaN-masking
        self.positions_for_labels = self.positions.copy()

        # Mask inactive peaks (index >= num_peaks[i]) -> set to NaN
        peak_indices = np.arange(max_peaks)
        mask = peak_indices < self.num_peaks[:, None]
        self.amplitudes[~mask] = np.nan
        self.positions[~mask] = np.nan
        self.widths[~mask] = np.nan

        x_ = x_values.reshape(1, 1, -1)
        pos_ = self.positions_for_labels[..., np.newaxis]
        wid_ = self.widths[..., np.newaxis]
        amp_ = self.amplitudes[..., np.newaxis]

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(self.num_peaks, max_peaks + 1, dtype=np.float32)

        return amp_ * np.exp(-0.5 * ((x_ - pos_) / wid_) ** 2)


@dataclass
class Lorentzian(BaseKernel):
    """
    Simple Lorentzian pulse model.

    Attributes
    ----------
    amplitude : float or (low, high)
        Peak amplitude (A). If a tuple is given, values are sampled uniformly in [low, high].
    position : float or (low, high)
        Center position (x0). If a tuple is given, values are sampled uniformly in [low, high].
    width : float or (low, high)
        Lorentzian half-width at half-maximum (HWHM), i.e. $\gamma$.
        If a tuple is given, values are sampled uniformly in [low, high].

    Notes
    -----
    The Lorentzian profile used is:
        L(x; A, x0, $\gamma$) = A * $\gamma$^2 / ((x - x0)^2 + $\gamma$^2) = A / (1 + ((x - x0)/$\gamma$)^2)
    """

    amplitude: float
    position: float
    width: float  # HWHM $\gamma$

    def __post_init__(self):
        self.amplitude = self._ensure_tuple(self.amplitude)
        self.position = self._ensure_tuple(self.position)
        self.width = self._ensure_tuple(self.width)

    def get_kwargs(self) -> dict:
        return {
            "amplitudes": self.amplitudes,
            "positions": self.positions,
            "widths": self.widths,
        }

    def evaluate(
        self,
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
    ) -> NDArray:
        """
        Evaluate a batch of Lorentzian pulses.

        Parameters
        ----------
        x_values : ndarray, shape (M,)
            1D array of x-values where the pulses are evaluated.
        n_samples : int
            Number of samples (signals) to generate.
        n_peaks : tuple
            (min_peaks, max_peaks) specifying the inclusive range of peak count per signal.
        categorical_peak_count : bool, optional
            If True, `self.num_peaks` is converted to one-hot (length = max_peaks+1).

        Returns
        -------
        ndarray, shape (n_samples, max_peaks, M)
            Evaluated Lorentzians for each (sample, peak). Inactive peaks are NaN-masked.
        """
        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])

        # Draw per-sample number of peaks
        self.num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)

        # Draw parameters (uniform in provided ranges)
        self.amplitudes = np.random.uniform(*self.amplitude, size=(n_samples, max_peaks))
        self.positions = np.random.uniform(*self.position, size=(n_samples, max_peaks))
        self.widths = np.random.uniform(*self.width, size=(n_samples, max_peaks))

        # Keep copy for labels before masking
        self.positions_for_labels = self.positions.copy()

        # Mask inactive peaks with NaN for downstream handling
        peak_indices = np.arange(max_peaks)
        mask = peak_indices < self.num_peaks[:, None]
        self.amplitudes[~mask] = np.nan
        self.positions[~mask] = np.nan
        self.widths[~mask] = np.nan

        # Broadcast to (n_samples, max_peaks, M)
        x_ = x_values.reshape(1, 1, -1)
        pos_ = self.positions[..., np.newaxis]
        gam_ = self.widths[..., np.newaxis]  # $\gamma$ (HWHM)
        amp_ = self.amplitudes[..., np.newaxis]

        # L(x) = A * $\gamma$^2 / ((x - x0)^2 + $\gamma$^2)
        y = amp_ * (gam_**2 / ((x_ - pos_) ** 2 + gam_**2))

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(self.num_peaks, max_peaks + 1, dtype=np.float32)

        return y


@dataclass
class Square(BaseKernel):
    """
    Simple square pulse model (batch-capable, Gaussian-style API).

    Attributes
    ----------
    amplitude : float or (low, high)
        Peak amplitude. If a tuple is given, values are sampled uniformly in [low, high].
    position : float or (low, high)
        Center position. If a tuple is given, values are sampled uniformly in [low, high].
    width : float or (low, high)
        Full width of the square pulse. If a tuple is given, values are sampled uniformly in [low, high].

    Notes
    -----
    The square pulse is:
        S(x; A, x0, w) = A · 1_{x ∈ [x0 - w/2, x0 + w/2]}
    with inclusive edges.
    """

    amplitude: float
    position: float
    width: float

    def __post_init__(self):
        # allow scalar or (low, high) tuples; unify as tuples
        self.amplitude = self._ensure_tuple(self.amplitude)
        self.position = self._ensure_tuple(self.position)
        self.width = self._ensure_tuple(self.width)

    def get_kwargs(self) -> dict:
        return {
            "amplitudes": self.amplitudes,
            "positions": self.positions,
            "widths": self.widths,
        }

    def evaluate(
        self,
        x_values: NDArray[np.float64],
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
    ) -> NDArray[np.float64]:
        """
        Evaluate a batch of square pulses.

        Parameters
        ----------
        x_values : ndarray, shape (M,)
            1D array of x-values where the pulses are evaluated.
        n_samples : int
            Number of samples (signals) to generate.
        n_peaks : tuple
            (min_peaks, max_peaks) specifying the inclusive range of peak count per signal.
        categorical_peak_count : bool, optional
            If True, `self.num_peaks` is converted to one-hot (length = max_peaks+1).

        Returns
        -------
        ndarray, shape (n_samples, max_peaks, M)
            Evaluated square pulses for each (sample, peak). Inactive peaks are NaN-masked.
        """
        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])

        # Draw per-sample number of peaks
        self.num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)

        # Draw parameters (uniform in provided ranges)
        self.amplitudes = np.random.uniform(*self.amplitude, size=(n_samples, max_peaks))
        self.positions = np.random.uniform(*self.position, size=(n_samples, max_peaks))
        self.widths = np.random.uniform(*self.width, size=(n_samples, max_peaks))

        # Keep copy for label computation before masking
        self.positions_for_labels = self.positions.copy()

        # Mask inactive peaks (index >= num_peaks[i]) -> set to NaN
        peak_indices = np.arange(max_peaks)
        active_mask = peak_indices < self.num_peaks[:, None]
        self.amplitudes[~active_mask] = np.nan
        self.positions[~active_mask] = np.nan
        self.widths[~active_mask] = np.nan

        # Broadcast to (n_samples, max_peaks, M)
        x_ = x_values.reshape(1, 1, -1)
        pos_ = self.positions[..., np.newaxis]
        wid_ = self.widths[..., np.newaxis]
        amp_ = self.amplitudes[..., np.newaxis]

        # Indicator for inclusive interval [x0 - w/2, x0 + w/2]
        left = pos_ - 0.5 * wid_
        right = pos_ + 0.5 * wid_
        rect = ((x_ >= left) & (x_ <= right)).astype(float)

        y = amp_ * rect  # shape: (n_samples, max_peaks, M)

        # Ensure inactive peaks are NaN across the whole row (consistent with Gaussian/Lorentzian)
        if np.any(~active_mask):
            inactive = (~active_mask)[..., np.newaxis]  # (n_samples, max_peaks, 1)
            y[inactive.repeat(y.shape[-1], axis=-1)] = np.nan

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(self.num_peaks, max_peaks + 1, dtype=np.float32)

        return y


@dataclass
class Dirac(BaseKernel):
    """
    Discrete Dirac pulse model (batch-capable, Gaussian-style API).

    Attributes
    ----------
    amplitude : float or (low, high)
        Impulse amplitude. If a tuple is given, sampled uniformly in [low, high].
    position : float or (low, high)
        Center position. If a tuple is given, sampled uniformly in [low, high].

    Notes
    -----
    Each active peak is placed at the nearest sample index on `x_values`.
    On a uniform grid with step `dt`, index ~= round((pos - x0)/dt) clamped to [0, M-1].
    Inactive peaks are NaN-masked across the full length (consistent with other kernels).
    """

    amplitude: float
    position: float

    def __post_init__(self):
        self.amplitude = self._ensure_tuple(self.amplitude)
        self.position = self._ensure_tuple(self.position)

    def get_kwargs(self) -> dict:
        return {
            "amplitudes": self.amplitudes,
            "positions": self.positions,
        }

    def evaluate(
        self,
        x_values: NDArray[np.float64],
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
    ) -> NDArray[np.float64]:
        """
        Evaluate a batch of Dirac impulses.

        Parameters
        ----------
        x_values : ndarray, shape (M,)
            1D grid where impulses are placed (assumed uniform & ascending).
        n_samples : int
            Number of signals to generate.
        n_peaks : tuple
            (min_peaks, max_peaks) inclusive range of peak count per signal.
        categorical_peak_count : bool, optional
            If True, `self.num_peaks` is converted to one-hot (length = max_peaks+1).

        Returns
        -------
        ndarray, shape (n_samples, max_peaks, M)
            For each (sample, peak), an array that is zero everywhere except at
            one index where it equals the amplitude. Inactive peaks are NaN.
        """
        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])

        # Draw per-sample number of peaks
        self.num_peaks = np.random.randint(low=min_peaks, high=max_peaks + 1, size=n_samples)

        # Draw parameters
        self.amplitudes = np.random.uniform(*self.amplitude, size=(n_samples, max_peaks))
        self.positions = np.random.uniform(*self.position, size=(n_samples, max_peaks))

        # Keep copy for labels before masking
        self.positions_for_labels = self.positions.copy()

        # Mask inactive peaks
        peak_idx = np.arange(max_peaks)
        active_mask = peak_idx < self.num_peaks[:, None]
        self.amplitudes[~active_mask] = np.nan
        self.positions[~active_mask] = np.nan

        # Prepare output
        M = int(x_values.size)
        y = np.full((n_samples, max_peaks, M), np.nan, dtype=float)

        if M == 0:
            if categorical_peak_count:
                self.num_peaks = self._one_hot_numpy(self.num_peaks, max_peaks + 1, dtype=np.float32)
            return y

        # Assume uniform, ascending grid
        dx = np.diff(x_values)
        if not (np.all(dx > 0) and np.allclose(dx, dx[0], rtol=1e-6, atol=1e-12)):
            raise ValueError("Dirac.evaluate expects a uniform, strictly ascending x_values grid.")
        dt = float(dx[0])
        x0 = float(x_values[0])

        # Place impulses at nearest sample for active peaks
        rows, cols = np.where(active_mask)  # indices of active (sample, peak)
        for s, p in zip(rows, cols):
            pos = float(self.positions[s, p])
            amp = float(self.amplitudes[s, p])
            idx = int(round((pos - x0) / dt))
            idx = 0 if idx < 0 else (M - 1 if idx >= M else idx)
            row = np.zeros(M, dtype=float)
            row[idx] = amp
            y[s, p, :] = row

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(self.num_peaks, max_peaks + 1, dtype=np.float32)

        return y
