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
        self._amplitude = self._ensure_tuple(self.amplitude)
        self._position = self._ensure_tuple(self.position)
        self._width = self._ensure_tuple(self.width)

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
    ) -> np.ndarray:
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

        self.num_peaks = np.random.randint(
            low=min_peaks, high=max_peaks + 1, size=n_samples
        )
        self.amplitudes = np.random.uniform(
            *self._amplitude, size=(n_samples, max_peaks)
        )

        # interpret user-supplied position range in normalized space [0,1]
        # map to real x-values
        self.positions = np.random.uniform(*self._position, size=(n_samples, max_peaks))

        self.widths = np.random.uniform(*self._width, size=(n_samples, max_peaks))

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
            self.num_peaks = self._one_hot_numpy(
                self.num_peaks, max_peaks + 1, dtype=np.float32
            )

        return self._kernel(x_values=x_, amplitudes=amp_, centers=pos_, widths=wid_)

    def _kernel(
        self, x_values: NDArray, amplitudes: NDArray, centers: NDArray, widths: NDArray
    ) -> NDArray:
        r"""
        Compute Gaussian kernel values.

        The Gaussian profile is given by the formula:
        ... math::
            G(x; A, x0, $\sigma$) = A * exp(-0.5 * ((x - x0) / $\sigma$)^2)

        Parameters
        ----------
        x_values : NDArray
            Input x-values, shape (1, 1, M).
        amplitudes : NDArray
            Amplitudes, shape (n_samples, max_peaks, 1).
        centers : NDArray
            Centers, shape (n_samples, max_peaks, 1).
        widths : NDArray
            Widths (standard deviations), shape (n_samples, max_peaks, 1).

        Returns
        -------
        NDArray
            Evaluated Gaussian values, shape (n_samples, max_peaks, M).
        """
        return amplitudes * np.exp(-0.5 * ((x_values - centers) / widths) ** 2)


@dataclass
class Lorentzian(BaseKernel):
    r"""
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
        L(x; A, x0, $\gamma$) = A * \gamma^2 / ((x - x0)^2 + $\gamma$^2) = A / (1 + ((x - x0)/$\gamma$)^2)
    """

    amplitude: float
    position: float
    width: float  # HWHM $\gamma$

    def __post_init__(self):
        self._amplitude = self._ensure_tuple(self.amplitude)
        self._position = self._ensure_tuple(self.position)
        self._width = self._ensure_tuple(self.width)

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
        self.num_peaks = np.random.randint(
            low=min_peaks, high=max_peaks + 1, size=n_samples
        )

        # Draw parameters (uniform in provided ranges)
        self.amplitudes = np.random.uniform(
            *self._amplitude, size=(n_samples, max_peaks)
        )
        # interpret user-supplied position range in normalized space [0,1]
        # map to real x-values
        self.positions = np.random.uniform(*self._position, size=(n_samples, max_peaks))

        self.widths = np.random.uniform(*self._width, size=(n_samples, max_peaks))

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

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(
                self.num_peaks, max_peaks + 1, dtype=np.float32
            )

        return self._kernel(x_values=x_, amplitudes=amp_, centers=pos_, widths=gam_)

    def _kernel(
        self, x_values: NDArray, amplitudes: NDArray, centers: NDArray, widths: NDArray
    ) -> NDArray:
        r"""
        Compute Lorentzian kernel values.

        The Lorentzian profile is given by the formula:
        ... math::
            L(x; A, x0, \gamma) = A \cdot \frac{\gamma^2}{(x - x0)^2 + \gamma^2}

        Parameters
        ----------
        x_values : NDArray
            Input x-values, shape (1, 1, M).
        amplitudes : NDArray
            Amplitudes, shape (n_samples, max_peaks, 1).
        centers : NDArray
            Centers, shape (n_samples, max_peaks, 1).
        widths : NDArray
            Widths (HWHM), shape (n_samples, max_peaks, 1).

        Returns
        -------
        NDArray
            Evaluated Lorentzian values, shape (n_samples, max_peaks, M).
        """
        return amplitudes * (widths**2 / ((x_values - centers) ** 2 + widths**2))


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
        self._amplitude = self._ensure_tuple(self.amplitude)
        self._position = self._ensure_tuple(self.position)
        self._width = self._ensure_tuple(self.width)

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
        self.num_peaks = np.random.randint(
            low=min_peaks, high=max_peaks + 1, size=n_samples
        )

        # Draw parameters (uniform in provided ranges)
        self.amplitudes = np.random.uniform(
            *self._amplitude, size=(n_samples, max_peaks)
        )
        # interpret user-supplied position range in normalized space [0,1]
        # map to real x-values
        self.positions = np.random.uniform(*self._position, size=(n_samples, max_peaks))

        self.widths = np.random.uniform(*self._width, size=(n_samples, max_peaks))

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
            self.num_peaks = self._one_hot_numpy(
                self.num_peaks, max_peaks + 1, dtype=np.float32
            )

        return self._kernel(x_values=x_, amplitudes=amp_, centers=pos_, widths=wid_)

    def _kernel(
        self, x_values: NDArray, amplitudes: NDArray, centers: NDArray, widths: NDArray
    ) -> NDArray:
        """
        Compute square pulse kernel values.

        The square pulse is:
            S(x; A, x0, w) = A · 1_{x ∈ [x0 - w/2, x0 + w/2]}

        Parameters
        ----------
        x_values : NDArray
            Input x-values, shape (1, 1, M).
        amplitudes : NDArray
            Amplitudes, shape (n_samples, max_peaks, 1).
        centers : NDArray
            Centers, shape (n_samples, max_peaks, 1).
        widths : NDArray
            Widths, shape (n_samples, max_peaks, 1).

        Returns
        -------
        NDArray
            Computed square pulse values, shape (n_samples, max_peaks, M).

        """
        # Indicator for inclusive interval [x0 - w/2, x0 + w/2]
        left = centers - 0.5 * widths
        right = centers + 0.5 * widths
        rect = ((x_values >= left) & (x_values <= right)).astype(float)

        return amplitudes * rect


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
        self._amplitude = self._ensure_tuple(self.amplitude)
        self._position = self._ensure_tuple(self.position)

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
        self.num_peaks = np.random.randint(
            low=min_peaks, high=max_peaks + 1, size=n_samples
        )

        # Draw parameters
        self.amplitudes = np.random.uniform(
            *self._amplitude, size=(n_samples, max_peaks)
        )

        # interpret user-supplied position range in normalized space [0,1]
        # map to real x-values
        self.positions = np.random.uniform(*self._position, size=(n_samples, max_peaks))

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
                self.num_peaks = self._one_hot_numpy(
                    self.num_peaks, max_peaks + 1, dtype=np.float32
                )
            return y

        # Assume uniform, ascending grid
        dx = np.diff(x_values)
        if not (np.all(dx > 0) and np.allclose(dx, dx[0], rtol=1e-6, atol=1e-12)):
            raise ValueError(
                "Dirac.evaluate expects a uniform, strictly ascending x_values grid."
            )
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
            self.num_peaks = self._one_hot_numpy(
                self.num_peaks, max_peaks + 1, dtype=np.float32
            )

        return y

    def _kernel(
        self, x_values: NDArray, amplitudes: NDArray, centers: NDArray, widths: NDArray
    ) -> NDArray:
        """
        Compute Dirac pulse kernel values.

        Parameters
        ----------
        x_values : NDArray
            Input x-values, shape (1, 1, M).
        amplitudes : NDArray
            Amplitudes, shape (n_samples, max_peaks, 1).
        centers : NDArray
            Centers, shape (n_samples, max_peaks, 1).
        widths : NDArray
            Widths, shape (n_samples, max_peaks, 1). (Unused for Dirac.)

        Returns
        -------
        NDArray
            Computed Dirac pulse values, shape (n_samples, max_peaks, M).

        """
        # Indicator for inclusive interval [x0 - w/2, x0 + w/2]
        left = centers - 0.5 * widths
        right = centers + 0.5 * widths
        rect = ((x_values >= left) & (x_values <= right)).astype(float)

        return amplitudes * rect


@dataclass
class CustomKernel(BaseKernel):
    """
    A pulse model that uses a user supplied kernel shape.

    The kernel is a one dimensional array that defines the pulse shape.
    The user can place scaled copies of this kernel at random positions,
    exactly like Gaussian or Lorentzian pulses.

    Attributes
    ----------
    kernel : ndarray
        One dimensional array representing the pulse shape. The array does not
        need to match the resolution of x_values. Linear interpolation is used.
    amplitude : float or (low, high)
        Amplitude range for random sampling.
    position : float or (low, high)
        Center position range for the kernel.
    """

    kernel: NDArray
    amplitude: float
    position: float

    def __post_init__(self):
        if self.kernel.ndim != 1:
            raise ValueError("kernel must be a one dimensional array")

        self._amplitude = self._ensure_tuple(self.amplitude)
        self._position = self._ensure_tuple(self.position)

        # Normalize kernel length for later interpolation
        self.kernel = np.asarray(self.kernel, dtype=float)
        self.kernel_x = np.linspace(0, 1, self.kernel.size)

    def get_kwargs(self) -> dict:
        return {
            "kernel": self.kernel,
            "amplitudes": self.amplitudes,
            "positions": self.positions,
        }

    def evaluate(
        self,
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
    ) -> NDArray:
        """
        Evaluate the custom kernel at random positions and amplitudes.

        The output has shape (n_samples, max_peaks, M)
        where M = len(x_values).
        """
        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])
        M = x_values.size

        # Draw number of peaks per sample
        self.num_peaks = np.random.randint(
            low=min_peaks, high=max_peaks + 1, size=n_samples
        )

        # Draw random amplitudes and positions
        self.amplitudes = np.random.uniform(
            *self._amplitude, size=(n_samples, max_peaks)
        )
        self.positions = np.random.uniform(*self._position, size=(n_samples, max_peaks))

        # Keep original unmasked positions for labels
        self.positions_for_labels = self.positions.copy()

        # Mask inactive peaks
        peak_indices = np.arange(max_peaks)
        active_mask = peak_indices < self.num_peaks[:, None]
        self.amplitudes[~active_mask] = np.nan
        self.positions[~active_mask] = np.nan

        # Broadcast arrays
        x_ = x_values.reshape(1, 1, M)
        pos_ = self.positions[..., np.newaxis]  # (n_samples, max_peaks, 1)
        amp_ = self.amplitudes[..., np.newaxis]  # (n_samples, max_peaks, 1)

        # Evaluate the kernel
        y = self._kernel(x_values=x_, amplitudes=amp_, centers=pos_)

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(
                self.num_peaks, max_peaks + 1, dtype=np.float32
            )

        return y

    def _kernel(
        self,
        x_values: NDArray,
        amplitudes: NDArray,
        centers: NDArray,
    ) -> NDArray:
        """
        Evaluate the user kernel at each center without truncation.

        The kernel is defined on self.kernel_x which spans [0, 1].
        We stretch this interval to match the actual kernel width in x_values.
        """
        n_samples, max_peaks = amplitudes.shape[0], amplitudes.shape[1]
        M = x_values.shape[-1]

        # True kernel support in x coordinates
        # Kernel width equals the median dx times kernel length
        # This preserves your recovered kernel exactly
        x_grid = x_values[0, 0, :]
        dx = float(np.median(np.diff(x_grid)))
        kernel_width = dx * self.kernel.size

        # Kernel coordinate in real x-space
        kernel_support_x = np.linspace(
            -0.5 * kernel_width, 0.5 * kernel_width, self.kernel.size
        )

        # Output
        y = np.zeros((n_samples, max_peaks, M), dtype=float)

        for i in range(n_samples):
            for j in range(max_peaks):

                A = amplitudes[i, j, 0]
                x0 = centers[i, j, 0]

                # Inactive peaks
                if np.isnan(A) or np.isnan(x0):
                    y[i, j, :] = np.nan
                    continue

                # Shift kernel to center x0
                shifted_support = kernel_support_x + x0

                # Interpolate without truncation
                vals = np.interp(
                    x_grid, shifted_support, self.kernel, left=0.0, right=0.0
                )

                y[i, j, :] = A * vals

        return y
