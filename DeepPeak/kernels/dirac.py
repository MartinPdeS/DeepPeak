from abc import ABC
from typing import Optional
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from .base import BaseKernel
from DeepPeak.peak_count import PeakCount


@dataclass(repr=False)
class Dirac(BaseKernel):
    """
    Discrete Dirac impulse model sampled on a finite grid.

    Parameters
    ----------
    amplitude : float or tuple[float, float]
        Impulse amplitude or inclusive sampling range for the impulse height.
    position : float or tuple[float, float]
        Impulse position or inclusive sampling range for the center location.

    Notes
    -----
    Each active peak is placed at the nearest sample index on `x_values`.
    On a uniform grid with step `dt`, index ~= round((pos - x0)/dt) clamped to [0, M-1].
    Inactive peaks are NaN-masked across the full length (consistent with other kernels).
    """

    amplitude: float
    position: float

    def __post_init__(self):
        self._initialize_common_ranges(has_width=False)

    def get_kwargs(self) -> dict:
        return self._state_dict("amplitudes", "positions")

    def evaluate(
        self,
        x_values: NDArray[np.float64],
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
        peak_count: PeakCount | None = None,
        peak_count_distribution: str = "uniform",
        peak_count_rate: Optional[float] = None,
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
        peak_count : PeakCount or None, optional
            Optional peak-count sampler overriding the legacy distribution arguments.
        peak_count_distribution : {"uniform", "poisson"}, default="uniform"
            Legacy peak-count distribution used when ``peak_count`` is not provided.
        peak_count_rate : float or tuple[float, float], optional
            Legacy Poisson rate configuration used when
            ``peak_count_distribution='poisson'``.

        Returns
        -------
        ndarray, shape (n_samples, max_peaks, M)
            For each (sample, peak), an array that is zero everywhere except at
            one index where it equals the amplitude. Inactive peaks are NaN.
        """
        x_values, _, _, _, active_mask, max_peaks = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            has_width=False,
        )

        # Prepare output
        x_grid = x_values[0, 0, :]
        M = int(x_grid.size)
        y = np.full((n_samples, max_peaks, M), np.nan, dtype=float)

        if M == 0:
            return y

        # Assume uniform, ascending grid
        dx = np.diff(x_grid)
        if not (np.all(dx > 0) and np.allclose(dx, dx[0], rtol=1e-6, atol=1e-12)):
            raise ValueError(
                "Dirac.evaluate expects a uniform, strictly ascending x_values grid."
            )
        dt = float(dx[0])
        x0 = float(x_grid[0])

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

        return y

    def _kernel(
        self, x_values: NDArray, amplitudes: NDArray, centers: NDArray, widths: NDArray
    ) -> NDArray:
        """
        Compute rectangle-indicator values for a Dirac-like pulse approximation.

        Parameters
        ----------
        x_values : NDArray
            Input x-values with shape ``(1, 1, M)``.
        amplitudes : NDArray
            Pulse amplitudes with shape ``(n_samples, max_peaks, 1)``.
        centers : NDArray
            Pulse centers with shape ``(n_samples, max_peaks, 1)``.
        widths : NDArray
            Auxiliary widths with shape ``(n_samples, max_peaks, 1)``.
            This argument is unused by ``Dirac.evaluate()`` but retained for
            signature compatibility with other kernels.

        Returns
        -------
        NDArray
            Rectangle-indicator values with shape ``(n_samples, max_peaks, M)``.

        Notes
        -----
        This helper is not used by ``Dirac.evaluate()``. The public evaluation path
        places impulses at the nearest grid index directly.

        """
        # Indicator for inclusive interval [x0 - w/2, x0 + w/2]
        left = centers - 0.5 * widths
        right = centers + 0.5 * widths
        rect = ((x_values >= left) & (x_values <= right)).astype(float)

        return amplitudes * rect
