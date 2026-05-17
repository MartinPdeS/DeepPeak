from typing import Optional
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from .base import BaseKernel, RangeValue, FloatRange
from DeepPeak.peak_count import PeakCount


@dataclass(repr=False)
class Square(BaseKernel):
    """
    Square pulse model with sampled amplitude, center, and full width.

    Parameters
    ----------
    amplitude : float or tuple[float, float]
        Peak amplitude or inclusive sampling range for the pulse amplitude.
    position : float or tuple[float, float]
        Pulse center or inclusive sampling range for the center position.
    width : float or tuple[float, float]
        Full pulse width or inclusive sampling range for the width.

    Notes
    -----
    Each active pulse is evaluated as a constant-amplitude segment on the inclusive
    interval ``[x0 - w/2, x0 + w/2]``.
    """

    amplitude: float
    position: float
    width: float

    def __post_init__(self):
        self._initialize_common_ranges(has_width=True)

    def get_kwargs(self) -> dict:
        return self._state_dict("amplitudes", "positions", "widths")

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
            Evaluated square pulses for each (sample, peak). Inactive peaks are NaN-masked.
        """
        x_, amp_, pos_, wid_, _, _ = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            has_width=True,
        )

        assert wid_ is not None
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
            Input x-values with shape ``(1, 1, M)``.
        amplitudes : NDArray
            Pulse amplitudes with shape ``(n_samples, max_peaks, 1)``.
        centers : NDArray
            Pulse centers with shape ``(n_samples, max_peaks, 1)``.
        widths : NDArray
            Full pulse widths with shape ``(n_samples, max_peaks, 1)``.

        Returns
        -------
        NDArray
            Evaluated square pulse values with shape ``(n_samples, max_peaks, M)``.

        Notes
        -----
        The square pulse is evaluated as

        ``S(x; A, x0, w) = A * 1_{x in [x0 - w/2, x0 + w/2]}``.
        """
        # Indicator for inclusive interval [x0 - w/2, x0 + w/2]
        left = centers - 0.5 * widths
        right = centers + 0.5 * widths
        rect = ((x_values >= left) & (x_values <= right)).astype(float)

        return amplitudes * rect
