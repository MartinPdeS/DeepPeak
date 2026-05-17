from typing import Optional
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from .base import BaseKernel, RangeValue, FloatRange
from DeepPeak.peak_count import PeakCount


@dataclass(repr=False)
class Gaussian(BaseKernel):
    """
    Gaussian pulse model with sampled amplitude, center, and width.

    Parameters
    ----------
    amplitude : float or tuple[float, float]
        Peak amplitude or inclusive sampling range for the pulse amplitude.
    position : float or tuple[float, float]
        Pulse center or inclusive sampling range for the center position.
    width : float or tuple[float, float]
        Standard deviation or inclusive sampling range for the Gaussian width.

    Notes
    -----
    Each active pulse is evaluated as

    ``A * exp(-0.5 * ((x - x0) / sigma) ** 2)``.
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
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
        peak_count: PeakCount | None = None,
        peak_count_distribution: str = "uniform",
        peak_count_rate: Optional[float] = None,
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
            Array of shape ``(n_samples, max_peaks, len(x_values))`` containing
            one Gaussian component per sampled peak. Inactive peaks are NaN-masked.
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
        r"""
        Compute Gaussian kernel values.

        Parameters
        ----------
        x_values : NDArray
            Input x-values with shape ``(1, 1, M)``.
        amplitudes : NDArray
            Pulse amplitudes with shape ``(n_samples, max_peaks, 1)``.
        centers : NDArray
            Pulse centers with shape ``(n_samples, max_peaks, 1)``.
        widths : NDArray
            Pulse standard deviations with shape ``(n_samples, max_peaks, 1)``.

        Returns
        -------
        NDArray
            Evaluated Gaussian pulse values with shape
            ``(n_samples, max_peaks, M)``.

        Notes
        -----
        The Gaussian profile is evaluated as

        ``G(x; A, x0, sigma) = A * exp(-0.5 * ((x - x0) / sigma) ** 2)``.
        """
        return amplitudes * np.exp(-0.5 * ((x_values - centers) / widths) ** 2)
