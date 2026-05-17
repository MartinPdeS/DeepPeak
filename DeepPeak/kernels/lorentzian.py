from typing import Optional
from dataclasses import dataclass
from numpy.typing import NDArray

from .base import BaseKernel, RangeValue, FloatRange
from DeepPeak.peak_count import PeakCount


@dataclass(repr=False)
class Lorentzian(BaseKernel):
    r"""
    Lorentzian pulse model with sampled amplitude, center, and half-width.

    Parameters
    ----------
    amplitude : float or tuple[float, float]
        Peak amplitude or inclusive sampling range for the pulse amplitude.
    position : float or tuple[float, float]
        Pulse center or inclusive sampling range for the center position.
    width : float or tuple[float, float]
        Half-width at half maximum ``gamma`` or inclusive sampling range for it.

    Notes
    -----
    Each active pulse is evaluated as

    ``L(x; A, x0, gamma) = A * gamma**2 / ((x - x0)**2 + gamma**2)``.
    """

    amplitude: float
    position: float
    width: float  # HWHM $\gamma$

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
            Evaluated Lorentzian components for each ``(sample, peak)`` pair.
            Inactive peaks are NaN-masked.
        """
        x_, amp_, pos_, gam_, _, _ = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            has_width=True,
        )

        assert gam_ is not None
        return self._kernel(x_values=x_, amplitudes=amp_, centers=pos_, widths=gam_)

    def _kernel(
        self, x_values: NDArray, amplitudes: NDArray, centers: NDArray, widths: NDArray
    ) -> NDArray:
        r"""
        Compute Lorentzian kernel values.

        Parameters
        ----------
        x_values : NDArray
            Input x-values with shape ``(1, 1, M)``.
        amplitudes : NDArray
            Pulse amplitudes with shape ``(n_samples, max_peaks, 1)``.
        centers : NDArray
            Pulse centers with shape ``(n_samples, max_peaks, 1)``.
        widths : NDArray
            Lorentzian half-widths with shape ``(n_samples, max_peaks, 1)``.

        Returns
        -------
        NDArray
            Evaluated Lorentzian pulse values with shape
            ``(n_samples, max_peaks, M)``.
        """
        return amplitudes * (widths**2 / ((x_values - centers) ** 2 + widths**2))
