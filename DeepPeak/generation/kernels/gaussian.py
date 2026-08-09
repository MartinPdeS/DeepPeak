from typing import Optional
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from .base import BaseKernel, RangeValue, FloatRange
from ..peak_count import PeakCount


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
    rise_scale: RangeValue = 1.0
    decay_scale: RangeValue = 1.0

    def __post_init__(self):
        self._initialize_common_ranges(has_width=True)
        self._rise_scale = self._normalize_range(
            "rise_scale", self.rise_scale, minimum=0.0, inclusive_minimum=False
        )
        self._decay_scale = self._normalize_range(
            "decay_scale", self.decay_scale, minimum=0.0, inclusive_minimum=False
        )

    def get_kwargs(self) -> dict:
        return self._state_dict(
            "amplitudes", "positions", "widths", "rise_scales", "decay_scales"
        )

    def evaluate(
        self,
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
        peak_count: PeakCount | None = None,
        peak_count_distribution: str = "uniform",
        peak_count_rate: Optional[float] = None,
        rng: np.random.Generator | None = None,
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
        rng = np.random.default_rng() if rng is None else rng
        x_, amp_, pos_, wid_, active_mask, _ = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            rng=rng,
            has_width=True,
        )

        assert wid_ is not None
        shape = self.amplitudes.shape
        self.rise_scales = self._sample_uniform(self._rise_scale, size=shape, rng=rng)
        self.decay_scales = self._sample_uniform(self._decay_scale, size=shape, rng=rng)
        self.rise_scales[~active_mask] = np.nan
        self.decay_scales[~active_mask] = np.nan

        return self._kernel(
            x_values=x_,
            amplitudes=amp_,
            centers=pos_,
            widths=wid_,
            rise_scales=self.rise_scales[..., np.newaxis],
            decay_scales=self.decay_scales[..., np.newaxis],
        )

    def _kernel(
        self,
        x_values: NDArray,
        amplitudes: NDArray,
        centers: NDArray,
        widths: NDArray,
        rise_scales: NDArray,
        decay_scales: NDArray,
    ) -> NDArray:
        return self._split_gaussian(
            x_values, amplitudes, centers, widths, rise_scales, decay_scales
        )
