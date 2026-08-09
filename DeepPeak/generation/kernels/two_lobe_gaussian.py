from typing import Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import BaseKernel, FloatRange, RangeValue
from ..peak_count import PeakCount


@dataclass(repr=False)
class TwoLobeGaussian(BaseKernel):
    """Gaussian pulse with an optional delayed secondary lobe.

    Parameters
    ----------
    amplitude : float or tuple[float, float]
        Main-lobe amplitude or inclusive sampling range for it.
    position : float or tuple[float, float]
        Main-lobe center or inclusive sampling range for the center position.
    width : float or tuple[float, float]
        Main-lobe standard deviation or inclusive sampling range for it.
    secondary_ratio : float or tuple[float, float], default=(0.0, 0.25)
        Ratio between the secondary-lobe amplitude and the main-lobe amplitude.
    secondary_offset : float or tuple[float, float], default=(0.0, 0.0)
        Offset applied to the secondary-lobe center relative to ``position``.
    secondary_width : float or tuple[float, float] or None, optional
        Secondary-lobe standard deviation. If omitted, the main-lobe width range is
        reused.
    secondary_presence : float or tuple[float, float], default=1.0
        Probability that the secondary lobe is active for each sampled peak.

    Notes
    -----
    Each sampled peak is the sum of a main Gaussian centered at ``position`` and
    a second Gaussian centered at ``position + secondary_offset``. The secondary
    lobe amplitude is expressed as a ratio of the main lobe amplitude and can be
    disabled per peak through ``secondary_presence``.
    """

    amplitude: RangeValue
    position: RangeValue
    width: RangeValue
    secondary_ratio: RangeValue = (0.0, 0.25)
    secondary_offset: RangeValue = (0.0, 0.0)
    secondary_width: RangeValue | None = None
    secondary_presence: RangeValue = 1.0
    rise_scale: RangeValue = 1.0
    decay_scale: RangeValue = 1.0

    def __post_init__(self):
        self._initialize_common_ranges(has_width=True)
        self._secondary_ratio = self._normalize_range(
            "secondary_ratio",
            self.secondary_ratio,
            minimum=0.0,
        )
        self._secondary_offset = self._normalize_range(
            "secondary_offset",
            self.secondary_offset,
        )
        secondary_width = (
            self.width if self.secondary_width is None else self.secondary_width
        )
        self._secondary_width = self._normalize_range(
            "secondary_width",
            secondary_width,
            minimum=0.0,
            inclusive_minimum=False,
        )
        self._secondary_presence = self._normalize_probability_range(
            "secondary_presence",
            self.secondary_presence,
        )
        self._rise_scale = self._normalize_range(
            "rise_scale", self.rise_scale, minimum=0.0, inclusive_minimum=False
        )
        self._decay_scale = self._normalize_range(
            "decay_scale", self.decay_scale, minimum=0.0, inclusive_minimum=False
        )

    def get_kwargs(self) -> dict:
        return self._state_dict(
            "amplitudes",
            "positions",
            "widths",
            "secondary_ratios",
            "secondary_offsets",
            "secondary_widths",
        )

    def _resolved_param_labels(self) -> list[str]:
        labels = super()._resolved_param_labels()
        if hasattr(self, "secondary_ratios"):
            labels.append(f"secondary_ratio = {self.secondary_ratios[0, 0]:.4g}")
        if hasattr(self, "secondary_offsets"):
            labels.append(f"secondary_offset = {self.secondary_offsets[0, 0]:.4g}")
        if hasattr(self, "secondary_widths"):
            labels.append(f"secondary_width = {self.secondary_widths[0, 0]:.4g}")
        return labels

    def _plot_bounds(self) -> tuple[float, float]:
        position_low, position_high = self._position
        _, main_width_high = self._width
        offset_low, offset_high = self._secondary_offset
        _, secondary_width_high = self._secondary_width

        left = float(position_low) - 4.0 * float(main_width_high)
        right = max(
            float(position_high) + 4.0 * float(main_width_high),
            float(position_high)
            + float(offset_high)
            + 4.0 * float(secondary_width_high),
        )
        return (left, right)

    @classmethod
    def _normalize_probability_range(
        cls,
        name: str,
        value: RangeValue,
    ) -> FloatRange:
        low, high = cls._normalize_range(name, value, minimum=0.0)
        if high > 1.0:
            raise ValueError(f"{name} values must be <= 1.0.")
        return (low, high)

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
    ) -> NDArray:
        """Evaluate a batch of two-lobe Gaussian pulses.

        Parameters
        ----------
        x_values : NDArray
            One-dimensional evaluation grid.
        n_samples : int
            Number of signals to generate.
        n_peaks : tuple[int, int]
            Inclusive lower and upper bounds for the number of active peaks.
        categorical_peak_count : bool, default=False
            If ``True``, encode the sampled peak count as one-hot values.
        peak_count : PeakCount or None, optional
            Optional peak-count sampler overriding the legacy distribution arguments.
        peak_count_distribution : {"uniform", "poisson"}, default="uniform"
            Legacy peak-count distribution used when ``peak_count`` is not provided.
        peak_count_rate : float or tuple[float, float], optional
            Legacy Poisson rate configuration used when
            ``peak_count_distribution='poisson'``.

        Returns
        -------
        NDArray
            Evaluated two-lobe Gaussian components with shape
            ``(n_samples, max_peaks, len(x_values))``. Inactive peaks are NaN-masked.
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

        secondary_ratios = self._sample_uniform(
            self._secondary_ratio, size=shape, rng=rng
        )
        secondary_offsets = self._sample_uniform(
            self._secondary_offset, size=shape, rng=rng
        )
        secondary_widths = self._sample_uniform(
            self._secondary_width, size=shape, rng=rng
        )
        secondary_presence = self._sample_uniform(
            self._secondary_presence, size=shape, rng=rng
        )
        secondary_active = rng.uniform(0.0, 1.0, size=shape) <= secondary_presence
        rise_scales = self._sample_uniform(self._rise_scale, size=shape, rng=rng)
        decay_scales = self._sample_uniform(self._decay_scale, size=shape, rng=rng)

        secondary_ratios = np.where(
            active_mask & secondary_active, secondary_ratios, 0.0
        )
        secondary_offsets = np.where(active_mask, secondary_offsets, np.nan)
        secondary_widths = np.where(active_mask, secondary_widths, np.nan)
        rise_scales[~active_mask] = np.nan
        decay_scales[~active_mask] = np.nan

        self.secondary_ratios = secondary_ratios
        self.secondary_offsets = secondary_offsets
        self.secondary_widths = secondary_widths
        self.rise_scales = rise_scales
        self.decay_scales = decay_scales

        return self._kernel(
            x_values=x_,
            amplitudes=amp_,
            centers=pos_,
            widths=wid_,
            secondary_ratios=secondary_ratios[..., np.newaxis],
            secondary_offsets=secondary_offsets[..., np.newaxis],
            secondary_widths=secondary_widths[..., np.newaxis],
            rise_scales=rise_scales[..., np.newaxis],
            decay_scales=decay_scales[..., np.newaxis],
        )

    def _kernel(
        self,
        x_values: NDArray,
        amplitudes: NDArray,
        centers: NDArray,
        widths: NDArray,
        secondary_ratios: NDArray,
        secondary_offsets: NDArray,
        secondary_widths: NDArray,
        rise_scales: NDArray,
        decay_scales: NDArray,
    ) -> NDArray:
        """Compute main and secondary Gaussian lobe values.

        Parameters
        ----------
        x_values : NDArray
            Input x-values with shape ``(1, 1, M)``.
        amplitudes : NDArray
            Main-lobe amplitudes with shape ``(n_samples, max_peaks, 1)``.
        centers : NDArray
            Main-lobe centers with shape ``(n_samples, max_peaks, 1)``.
        widths : NDArray
            Main-lobe standard deviations with shape ``(n_samples, max_peaks, 1)``.
        secondary_ratios : NDArray
            Secondary-to-main amplitude ratios with shape
            ``(n_samples, max_peaks, 1)``.
        secondary_offsets : NDArray
            Secondary-center offsets with shape ``(n_samples, max_peaks, 1)``.
        secondary_widths : NDArray
            Secondary standard deviations with shape
            ``(n_samples, max_peaks, 1)``.
        rise_scales : NDArray
            Scale factor for the left (rising) half of the main lobe,
            shape ``(n_samples, max_peaks, 1)``.
        decay_scales : NDArray
            Scale factor for the right (decaying) half of the main lobe,
            shape ``(n_samples, max_peaks, 1)``.

        Returns
        -------
        NDArray
            Sum of the main and secondary Gaussian lobe values with shape
            ``(n_samples, max_peaks, M)``.
        """
        main_lobe = self._split_gaussian(
            x_values, amplitudes, centers, widths, rise_scales, decay_scales
        )

        secondary_centers = centers + secondary_offsets
        secondary_amplitudes = amplitudes * secondary_ratios
        secondary_lobe = secondary_amplitudes * np.exp(
            -0.5 * ((x_values - secondary_centers) / secondary_widths) ** 2
        )

        return main_lobe + secondary_lobe
