from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MPSPlots import helper
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

from DeepPeak import utils

ArrayLike = Union[np.ndarray, Sequence[float]]


def _trapezoid(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """Integrate one curve with NumPy's trapezoidal rule across versions."""

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y_values, x_values))


def _crossing_time(
    time: np.ndarray,
    values: np.ndarray,
    level: float,
    *,
    side: str,
) -> float:
    """Estimate the time at which a waveform crosses a given level."""

    indices = np.where(values >= float(level))[0]
    if indices.size == 0:
        return float("nan")

    if side == "left":
        if int(indices[0]) == 0:
            return float("nan")
        right_index = int(indices[0])
        left_index = right_index - 1
    elif side == "right":
        if int(indices[-1]) >= values.size - 1:
            return float("nan")
        left_index = int(indices[-1])
        right_index = left_index + 1
    else:
        raise ValueError(f"Unknown side {side!r}; expected 'left' or 'right'.")

    x0 = float(time[left_index])
    x1 = float(time[right_index])
    y0 = float(values[left_index])
    y1 = float(values[right_index])

    if y1 == y0:
        return x0

    return float(x0 + (float(level) - y0) * (x1 - x0) / (y1 - y0))


def _apply_cosine_taper(kernel: np.ndarray, fraction: float) -> np.ndarray:
    """Apply a half-cosine taper to both ends of a 1-D kernel array.

    Parameters
    ----------
    kernel : ndarray
        One-dimensional kernel, already normalized.
    fraction : float
        Fraction of samples at each end to taper (0.0 = no taper, 0.1 = 10 %).

    Returns
    -------
    ndarray
        Tapered copy of the input array.
    """
    n = len(kernel)
    n_taper = max(1, int(round(fraction * n)))
    window = np.ones(n, dtype=float)
    # left fade-in: 0 → 1 over n_taper samples
    window[:n_taper] = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / n_taper))
    # right fade-out: 1 → 0 over n_taper samples
    window[-n_taper:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_taper) / n_taper))
    return kernel * window


def _gaussian_reference(time: np.ndarray, centroid: float, sigma: float) -> np.ndarray:
    """Build a unit-height Gaussian reference curve."""

    if not np.isfinite(sigma) or float(sigma) <= 0.0:
        return np.full_like(time, np.nan, dtype=float)

    gaussian = np.exp(
        -0.5 * ((np.asarray(time, dtype=float) - float(centroid)) / float(sigma)) ** 2
    )
    maximum = float(np.max(gaussian))
    if maximum <= 0.0:
        return np.full_like(time, np.nan, dtype=float)
    return gaussian / maximum


def _shift_with_fill(
    values: np.ndarray, shift: int, *, fill_value: float = 0.0
) -> np.ndarray:
    """Shift a 1-D array without wrap-around, filling vacated samples."""

    values = np.asarray(values, dtype=float).ravel()
    shift = int(shift)
    if shift == 0 or values.size == 0:
        return values.copy()

    shifted = np.full(values.shape, float(fill_value), dtype=float)
    if abs(shift) >= values.size:
        return shifted

    if shift > 0:
        shifted[shift:] = values[:-shift]
    else:
        shifted[:shift] = values[-shift:]
    return shifted


def _recenter_kernel(kernel: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Align one kernel to a common reference point before export."""

    kernel = np.asarray(kernel, dtype=float).ravel()
    if kernel.size == 0:
        return kernel.copy()

    if method is None:
        return kernel.copy()

    normalized_method = str(method).strip().lower()
    if normalized_method in {"none", "original"}:
        return kernel.copy()

    target_index = kernel.size // 2
    positive_kernel = np.clip(kernel, 0.0, None)

    if normalized_method in {"max", "peak"}:
        source_index = int(np.argmax(positive_kernel))
    elif normalized_method in {"centroid", "com", "center_of_mass"}:
        total_weight = float(np.sum(positive_kernel))
        if total_weight <= 0.0:
            return kernel.copy()
        sample_indices = np.arange(kernel.size, dtype=float)
        source_index = int(
            np.rint(np.sum(sample_indices * positive_kernel) / total_weight)
        )
    else:
        raise ValueError(
            "recenter must be one of None, 'none', 'max', or 'centroid'. "
            f"Got {method!r}."
        )

    return _shift_with_fill(kernel, target_index - source_index, fill_value=0.0)


class PulseShapeAnalyzer:
    """Inspect detected peaks as aligned windows and summarize their shapes.

    The class is designed for notebook workflows:

    1. create an analyzer from a :class:`CsvTrace` or raw arrays
    2. detect or select peaks
    3. extract baseline-corrected windows around those peaks
    4. inspect per-pulse and aggregate shape diagnostics
    """

    def __init__(
        self,
        signal: ArrayLike,
        *,
        sampling_rate: Optional[float] = None,
        time: Optional[ArrayLike] = None,
        dx: Optional[float] = None,
    ) -> None:
        self.signal = np.asarray(signal, dtype=float).ravel()
        if self.signal.ndim != 1:
            raise ValueError("signal must be one-dimensional.")

        if sampling_rate is not None:
            sampling_rate = float(sampling_rate)
            if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
                raise ValueError("sampling_rate must be a finite positive number.")
            if dx is not None:
                raise ValueError("Provide either sampling_rate or dx, not both.")

            self.sampling_rate = sampling_rate
            self.dx = 1.0 / sampling_rate
            self.unit_label = "s"

            if time is None:
                self.time = np.arange(self.signal.size, dtype=float) * self.dx
            else:
                self.time = np.asarray(time, dtype=float).ravel()
                if self.time.shape != self.signal.shape:
                    raise ValueError("time and signal must have the same shape.")
        else:
            self.sampling_rate = None
            self.unit_label = "sample"

            if time is None:
                if dx is None:
                    self.dx = 1.0
                    self.time = np.arange(self.signal.size, dtype=float)
                else:
                    self.dx = float(dx)
                    self.time = np.arange(self.signal.size, dtype=float) * self.dx
            else:
                self.time = np.asarray(time, dtype=float).ravel()
                if self.time.shape != self.signal.shape:
                    raise ValueError("time and signal must have the same shape.")
                if self.time.size >= 2:
                    inferred_dx = float(np.median(np.diff(self.time)))
                elif dx is not None:
                    inferred_dx = float(dx)
                else:
                    inferred_dx = 1.0
                self.dx = float(inferred_dx if dx is None else dx)

        self._detected_peak_indices = np.asarray([], dtype=int)
        self._detected_peak_widths_samples = np.asarray([], dtype=float)
        self._selected_peak_mask = np.asarray([], dtype=bool)
        self._window_peak_indices = np.asarray([], dtype=int)
        self._window_peak_widths_samples = np.asarray([], dtype=float)
        self._window_baselines = np.asarray([], dtype=float)
        self._pulse_windows = np.empty((0, 0), dtype=float)
        self._local_time = np.asarray([], dtype=float)
        self._shape_statistics_cache: Optional[pd.DataFrame] = None

    @classmethod
    def from_trace(
        cls,
        trace: Any,
        *,
        use_processed: bool = True,
        sampling_rate: Optional[float] = None,
    ) -> "PulseShapeAnalyzer":
        """Build an analyzer from a ``CsvTrace``-like object.

        By default, widths and timings are reported in sample units. Pass
        ``sampling_rate=trace.sampling_rate`` to express them in seconds.
        """

        signal = trace.y_processed if use_processed else trace.y_raw
        return cls(signal=signal, sampling_rate=sampling_rate)

    @property
    def peak_indices(self) -> np.ndarray:
        """Return the currently selected peak indices."""

        return np.asarray(
            self._detected_peak_indices[self._selected_peak_mask], dtype=int
        )

    @property
    def peak_times(self) -> np.ndarray:
        """Return the currently selected peak positions in the analyzer's active unit."""

        return np.asarray(self.time[self.peak_indices], dtype=float)

    @property
    def peak_amplitudes(self) -> np.ndarray:
        """Return the signal values at the currently selected peaks."""

        return np.asarray(self.signal[self.peak_indices], dtype=float)

    @property
    def peak_widths_samples(self) -> np.ndarray:
        """Return the widths of the currently selected peaks in samples."""

        return np.asarray(
            self._detected_peak_widths_samples[self._selected_peak_mask], dtype=float
        )

    @property
    def peak_widths_time(self) -> np.ndarray:
        """Return the widths of the currently selected peaks in the analyzer's active unit."""

        return np.asarray(self.peak_widths_samples * self.dx, dtype=float)

    @property
    def peak_widths(self) -> np.ndarray:
        """Return the widths of the currently selected peaks in the analyzer's active unit."""

        return self.peak_widths_time

    @property
    def pulse_windows(self) -> np.ndarray:
        """Return the extracted baseline-corrected pulse windows."""

        return np.asarray(self._pulse_windows, dtype=float)

    @property
    def local_time(self) -> np.ndarray:
        """Return the local time axis shared by the extracted pulse windows."""

        return np.asarray(self._local_time, dtype=float)

    @property
    def window_peak_indices(self) -> np.ndarray:
        """Return the peak indices corresponding to the extracted windows."""

        return np.asarray(self._window_peak_indices, dtype=int)

    def detect_peaks(
        self,
        height: Union[float, str],
        *,
        rel_height: float = 0.15,
        **find_peaks_kwargs: Any,
    ) -> "PulseShapeAnalyzer":
        """Detect peaks on the current signal using the same convention as ``CsvTrace``.

        The ``height`` argument is interpreted as a prominence threshold. Strings of
        the form ``"<n>sigma"`` use the repository's robust noise estimator.
        """

        prominence = self._resolve_threshold(height)
        peak_indices, _ = find_peaks(
            self.signal, prominence=prominence, **find_peaks_kwargs
        )
        widths_samples, _, _, _ = peak_widths(
            self.signal, peak_indices, rel_height=float(rel_height)
        )

        self._detected_peak_indices = np.asarray(peak_indices, dtype=int)
        self._detected_peak_widths_samples = np.asarray(widths_samples, dtype=float)
        self._selected_peak_mask = np.ones(self._detected_peak_indices.size, dtype=bool)
        self._reset_windows()
        return self

    def reset_peak_selection(self) -> "PulseShapeAnalyzer":
        """Reset peak filtering so all detected peaks are selected again."""

        if self._detected_peak_indices.size == 0:
            self._selected_peak_mask = np.asarray([], dtype=bool)
        else:
            self._selected_peak_mask = np.ones(
                self._detected_peak_indices.size, dtype=bool
            )
        self._reset_windows()
        return self

    def select_peaks(
        self,
        *,
        min_width_samples: Optional[float] = None,
        max_width_samples: Optional[float] = None,
        min_amplitude: Optional[float] = None,
        max_amplitude: Optional[float] = None,
    ) -> "PulseShapeAnalyzer":
        """Filter the detected peaks by width and/or amplitude."""

        self._require_detected_peaks()

        peak_indices = self._detected_peak_indices
        widths = self._detected_peak_widths_samples
        amplitudes = self.signal[peak_indices]
        mask = np.asarray(self._selected_peak_mask, dtype=bool).copy()

        if min_width_samples is not None:
            mask &= widths >= float(min_width_samples)
        if max_width_samples is not None:
            mask &= widths <= float(max_width_samples)
        if min_amplitude is not None:
            mask &= amplitudes >= float(min_amplitude)
        if max_amplitude is not None:
            mask &= amplitudes <= float(max_amplitude)

        self._selected_peak_mask = mask
        self._reset_windows()
        return self

    def extract_windows(
        self,
        *,
        left_samples: int = 80,
        right_samples: int = 120,
        baseline_samples: Optional[int] = None,
    ) -> "PulseShapeAnalyzer":
        """Extract baseline-corrected windows centered around the selected peaks."""

        self._require_detected_peaks()
        left_samples = int(left_samples)
        right_samples = int(right_samples)
        if left_samples < 0 or right_samples < 0:
            raise ValueError(
                "left_samples and right_samples must be non-negative integers."
            )

        if baseline_samples is None:
            baseline_samples = max(1, left_samples)
        baseline_samples = int(baseline_samples)
        if baseline_samples <= 0:
            raise ValueError("baseline_samples must be a strictly positive integer.")

        selected_indices = self.peak_indices
        selected_widths = self.peak_widths_samples
        window_size = left_samples + right_samples + 1

        windows = []
        window_peak_indices = []
        window_peak_widths = []
        baselines = []

        for peak_index, peak_width in zip(selected_indices, selected_widths):
            start = int(peak_index) - left_samples
            end = int(peak_index) + right_samples + 1
            if start < 0 or end > self.signal.size:
                continue

            window = np.asarray(self.signal[start:end], dtype=float).copy()
            if window.size != window_size:
                continue

            baseline_end = min(left_samples, window.size)
            baseline_slice = window[:baseline_end]
            if baseline_slice.size == 0:
                baseline_slice = window[: min(baseline_samples, window.size)]

            if baseline_slice.size == 0:
                baseline = 0.0
            else:
                baseline = float(
                    np.median(
                        baseline_slice[: min(baseline_samples, baseline_slice.size)]
                    )
                )

            windows.append(window - baseline)
            window_peak_indices.append(int(peak_index))
            window_peak_widths.append(float(peak_width))
            baselines.append(baseline)

        self._pulse_windows = (
            np.asarray(windows, dtype=float)
            if windows
            else np.empty((0, window_size), dtype=float)
        )
        self._window_peak_indices = np.asarray(window_peak_indices, dtype=int)
        self._window_peak_widths_samples = np.asarray(window_peak_widths, dtype=float)
        self._window_baselines = np.asarray(baselines, dtype=float)
        self._local_time = (
            np.arange(window_size, dtype=float) - float(left_samples)
        ) * self.dx
        self._shape_statistics_cache = None
        return self

    def normalized_windows(self) -> np.ndarray:
        """Return extracted windows normalized to unit peak height."""

        self._require_extracted_windows()
        peak_heights = np.max(self._pulse_windows, axis=1, keepdims=True)
        return self._pulse_windows / np.clip(peak_heights, 1e-12, None)

    def mean_pulse(self, *, normalize: bool = True) -> np.ndarray:
        """Return the mean extracted pulse."""

        windows = self.normalized_windows() if normalize else self.pulse_windows
        if windows.size == 0:
            return np.asarray([], dtype=float)
        return np.mean(windows, axis=0)

    def median_pulse(self, *, normalize: bool = True) -> np.ndarray:
        """Return the median extracted pulse."""

        windows = self.normalized_windows() if normalize else self.pulse_windows
        if windows.size == 0:
            return np.asarray([], dtype=float)
        return np.median(windows, axis=0)

    def shape_statistics(self) -> pd.DataFrame:
        """Return one row of pulse-shape metrics per extracted window."""

        self._require_extracted_windows()
        if self._shape_statistics_cache is not None:
            return self._shape_statistics_cache.copy()

        rows = []
        for row_index, window in enumerate(self._pulse_windows):
            statistics = self._shape_statistics_for_window(window, self._local_time)
            if statistics is None:
                continue

            peak_index = int(self._window_peak_indices[row_index])
            statistics.update(
                {
                    "peak_index": peak_index,
                    "peak_time": float(self.time[peak_index]),
                    "peak_value": float(self.signal[peak_index]),
                    "peak_width_samples": float(
                        self._window_peak_widths_samples[row_index]
                    ),
                    "peak_width_time": float(
                        self._window_peak_widths_samples[row_index] * self.dx
                    ),
                    "peak_width": float(
                        self._window_peak_widths_samples[row_index] * self.dx
                    ),
                    "baseline": float(self._window_baselines[row_index]),
                }
            )
            rows.append(statistics)

        self._shape_statistics_cache = pd.DataFrame(rows)
        return self._shape_statistics_cache.copy()

    def shape_summary(
        self,
        *,
        percentiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        """Return a compact descriptive summary of the per-pulse shape statistics."""

        statistics = self.shape_statistics()
        if statistics.empty:
            return pd.DataFrame()
        return statistics.describe(percentiles=list(percentiles)).T

    @helper.post_mpl_plot
    def plot_pulses(
        self,
        *,
        normalize: bool = False,
        max_pulses: int = 80,
        show_mean: bool = False,
        show_gaussian_reference: bool = False,
    ) -> plt.Figure:
        """Overlay individual pulse windows and optional aggregate references."""

        self._require_extracted_windows()
        pulse_windows = self.normalized_windows() if normalize else self.pulse_windows
        figure, ax = plt.subplots(1, 1, figsize=(9, 4.8))

        for window in pulse_windows[: int(max_pulses)]:
            ax.plot(self._local_time, window, color="C0", alpha=0.14, lw=1.0)

        median_pulse = np.median(pulse_windows, axis=0)
        ax.plot(
            self._local_time, median_pulse, color="black", lw=2.2, label="median pulse"
        )

        if show_mean:
            ax.plot(
                self._local_time,
                np.mean(pulse_windows, axis=0),
                color="C3",
                lw=2.0,
                label="mean pulse",
            )

        if show_gaussian_reference:
            reference_statistics = self._shape_statistics_for_window(
                median_pulse, self._local_time
            )
            if reference_statistics is not None:
                gaussian = _gaussian_reference(
                    self._local_time,
                    centroid=float(reference_statistics["centroid"]),
                    sigma=float(reference_statistics["sigma_moment"]),
                )
                if np.all(np.isfinite(gaussian)):
                    ax.plot(
                        self._local_time,
                        gaussian,
                        color="C2",
                        ls="--",
                        lw=2.0,
                        label="Gaussian reference",
                    )

        ax.axvline(0.0, color="0.3", ls=":", lw=1.0)
        ax.set_title("Extracted pulse shapes")
        ax.set_xlabel(f"Time relative to peak [{self.unit_label}]")
        ax.set_ylabel(
            "Normalized amplitude" if normalize else "Amplitude above local baseline"
        )
        ax.legend(loc="best")
        return figure

    @helper.post_mpl_plot
    def plot_amplitude_vs_width(self) -> plt.Figure:
        """Plot pulse height above baseline against measured peak width."""

        statistics = self.shape_statistics()
        figure, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
        if not statistics.empty:
            ax.scatter(
                statistics["peak_width"],
                statistics["peak_height"],
                alpha=0.8,
                s=24,
            )
        ax.set_title("Pulse height versus width")
        ax.set_xlabel(f"Width [{self.unit_label}]")
        ax.set_ylabel("Peak height above local baseline")
        return figure

    def extract_kernel(
        self,
        *,
        aggregation: str = "median",
        noise_floor: float = 0.01,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Extract a normalized kernel shape from the extracted pulse windows.

        The kernel is the median (or mean) normalized pulse, trimmed so that
        samples below ``noise_floor`` on both sides are removed. The result is
        normalized to unit peak height and is ready to pass directly into
        :class:`~DeepPeak.generation.CustomKernel`.

        Parameters
        ----------
        aggregation : {"median", "mean"}, default "median"
            How to combine the normalized windows into a single representative
            pulse. ``"median"`` is more robust to outliers.
        noise_floor : float, default 0.01
            Fraction of the peak height below which leading and trailing samples
            are trimmed. A value of ``0.01`` keeps everything above 1% of the
            peak. Set to ``0.0`` to keep the full window.
        save_path : str or None, optional
            If provided, save the kernel array to this path with
            ``np.save(save_path, kernel)``.

        Returns
        -------
        numpy.ndarray
            One-dimensional kernel array normalized to unit peak height, ready
        for use with :class:`~DeepPeak.generation.CustomKernel`.
        """
        self._require_extracted_windows()

        aggregation = str(aggregation).strip().lower()
        if aggregation == "median":
            kernel = self.median_pulse(normalize=True)
        elif aggregation == "mean":
            kernel = self.mean_pulse(normalize=True)
        else:
            raise ValueError(
                f"aggregation must be 'median' or 'mean', got {aggregation!r}."
            )

        if kernel.size == 0:
            raise RuntimeError("No pulse windows available after aggregation.")

        kernel = np.asarray(kernel, dtype=float)
        kernel_max = float(np.max(kernel))
        if not np.isfinite(kernel_max) or kernel_max <= 0.0:
            raise RuntimeError("Aggregated kernel has no positive peak.")

        kernel = kernel / kernel_max

        if float(noise_floor) > 0.0:
            above = np.where(kernel >= float(noise_floor))[0]
            if above.size > 0:
                kernel = kernel[int(above[0]) : int(above[-1]) + 1]

        if save_path is not None:
            np.save(str(save_path), kernel)

        return kernel

    def extract_kernel_library(
        self,
        *,
        smooth_sigma: float = 2.0,
        baseline_samples: int = 20,
        max_kernels: int = 10,
        reject_saturated: bool = True,
        random_selection: bool = True,
        recenter: Optional[str] = None,
        taper_fraction: float = 0.0,
        save_path: Optional[str] = None,
        plot: bool = False,
    ) -> np.ndarray:
        """Extract a library of individual kernels from the pulse windows.

        Each window is individually smoothed, baseline-corrected, clipped to
        non-negative values, and normalized to unit height.  Optionally
        saturated (flat-top) pulses are rejected.  Up to ``max_kernels`` are
        returned, sampled randomly (or by even spacing) from the accepted set.

        The result is a 2-D array of shape ``(N, K)`` suitable for
        :class:`~DeepPeak.generation.CustomKernels`.

        Parameters
        ----------
        smooth_sigma : float, default 2.0
            Standard deviation of the Gaussian smoothing filter (in samples)
            applied to each window before baseline correction.
        baseline_samples : int, default 20
            Number of leading samples used to estimate the per-window baseline
            via a median.
        max_kernels : int, default 10
            Maximum number of kernels to return.
        reject_saturated : bool, default True
            If ``True``, pulses whose flat-top plateau (samples > 95 % of peak)
            is wider than half the FWHM are discarded as likely saturated.
        random_selection : bool, default True
            If ``True``, select ``max_kernels`` from the accepted pool randomly
            (without replacement).  If ``False``, select evenly spaced indices.
        recenter : {None, "none", "max", "centroid"}, optional
            Optional alignment applied to each accepted kernel before tapering
            and export. ``"max"`` shifts the kernel so its maximum lands at the
            central sample. ``"centroid"`` aligns the center of mass of the
            positive kernel values instead. The default preserves the original
            extracted window alignment.
        taper_fraction : float, default 0.0
            Fraction of the kernel length to apply a half-cosine taper on each
            end so that the kernel decays smoothly to zero at both boundaries.
            ``0.0`` disables tapering; ``0.1`` tapers the first and last 10 % of
            samples. Useful when the extracted window doesn't reach zero
            naturally at its edges.
        save_path : str or None, optional
            If provided, save the kernel library array with
            ``np.save(save_path, library)``.
        plot : bool, default False
            If ``True``, plot all extracted kernels in a grid.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N, K)`` where N ≤ ``max_kernels``.

        Raises
        ------
        RuntimeError
            If no pulse windows have been extracted yet, or if no kernels pass
            the quality filters.
        """
        self._require_extracted_windows()

        if recenter is not None and str(recenter).strip().lower() not in {
            "none",
            "original",
            "max",
            "peak",
            "centroid",
            "com",
            "center_of_mass",
        }:
            raise ValueError(
                "recenter must be one of None, 'none', 'max', or 'centroid'. "
                f"Got {recenter!r}."
            )
        smooth_sigma = float(smooth_sigma)
        if smooth_sigma < 0.0:
            raise ValueError("smooth_sigma must be non-negative.")

        accepted = []
        for window in self._pulse_windows:
            k = (
                window.astype(float).copy()
                if smooth_sigma == 0.0
                else gaussian_filter1d(window.astype(float), sigma=smooth_sigma)
            )
            baseline = float(np.median(k[: int(baseline_samples)]))
            k = k - baseline
            k = np.clip(k, 0.0, None)
            peak_val = float(k.max())
            if peak_val <= 0.0:
                continue
            k /= peak_val

            if reject_saturated:
                fwhm_approx = int(np.sum(k > 0.5))
                plateau_width = int(np.sum(k > 0.95))
                if fwhm_approx > 0 and plateau_width > 0.5 * fwhm_approx:
                    continue

            k = _recenter_kernel(k, recenter)

            if float(taper_fraction) > 0.0:
                k = _apply_cosine_taper(k, float(taper_fraction))

            accepted.append(k)

        if not accepted:
            raise RuntimeError(
                "No kernels passed quality filters. "
                "Try lowering smooth_sigma, disabling reject_saturated, or "
                "checking that extract_windows() captured the full pulse shape."
            )

        library = np.array(accepted, dtype=float)

        if len(library) > int(max_kernels):
            n = int(max_kernels)
            if random_selection:
                idx = np.random.choice(len(library), size=n, replace=False)
            else:
                idx = np.linspace(0, len(library) - 1, n, dtype=int)
            library = library[idx]

        if save_path is not None:
            np.save(str(save_path), library)

        if plot:
            n_kernels = len(library)
            n_cols = min(5, n_kernels)
            n_rows = int(np.ceil(n_kernels / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True
            )
            axes_flat = np.array(axes).ravel()
            for i, (ax, k) in enumerate(zip(axes_flat, library)):
                ax.plot(k)
                ax.set_title(f"kernel {i}")
                ax.axhline(0, color="k", lw=0.5, ls="--")
            for ax in axes_flat[n_kernels:]:
                ax.set_visible(False)
            plt.tight_layout()
            plt.show()

        return library

    def _resolve_threshold(self, height: Union[float, str]) -> float:
        """Resolve an absolute or sigma-scaled peak threshold."""

        if isinstance(height, str):
            normalized = height.strip().lower().replace(" ", "")
            if not normalized.endswith("sigma"):
                raise ValueError(
                    "String thresholds must be of the form '<number>sigma', for example '5sigma'."
                )
            sigma_multiplier = float(normalized[:-5])
            return float(sigma_multiplier) * utils.robust_sigma_from_diff(self.signal)
        return float(height)

    def _reset_windows(self) -> None:
        """Clear cached window and statistics state after peak changes."""

        self._window_peak_indices = np.asarray([], dtype=int)
        self._window_peak_widths_samples = np.asarray([], dtype=float)
        self._window_baselines = np.asarray([], dtype=float)
        self._pulse_windows = np.empty((0, 0), dtype=float)
        self._local_time = np.asarray([], dtype=float)
        self._shape_statistics_cache = None

    def _require_detected_peaks(self) -> None:
        """Ensure a peak set exists before selection or extraction."""

        if self._detected_peak_indices.size == 0:
            raise RuntimeError("No peaks are available. Call detect_peaks(...) first.")

    def _require_extracted_windows(self) -> None:
        """Ensure pulse windows were extracted before downstream analysis."""

        if self._pulse_windows.size == 0:
            raise RuntimeError(
                "No pulse windows are available. Call extract_windows(...) after selecting peaks."
            )

    @staticmethod
    def _shape_statistics_for_window(
        window: np.ndarray, local_time: np.ndarray
    ) -> Optional[dict[str, float]]:
        """Compute shape metrics for one baseline-corrected pulse window."""

        values = np.asarray(window, dtype=float)
        time = np.asarray(local_time, dtype=float)

        peak_height = float(np.max(values))
        if not np.isfinite(peak_height) or peak_height <= 0.0:
            return None

        normalized = values / peak_height
        normalized_positive = np.clip(normalized, 0.0, None)

        area = _trapezoid(normalized_positive, time)
        if area <= 0.0:
            return None

        peak_index = int(np.argmax(normalized))
        peak_time = float(time[peak_index])
        centroid = _trapezoid(time * normalized_positive, time) / area
        variance = (
            _trapezoid(((time - centroid) ** 2) * normalized_positive, time) / area
        )
        sigma_moment = float(np.sqrt(max(variance, 0.0)))

        t10_left = _crossing_time(time, normalized, 0.10, side="left")
        t50_left = _crossing_time(time, normalized, 0.50, side="left")
        t90_left = _crossing_time(time, normalized, 0.90, side="left")
        t90_right = _crossing_time(time, normalized, 0.90, side="right")
        t50_right = _crossing_time(time, normalized, 0.50, side="right")
        t10_right = _crossing_time(time, normalized, 0.10, side="right")

        rise_10_90 = (
            float(t90_left - t10_left)
            if np.isfinite(t10_left) and np.isfinite(t90_left)
            else float("nan")
        )
        decay_90_10 = (
            float(t10_right - t90_right)
            if np.isfinite(t10_right) and np.isfinite(t90_right)
            else float("nan")
        )
        fwhm = (
            float(t50_right - t50_left)
            if np.isfinite(t50_left) and np.isfinite(t50_right)
            else float("nan")
        )
        width_10pct = (
            float(t10_right - t10_left)
            if np.isfinite(t10_left) and np.isfinite(t10_right)
            else float("nan")
        )

        left_mask = time <= 0.0
        right_mask = time >= 0.0
        left_area = (
            _trapezoid(normalized_positive[left_mask], time[left_mask])
            if np.any(left_mask)
            else float("nan")
        )
        right_area = (
            _trapezoid(normalized_positive[right_mask], time[right_mask])
            if np.any(right_mask)
            else float("nan")
        )

        asymmetry_ratio = (
            float(decay_90_10 / rise_10_90)
            if np.isfinite(rise_10_90) and rise_10_90 > 0.0 and np.isfinite(decay_90_10)
            else float("nan")
        )
        area_ratio_right_left = (
            float(right_area / left_area)
            if np.isfinite(left_area) and left_area > 0.0 and np.isfinite(right_area)
            else float("nan")
        )
        tail_ratio = (
            float(width_10pct / fwhm)
            if np.isfinite(fwhm) and fwhm > 0.0 and np.isfinite(width_10pct)
            else float("nan")
        )

        gaussian = _gaussian_reference(time, centroid=centroid, sigma=sigma_moment)
        gaussian_rmse = (
            float(np.sqrt(np.mean((normalized - gaussian) ** 2)))
            if np.all(np.isfinite(gaussian))
            else float("nan")
        )

        return {
            "peak_height": peak_height,
            "peak_time_local": peak_time,
            "centroid": float(centroid),
            "centroid_shift": float(centroid - peak_time),
            "sigma_moment": sigma_moment,
            "fwhm": fwhm,
            "width_10pct": width_10pct,
            "rise_10_90": rise_10_90,
            "decay_90_10": decay_90_10,
            "asymmetry_ratio": asymmetry_ratio,
            "left_area": float(left_area),
            "right_area": float(right_area),
            "area_ratio_right_left": area_ratio_right_left,
            "tail_ratio": tail_ratio,
            "gaussian_rmse": gaussian_rmse,
        }
