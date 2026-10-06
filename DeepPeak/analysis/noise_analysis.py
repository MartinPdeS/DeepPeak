from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MPSPlots import helper
from scipy import stats
from scipy.signal import find_peaks, welch

from DeepPeak import utils

ArrayLike = Union[np.ndarray, Sequence[float]]


class NoiseAnalyzer:
    """Analyze noise-only regions in traces."""

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
            raise ValueError("signal must be one dimensional.")
        if self.signal.size == 0:
            raise ValueError("signal must contain at least one sample.")

        self.sampling_rate = None if sampling_rate is None else float(sampling_rate)
        if self.sampling_rate is not None and self.sampling_rate <= 0.0:
            raise ValueError("sampling_rate must be positive when provided.")

        if time is not None:
            self.time = np.asarray(time, dtype=float).ravel()
            if self.time.shape != self.signal.shape:
                raise ValueError("time must have the same shape as signal.")
        else:
            if self.sampling_rate is not None:
                inferred_dx = 1.0 / self.sampling_rate
            elif dx is not None:
                inferred_dx = float(dx)
            else:
                inferred_dx = 1.0
            self.time = np.arange(self.signal.size, dtype=float) * inferred_dx

        if dx is not None:
            self.dx = float(dx)
        elif self.sampling_rate is not None:
            self.dx = 1.0 / self.sampling_rate
        elif self.time.size > 1:
            self.dx = float(np.median(np.diff(self.time)))
        else:
            self.dx = 1.0

        self.unit_label = "s" if self.sampling_rate is not None else "sample"

        self._peak_indices = np.asarray([], dtype=int)
        self._noise_mask = np.asarray([], dtype=bool)
        self._detection_threshold: Optional[float] = None
        self._detection_threshold_label: Optional[str] = None
        self._noise_statistics_cache: Optional[pd.Series] = None
        self._autocorrelation_cache: dict[int, pd.DataFrame] = {}
        self._psd_cache: dict[Optional[int], pd.DataFrame] = {}

    @classmethod
    def from_trace(
        cls,
        trace: Any,
        use_processed: bool = True,
        sampling_rate: Optional[float] = None,
    ) -> "NoiseAnalyzer":
        """Build a noise analyzer from a ``CsvTrace``-like object."""
        signal_name = "y_processed" if use_processed else "y_raw"
        signal = getattr(trace, signal_name)
        time = getattr(trace, "x", None)
        dx = getattr(trace, "dx", None)
        return cls(signal=signal, sampling_rate=sampling_rate, time=time, dx=dx)

    @property
    def noise_mask(self) -> np.ndarray:
        self._require_noise_detected()
        return self._noise_mask.copy()

    @property
    def noise_samples(self) -> np.ndarray:
        self._require_noise_detected()
        return self.signal[self._noise_mask].copy()

    @property
    def noise_time(self) -> np.ndarray:
        self._require_noise_detected()
        return self.time[self._noise_mask].copy()

    def detect_noise(
        self,
        height: Union[float, str] = "15sigma",
        *,
        distance: int = 10,
        left_guard: int = 150,
        right_guard: int = 200,
        use_peaks: Optional[ArrayLike] = None,
        **find_peaks_kwargs,
    ) -> "NoiseAnalyzer":
        """Detect pulse-like peaks and keep samples outside guarded neighborhoods."""
        left_guard = int(left_guard)
        right_guard = int(right_guard)
        if left_guard < 0 or right_guard < 0:
            raise ValueError("left_guard and right_guard must be non-negative.")

        if use_peaks is None:
            threshold = self._resolve_threshold(height)
            peak_indices, _ = find_peaks(
                self.signal,
                height=threshold,
                distance=int(distance),
                **find_peaks_kwargs,
            )
            self._detection_threshold = float(threshold)
            self._detection_threshold_label = str(height)
        else:
            peak_indices = np.asarray(use_peaks, dtype=int).ravel()
            self._detection_threshold = None
            self._detection_threshold_label = None

        mask = np.ones(self.signal.size, dtype=bool)
        for peak_index in peak_indices:
            start = max(0, int(peak_index) - left_guard)
            end = min(self.signal.size, int(peak_index) + right_guard + 1)
            mask[start:end] = False

        self._peak_indices = np.asarray(peak_indices, dtype=int)
        self._noise_mask = mask
        self._clear_caches()
        return self

    def noise_statistics(self) -> pd.Series:
        """Return descriptive statistics for the retained noise samples."""
        self._require_noise_detected()
        if self._noise_statistics_cache is not None:
            return self._noise_statistics_cache.copy()

        noise = self.noise_samples
        result = pd.Series(
            {
                "count": int(noise.size),
                "mean": float(np.mean(noise)),
                "median": float(np.median(noise)),
                "std": float(np.std(noise, ddof=1)) if noise.size > 1 else 0.0,
                "robust_sigma": float(utils.robust_sigma_from_diff(noise)),
                "skewness": (
                    float(stats.skew(noise, bias=False)) if noise.size > 2 else np.nan
                ),
                "kurtosis": (
                    float(stats.kurtosis(noise, fisher=True, bias=False))
                    if noise.size > 3
                    else np.nan
                ),
                "min": float(np.min(noise)),
                "max": float(np.max(noise)),
            },
            dtype=float,
        )
        self._noise_statistics_cache = result
        return result.copy()

    def autocorrelation(self, max_lag: int = 200) -> pd.DataFrame:
        """Return the normalized autocorrelation of the retained noise."""
        self._require_noise_detected()
        max_lag = int(max_lag)
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        if max_lag in self._autocorrelation_cache:
            return self._autocorrelation_cache[max_lag].copy()

        noise = self.noise_samples
        centered = noise - float(np.mean(noise))
        denominator = float(np.dot(centered, centered))
        autocorrelation = np.empty(max_lag + 1, dtype=float)

        if denominator <= 0.0:
            autocorrelation.fill(np.nan)
            autocorrelation[0] = 1.0
        else:
            autocorrelation[0] = 1.0
            for lag in range(1, max_lag + 1):
                if lag >= centered.size:
                    autocorrelation[lag] = np.nan
                else:
                    autocorrelation[lag] = float(
                        np.dot(centered[:-lag], centered[lag:]) / denominator
                    )

        lags = np.arange(max_lag + 1, dtype=int)
        frame = pd.DataFrame(
            {
                "lag_samples": lags,
                "lag": lags.astype(float) * self.dx,
                "autocorrelation": autocorrelation,
            }
        )
        self._autocorrelation_cache[max_lag] = frame
        return frame.copy()

    def power_spectral_density(self, nperseg: Optional[int] = None) -> pd.DataFrame:
        """Estimate the retained-noise PSD with Welch's method."""
        self._require_noise_detected()
        key = None if nperseg is None else int(nperseg)
        if key in self._psd_cache:
            return self._psd_cache[key].copy()

        noise = self.noise_samples
        nperseg = min(noise.size, 256 if nperseg is None else int(nperseg))
        if nperseg < 2:
            raise RuntimeError(
                "Need at least two retained noise samples for a power spectrum."
            )

        sampling_rate = self.sampling_rate if self.sampling_rate is not None else 1.0
        frequency, power = welch(noise, fs=sampling_rate, nperseg=nperseg)
        frame = pd.DataFrame({"frequency": frequency, "power": power})
        self._psd_cache[key] = frame
        return frame.copy()

    def summary(self, percentiles: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        """Return a compact summary table for the retained noise."""
        self._require_noise_detected()
        summary = (
            pd.DataFrame({"noise": self.noise_samples})
            .describe(percentiles=list(percentiles))
            .T
        )
        statistics = self.noise_statistics()
        summary["robust_sigma"] = statistics["robust_sigma"]
        summary["skewness"] = statistics["skewness"]
        summary["kurtosis"] = statistics["kurtosis"]
        return summary

    @helper.post_mpl_plot
    def plot_trace(self, *, show_threshold: bool = True) -> plt.Figure:
        """Plot the trace with kept noise regions in green and masked regions in red."""
        self._require_noise_detected()

        figure, ax = plt.subplots(figsize=(10, 4.5))
        baseline = float(np.min(self.signal))
        ax.plot(self.time, self.signal, color="black", lw=1.0, alpha=0.8, label="trace")
        ax.fill_between(
            self.time,
            baseline,
            self.signal,
            where=self._noise_mask,
            color="green",
            alpha=0.25,
            interpolate=True,
            label="kept noise",
        )
        ax.fill_between(
            self.time,
            baseline,
            self.signal,
            where=~self._noise_mask,
            color="red",
            alpha=0.25,
            interpolate=True,
            label="masked",
        )
        if show_threshold and self._detection_threshold is not None:
            label = "threshold"
            if self._detection_threshold_label is not None:
                label = f"threshold ({self._detection_threshold_label})"
            ax.axhline(
                self._detection_threshold,
                color="C3",
                lw=1.5,
                ls="--",
                label=label,
            )
        ax.set_title("Trace with retained and masked regions")
        ax.set_xlabel(f"Time [{self.unit_label}]")
        ax.set_ylabel("Amplitude")
        ax.legend()
        figure.tight_layout()
        return figure

    @helper.post_mpl_plot
    def plot_distribution(
        self,
        *,
        bins: int = 150,
        show_gaussian_reference: bool = True,
    ) -> plt.Figure:
        """Plot the retained-noise histogram."""
        self._require_noise_detected()

        noise = self.noise_samples
        figure, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(noise, bins=int(bins), density=True, alpha=0.7, color="C0")

        if show_gaussian_reference and noise.size > 1:
            mean = float(np.mean(noise))
            std = float(np.std(noise, ddof=1))
            if std > 0.0:
                x_values = np.linspace(mean - 5.0 * std, mean + 5.0 * std, 400)
                ax.plot(
                    x_values,
                    stats.norm.pdf(x_values, loc=mean, scale=std),
                    color="black",
                    lw=2,
                    label="matched Gaussian",
                )
                ax.legend()

        ax.set_title("Noise distribution")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Probability density")
        figure.tight_layout()
        return figure

    @helper.post_mpl_plot
    def plot_autocorrelation(self, *, max_lag: int = 200) -> plt.Figure:
        """Plot the retained-noise autocorrelation."""
        frame = self.autocorrelation(max_lag=max_lag)

        figure, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(frame["lag"], frame["autocorrelation"], color="C1", lw=1.5)
        ax.axhline(0.0, color="black", ls="--", lw=1.0)
        ax.set_title("Noise autocorrelation")
        ax.set_xlabel(f"Lag [{self.unit_label}]")
        ax.set_ylabel("Autocorrelation")
        figure.tight_layout()
        return figure

    @helper.post_mpl_plot
    def plot_power_spectral_density(
        self,
        *,
        nperseg: Optional[int] = None,
        loglog: bool = True,
    ) -> plt.Figure:
        """Plot the retained-noise power spectral density."""
        frame = self.power_spectral_density(nperseg=nperseg)

        figure, ax = plt.subplots(figsize=(8, 4.5))
        plot = ax.loglog if loglog else ax.plot
        if len(frame) > 1:
            plot(
                frame["frequency"].iloc[1:], frame["power"].iloc[1:], color="C2", lw=1.5
            )
        else:
            plot(frame["frequency"], frame["power"], color="C2", lw=1.5)

        xlabel = (
            "Frequency [Hz]"
            if self.sampling_rate is not None
            else "Frequency [cycles/sample]"
        )
        ax.set_title("Noise power spectral density")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power")
        figure.tight_layout()
        return figure

    def _resolve_threshold(self, height: Union[float, str]) -> float:
        if isinstance(height, str):
            normalized = height.strip().lower().replace(" ", "")
            if not normalized.endswith("sigma"):
                raise ValueError(
                    "String thresholds must be of the form '<number>sigma', for example '5sigma'."
                )
            sigma_multiplier = float(normalized[:-5])
            return float(sigma_multiplier) * utils.robust_sigma_from_diff(self.signal)
        return float(height)

    def _clear_caches(self) -> None:
        self._noise_statistics_cache = None
        self._autocorrelation_cache = {}
        self._psd_cache = {}

    def _require_noise_detected(self) -> None:
        if self._noise_mask.size == 0:
            raise RuntimeError(
                "Noise regions have not been detected yet. Call detect_noise(...) first."
            )
        if not np.any(self._noise_mask):
            raise RuntimeError(
                "No retained noise samples remain after masking. Reduce the guard region or threshold."
            )
