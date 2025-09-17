import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from MPSPlots import helper
from dataclasses import dataclass


@dataclass
class SingleResult:
    """
    Container for single-signal NMS results with analysis & plotting utilities.
    Returned by `NonMaximumSuppression.run(...)`.

    Parameters
    ----------
    detector : NonMaximumSuppression
        The detector instance used to generate these results.
    signal : ndarray, shape (N,)
        Input signal.
    time_samples : ndarray, shape (N,)
        Uniform sample times.
    matched_filter_output : ndarray, shape (N,)
        Matched filter output.
    gaussian_kernel : ndarray, shape (L,)
        The Gaussian kernel used for matched filtering.
    threshold_used : float
        The threshold value used for peak detection.
    suppression_half_window_in_samples : int
        The half-width of the suppression window (in samples).
    peak_indices : ndarray, shape (K,)
        Indices of the detected peaks.
    peak_times : ndarray, shape (K,)
        Times of the detected peaks.
    peak_amplitude_raw : ndarray, shape (K,)
        Signal values at the detected peak indices.
    peak_amplitude_matched : ndarray, shape (K,)
        Matched-filter output values at the detected peak indices.
    """

    detector: "NonMaximumSuppression"
    signal: NDArray[np.float64]
    time_samples: NDArray[np.float64]
    matched_filter_output: NDArray[np.float64]
    gaussian_kernel: NDArray[np.float64]
    threshold_used: float
    suppression_half_window_in_samples: int
    peak_indices: NDArray[np.int_]
    peak_times: NDArray[np.float64]
    peak_amplitude_raw: NDArray[np.float64]
    peak_amplitude_matched: NDArray[np.float64]

    # -------- analysis helpers --------
    @property
    def sequence_length(self) -> int:
        return int(self.signal.size)

    @property
    def number_of_peaks(self) -> int:
        return int(self.peak_indices.size)

    def summary(self) -> dict:
        """Quick diagnostic summary for this sample."""
        return {
            "N": self.sequence_length,
            "K_detected": self.number_of_peaks,
            "threshold_used": float(self.threshold_used),
            "win_samples": int(self.suppression_half_window_in_samples),
            "sigma": float(self.detector.gaussian_sigma),
            "max_matched": float(np.max(self.matched_filter_output)) if self.sequence_length else np.nan,
        }

    def to_dict(self) -> dict[str, object]:
        """Export as a plain dict (compatible with previous API)."""
        return {
            "signal": self.signal,
            "time_samples": self.time_samples,
            "matched_filter_output": self.matched_filter_output,
            "gaussian_kernel": self.gaussian_kernel,
            "threshold_used": self.threshold_used,
            "suppression_half_window_in_samples": self.suppression_half_window_in_samples,
            "peak_indices": self.peak_indices,
            "peak_times": self.peak_times,
            "peak_amplitude": self.peak_amplitude_raw,  # raw signal at indices
            "peak_amplitude_matched": self.peak_amplitude_matched,
        }

    # -------- plotting --------
    @helper.post_mpl_plot
    def plot(self, *, show_matched_filter: bool = True, show_peaks: bool = True) -> plt.Figure:
        """Signal (+ optional MF) with vertical lines at detected peak times."""
        t = self.time_samples
        y = self.signal
        r = self.matched_filter_output
        peaks_t = self.peak_times

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
        ax.plot(t, y, label="signal")
        if show_matched_filter:
            ax.plot(t, r, label="matched filter")

        if show_peaks and peaks_t.size:
            for m in peaks_t:
                ax.axvline(m, linestyle="--", alpha=0.6)
            # single legend entry for peaks (avoid duplicates)
            peak_proxy = plt.Line2D([0], [0], linestyle="--", color="C2", alpha=0.6)
            handles, labels = ax.get_legend_handles_labels()
            handles.append(peak_proxy)
            labels.append("peaks")
            ax.legend(handles, labels, loc="best")
        else:
            ax.legend(loc="best")

        ax.set(title=f"Detected peaks: {self.number_of_peaks}", xlabel="t", ylabel="amplitude")
        return fig

    def _compute_average_pulse(self, *, source: str = "raw", baseline: str = "median") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute the average pulse centered on detected peaks and the per-sample std.

        Parameters
        ----------
        source : {"raw", "matched"}
            Use the raw input signal or the matched-filter output.
        baseline : {"median", "none"}
            Subtract a per-window baseline before averaging. "median" is robust.

        Returns
        -------
        t_rel : ndarray, shape (K,)
            Time axis relative to the peak center (seconds).
        avg : ndarray, shape (K,)
            Average pulse (after optional baseline removal).
        std : ndarray, shape (K,)
            Pointwise standard deviation across windows (after baseline removal).
        """
        if self.peak_indices.size == 0:
            raise ValueError("No peaks to average.")

        y = self.signal if source == "raw" else self.matched_filter_output
        k = self.gaussian_kernel
        if y is None or k is None:
            raise ValueError("Required arrays are missing on SingleResult.")

        N = y.size
        K = k.size
        half = (K - 1) // 2
        dt = float(self.time_samples[1] - self.time_samples[0])
        t_rel = np.arange(-half, half + 1, dtype=float) * dt

        # Collect windows fully contained in the signal
        windows = []
        for idx in self.peak_indices:
            i = int(idx)
            if i - half < 0 or i + half >= N:
                continue
            w = y[i - half : i + half + 1].astype(float, copy=True)
            if baseline == "median":
                w -= np.median(w)
            windows.append(w)

        if not windows:
            raise ValueError("No valid windows (peaks too close to edges for kernel length).")

        W = np.vstack(windows)  # (M, K)
        avg = W.mean(axis=0)
        std = W.std(axis=0, ddof=1) if W.shape[0] > 1 else np.zeros_like(avg)
        return t_rel, avg, std

    @staticmethod
    def _best_fit_scale(avg: np.ndarray, kernel: np.ndarray) -> float:
        r"""
        Least-squares scale factor α that minimizes ||avg - α k||_2^2.
        Kernel is unit-energy in this project, so α = <avg, k>.
        """
        denom = float(np.dot(kernel, kernel))
        return float(np.dot(avg, kernel) / denom) if denom > 0 else 1.0

    @staticmethod
    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        a0 = a - a.mean()
        b0 = b - b.mean()
        na = np.linalg.norm(a0)
        nb = np.linalg.norm(b0)
        return float(np.dot(a0, b0) / (na * nb)) if (na > 0 and nb > 0) else 0.0

    @staticmethod
    def _sigma_and_fwhm_from_profile(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        r"""
        Estimate σ and FWHM of a (roughly) Gaussian, from its averaged shape.

        - σ via second central moment with nonnegative weights.
        - FWHM via linear interpolation of half-maximum crossings.
        """
        w = y - np.min(y)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            return np.nan, np.nan
        w /= s
        mu = float((t * w).sum())
        sigma = float(np.sqrt(((t - mu) ** 2 * w).sum()))

        ymax = float(y.max())
        if ymax <= 0:
            return sigma, np.nan
        half = 0.5 * ymax

        i0 = int(np.argmax(y))

        i_left = None
        for i in range(i0, 0, -1):
            if y[i - 1] <= half <= y[i]:
                frac = (half - y[i - 1]) / (y[i] - y[i - 1] + 1e-12)
                t_left = t[i - 1] + frac * (t[i] - t[i - 1])
                i_left = t_left
                break

        i_right = None
        for i in range(i0, len(y) - 1):
            if y[i] >= half >= y[i + 1]:
                frac = (half - y[i + 1]) / (y[i] - y[i + 1] + 1e-12)
                t_right = t[i + 1] + frac * (t[i] - t[i + 1])
                i_right = t_right
                break

        fwhm = float(i_right - i_left) if (i_left is not None and i_right is not None) else np.nan
        return sigma, fwhm

    @helper.post_mpl_plot
    def plot_kernel_vs_average_pulse(self, source: str = "raw", baseline: str = "median", show_spread: bool = True, label_kernel: str = "Gaussian kernel (scaled)", label_avg: str = "Average pulse") -> dict[str, float]:
        r"""
        Plot the average detected pulse against the Gaussian kernel (best-fit scaled).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        source : {"raw", "matched"}
            Compare against the raw signal or matched-filter output.
        baseline : {"median", "none"}
            Per-window baseline removal before averaging.
        show_spread : bool
            If True, display ±1 std envelope around the average pulse.
        label_kernel : str
            Legend label for the scaled kernel.
        label_avg : str
            Legend label for the averaged pulse.

        Returns
        -------
        dict
            Dictionary of fit statistics: {"alpha", "rmse", "corr", "sigma_est", "fwhm_est"}.
        """
        t_rel, avg, std = self._compute_average_pulse(source=source, baseline=baseline)
        k = self.gaussian_kernel
        if k is None:
            raise ValueError("gaussian_kernel not present on SingleResult.")

        # Best-fit scaling of unit-energy kernel to average pulse
        alpha = self._best_fit_scale(avg, k)

        figure, ax = plt.subplots()

        ax.plot(t_rel, avg, lw=2, label=label_avg)
        if show_spread and np.any(std > 0):
            ax.fill_between(t_rel, avg - std, avg + std, alpha=0.2, linewidth=0)

        ax.plot(t_rel, alpha * k, lw=2, linestyle="--", label=label_kernel)

        ax.set_xlabel("Time relative to peak (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Kernel vs average pulse ({source})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return figure

    def print_kernel_vs_average_pulse_stats(self, *, source: str = "raw", baseline: str = "median") -> None:
        r"""
        Print a compact report comparing the averaged pulse to the Gaussian kernel.

        Parameters
        ----------
        source : {"raw", "matched"}
            Use raw signal or matched-filter output to build the average.
        baseline : {"median", "none"}
            Per-window baseline removal before averaging.
        """
        t_rel, avg, _ = self._compute_average_pulse(source=source, baseline=baseline)
        k = self.gaussian_kernel
        if k is None:
            raise ValueError("gaussian_kernel not present on SingleResult.")

        alpha = self._best_fit_scale(avg, k)
        rmse = float(np.sqrt(np.mean((avg - alpha * k) ** 2)))
        corr = self._pearson_corr(avg, k)
        sigma_est, fwhm_est = self._sigma_and_fwhm_from_profile(t_rel, avg)

        dt = float(self.time_samples[1] - self.time_samples[0])
        print("=== Kernel vs Average Pulse ===")
        print(f"Number of peaks used     : {np.isfinite(self.peak_times).sum()}")
        print(f"Kernel length (samples)  : {k.size}  | dt = {dt:.6g} s")
        print(f"Best-fit scale $\alpha$         : {alpha:.6g}")
        print(f"RMSE                     : {rmse:.6g}")
        print(f"Pearson corr (zero-mean) : {corr:.6g}")
        print(f"Estimated $\sigma$ (average)    : {sigma_est:.6g} s")
        if np.isfinite(fwhm_est):
            print(f"Estimated FWHM (average) : {fwhm_est:.6g} s  ($\sigma$≈FWHM/2√(2ln2) → {fwhm_est/(2*np.sqrt(2*np.log(2))):.6g} s)")
        else:
            print("Estimated FWHM (average) : NaN (could not determine half-maximum)")


@dataclass
class BatchResult:
    """
    Container for batched NMS results with analysis & plotting utilities.
    Returned by `NonMaximumSuppression.run_batch(...)`.

    Parameters
    ----------

    detector : NonMaximumSuppression
        The detector instance used to generate these results.
    signals : ndarray, shape (B, N)
        Input signals.
    time_samples : ndarray, shape (N,)
        Uniform sample times.
    matched_filter_output : ndarray, shape (B, N)
        Matched filter outputs.
    gaussian_kernel : ndarray, shape (L,)
        The Gaussian kernel used for matched filtering.
    threshold_used : ndarray, shape (B,)
        The thresholds used for peak detection.
    suppression_half_window_in_samples : int
        The half-window size used for non-maximum suppression.
    peak_indices : ndarray, shape (B, K)
        Detected peak indices (sample-aligned), -1 for missing peaks.
    peak_times : ndarray, shape (B, K)
        Detected peak times (sample-aligned), NaN for missing peaks.
    peak_amplitude_raw : ndarray, shape (B, K)
        Signal values at detected peak indices, NaN for missing peaks.
    """

    detector: "NonMaximumSuppression"
    signals: NDArray[np.float64]
    time_samples: NDArray[np.float64]
    matched_filter_output: NDArray[np.float64]
    gaussian_kernel: NDArray[np.float64]
    threshold_used: NDArray[np.float64]
    suppression_half_window_in_samples: int
    peak_indices: NDArray[np.int_]
    peak_times: NDArray[np.float64]
    peak_amplitude_raw: NDArray[np.float64]

    # ---------- analysis helpers ----------
    @property
    def batch_size(self) -> int:
        return int(self.signals.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.signals.shape[1])

    @property
    def number_of_peaks(self) -> int:
        return int(self.peak_times.shape[1])

    @property
    def peak_count(self) -> NDArray[np.int_]:
        """Number of detected peaks per sample, shape (B,)."""
        return np.sum(~np.isnan(self.peak_times), axis=1).astype(int)

    def summary(self) -> dict:
        """Quick diagnostic summary."""
        counts = self.peak_count
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "K": self.number_of_peaks,
            "mean_peaks": float(np.mean(counts)),
            "std_peaks": float(np.std(counts)),
            "min_peaks": int(np.min(counts)),
            "max_peaks": int(np.max(counts)),
            "threshold_min": float(np.min(self.threshold_used)),
            "threshold_max": float(np.max(self.threshold_used)),
            "win_samples": int(self.suppression_half_window_in_samples),
            "sigma": float(self.detector.gaussian_sigma),
        }

    def to_dict(self) -> dict[str, object]:
        """Export as a plain dict (e.g., for serialization)."""
        return {
            "signals": self.signals,
            "time_samples": self.time_samples,
            "matched_filter_output": self.matched_filter_output,
            "gaussian_kernel": self.gaussian_kernel,
            "threshold_used": self.threshold_used,
            "suppression_half_window_in_samples": self.suppression_half_window_in_samples,
            "peak_indices": self.peak_indices,
            "peak_times": self.peak_times,
            "peak_amplitude_raw": self.peak_amplitude_raw,
        }

    # ---------- plotting ----------
    @helper.post_mpl_plot
    def plot(
        self,
        indices: NDArray[np.int_] | None = None,
        *,
        ncols: int = 1,
        max_plots: int | None = 12,
        ground_truth: NDArray[np.float64] | None = None,
    ) -> plt.Figure:
        """
        Small multiples of several samples.

        Parameters
        ----------
        indices : array of int, optional
            Which samples to plot. If None, the first ``max_plots`` samples are used.
        ncols : int
            Number of columns for the plot grid.
        max_plots : int, optional
            Maximum number of samples to plot if ``indices`` is None. If None, plot all samples.
        ground_truth : ndarray, shape (B, K_true), optional
            If provided, vertical lines are drawn at the ground-truth peak times for each sample.
        """
        if indices is None:
            batch_selection = min(self.batch_size, max_plots or self.batch_size)
            batch_index = np.arange(batch_selection, dtype=int)
        else:
            batch_index = np.asarray(indices, dtype=int)

        n = batch_index.size

        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 2.8 * nrows), squeeze=False)
        axes_flat = axes.ravel()

        for k, (example_number, ax) in enumerate(zip(batch_index, axes_flat)):

            peaks_t = self.peak_times[example_number]

            ax.plot(self.time_samples, self.signals[example_number], label="signal")

            ax.plot(self.time_samples, self.matched_filter_output[example_number], label="matched filter")

            for peak_time in peaks_t[np.isfinite(peaks_t)]:
                ax.axvline(peak_time, linestyle="-", alpha=0.6)

            if ground_truth is not None:
                ground_truth_times = ground_truth[example_number]
                for idx, ground_truth_time in enumerate(ground_truth_times[np.isfinite(ground_truth_times)]):
                    ax.axvline(ground_truth_time, color="black", linestyle=":", alpha=0.6, label="ground truth" if idx == 0 else None)

            ax.set(title=f"Sample #{example_number} (K={self.number_of_peaks}, detected={np.sum(np.isfinite(peaks_t))})", xlabel="time", ylabel="amplitude")

            if k == 0:
                ax.legend(loc="best")

        fig.tight_layout()
        return fig

    @helper.post_mpl_plot
    def plot_histogram_counts(self, bins: int = 1) -> plt.Figure:
        """
        Histogram of detected peak counts per sample.

        Parameters
        ----------
        bins : int
            Number of bins to use for the histogram.
        """
        counts = self.peak_count
        # bins default: integers from 0..K
        if bins == 1:
            bins = np.arange(-0.5, self.number_of_peaks + 1.5, 1)
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.4))
        ax.hist(counts, bins=bins, edgecolor="black", alpha=0.8)
        ax.set(title="Detected peaks per sample", xlabel="#peaks", ylabel="frequency")
        return fig


class NonMaximumSuppression:
    r"""
    Detect up to three equal-width Gaussian pulses in a one-dimensional signal.

    The detector operates in two stages:

    1. **Matched filtering**
       The input signal is correlated with a unit-energy Gaussian kernel.

    2. **Non-maximum suppression**
       Candidate peaks are selected as local maxima above a threshold.

    .. math::

        y(t) = \sum_{k=1}^A a_k \exp\!\left(-\frac{(t - \mu_k)^2}{2\sigma^2}\right) + \eta(t)

    where all pulses share the same width :math:`\sigma`.
    """

    def __init__(self, gaussian_sigma: float, *, threshold: float | str = "auto", minimum_separation: float | None = None, maximum_number_of_pulses: int = 3, kernel_truncation_radius_in_sigmas: float = 3.5) -> None:
        r"""
        Parameters
        ----------
        gaussian_sigma : float
            The known common Gaussian standard deviation :math:`\sigma`.
        threshold : float | "auto"
            Threshold on the matched-filter output.
            If ``"auto"``, it is set to :math:`4.5 \,\hat\sigma_n` wheredd"``, it is set to :math:`4.5 \,\hat\sigma_n` wheredd"``, it is set to :math:`4.5 \,\hat\sigma_n` wheredd"``, it is set to :math:`4.5 \,\hat\sigma_n` wheredd"``, it is set to :math:`4.5 \,\hat\sigma_n` wheredd
            :math:`\hat\sigma_n` is a robust noise estimate.
        minimum_separation : float | None
            Minimum allowed peak separation in time units.
            Defaults to :math:`\sigma` if None.
        maximum_number_of_pulses : int
            Maximum number of pulses to return (1-N).
        kernel_truncation_radius_in_sigmas : float
            Radius of the Gaussian FIR kernel in multiples of :math:`\sigma`.
        """
        self.gaussian_sigma = float(gaussian_sigma)
        self.threshold = threshold
        self.minimum_separation = minimum_separation
        self.maximum_number_of_pulses = int(maximum_number_of_pulses)
        self.kernel_truncation_radius_in_sigmas = float(kernel_truncation_radius_in_sigmas)

        # Results after detection (coarse, no quadratic refinement)
        self.gaussian_kernel_: NDArray[np.float64] | None = None
        self.matched_filter_output_: NDArray[np.float64] | None = None
        self.peak_indices_: NDArray[np.int_] | None = None
        self.peak_times_: NDArray[np.float64] | None = None
        self.peak_heights_: NDArray[np.float64] | None = None
        self.threshold_used_: float | None = None
        self.suppression_half_window_in_samples_: int | None = None

    def run(self, time_samples: NDArray[np.float64], signal: NDArray[np.float64]) -> SingleResult:
        r"""
        Run the detection pipeline (matched filter + non-maximum suppression) on a single signal.

        Parameters
        ----------
        time_samples : ndarray, shape (N,)
            Uniform sample times.
        signal : ndarray, shape (N,)
            Input signal.

        Returns
        -------
        SingleResult
            Rich result object with analysis and plotting utilities.
        """
        signal = signal.squeeze()
        time_samples = time_samples.squeeze()
        assert signal.ndim == 1 and time_samples.ndim == 1 and len(signal) == len(time_samples), "signal and time_samples must be one-dimensional arrays of the same length"

        sample_interval = float(time_samples[1] - time_samples[0])

        # 1) Matched filter
        gaussian_kernel = self._build_gaussian_kernel(
            sample_interval=sample_interval,
            gaussian_sigma=self.gaussian_sigma,
            truncation_radius_in_sigmas=self.kernel_truncation_radius_in_sigmas,
        )
        matched_filter_output = self._correlate(signal, gaussian_kernel)

        # 2) Window & threshold
        minimum_separation = self.gaussian_sigma if self.minimum_separation is None else self.minimum_separation
        win = int(max(1, np.round(minimum_separation / sample_interval / 2.0)))

        if self.threshold == "auto":
            noise_sigma = self._estimate_noise_std(matched_filter_output)
            threshold_value = 4.5 * noise_sigma
        else:
            threshold_value = float(self.threshold)

        # 3) NMS
        peak_indices = self._non_maximum_suppression(
            values=matched_filter_output,
            half_window=win,
            threshold=threshold_value,
            max_peaks=self.maximum_number_of_pulses,
        )

        # Sort peaks by time
        if peak_indices.size:
            order = np.argsort(time_samples[peak_indices])
            peak_indices = peak_indices[order]
            peak_times = time_samples[peak_indices]
            peak_heights_mf = matched_filter_output[peak_indices]
            peak_heights_raw = signal[peak_indices]
        else:
            peak_times = np.empty(0, dtype=float)
            peak_heights_mf = np.empty(0, dtype=float)
            peak_heights_raw = np.empty(0, dtype=float)

        # Build rich result
        result = SingleResult(
            detector=self,
            signal=signal,
            time_samples=time_samples,
            matched_filter_output=matched_filter_output,
            gaussian_kernel=gaussian_kernel,
            threshold_used=float(threshold_value),
            suppression_half_window_in_samples=int(win),
            peak_indices=peak_indices,
            peak_times=peak_times,
            peak_amplitude_raw=peak_heights_raw,
            peak_amplitude_matched=peak_heights_mf,
        )
        return result

    # ---------------- Static helper methods ----------------
    @staticmethod
    def full_width_half_maximum_to_sigma(fwhm: float) -> float:
        r"""
        Convert full width at half maximum (FWHM) to Gaussian standard deviation.

        .. math::

            \text{FWHM} = 2 \sqrt{2 \ln 2} \,\sigma \;\;\Rightarrow\;\;
            \sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}

        Parameters
        ----------
        fwhm : float
            Full width at half maximum.
        """
        return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # ---------------- Private methods (implementation) ----------------

    @staticmethod
    def _build_gaussian_kernel(
        sample_interval: float,
        gaussian_sigma: float,
        truncation_radius_in_sigmas: float,
    ) -> NDArray[np.float64]:
        r"""
        Construct a discrete Gaussian kernel and normalize to unit energy.

        .. math::

            g[k] = \exp\!\left(-\tfrac{1}{2} \left(\frac{k \,\Delta t}{\sigma}\right)^2\right),
            \quad k = -L, \dots, L,

        where :math:`L = \left\lceil \dfrac{\text{radius}\,\sigma}{\Delta t} \right\rceil`
        and the discrete energy satisfies :math:`\sum_k g[k]^2 = 1`.

        Parameters
        ----------
        sample_interval : float
            Sample spacing :math:`\Delta t`.
        gaussian_sigma : float
            Gaussian standard deviation :math:`\sigma`.
        truncation_radius_in_sigmas : float
            Kernel radius in multiples of :math:`\sigma`.
        """
        half_length = int(np.ceil(truncation_radius_in_sigmas * gaussian_sigma / sample_interval))
        time_axis = np.arange(-half_length, half_length + 1, dtype=float) * sample_interval
        kernel = np.exp(-0.5 * (time_axis / gaussian_sigma) ** 2)
        kernel /= np.sqrt(np.sum(kernel**2))
        return kernel

    @staticmethod
    def _correlate(signal: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Discrete correlation (matched filter):

        .. math::

            r[n] = \sum_m y[m] \, g[m-n].

        Implemented as convolution with the reversed kernel.

        Parameters
        ----------
        signal : array
            Input signal samples :math:`y[n]`.
        kernel : array
            Correlation kernel :math:`g[k]`.
        """
        return np.convolve(signal, kernel[::-1], mode="same")

    @staticmethod
    def _non_maximum_suppression(values: NDArray[np.float64], half_window: int, threshold: float, max_peaks: int) -> NDArray[np.int_]:
        r"""
        Non-maximum suppression.

        Keep index :math:`n` if

        .. math::

            r[n] = \max_{|k-n|\leq W} r[k], \quad r[n] \ge \tau,

        where :math:`W` is the half-window and :math:`\tau` is the threshold.

        Returns at most ``max_peaks`` indices with the largest responses.

        Parameters
        ----------
        values : ndarray, shape (N,)
            Input values :math:`r[n]`.
        half_window : int
            Half-window :math:`W` in samples (must be ≥ 1).
        threshold : float
            Minimum value :math:`\tau` to be considered a peak.
        max_peaks : int
            Maximum number of peaks to return.  If more are found, the top ``max_peaks``
        """
        if half_window < 1:
            core = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] >= threshold)
            idx = np.where(core)[0] + 1
        else:
            window_len = 2 * half_window + 1
            padded = np.pad(values, (half_window, half_window), mode="edge")
            windows = NonMaximumSuppression._sliding_window_view_1d(padded, window_len)
            local_max = windows.max(axis=1)
            idx = np.where((values >= local_max) & (values >= threshold))[0]

        if idx.size > max_peaks:
            keep = np.argpartition(values[idx], -max_peaks)[-max_peaks:]
            idx = idx[keep]
            idx = idx[np.argsort(values[idx])]

        return np.sort(idx)

    @staticmethod
    def _estimate_noise_std(values: NDArray[np.float64]) -> float:
        r"""
        Estimate noise standard deviation from median absolute deviation (MAD).

        .. math::

            m = \text{median}(x), \quad MAD = \text{median}(|x-m|), \quad \hat\sigma_n \approx 1.4826 \, MAD

        This estimator is robust to outliers (e.g., signal peaks).

        Parameters
        ----------
        values : ndarray, shape (N,)
            Input values (e.g., matched filter output).
        """
        m = np.median(values)
        mad = np.median(np.abs(values - m))
        return 1.4826 * mad

    # ---------------- Local replacement for sliding_window_view ----------------
    @staticmethod
    def _sliding_window_view_1d(array: NDArray[np.float64], window_length: int) -> NDArray[np.float64]:
        r"""
        Create a 2D strided view of 1D ``array`` with a moving window of length ``window_length``.

        The returned view has shape :math:`(N - L + 1, L)` where
        :math:`N` is the length of ``array`` and :math:`L` is ``window_length``.
        No data is copied.

        This function replicates the essential behavior of
        :code:`numpy.lib.stride_tricks.sliding_window_view` for the 1D case,
        without importing it directly.

        Parameters
        ----------
        array : array
            One-dimensional input array.
        window_length : int
            Length :math:`L` of each sliding window (must satisfy :math:`1 \le L \le N`).

        Returns
        -------
        array
            A read-only view of shape :math:`(N-L+1, L)`.
        """
        if array.ndim != 1:
            raise ValueError("array must be one-dimensional")
        if not (1 <= window_length <= array.shape[0]):
            raise ValueError("window_length must satisfy 1 <= L <= len(array)")

        N = array.shape[0]
        stride = array.strides[0]
        shape = (N - window_length + 1, window_length)
        strides = (stride, stride)
        view = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
        view.setflags(write=False)
        return view

    # ---------------- Batch methods ----------------
    def run_batch(self, time_samples: NDArray[np.float64], signal: NDArray[np.float64]) -> BatchResult:
        r"""
        Run the detection pipeline on one or many signals.

        Parameters
        ----------
        time_samples : ndarray, shape (N,)
            Shared uniform sample times.
        signal : ndarray, shape (N,) or (B, N)
            Single signal or batch.

        Returns
        -------
        BatchResult
            Rich result object containing arrays, analysis helpers, and plotting methods.
        """
        # ---- coerce shapes
        time_samples = np.asarray(time_samples, dtype=float)
        assert time_samples.ndim == 1, "time_samples must be a 1D array (shared grid)."
        sig = np.asarray(signal, dtype=float)
        if sig.ndim == 1:
            sig = sig[None, :]
        assert sig.ndim == 2, "signal must have shape (N,) or (B, N)"
        B, N = sig.shape
        assert N == time_samples.size, "signal and time_samples length mismatch."

        dt = float(time_samples[1] - time_samples[0])

        # ---- matched filter
        gaussian_kernel = self._build_gaussian_kernel(
            sample_interval=dt,
            gaussian_sigma=self.gaussian_sigma,
            truncation_radius_in_sigmas=self.kernel_truncation_radius_in_sigmas,
        )
        r = self._correlate_batch(sig, gaussian_kernel[::-1])  # (B, N)

        # ---- window & threshold per sample
        min_sep = self.gaussian_sigma if self.minimum_separation is None else self.minimum_separation
        win = int(max(1, np.round(min_sep / dt / 2.0)))  # NMS half-window

        if self.threshold == "auto":
            noise_sigma = self._robust_noise_std_batch(r)  # (B,)
            tau = 4.5 * noise_sigma
        else:
            tau = np.full(B, float(self.threshold), dtype=float)

        # ---- NMS (batched, vectorized)
        padded = np.pad(r, ((0, 0), (win, win)), mode="edge")  # (B, N + 2*win)
        blocks = self._sliding_window_last_axis(padded, 2 * win + 1)  # (B, N, 2*win+1)
        locmax = blocks.max(axis=-1)  # (B, N)

        mask = (r >= locmax) & (r >= tau[:, None])  # (B, N)

        K = int(self.maximum_number_of_pulses)
        masked_vals = np.where(mask, r, -np.inf)  # (B, N)
        idx_sorted_desc = np.argsort(masked_vals, axis=1)[:, ::-1]  # (B, N)
        idx_topk = idx_sorted_desc[:, : min(K, N)]  # (B, K')
        vals_topk = np.take_along_axis(masked_vals, idx_topk, axis=1)  # (B, K')
        valid_topk = np.isfinite(vals_topk)  # (B, K')

        # Pad to K
        if idx_topk.shape[1] < K:
            pad_w = K - idx_topk.shape[1]
            idx_topk = np.pad(idx_topk, ((0, 0), (0, pad_w)), constant_values=0)
            valid_topk = np.pad(valid_topk, ((0, 0), (0, pad_w)), constant_values=False)

        bigN = N + 1
        idx_for_sort = np.where(valid_topk, idx_topk, bigN)
        idx_time_sorted = np.sort(idx_for_sort, axis=1)  # (B, K)
        valid_sorted = idx_time_sorted < N  # (B, K)

        # Final peak indices
        peak_indices = np.where(valid_sorted, idx_time_sorted, -1).astype(int)  # (B, K)

        # Safe gathers (avoid OOB), then mask
        safe_idx = np.where(valid_sorted, idx_time_sorted, 0)
        times_at_safe = time_samples[safe_idx]  # (B, K)
        peak_times = np.where(valid_sorted, times_at_safe, np.nan)

        amps_at_safe = np.take_along_axis(sig, safe_idx, axis=1)  # (B, K)
        peak_amplitudes = np.where(valid_sorted, amps_at_safe, np.nan)

        # ---- pack into BatchResult
        result = BatchResult(
            detector=self,
            signals=sig,
            time_samples=time_samples,
            matched_filter_output=r,
            gaussian_kernel=gaussian_kernel,
            threshold_used=tau,
            suppression_half_window_in_samples=win,
            peak_indices=peak_indices,
            peak_times=peak_times,
            peak_amplitude_raw=peak_amplitudes,
        )
        return result

    @staticmethod
    def _sliding_window_last_axis(x: np.ndarray, window: int) -> np.ndarray:
        """
        Return a strided sliding window view over the last axis.

        x : (B, N)
        window : int
        returns : (B, N - window + 1, window)
        """
        if x.ndim != 2:
            raise ValueError("x must be 2D (B, N).")
        B, N = x.shape
        if not (1 <= window <= N):
            raise ValueError("window must satisfy 1 <= window <= N")
        stride_b, stride_n = x.strides
        shape = (B, N - window + 1, window)
        strides = (stride_b, stride_n, stride_n)
        view = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        view.setflags(write=False)
        return view

    @staticmethod
    def _correlate_batch(signals: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Batched 'same' correlation using zero-padding (no Python loops).

        signals : (B, N)
        kernel  : (K,)  (pass REVERSED kernel for correlation, i.e. g[::-1])
        returns : (B, N)
        """
        signals = np.asarray(signals, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        K = kernel.size
        pad = K // 2  # odd K assumed (your Gaussian is 2L+1)
        padded = np.pad(signals, ((0, 0), (pad, pad)), mode="constant")
        # windows: (B, N, K)
        windows = NonMaximumSuppression._sliding_window_last_axis(padded, K)
        # dot each window with kernel -> (B, N)
        return np.einsum("b n k, k -> b n", windows, kernel)

    @staticmethod
    def _robust_noise_std_batch(values: np.ndarray) -> np.ndarray:
        """
        Robust per-row noise estimate via MAD: sigma ≈ 1.4826 * median(|x - median(x)|).
        values : (B, N)
        returns : (B,)
        """
        med = np.median(values, axis=1, keepdims=True)
        mad = np.median(np.abs(values - med), axis=1)
        return 1.4826 * mad
