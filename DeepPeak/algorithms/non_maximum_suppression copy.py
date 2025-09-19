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
    def plot(
        self,
        *,
        show_matched_filter: bool = True,
        show_peaks: bool = True,
        show_kernel: bool = True,
    ) -> plt.Figure:
        """Signal (+ optional MF) with vertical lines at detected peak times.
        If show_kernel=True, overlay the Gaussian kernel at each detected peak,
        scaled so max(kernel) matches the raw peak amplitude.
        """
        t = self.time_samples
        y = self.signal
        r = self.matched_filter_output
        peaks_t = self.peak_times

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
        ax.plot(t, y, label="signal")
        if show_matched_filter:
            ax.plot(t, r, label="matched filter")

        # Vertical lines at peaks
        if show_peaks and peaks_t.size:
            for m in peaks_t:
                ax.axvline(m, linestyle="--", alpha=0.6)
        # Kernel overlay (centered at each detected peak, scaled to raw peak height)
        if show_kernel and peaks_t.size and self.gaussian_kernel is not None:
            k = self.gaussian_kernel
            K = k.size
            half = (K - 1) // 2
            dt = float(self.time_samples[1] - self.time_samples[0])
            kmax = float(np.max(np.abs(k))) if np.any(k) else 1.0
            color_kernel = "C3"

            # use raw amplitudes at peak indices for scaling
            for m, amp in zip(peaks_t, self.peak_amplitude_raw):
                alpha = (amp / kmax) if kmax > 0 else amp
                t_k = m + (np.arange(-half, half + 1, dtype=float) * dt)
                y_k = alpha * k
                ax.plot(t_k, y_k, linestyle="--", linewidth=1.4, alpha=0.85, color=color_kernel)

            # add a single legend entry for the kernel
            kernel_proxy = plt.Line2D([0], [0], linestyle="--", color=color_kernel, linewidth=1.4, alpha=0.85)
        else:
            kernel_proxy = None

        # Compose legend once
        handles, labels = ax.get_legend_handles_labels()
        if show_peaks and peaks_t.size:
            peak_proxy = plt.Line2D([0], [0], linestyle="--", color="C2", alpha=0.6)
            handles.append(peak_proxy)
            labels.append("peaks")
        if kernel_proxy is not None:
            handles.append(kernel_proxy)
            labels.append("kernel @ peaks")
        ax.legend(handles, labels, loc="best")

        ax.set(title=f"Detected peaks: {self.number_of_peaks}", xlabel="t", ylabel="amplitude")
        return fig


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
    def _plot_kernel_overlay(self, ax, example_number: int) -> None:
        """
        Overlay the detection kernel at detected peak positions for a given sample.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to draw on.
        example_number : int
            Index of the sample in the batch.
        """
        k = self.gaussian_kernel
        if k is None or k.size == 0:
            return

        idxs = self.peak_indices[example_number] if hasattr(self, "peak_indices") else None
        if idxs is None:
            return

        valid = idxs >= 0
        if not np.any(valid):
            return

        K = k.size
        half = (K - 1) // 2
        dt = float(self.time_samples[1] - self.time_samples[0])
        kmax = float(np.max(np.abs(k))) if np.any(k) else 1.0
        color_kernel = "C3"

        for i in idxs[valid]:
            m = self.time_samples[i]
            amp = self.signals[example_number, i]
            alpha = (amp / kmax) if kmax > 0 else amp
            t_k = m + (np.arange(-half, half + 1, dtype=float) * dt)
            y_k = alpha * k
            ax.plot(t_k, y_k, linestyle="--", linewidth=1.4, alpha=0.85, color=color_kernel)

    def _plot_signal_and_filter(self, ax, example_number: int) -> None:
        """Plot raw signal and matched filter output for one sample."""
        ax.plot(self.time_samples, self.signals[example_number], label="signal")
        ax.plot(self.time_samples, self.matched_filter_output[example_number], label="matched filter")

    def _plot_peaks(self, ax, example_number: int) -> None:
        """Mark detected peaks with vertical lines."""
        peaks_t = self.peak_times[example_number]
        for peak_time in peaks_t[np.isfinite(peaks_t)]:
            ax.axvline(peak_time, linestyle="-", alpha=0.6)

    def _plot_kernel_overlay(self, ax, example_number: int) -> None:
        """Overlay the detection kernel at detected peak positions for one sample."""
        k = self.gaussian_kernel
        if k is None or k.size == 0:
            return

        idxs = self.peak_indices[example_number] if hasattr(self, "peak_indices") else None
        if idxs is None or not np.any(idxs >= 0):
            return

        K = k.size
        half = (K - 1) // 2
        dt = float(self.time_samples[1] - self.time_samples[0])
        kmax = float(np.max(np.abs(k))) if np.any(k) else 1.0

        for i in idxs[idxs >= 0]:
            m = self.time_samples[i]
            amp = self.signals[example_number, i]
            alpha = (amp / kmax) if kmax > 0 else amp
            t_k = m + (np.arange(-half, half + 1, dtype=float) * dt)
            y_k = alpha * k
            ax.plot(t_k, y_k, linestyle="--", linewidth=1.4, alpha=0.85, color="C3")

    def _plot_true_positions(self, ax, example_number: int, true_position: np.ndarray) -> None:
        """Overlay ground-truth positions if provided."""
        gt = true_position[example_number]
        for j, gt_t in enumerate(gt[np.isfinite(gt)]):
            ax.axvline(gt_t, color="black", linestyle=":", alpha=0.6, label="true position" if j == 0 else None)

    def _format_axes(self, ax, example_number: int) -> None:
        """Set title and labels for one subplot."""
        peaks_t = self.peak_times[example_number]
        ax.set(
            title=f"Sample #{example_number} (K={self.number_of_peaks}, detected={np.sum(np.isfinite(peaks_t))})",
            xlabel="time",
            ylabel="amplitude",
        )

    @helper.post_mpl_plot
    def plot(
        self,
        indices: NDArray[np.int_] | None = None,
        *,
        ncols: int = 1,
        max_plots: int | None = 12,
        true_position: NDArray[np.float64] | None = None,
        show_kernel: bool = False,
    ) -> plt.Figure:
        """Small multiples of several samples."""
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

        for panel_idx, (example_number, ax) in enumerate(zip(batch_index, axes_flat)):
            self._plot_signal_and_filter(ax, example_number)
            self._plot_peaks(ax, example_number)
            if show_kernel:
                self._plot_kernel_overlay(ax, example_number)
            if true_position is not None:
                self._plot_true_positions(ax, example_number, true_position)
            self._format_axes(ax, example_number)

            if panel_idx == 0:
                ax.legend(loc="best")

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
