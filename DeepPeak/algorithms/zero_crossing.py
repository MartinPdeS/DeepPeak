import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray
from MPSPlots import helper


@dataclass
class ZCSingleResult:
    """
    Result container for zero-crossing detection (no NMS).

    Attributes
    ----------
    detector : ZeroCrossing
        The detector instance that produced this result.
    signal : NDArray[np.float64]
        The raw input signal.
    time_samples : NDArray[np.float64]
        The time samples corresponding to the signal.
    smoothed_signal : NDArray[np.float64]
        The smoothed version of the input signal.
    derivative : NDArray[np.float64]
        The numerical derivative of the smoothed signal.
    threshold_used : float
        The threshold value used for peak detection.
    peak_indices : NDArray[np.int_]
        Indices of detected peaks in the signal array.
    peak_times : NDArray[np.float64]
        Times of detected peaks.
    peak_amplitude_raw : NDArray[np.float64]
        Amplitudes of detected peaks in the raw signal.
    peak_amplitude_smoothed : NDArray[np.float64]
        Amplitudes of detected peaks in the smoothed signal.
    """

    detector: "ZeroCrossing"
    signal: NDArray[np.float64]
    time_samples: NDArray[np.float64]
    smoothed_signal: NDArray[np.float64]
    derivative: NDArray[np.float64]
    threshold_used: float
    peak_indices: NDArray[np.int_]
    peak_times: NDArray[np.float64]
    peak_amplitude_raw: NDArray[np.float64]
    peak_amplitude_smoothed: NDArray[np.float64]

    @property
    def number_of_peaks(self) -> int:
        return int(self.peak_indices.size)

    def summary(self) -> dict:
        return {
            "N": int(self.signal.size),
            "K_detected": self.number_of_peaks,
            "sigma_smooth": float(self.detector.gaussian_sigma),
            "threshold_used": float(self.threshold_used),
            "require_negative_curvature": bool(self.detector.require_negative_curvature),
        }

    @helper.post_mpl_plot
    def plot(self, *, title: Optional[str] = None) -> plt.Figure:
        """
        Plot the raw signal, smoothed signal, threshold, detected peaks,
        and derivative in a single figure with subplots.
        """
        t = self.time_samples

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        # --- Top subplot: signal, smoothed, threshold, peaks ---
        ax1.plot(t, self.signal, label="signal", linewidth=1.2)
        ax1.plot(t, self.smoothed_signal, label="smoothed", linewidth=1.2)
        ax1.axhline(self.threshold_used, linestyle="--", alpha=0.7, label="threshold")

        if self.peak_times.size:
            for tt in self.peak_times:
                ax1.axvline(tt, linestyle=":", alpha=0.7)
            # single legend entry for peaks
            peak_proxy = plt.Line2D([0], [0], linestyle=":", color="C2", alpha=0.8)
            handles, labels = ax1.get_legend_handles_labels()
            handles.append(peak_proxy)
            labels.append("peaks")
            ax1.legend(handles, labels, loc="best")
        else:
            ax1.legend(loc="best")

        ax1.set(
            title=title or f"Zero-crossing peaks (K={self.number_of_peaks})",
            ylabel="amplitude",
        )
        ax1.grid(True, alpha=0.3)

        # --- Bottom subplot: derivative ---
        ax2.plot(t, self.derivative, label="derivative dy/dt", linewidth=1.0)
        ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)

        if self.peak_times.size:
            for tt in self.peak_times:
                ax2.axvline(tt, linestyle=":", alpha=0.7)

        ax2.set(xlabel="time", ylabel="dy/dt", title="Derivative and zero-crossings")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

        fig.tight_layout()
        return fig


@dataclass
class ZCBatchResult:
    """
    Container for batched zero-crossing results with plotting utilities.

    Parameters
    ----------
    detector : ZeroCrossing
    signals : (B, N)
    time_samples : (N,)
    smoothed_signals : (B, N)
    derivative : (B, N)
    smoothing_kernel : (L,)
    threshold_used : (B,)
    peak_indices : (B, K)   (-1 for missing)
    peak_times : (B, K)     (NaN for missing)
    peak_amplitude_raw : (B, K)         (NaN for missing)
    peak_amplitude_smoothed : (B, K)    (NaN for missing)
    """

    detector: "ZeroCrossing"
    signals: NDArray[np.float64]
    time_samples: NDArray[np.float64]
    smoothed_signals: NDArray[np.float64]
    derivative: NDArray[np.float64]
    # smoothing_kernel: NDArray[np.float64]
    threshold_used: NDArray[np.float64]
    peak_indices: NDArray[np.int_]
    peak_times: NDArray[np.float64]
    peak_amplitude_raw: NDArray[np.float64]
    peak_amplitude_smoothed: NDArray[np.float64]

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
            "sigma_smooth": float(self.detector.gaussian_sigma),
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
        Small multiples of several samples showing signal (+ optional smoothed) and detected peaks.
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

            ax.plot(self.time_samples, self.smoothed_signals[example_number], label="smoothed")
            ax.axhline(self.threshold_used[example_number], linestyle="--", alpha=0.7, label="threshold" if k == 0 else None)

            for peak_time in peaks_t[np.isfinite(peaks_t)]:
                ax.axvline(peak_time, linestyle=":", alpha=0.7)

            ax.set(
                title=f"Sample #{example_number} (detected={np.sum(np.isfinite(peaks_t))})",
                xlabel="time",
                ylabel="amplitude",
            )

            if k == 0:
                ax.legend(loc="best")

            if ground_truth is not None:
                ground_truth_times = ground_truth[example_number]
                for idx, ground_truth_time in enumerate(ground_truth_times[np.isfinite(ground_truth_times)]):
                    ax.axvline(ground_truth_time, color="black", linestyle=":", alpha=0.6, label="ground truth" if idx == 0 else None)

        return fig


class ZeroCrossing:
    r"""
    Detect peaks using the **zero-crossing of the derivative** of a smoothed signal.

    Parameters
    ----------
    gaussian_sigma : float
        Standard deviation of the Gaussian kernel used for smoothing (in the same units as `time_samples`).
    threshold : float or 'auto', optional
        Amplitude threshold for peak detection on the smoothed signal.
        If 'auto', the threshold is set to `median(smoothed) + k * MAD(smoothed)`,
        where `k` is given by `threshold_k`. Default is 'auto'.
    threshold_k : float, optional
        The multiplier `k` used in the automatic threshold calculation when `threshold='auto'`.
        Default is 3.0.
    kernel_truncation_radius_in_sigmas : float, optional
        The truncation radius of the Gaussian kernel in units of `gaussian_sigma`.
        Default is 3.5.
    require_negative_curvature : bool, optional
        If True, keep only candidates with negative second derivative (concave).
    maximum_number_of_peaks : int, optional
        Maximum number of peaks to return per signal (for consistent batch shapes). Default 3.

    Pipeline:
      1) Smooth the input signal with an L1-normalized Gaussian of width ``gaussian_sigma``.
      2) Compute the derivative via ``np.gradient``.
      3) Candidate peaks = indices where derivative crosses zero from positive → non-positive.
      4) Optional curvature check: keep only points with negative second derivative (concave).
      5) Threshold on the **smoothed** amplitude at candidate indices.
      6) If more than ``maximum_number_of_peaks`` are found, keep the top-K by smoothed amplitude; sort by time.
    """

    def __init__(
        self,
        gaussian_sigma: float,
        *,
        threshold: float | str = "auto",
        threshold_k: float = 3.0,
        kernel_truncation_radius_in_sigmas: float = 3.5,
        require_negative_curvature: bool = True,
        maximum_number_of_peaks: int = 3,
    ) -> None:
        self.gaussian_sigma = float(gaussian_sigma)
        self.threshold = threshold
        self.threshold_k = float(threshold_k)
        self.kernel_truncation_radius_in_sigmas = float(kernel_truncation_radius_in_sigmas)
        self.require_negative_curvature = bool(require_negative_curvature)
        self.maximum_number_of_peaks = int(maximum_number_of_peaks)

    # -------- public API --------
    def run(self, time_samples: NDArray[np.float64], signal: NDArray[np.float64]) -> ZCSingleResult:
        """
        Run zero-crossing detection (no NMS).

        Returns
        -------
        ZCSingleResult
        """
        y = np.asarray(signal, dtype=float).squeeze()
        t = np.asarray(time_samples, dtype=float).squeeze()
        assert y.ndim == 1 and t.ndim == 1 and y.size == t.size, "signal and time_samples must be 1D and same length"

        dt = float(t[1] - t[0])

        # 1) Smooth with L1-normalized Gaussian
        g = self._build_gaussian_smoothing_kernel(
            sample_interval=dt,
            gaussian_sigma=self.gaussian_sigma,
            truncation_radius_in_sigmas=self.kernel_truncation_radius_in_sigmas,
        )
        y_s = np.convolve(y, g, mode="same")

        # 2) Derivative (central difference)
        dy = np.gradient(y_s, dt)

        # 3) Candidate maxima: derivative crosses zero from + to <= 0
        cand = np.where((dy[:-1] > 0.0) & (dy[1:] <= 0.0))[0] + 1

        # 4) Optional curvature: keep only points with negative second derivative
        if self.require_negative_curvature and cand.size:
            d2y = np.gradient(dy, dt)
            cand = cand[d2y[cand] < 0.0]

        # 5) Threshold on smoothed amplitude
        if isinstance(self.threshold, str):
            assert self.threshold == "auto", "threshold must be 'auto' or a float"
            tau = self._median_plus_kmad(y_s, k=self.threshold_k)
        else:
            tau = float(self.threshold)

        if cand.size:
            keep = cand[y_s[cand] >= tau]
        else:
            keep = cand

        # 6) Enforce top-K (for consistent shapes), then sort by time
        if keep.size > self.maximum_number_of_peaks:
            top = np.argpartition(y_s[keep], -self.maximum_number_of_peaks)[-self.maximum_number_of_peaks :]
            keep = keep[top]
        keep = np.sort(keep)

        peak_idx = np.asarray(keep, dtype=int)
        peak_times = t[peak_idx] if peak_idx.size else np.empty(0, dtype=float)
        peak_amp_raw = y[peak_idx] if peak_idx.size else np.empty(0, dtype=float)
        peak_amp_smooth = y_s[peak_idx] if peak_idx.size else np.empty(0, dtype=float)

        return ZCSingleResult(
            detector=self,
            signal=y,
            time_samples=t,
            smoothed_signal=y_s,
            derivative=dy,
            threshold_used=float(tau),
            peak_indices=peak_idx,
            peak_times=peak_times,
            peak_amplitude_raw=peak_amp_raw,
            peak_amplitude_smoothed=peak_amp_smooth,
        )

    def run_batch(self, time_samples: NDArray[np.float64], signal: NDArray[np.float64]) -> ZCBatchResult:
        """
        Run zero-crossing detection on a batch of signals.

        Parameters
        ----------
        time_samples : (N,)
            Shared uniform sample times.
        signal : (N,) or (B, N)
            Single signal or batch.
        """
        # ---- coerce shapes
        t = np.asarray(time_samples, dtype=float)
        assert t.ndim == 1, "time_samples must be a 1D array (shared grid)."
        sig = np.asarray(signal, dtype=float)
        if sig.ndim == 1:
            sig = sig[None, :]
        assert sig.ndim == 2, "signal must have shape (N,) or (B, N)"
        B, N = sig.shape
        assert N == t.size, "signal and time_samples length mismatch."
        dt = float(t[1] - t[0])

        # ---- smoothing kernel (same for all rows)
        g = self._build_gaussian_smoothing_kernel(
            sample_interval=dt,
            gaussian_sigma=self.gaussian_sigma,
            truncation_radius_in_sigmas=self.kernel_truncation_radius_in_sigmas,
        )

        # ---- batched smoothing via sliding windows + einsum
        K = g.size
        pad = K // 2
        padded = np.pad(sig, ((0, 0), (pad, pad)), mode="constant")
        windows = self._sliding_window_last_axis(padded, K)  # (B, N, K)
        y_s = np.einsum("bnk,k->bn", windows, g)  # (B, N)

        # ---- derivative(s)
        dy = np.gradient(y_s, dt, axis=1)
        if self.require_negative_curvature:
            d2y = np.gradient(dy, dt, axis=1)

        # ---- zero-crossing candidates per row: + -> <= 0
        mask = (dy[:, :-1] > 0.0) & (dy[:, 1:] <= 0.0)
        cand_lists = [np.where(mask[b])[0] + 1 for b in range(B)]  # list of 1D int arrays

        # ---- curvature check (FIX: no Python and/or with arrays)
        if self.require_negative_curvature:
            filtered = []
            for b, c in enumerate(cand_lists):
                if c.size:
                    sel = d2y[b, c] < 0.0  # boolean mask over candidate indices
                    filtered.append(c[sel])  # keep only concave points
                else:
                    filtered.append(np.empty(0, dtype=int))
            cand_lists = filtered

        # ---- thresholds (auto or fixed), per row
        if isinstance(self.threshold, str):
            med = np.median(y_s, axis=1)  # (B,)
            mad = np.median(np.abs(y_s - med[:, None]), axis=1)  # (B,)
            sigma = 1.4826 * mad
            tau = med + self.threshold_k * sigma  # (B,)
        else:
            tau = np.full(B, float(self.threshold), dtype=float)

        # ---- keep only candidates above threshold on smoothed amplitude; top-K; sort by time
        kept_lists = []
        for b in range(B):
            c = cand_lists[b]
            if c.size:
                kept = c[y_s[b, c] >= tau[b]]
            else:
                kept = c
            if kept.size > self.maximum_number_of_peaks:
                top = np.argpartition(y_s[b, kept], -self.maximum_number_of_peaks)[-self.maximum_number_of_peaks :]
                kept = kept[top]
            kept_lists.append(np.sort(kept))

        # ---- pack to fixed (B, K)
        Kmax = int(self.maximum_number_of_peaks)
        peak_indices = -np.ones((B, Kmax), dtype=int)
        peak_times = np.full((B, Kmax), np.nan, dtype=float)
        peak_amp_raw = np.full((B, Kmax), np.nan, dtype=float)
        peak_amp_smooth = np.full((B, Kmax), np.nan, dtype=float)

        for b in range(B):
            idx = kept_lists[b]
            fill = min(idx.size, Kmax)
            if fill > 0:
                peak_indices[b, :fill] = idx[:fill]
                peak_times[b, :fill] = t[idx[:fill]]
                peak_amp_raw[b, :fill] = sig[b, idx[:fill]]
                peak_amp_smooth[b, :fill] = y_s[b, idx[:fill]]

        return ZCBatchResult(
            detector=self,
            signals=sig,
            time_samples=t,
            smoothed_signals=y_s,
            derivative=dy,
            threshold_used=tau.astype(float),
            peak_indices=peak_indices,
            peak_times=peak_times,
            peak_amplitude_raw=peak_amp_raw,
            peak_amplitude_smoothed=peak_amp_smooth,
        )

    # -------- helpers --------
    @staticmethod
    def _median_plus_kmad(values: NDArray[np.float64], k: float) -> float:
        m = float(np.median(values))
        mad = float(np.median(np.abs(values - m)))
        sigma = 1.4826 * mad
        return m + k * sigma

    @staticmethod
    def _build_gaussian_smoothing_kernel(
        sample_interval: float,
        gaussian_sigma: float,
        truncation_radius_in_sigmas: float,
    ) -> NDArray[np.float64]:
        """
        L1-normalized Gaussian FIR for smoothing (area = 1).
        """
        half_len = int(np.ceil(truncation_radius_in_sigmas * gaussian_sigma / sample_interval))
        t_axis = np.arange(-half_len, half_len + 1, dtype=float) * sample_interval
        k = np.exp(-0.5 * (t_axis / gaussian_sigma) ** 2)
        k /= np.sum(k) if np.sum(k) > 0 else 1.0
        return k

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
