from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


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
        self.results: dict[str, object] | None = None

    def run(self, time_samples: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, object]:
        r"""
        Run the detection pipeline (matched filter + non-maximum suppression).

        Parameters
        ----------
        time_samples : array
            Uniform sample times :math:`t[n]` with spacing :math:`\Delta t`.
        signal : array
            Input signal samples :math:`y[n]`.

        Returns
        -------
        dict
            Results including coarse peak times (sample-aligned), their heights,
            the matched filter output, and configuration used.

        Notes
        -----
        Matched filter correlation:

        .. math::

            r[n] = \sum_m y[m] \, g[m-n]

        Non-maximum suppression keeps peaks :math:`n` such that

        .. math::

            r[n] = \max_{|k-n|\leq W} r[k], \qquad r[n] \ge \tau.
        """
        assert signal.ndim == 1 and time_samples.ndim == 1 and len(signal) == len(time_samples), "signal and time_samples must be one-dimensional arrays of the same length"

        sample_interval = float(time_samples[1] - time_samples[0])

        # 1) Build unit-energy Gaussian kernel and apply matched filter (correlation)
        gaussian_kernel = self._build_gaussian_kernel(
            sample_interval=sample_interval,
            gaussian_sigma=self.gaussian_sigma,
            truncation_radius_in_sigmas=self.kernel_truncation_radius_in_sigmas,
        )
        matched_filter_output = self._correlate(signal, gaussian_kernel)

        # 2) Determine suppression window and threshold
        if self.minimum_separation is None:
            minimum_separation = self.gaussian_sigma
        else:
            minimum_separation = self.minimum_separation

        suppression_half_window_in_samples = int(max(1, np.round(minimum_separation / sample_interval / 2.0)))

        if self.threshold == "auto":
            noise_sigma = self._estimate_noise_std(matched_filter_output)
            threshold_value = 4.5 * noise_sigma
        else:
            threshold_value = float(self.threshold)

        # 3) Non-maximum suppression (coarse indices only)
        peak_indices = self._non_maximum_suppression(
            values=matched_filter_output,
            half_window=suppression_half_window_in_samples,
            threshold=threshold_value,
            max_peaks=self.maximum_number_of_pulses,
        )

        # Coarse peak times and heights (no sub-sample refinement)
        peak_times = time_samples[peak_indices] if peak_indices.size else np.empty(0)
        peak_heights = matched_filter_output[peak_indices] if peak_indices.size else np.empty(0)

        # Save results on the instance
        self.gaussian_kernel_ = gaussian_kernel
        self.matched_filter_output_ = matched_filter_output
        self.peak_indices_ = np.sort(peak_indices)
        order = np.argsort(peak_times)
        self.peak_times_ = peak_times[order]
        self.peak_heights_ = peak_heights[order]
        self.threshold_used_ = float(threshold_value)
        self.suppression_half_window_in_samples_ = int(suppression_half_window_in_samples)

        self.results = {
            "signal": signal,
            "time_samples": time_samples,
            "peak_indices": self.peak_indices_,
            "peak_times": self.peak_times_,
            "peak_heights": self.peak_heights_,
            "matched_filter_output": matched_filter_output,
            "gaussian_kernel": gaussian_kernel,
            "threshold_used": float(threshold_value),
            "suppression_half_window_in_samples": int(suppression_half_window_in_samples),
        }
        return self.results

    def plot(self) -> None:
        r"""
        Plot the input signal, matched-filter output, and coarse peak locations.
        """
        from MPSPlots.styles import mps as plot_style

        if not getattr(self, "results", None):
            raise ValueError("Computation not done yet, use .run() first.")

        with plt.style.context(plot_style):
            t = self.results["time_samples"]
            y = self.results["signal"]
            r = self.results["matched_filter_output"]
            peaks_t = self.results["peak_times"]

            plt.figure()
            plt.plot(t, y, label="signal y(t)")
            plt.plot(t, r, label="matched filter output r(t)")
            for m in peaks_t:
                plt.axvline(m, linestyle="--", alpha=0.6, label=None)
            plt.title("Equal-width Gaussian pulse detection (coarse)")
            plt.xlabel("t")
            plt.ylabel("amplitude")
            plt.legend()
            plt.show()

    # ---------------- Static helper methods ----------------

    @staticmethod
    def full_width_half_maximum_to_sigma(fwhm: float) -> float:
        r"""
        Convert full width at half maximum (FWHM) to Gaussian standard deviation.

        .. math::

            \text{FWHM} = 2 \sqrt{2 \ln 2} \,\sigma \;\;\Rightarrow\;\;
            \sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}
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
