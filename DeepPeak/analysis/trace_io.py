"""CSV trace loading and lightweight preprocessing utilities.

``CsvTrace`` is the file-oriented bridge between raw exported acquisitions and
the analysis pipeline. It intentionally stays small: load the trace, provide a
few common signal manipulations, and expose basic peak-finding helpers for
interactive inspection.
"""

from MPSPlots import helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


class CsvTrace:
    """Load one CSV trace exported with a two-line header.

    Parameters
    ----------
    filename:
        Path to the CSV file to load.
    n_rows:
        Optional row limit for partial loading during exploration.

    Attributes
    ----------
    x : numpy.ndarray
        Trace x-axis values after conversion to seconds.
    y_raw : numpy.ndarray
        Raw signal values loaded from the file.
    y_processed : numpy.ndarray
        Working signal used by processing helpers and plotting.
    """

    def __init__(self, filename, n_rows: int | None = None):
        self.x, self.y_raw = self.load_csv_two_line_header(
            filename=filename, n_rows=n_rows
        )
        self.x *= 1e-3
        self.y_processed = self.y_raw

    @property
    def dx(self):
        """Return the sampling interval in the trace's time units.

        Returns
        -------
        float
            Difference between consecutive x-axis samples.
        """

        return self.x[1] - self.x[0]

    @property
    def delta_x(self):
        """Return the span covered by the loaded trace.

        Returns
        -------
        float
            Difference between the last and first x-axis samples.
        """

        return self.x[-1] - self.x[0]

    @property
    def sampling_rate(self):
        """Return the sampling frequency inferred from ``dx``.

        Returns
        -------
        float
            Inverse of the sampling interval.
        """

        return 1 / self.dx

    def remove_dc(self):
        """Subtract the mean value from ``y_processed`` in place.

        Returns
        -------
        None
            The processed signal is modified in place.
        """

        if not hasattr(self, "y_processed"):
            raise RuntimeError(
                "Call low_pass_filter or create y_processed before DC removal"
            )
        mean_value = float(np.mean(self.y_processed))
        self.y_processed = self.y_processed - mean_value

    def load_csv_two_line_header(self, filename: str, n_rows: int | None = None):
        """Load the trace assuming the first line stores column names.

        Parameters
        ----------
        filename : str
            Path to the CSV file.
        n_rows : int, optional
            Optional row limit for partial loading.

        Returns
        -------
        x_raw : numpy.ndarray
            Raw time-axis values from the file.
        y_raw : numpy.ndarray
            Raw signal values from channel A.
        """

        header_only = pd.read_csv(filename, nrows=1, header=None)
        col_names = [str(col).strip() for col in header_only.iloc[0]]

        df = pd.read_csv(
            filename,
            header=None,
            skiprows=2,
            names=col_names,
            low_memory=False,
            nrows=n_rows,
        )

        y_raw = (
            df["Channel A"]
            .astype(str)
            .str.replace("∞", "0", regex=False)
            .astype(float)
            .to_numpy()
        )
        x_raw = df["Time"].astype(float).to_numpy()

        assert not np.any(np.isnan(y_raw)), "NaN value found in y array"
        return x_raw, y_raw

    def robust_sigma_from_diff(self):
        """Estimate the noise level from first differences using a robust MAD rule.

        Returns
        -------
        float
            Robust estimate of the signal noise standard deviation.
        """

        y = self.y_raw
        dy = np.diff(y)
        mad = np.median(np.abs(dy - np.median(dy)))
        sigma_dy = 1.4826 * mad
        sigma_y = sigma_dy / np.sqrt(2)
        return sigma_y

    def get_height_based_on_noise(self, sigma):
        """Convert a sigma multiple into an absolute amplitude threshold.

        Parameters
        ----------
        sigma : float
            Number of estimated noise standard deviations above the median.

        Returns
        -------
        float
            Absolute height threshold.
        """

        median = np.median(self.y_raw)
        mad = np.median(np.abs(self.y_raw - median))
        sigma_noise = mad * 1.4826
        return median + sigma * sigma_noise

    def low_pass_filter(self, bandlimit: int = 1000) -> None:
        """Apply a simple FFT low-pass filter to the raw trace in place.

        Parameters
        ----------
        bandlimit : int, default=1000
            Frequency limit preserved by the FFT filter.

        Returns
        -------
        None
            The processed signal is modified in place.
        """

        bandlimit_index = int(bandlimit * self.y_raw.size / self.sampling_rate)
        fsig = np.fft.fft(self.y_raw)
        for i in range(bandlimit_index + 1, len(fsig) - bandlimit_index):
            fsig[i] = 0
        self.y_processed = np.real(np.fft.ifft(fsig))

    def _process_height(self, height):
        """Normalize height specifications for the ad hoc ``find_peaks`` helper.

        Parameters
        ----------
        height : float or str
            Height specification as an absolute threshold or ``"<n>sigma"``.

        Returns
        -------
        float
            Absolute height threshold.
        """

        if isinstance(height, str):
            height_sigma = height.strip("sigma")
            return float(height_sigma) * self.robust_sigma_from_diff()
        return height

    def find_peaks(self, height: float, use_processed: bool = True, **kwargs):
        """Run SciPy peak finding and cache peak positions, widths, and heights.

        Parameters
        ----------
        height : float
            Absolute threshold or sigma-derived threshold specification.
        use_processed : bool, default=True
            If ``True``, run detection on ``y_processed`` instead of ``y_raw``.
        **kwargs
            Additional keyword arguments forwarded to :func:`scipy.signal.find_peaks`.

        Returns
        -------
        peaks_positions : numpy.ndarray
            Peak positions on the x-axis.
        peaks_heights : numpy.ndarray
            Peak amplitudes at the detected positions.
        peaks_widths : numpy.ndarray
            Approximate peak widths returned by :func:`scipy.signal.peak_widths`.
        """

        height_threshold = self._process_height(height)
        y = self.y_processed if use_processed else self.y_raw
        self.peaks_index, _ = find_peaks(y, prominence=height_threshold, **kwargs)
        self.peaks_widths, _, _, _ = peak_widths(
            self.y_processed, self.peaks_index, rel_height=0.15
        )
        self.peaks_heights = y[self.peaks_index]
        self.peaks_positions = self.x[self.peaks_index]

        return self.peaks_positions, self.peaks_heights, self.peaks_widths

    @property
    def particle_flow(self):
        """Return the number of detected peaks per unit of trace duration.

        Returns
        -------
        float
            Detected peak count divided by the trace span.
        """

        return len(self.peaks_heights) / self.delta_x

    @helper.post_mpl_plot
    def plot_overview(self, end_idx=100000, nbins=None):
        """Plot the signal, detected peaks, and amplitude distribution for inspection.

        Parameters
        ----------
        end_idx : int, default=100000
            Last sample index displayed in the overview.
        nbins : int, optional
            Number of histogram bins used for the amplitude distribution.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the overview plots.
        """

        figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))

        axes[0].plot(self.x[:end_idx], self.y_raw[:end_idx], label="raw")
        if self.y_processed is not self.y_raw:
            axes[0].plot(
                self.x[:end_idx],
                self.y_processed[:end_idx],
                linestyle="--",
                label="filtered",
            )

        if hasattr(self, "peaks_positions"):
            mask = self.peaks_index < end_idx
            px = self.peaks_positions[mask]
            ph = self.peaks_heights[mask]
            axes[0].scatter(px, ph, color="black", zorder=20, label="peaks")

        axes[0].set_title(
            f"Signal and detected peaks [particle flow: {self.particle_flow}]"
        )
        axes[0].legend()

        axes[1].hist(self.y_raw[:end_idx], bins=nbins)
        axes[1].set_title("Distribution of signal amplitude")
        axes[1].set_xlabel("Amplitude")

        return figure


Data = CsvTrace

__all__ = ["CsvTrace", "Data"]
