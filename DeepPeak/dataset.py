import matplotlib.pyplot as plt
from MPSPlots import helper
import numpy as np


class DataSet:
    """
    A simple container class for datasets.

    This class dynamically sets attributes based on the provided keyword arguments,
    allowing for flexible storage of various dataset components.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to be set as attributes of the instance.
    """

    list_of_attributes = None

    def __init__(self, **kwargs):
        self.list_of_attributes = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.list_of_attributes.append(key)

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ", ".join(f"{key}" for key in self.list_of_attributes)
        return f"{class_name}({attributes})"

    def get_normalized_signal(self, normalization: str = "l1"):
        """
        Normalize the signals in the dataset to have values between 0 and 1.
        """
        if normalization.lower() == "l1":
            sum_vals = np.sum(self.signals, axis=1, keepdims=True)
            return self.signals / (sum_vals + 1e-8)
        elif normalization.lower() == "l2":
            norm_vals = np.linalg.norm(self.signals, axis=1, keepdims=True)
            return self.signals / (norm_vals + 1e-8)
        elif normalization.lower() == "min-max":
            min_vals = np.min(self.signals, axis=1, keepdims=True)
            max_vals = np.max(self.signals, axis=1, keepdims=True)
            return (self.signals - min_vals) / (max_vals - min_vals + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    @helper.post_mpl_plot
    def plot(
        self,
        number_of_samples: int | None = 3,
        number_of_columns: int = 1,
        randomize_signal: bool = False,
    ):
        """
        Plot the predicted Regions of Interest (ROIs) for several sample signals.

        Parameters
        ----------
        number_of_samples : int, default=3
            Number of signals to visualize.
        randomize_signal : bool, default=False
            If True, randomly select signals from the dataset instead of taking
            the first N samples.
        number_of_columns : int, default=1
            Number of columns in the subplot grid.
        """
        sample_count = self.signals.shape[0]

        if number_of_samples is None:
            number_of_samples = sample_count

        # Select which samples to display
        if randomize_signal:
            indices = np.random.choice(
                sample_count, size=number_of_samples, replace=False
            )
        else:
            indices = np.arange(min(number_of_samples, sample_count))

        number_of_rows = int(np.ceil(len(indices) / number_of_columns))

        figure, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(8 * number_of_columns, 3 * number_of_rows),
            squeeze=False,
        )

        for plot_index, ax in zip(indices, axes.flatten()):
            signal = self.signals[plot_index]

            # Plot signal
            ax.plot(self.x_values, signal, label="signal", color="black")

            # Highlight predicted region of interest

            handles, labels = ax.get_legend_handles_labels()

            if self.region_of_interest is not None:
                roi_patch = ax.fill_between(
                    self.x_values,
                    y1=0,
                    y2=1,
                    where=(self.region_of_interest[plot_index] != 0),
                    color="lightblue",
                    alpha=1.0,
                    transform=ax.get_xaxis_transform(),
                )

                handles.append(roi_patch)
                labels.append("Predicted ROI")

            # Build legend (consistent with your existing plotting logic)
            by_label = {}
            for h, l in zip(handles, labels):
                if l and not l.startswith("_") and l not in by_label:
                    by_label[l] = h

            ax.legend(by_label.values(), by_label.keys())
            ax.set_title(f"Sample {plot_index}")

        figure.supxlabel("Time step [AU]", y=0)
        figure.supylabel("Signal [AU]", x=0)

        return figure

    def low_pass(
        self,
        cutoff_fraction: float = 0.2,
        method: str = "fft",  # "fft" or "moving_average"
        window_size: int | None = None,  # used when method == "moving_average"
        inplace: bool = False,
    ):
        """
        Low pass filter the dataset signals.

        Parameters
        ----------
        cutoff_fraction : float
            Fraction of the Nyquist frequency to keep (0 < cutoff_fraction < 0.5).
            Used when method == "fft".
        method : {"fft", "moving_average"}
            "fft": zero out high frequency bins in rFFT.
            "moving_average": simple boxcar smoothing over window_size samples.
        window_size : int or None
            Length of the moving average window when method == "moving_average".
            If None, defaults to max(3, L//100).
        inplace : bool
            If True, overwrite self.signals with the filtered version.
            If False, return a new filtered array.

        Returns
        -------
        np.ndarray
            The filtered signals (also written to self.signals if inplace=True).
        """
        import numpy as np

        signals = np.asarray(self.signals, dtype=float)
        if signals.ndim != 2:
            raise ValueError("signals must be a 2D array of shape (N, L)")

        N, L = signals.shape

        # check that x_values is evenly spaced if using FFT
        if method == "fft":
            dx = np.diff(self.x_values)
            if not np.allclose(dx, dx[0], rtol=1e-3, atol=1e-9):
                raise ValueError("x_values must be evenly spaced for FFT low pass")

            if not (0.0 < cutoff_fraction < 0.5):
                raise ValueError("cutoff_fraction must be in (0, 0.5)")

            # rFFT bins: indices 0..K where K=L//2
            K = L // 2
            k_cut = int(np.floor(cutoff_fraction * K))
            if k_cut < 1:
                k_cut = 1

            filtered = np.empty_like(signals)
            for i in range(N):
                spec = np.fft.rfft(signals[i])
                spec[k_cut + 1 :] = 0.0
                filtered[i] = np.fft.irfft(spec, n=L)

        elif method == "moving_average":
            if window_size is None:
                window_size = max(3, L // 100)
            window_size = int(window_size)
            if window_size < 1:
                window_size = 1
            # simple symmetric boxcar with reflection at edges
            kernel = np.ones(window_size, dtype=float) / float(window_size)
            pad = window_size // 2
            filtered = np.empty_like(signals)
            for i in range(N):
                x = signals[i]
                xpad = np.pad(x, pad_width=pad, mode="reflect")
                filtered[i] = np.convolve(xpad, kernel, mode="valid")
                # ensure length L (valid yields L when pad == window//2)
                if filtered[i].shape[0] != L:
                    filtered[i] = filtered[i][:L]
        else:
            raise ValueError("method must be 'fft' or 'moving_average'")

        if inplace:
            self.signals = filtered
        return filtered

    def compute_region_of_interest(
        self,
        width_in_pixels: int = 4,
    ) -> np.ndarray:
        """
        ROI builder robust to any x sampling grid.
        Positions are already in real x coordinates.

        Parameters
        ----------
        dataset : DataSet
            Dataset to which to add the region_of_interest attribute.
        width_in_pixels : int
            Full width (in samples) of ROI around each peak center.
        """
        n_samples, sequence_length = self.signals.shape

        # Map true x positions -> nearest index
        diff = np.abs(self.positions[..., None] - self.x_values[None, None, :])
        centers = diff.argmin(axis=-1).astype(np.int64)

        np.clip(centers, 0, sequence_length - 1, out=centers)

        # Valid peaks = finite position and finite amplitude
        valid_pos = np.isfinite(self.positions)
        valid_amp = np.isfinite(self.amplitudes) & (self.amplitudes != 0)
        valid = valid_pos & valid_amp

        w = int(width_in_pixels)
        if w < 0:
            raise ValueError("width_in_pixels must be non-negative")
        half = w // 2

        starts = np.clip(centers - half, 0, sequence_length)
        ends = np.clip(centers + half + 1, 0, sequence_length)

        diff = np.zeros((n_samples, sequence_length + 1), dtype=np.int32)

        ii, jj = np.nonzero(valid)
        if ii.size > 0:
            np.add.at(diff, (ii, starts[ii, jj]), 1)
            np.add.at(diff, (ii, ends[ii, jj]), -1)

        rois = (np.cumsum(diff[:, :sequence_length], axis=1) > 0).astype(np.int32)

        self.region_of_interest = rois

        return self
