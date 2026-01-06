from __future__ import annotations

from typing import Literal
import matplotlib.pyplot as plt
from MPSPlots import helper
import numpy as np
from DeepPeak import processing


Profile = Literal["gaussian", "lorentzian"]
WidthDefinition = Literal["fwhm", "sigma", "gamma"]
NormalizationMode = Literal["analytic", "sampled"]


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

    def get_normalized_signal(self, normalization: str = "zscore"):
        """
        Normalize dataset signals.

        Supported normalization modes
        -----------------------------
        "none"
            Return a float copy of the signals.
        "l1"
            Divide each signal by its L1 norm (sum of absolute values).
        "l2"
            Divide each signal by its L2 norm.
        "minmax"
            Map each signal to [0, 1] using per signal min and max.
        "zscore"
            Per signal standardization: (x - mean) / std.
        "robust_zscore"
            Per signal robust standardization: (x - median) / (1.4826 * MAD).
        "maxabs"
            Divide each signal by its max absolute value.

        Notes
        -----
        - "minmax" guarantees an output in [0, 1] (per signal).
        - "zscore" and "robust_zscore" are usually better for neural network training.
        """
        return processing.normalize_signal(
            self.signals, normalization=normalization, axis=1
        )

    @helper.post_mpl_plot
    def plot(
        self,
        number_of_samples: int | None = 3,
        number_of_columns: int = 1,
        randomize_signal: bool = False,
        region_of_interest: np.ndarray | None = None,
        reference_pulse_trace: np.ndarray | None = None,
        reference_pulse_scale: str = "auto",  # "auto", "signal_max", "none"
    ):
        """
        Plot signals and optional labels for several sample signals.

        Parameters
        ----------
        number_of_samples : int, default=3
            Number of signals to visualize.
        randomize_signal : bool, default=False
            If True, randomly select signals from the dataset instead of taking
            the first N samples.
        number_of_columns : int, default=1
            Number of columns in the subplot grid.
        region_of_interest : np.ndarray or None
            Optional ROI mask of shape (n_samples, sequence_length).
            Nonzero values are highlighted as a band.
        reference_pulse_trace : np.ndarray or None
            Optional reference pulse trace of shape (n_samples, sequence_length).
            Plotted as an overlay line.
        reference_pulse_scale : {"auto", "signal_max", "none"}
            How to scale the reference pulse trace for display:
            "auto"      : scale each reference trace to the local signal max.
            "signal_max": same as "auto" (kept for explicitness).
            "none"      : plot reference as is.
        """
        sample_count = self.signals.shape[0]

        if number_of_samples is None:
            number_of_samples = sample_count

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

            ax.plot(self.x_values, signal, label="signal", color="black")

            handles, labels = ax.get_legend_handles_labels()

            if region_of_interest is not None:
                roi_patch = ax.fill_between(
                    self.x_values,
                    y1=0,
                    y2=1,
                    where=(region_of_interest[plot_index] != 0),
                    color="lightblue",
                    alpha=1.0,
                    transform=ax.get_xaxis_transform(),
                )
                handles.append(roi_patch)
                labels.append("Predicted ROI")

            if reference_pulse_trace is not None:
                reference = np.asarray(reference_pulse_trace[plot_index], dtype=float)

                if reference_pulse_scale.lower() in ["auto", "signal_max"]:
                    reference_max = float(np.max(reference))
                    signal_max = float(np.max(signal))
                    if reference_max > 0.0:
                        reference = reference / reference_max * signal_max

                reference_handle = ax.plot(
                    self.x_values,
                    reference,
                    label="Reference pulse",
                )[0]
                handles.append(reference_handle)
                labels.append("Reference pulse")

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

    def get_region_of_interest(
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

        return rois

    def _validate_reference_pulse_inputs(
        self,
        signals: np.ndarray,
        x_values: np.ndarray,
        positions: np.ndarray,
        amplitudes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate and standardize arrays needed to build a reference pulse trace.

        Expected shapes
        ---------------
        signals:
            (n_samples, sequence_length)
        x_values:
            (sequence_length,)
        positions:
            (n_samples, n_peaks)
        amplitudes:
            (n_samples, n_peaks)

        Returns
        -------
        signals, x_values, positions, amplitudes
            The same arrays converted to float ndarrays (no copies unless needed).

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 2:
            raise ValueError(
                "signals must be 2D with shape (n_samples, sequence_length)"
            )

        x_values = np.asarray(x_values, dtype=float)
        if x_values.ndim != 1:
            raise ValueError("x_values must be 1D with shape (sequence_length,)")

        positions = np.asarray(positions, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)

        if positions.ndim != 2 or amplitudes.ndim != 2:
            raise ValueError(
                "positions and amplitudes must both be 2D with shape (n_samples, n_peaks)"
            )

        if positions.shape != amplitudes.shape:
            raise ValueError("positions and amplitudes must have the same shape")

        n_samples, sequence_length = signals.shape
        if x_values.shape[0] != sequence_length:
            raise ValueError(
                "x_values length must match signals second dimension (sequence_length)"
            )

        if positions.shape[0] != n_samples:
            raise ValueError(
                "positions and amplitudes must have n_samples rows, matching signals"
            )

        return signals, x_values, positions, amplitudes

    def _gaussian_pulse(
        self,
        delta: np.ndarray,
        width: float,
        width_definition: Literal["fwhm", "sigma"],
    ) -> np.ndarray:
        """
        Evaluate a Gaussian pulse on a delta grid.

        Parameters
        ----------
        delta
            x minus center, with shape (..., sequence_length) or any broadcastable shape.
        width
            Width parameter. Interpreted by width_definition.
        width_definition
            "fwhm" interprets `width` as full width at half maximum.
            "sigma" interprets `width` as standard deviation.

        Returns
        -------
        np.ndarray
            Gaussian values with the same shape as `delta`.

        Raises
        ------
        ValueError
            If width is non positive or width_definition is invalid.
        """
        width = float(width)

        if width_definition.lower() == "fwhm":
            if width <= 0.0:
                raise ValueError("Gaussian FWHM must be positive")
            # Direct FWHM form avoids conversion drift: exp(-4 ln(2) (delta / fwhm)^2)
            return np.exp(-4.0 * np.log(2.0) * (delta / width) ** 2)

        if width_definition.lower() == "sigma":
            sigma = width
            if sigma <= 0.0:
                raise ValueError("Gaussian sigma must be positive")
            return np.exp(-0.5 * (delta / sigma) ** 2)

        raise ValueError("For gaussian, width_definition must be 'fwhm' or 'sigma'")

    def _lorentzian_pulse(
        self,
        delta: np.ndarray,
        width: float,
        width_definition: Literal["fwhm", "gamma"],
    ) -> np.ndarray:
        """
        Evaluate a Lorentzian pulse on a delta grid.

        Parameters
        ----------
        delta
            x minus center, with shape (..., sequence_length) or any broadcastable shape.
        width
            Width parameter. Interpreted by width_definition.
        width_definition
            "fwhm" interprets `width` as full width at half maximum (FWHM = 2 * gamma).
            "gamma" interprets `width` as the half width at half maximum.

        Returns
        -------
        np.ndarray
            Lorentzian values with the same shape as `delta`.

        Raises
        ------
        ValueError
            If width is non positive or width_definition is invalid.
        """
        width = float(width)

        if width_definition.lower() == "fwhm":
            if width <= 0.0:
                raise ValueError("Lorentzian FWHM must be positive")
            gamma = width / 2.0
            return 1.0 / (1.0 + (delta / gamma) ** 2)

        if width_definition.lower() == "gamma":
            gamma = width
            if gamma <= 0.0:
                raise ValueError("Lorentzian gamma must be positive")
            return 1.0 / (1.0 + (delta / gamma) ** 2)

        raise ValueError("For lorentzian, width_definition must be 'fwhm' or 'gamma'")

    def get_reference_pulse_trace(
        self,
        width: float,
        amplitude: float | None = None,
        profile: Profile = "gaussian",
        width_definition: WidthDefinition = "fwhm",
        normalize_peak_to_one: bool = False,
        *,
        normalization_mode: NormalizationMode = "analytic",
        sampled_normalization_epsilon: float = 1e-12,
    ) -> np.ndarray:
        """
        Build an idealized reference trace by summing analytic pulses at known peak positions.

        For each valid peak (finite position and finite nonzero amplitude), an analytic pulse
        is evaluated on `self.x_values` and summed into a clean reference trace for each sample.

        Parameters
        ----------
        width
            Pulse width parameter in the same units as `self.x_values`.
        amplitude
            If None, use `self.amplitudes` per peak. If provided, overrides all valid peak amplitudes.
        profile
            "gaussian" or "lorentzian".
        width_definition
            For gaussian: "fwhm" or "sigma".
            For lorentzian: "fwhm" or "gamma".
        normalize_peak_to_one
            If True, normalize the pulse so each peak has maximum 1 before amplitude scaling.
        normalization_mode
            "analytic" assumes the analytic maximum is 1 at delta=0.
            "sampled" normalizes by the sampled maximum along x, robust to sub sample peak centers.
        sampled_normalization_epsilon
            Floor to avoid division by zero in sampled normalization.

        Returns
        -------
        np.ndarray
            Reference pulse trace of shape (n_samples, sequence_length).
        """
        _, x_values, positions, amplitudes = self._validate_reference_pulse_inputs(
            signals=self.signals,
            x_values=self.x_values,
            positions=self.positions,
            amplitudes=self.amplitudes,
        )

        valid_pos = np.isfinite(positions)
        valid_amp = np.isfinite(amplitudes) & (amplitudes != 0.0)
        valid = valid_pos & valid_amp  # (n_samples, n_peaks)

        if amplitude is None:
            amplitude_per_peak = amplitudes
        else:
            amplitude_per_peak = np.full_like(amplitudes, float(amplitude))

        amplitude_per_peak = np.where(valid, amplitude_per_peak, 0.0)

        # NaN safe positions to keep delta finite everywhere
        positions_safe = np.where(valid, positions, 0.0)

        # delta shape: (n_samples, n_peaks, sequence_length)
        delta = x_values[None, None, :] - positions_safe[:, :, None]

        if profile.lower() == "gaussian":
            pulse = self._gaussian_pulse(
                delta=delta, width=width, width_definition=width_definition
            )  # type: ignore[arg-type]
        elif profile.lower() == "lorentzian":
            pulse = self._lorentzian_pulse(
                delta=delta, width=width, width_definition=width_definition
            )  # type: ignore[arg-type]
        else:
            raise ValueError("profile must be 'gaussian' or 'lorentzian'")

        if normalize_peak_to_one:
            mode = normalization_mode.lower()
            if mode not in ("analytic", "sampled"):
                raise ValueError("normalization_mode must be 'analytic' or 'sampled'")

            if mode == "sampled":
                sampled_max = np.max(pulse, axis=2)  # (n_samples, n_peaks)
                sampled_max = np.where(valid, sampled_max, 1.0)
                sampled_max = np.maximum(
                    sampled_max, float(sampled_normalization_epsilon)
                )
                pulse = pulse / sampled_max[:, :, None]

        # Ensure invalid peaks contribute exactly 0.
        pulse = np.where(valid[:, :, None], pulse, 0.0)

        reference_pulse_trace = np.sum(amplitude_per_peak[:, :, None] * pulse, axis=1)
        return reference_pulse_trace
