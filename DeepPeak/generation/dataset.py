from typing import Any, Literal
import numpy as np
import matplotlib.pyplot as plt
from MPSPlots import helper

from .. import processing


Profile = Literal["gaussian", "lorentzian"]
WidthDefinition = Literal["fwhm", "sigma", "gamma"]
NormalizationMode = Literal["analytic", "sampled"]
AmplitudeThresholdReference = Literal["absolute", "sample_max_amplitude", "signal_max"]


class DataSet:
    """Container for sample-aligned one-dimensional signal data.

    ``DataSet`` stores signals with shape ``(n_samples, sequence_length)`` and
    any additional sample-aligned arrays, such as clean traces, labels, peak
    positions, or amplitudes. Common fields are validated at construction time.
    Extra keyword arguments are retained as public attributes for domain-specific
    metadata.

    Parameters
    ----------
    n_samples : int, optional
        Number of signal traces. Inferred from ``signals`` when omitted.
    sequence_length : int, optional
        Number of samples per trace. Inferred from ``signals`` or ``x_values``
        when omitted.
    seed : int, optional
        Seed associated with the dataset generation process. This value is
        recorded for provenance; it is not used to seed operations on the
        dataset itself.
    metadata : dict, optional
        Additional provenance or experiment metadata. A shallow copy is stored.
    **kwargs : object
        Additional dataset fields. ``signals`` is expected to have shape
        ``(n_samples, sequence_length)``. Arrays whose first dimension equals
        ``n_samples`` are treated as sample-aligned by methods such as
        :meth:`shuffle` and :meth:`train_test_split`.

    Raises
    ------
    ValueError
        If signals, coordinates, labels, or clean traces have incompatible
        shapes.

    Examples
    --------
    >>> dataset = DataSet(
    ...     signals=np.zeros((8, 128)),
    ...     labels=np.zeros((8, 128)),
    ...     x_values=np.arange(128),
    ... )
    >>> dataset.to_model_inputs().shape
    (8, 128, 1)
    """

    list_of_attributes = None

    def __init__(
        self,
        *,
        n_samples: int | None = None,
        sequence_length: int | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize the dataset and validate its common array dimensions.

        Parameters
        ----------
        n_samples, sequence_length, seed, metadata, **kwargs
            See the :class:`DataSet` class documentation. In particular,
            ``signals``, ``labels``, and ``clean_signals`` should be supplied
            as sample-aligned arrays when available.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If supplied arrays do not satisfy the dataset shape contract.
        """
        self.list_of_attributes = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.list_of_attributes.append(key)

        inferred_n_samples = n_samples
        inferred_sequence_length = sequence_length

        signals = getattr(self, "signals", None)
        if isinstance(signals, np.ndarray) and signals.ndim >= 2:
            if inferred_n_samples is None:
                inferred_n_samples = int(signals.shape[0])
            if inferred_sequence_length is None:
                inferred_sequence_length = int(signals.shape[-1])

        x_values = getattr(self, "x_values", None)
        if (
            inferred_sequence_length is None
            and isinstance(x_values, np.ndarray)
            and x_values.ndim == 1
        ):
            inferred_sequence_length = int(x_values.size)

        if inferred_n_samples is not None:
            self.n_samples = int(inferred_n_samples)
            self.list_of_attributes.append("n_samples")
        if inferred_sequence_length is not None:
            self.sequence_length = int(inferred_sequence_length)
            self.list_of_attributes.append("sequence_length")

        self.seed = seed
        self.metadata = dict(metadata or {})
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        """Validate the common sample and time-axis contract.

        Raises
        ------
        ValueError
            If the signal matrix, coordinate vector, clean traces, or labels
            do not match the dataset dimensions.
        """
        signals = getattr(self, "signals", None)
        if signals is None:
            return
        signals = np.asarray(signals)
        if signals.ndim != 2:
            raise ValueError(
                "signals must have shape (n_samples, sequence_length); "
                f"received shape {signals.shape}."
            )
        if (
            signals.shape[0] != self.n_samples
            or signals.shape[1] != self.sequence_length
        ):
            raise ValueError(
                "signals shape does not match n_samples and sequence_length."
            )

        x_values = getattr(self, "x_values", None)
        if x_values is not None:
            x_values = np.asarray(x_values)
            if x_values.ndim != 1 or x_values.size != self.sequence_length:
                raise ValueError(
                    "x_values must have shape (sequence_length,) and match signals."
                )

        for name in ("clean_signals", "labels"):
            values = getattr(self, name, None)
            if values is not None:
                values = np.asarray(values)
                if (
                    values.shape[0] != self.n_samples
                    or values.shape[-1] != self.sequence_length
                ):
                    raise ValueError(
                        f"{name} must be sample-aligned with shape "
                        "(n_samples, sequence_length)."
                    )

    def to_model_inputs(
        self, *, normalization: str = "none", add_channel: bool = True
    ) -> np.ndarray:
        """Convert signals to the channel-last shape expected by neural models.

        Parameters
        ----------
        normalization : str, default="none"
            Normalization mode passed to :meth:`get_normalized_signal`.
            Supported modes include ``"none"``, ``"zscore"``, ``"minmax"``,
            ``"l1"``, ``"l2"``, ``"robust_zscore"``, and ``"maxabs"``.
        add_channel : bool, default=True
            If true, append a singleton channel dimension and return shape
            ``(n_samples, sequence_length, 1)``. If false, return the two-
            dimensional shape ``(n_samples, sequence_length)``.

        Returns
        -------
        numpy.ndarray
            Floating-point copy of the signal data, optionally normalized.

        Raises
        ------
        AttributeError
            If the dataset has no ``signals`` field.
        ValueError
            If signals are not a two-dimensional matrix.

        Examples
        --------
        >>> inputs = dataset.to_model_inputs(normalization="zscore")
        >>> inputs.shape
        (8, 128, 1)
        """
        values = (
            self.get_normalized_signal(normalization)
            if normalization != "none"
            else np.asarray(self.signals, dtype=float).copy()
        )
        if values.ndim != 2:
            raise ValueError("Model inputs require signals with two dimensions.")
        return values[..., None] if add_channel else values

    def targets(
        self,
        target: str = "auto",
        *,
        add_channel: bool = True,
        width: float | None = None,
        profile: Profile = "gaussian",
        width_definition: WidthDefinition = "fwhm",
        normalize_peak_to_one: bool = False,
        normalization_mode: NormalizationMode = "analytic",
        **reference_kwargs: Any,
    ) -> np.ndarray:
        """Return model targets from stored arrays or shaped references.

        Parameters
        ----------
        target : {"auto", "labels", "clean_signals", "reference"}, default="auto"
            Target field to return. ``"labels"`` contains pulse-location
            targets, typically binary values with one or more marked samples
            per trace. ``"clean_signals"`` contains reconstructed clean pulse
            traces. ``"reference"`` builds a shaped target from the stored
            ``positions`` and ``amplitudes`` using
            :meth:`get_reference_pulse_trace`. ``"auto"`` selects ``labels``
            when present and otherwise falls back to ``clean_signals``.
        add_channel : bool, default=True
            If true, append a singleton channel dimension and return shape
            ``(n_samples, sequence_length, 1)``. If false, return shape
            ``(n_samples, sequence_length)``.
        width : float, optional
            Width of a ``"reference"`` target, in the same units as
            ``x_values``. Required when ``target="reference"``.
        profile : {"gaussian", "lorentzian"}, default="gaussian"
            Shape used for a ``"reference"`` target.
        width_definition : {"fwhm", "sigma", "gamma"}, default="fwhm"
            Meaning of ``width``. Gaussian targets accept ``"fwhm"`` or
            ``"sigma"``; Lorentzian targets accept ``"fwhm"`` or ``"gamma"``.
        normalize_peak_to_one : bool, default=False
            Normalize each generated reference pulse to unit height before
            applying its amplitude.
        normalization_mode : {"analytic", "sampled"}, default="analytic"
            Normalization strategy used for shaped reference targets.
        **reference_kwargs
            Additional arguments forwarded to
            :meth:`get_reference_pulse_trace`, such as ``amplitude``,
            ``min_peak_distance``, or ``max_peak_overlap``.

        Returns
        -------
        numpy.ndarray
            Floating-point target array selected from the dataset.

        Raises
        ------
        ValueError
            If ``target`` is invalid, a requested field is absent, the target
            shape is invalid, or reference parameters are incomplete.

        Examples
        --------
        >>> labels = dataset.targets("labels")
        >>> labels.shape
        (8, 128, 1)
        >>> reference = dataset.targets("reference", width=6.0)
        >>> reference.shape
        (8, 128, 1)
        """
        if target == "auto":
            target = "labels" if hasattr(self, "labels") else "clean_signals"
        if target == "reference":
            if width is None:
                raise ValueError("width is required when target='reference'.")
            values = self.get_reference_pulse_trace(
                width=width,
                profile=profile,
                width_definition=width_definition,
                normalize_peak_to_one=normalize_peak_to_one,
                normalization_mode=normalization_mode,
                **reference_kwargs,
            )
        else:
            if target not in {"labels", "clean_signals"}:
                raise ValueError(
                    "target must be 'auto', 'labels', 'clean_signals', or 'reference'."
                )
            if not hasattr(self, target):
                raise ValueError(f"Dataset does not contain {target!r} targets.")
            values = np.asarray(getattr(self, target), dtype=float)
        if values.ndim != 2 or values.shape != (self.n_samples, self.sequence_length):
            raise ValueError(f"{target} must have shape (n_samples, sequence_length).")
        return values[..., None] if add_channel else values

    def train_test_split(
        self,
        test_size: float | int = 0.2,
        *,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> tuple["DataSet", "DataSet"]:
        """Split all sample-aligned fields into train and test datasets.

        Parameters
        ----------
        test_size : float or int, default=0.2
            If a float, the fraction of samples assigned to the test set. If
            an integer, the exact number of test samples.
        seed : int, optional
            Seed for the permutation used when ``shuffle=True``.
        shuffle : bool, default=True
            Whether to permute samples before splitting. Set false to preserve
            the original order and use the final samples as the test set.

        Returns
        -------
        train : DataSet
            Training subset containing every sample-aligned field.
        test : DataSet
            Test subset containing every sample-aligned field.

        Raises
        ------
        TypeError
            If ``test_size`` is neither an integer nor a float.
        ValueError
            If ``test_size`` does not define a non-empty test set smaller than
            the complete dataset.

        Notes
        -----
        The original dataset's ``seed`` and metadata are copied to both
        subsets. The split seed is added to each subset's metadata under
        ``"split_seed"``.
        """
        if isinstance(test_size, float):
            if not 0.0 < test_size < 1.0:
                raise ValueError("A float test_size must be between 0 and 1.")
            n_test = max(1, int(np.ceil(self.n_samples * test_size)))
        elif isinstance(test_size, (int, np.integer)):
            n_test = int(test_size)
            if not 0 < n_test < self.n_samples:
                raise ValueError(
                    "An integer test_size must be between 1 and n_samples - 1."
                )
        else:
            raise TypeError("test_size must be a float or integer.")

        indices = np.arange(self.n_samples)
        if shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        train_indices, test_indices = indices[:-n_test], indices[-n_test:]

        def subset(selected: np.ndarray) -> "DataSet":
            values = {}
            for name in self.list_of_attributes:
                value = getattr(self, name)
                if (
                    isinstance(value, np.ndarray)
                    and value.ndim >= 1
                    and value.shape[0] == self.n_samples
                ):
                    values[name] = value[selected].copy()
                elif name not in {"n_samples", "sequence_length"}:
                    values[name] = self._copy_attribute_value(value)
            return DataSet(
                **values,
                seed=self.seed,
                metadata={**self.metadata, "split_seed": seed},
            )

        return subset(train_indices), subset(test_indices)

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ", ".join(f"{key}" for key in self.list_of_attributes)
        return f"{class_name}({attributes})"

    def shuffle(self, seed: int | None = None, inplace: bool = False) -> "DataSet":
        """Shuffle all sample-aligned fields with one shared permutation.

        Parameters
        ----------
        seed : int, optional
            Seed used to create the sample permutation.
        inplace : bool, default=False
            If true, mutate and return this dataset. If false, return a new
            dataset and leave the original unchanged.

        Returns
        -------
        DataSet
            Shuffled dataset. All arrays whose first dimension equals
            ``n_samples`` retain their alignment.

        Raises
        ------
        AttributeError
            If the number of samples cannot be inferred.
        """

        n_samples = self._resolve_n_samples()
        permutation = np.random.default_rng(seed).permutation(n_samples)

        if inplace:
            target = self
        else:
            copied_attributes = {
                key: self._copy_attribute_value(getattr(self, key))
                for key in self.list_of_attributes
                if hasattr(self, key)
            }
            target = DataSet(**copied_attributes)

        for key in target.list_of_attributes:
            value = getattr(target, key)
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == n_samples
            ):
                setattr(target, key, value[permutation].copy())

        return target

    def _resolve_n_samples(self) -> int:
        if hasattr(self, "n_samples"):
            return int(self.n_samples)

        signals = getattr(self, "signals", None)
        if isinstance(signals, np.ndarray) and signals.ndim >= 1:
            return int(signals.shape[0])

        raise AttributeError(
            "Cannot determine n_samples for shuffle(); provide n_samples in the DataSet constructor or include a sample-aligned signals array."
        )

    @staticmethod
    def _copy_attribute_value(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def get_normalized_signal(self, normalization: str = "zscore"):
        """Normalize each signal trace independently.

        Parameters
        ----------
        normalization : str, default="zscore"
            Normalization strategy.

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

        Returns
        -------
        numpy.ndarray
            Normalized signal array with shape
            ``(n_samples, sequence_length)``.

        Raises
        ------
        ValueError
            If ``normalization`` is not a supported mode or the signal array
            has an incompatible shape.

        Examples
        --------
        >>> normalized = dataset.get_normalized_signal("zscore")
        >>> normalized.shape
        (8, 128)
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
        reference_pulse_trace: np.ndarray | None = None,
        reference_pulse_scale: str = "auto",  # "auto", "signal_max", "none"
    ):
        """
        Plot observed signals and optional clean reference traces.

        Parameters
        ----------
        number_of_samples : int, default=3
            Number of signals to visualize.
        randomize_signal : bool, default=False
            If True, randomly select signals from the dataset instead of taking
            the first N samples.
        number_of_columns : int, default=1
            Number of columns in the subplot grid.
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
            indices = np.random.default_rng().choice(
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

        for panel_index, (plot_index, ax) in enumerate(zip(indices, axes.flatten())):
            signal = self.signals[plot_index]

            ax.plot(self.x_values, signal, label="signal", color="black")

            handles, labels = ax.get_legend_handles_labels()

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

            if panel_index == 0:
                by_label = {}
                for handle, label in zip(handles, labels):
                    if label and not label.startswith("_") and label not in by_label:
                        by_label[label] = handle
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
        min_peak_distance: float | None = None,
        min_peak_amplitude: float | None = None,
        amplitude_threshold_reference: AmplitudeThresholdReference = "absolute",
        max_peak_overlap: float | None = None,
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
        min_peak_distance
            If provided, suppress reference pulses for peaks that are closer than this
            distance in `self.x_values` units to any other valid peak in the same sample.
            Both peaks in each too-close pair are suppressed.
        min_peak_amplitude
            If provided, suppress reference pulses for peaks whose absolute amplitude is
            below the computed threshold.
        amplitude_threshold_reference
            Reference scale used with `min_peak_amplitude`:
            "absolute" compares against the raw peak amplitude,
            "sample_max_amplitude" compares against the largest absolute peak amplitude
            in the same sample, and
            "signal_max" compares against the largest absolute signal value in the same
            sample.
        max_peak_overlap
            If provided, suppress reference pulses whose neighboring reference pulses
            contribute more than this fraction of the peak's own reference height at
            that peak center. Overlap is computed from the selected pulse profile,
            width, and reference amplitudes.

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

        if min_peak_distance is not None:
            min_peak_distance = float(min_peak_distance)
            if min_peak_distance < 0.0:
                raise ValueError("min_peak_distance must be non-negative")

            distance = np.abs(positions[:, :, None] - positions[:, None, :])
            close_pairs = distance < min_peak_distance
            diagonal = np.eye(close_pairs.shape[1], dtype=bool)[None, :, :]
            close_pairs &= ~diagonal
            close_pairs &= valid[:, :, None] & valid[:, None, :]

            crowded_peaks = np.any(close_pairs, axis=2)
            valid &= ~crowded_peaks

        if min_peak_amplitude is not None:
            min_peak_amplitude = float(min_peak_amplitude)
            if min_peak_amplitude < 0.0:
                raise ValueError("min_peak_amplitude must be non-negative")

            abs_amplitudes = np.abs(amplitudes)
            threshold_mode = amplitude_threshold_reference.lower()

            if threshold_mode == "absolute":
                threshold = np.full_like(abs_amplitudes, min_peak_amplitude)
            elif threshold_mode == "sample_max_amplitude":
                sample_max_amplitude = np.max(
                    np.where(valid, abs_amplitudes, 0.0), axis=1, keepdims=True
                )
                threshold = min_peak_amplitude * sample_max_amplitude
            elif threshold_mode == "signal_max":
                signal_max = np.max(np.abs(self.signals), axis=1, keepdims=True)
                threshold = min_peak_amplitude * signal_max
            else:
                raise ValueError(
                    "amplitude_threshold_reference must be 'absolute', "
                    "'sample_max_amplitude', or 'signal_max'"
                )

            valid &= abs_amplitudes >= threshold

        if amplitude is None:
            amplitude_per_peak = amplitudes
        else:
            amplitude_per_peak = np.full_like(amplitudes, float(amplitude))

        if max_peak_overlap is not None:
            max_peak_overlap = float(max_peak_overlap)
            if max_peak_overlap < 0.0:
                raise ValueError("max_peak_overlap must be non-negative")

            overlap_valid = valid.copy()
            positions_overlap = np.where(overlap_valid, positions, 0.0)
            center_delta = positions_overlap[:, :, None] - positions_overlap[:, None, :]

            if profile.lower() == "gaussian":
                pairwise_pulse = self._gaussian_pulse(
                    delta=center_delta,
                    width=width,
                    width_definition=width_definition,
                )  # type: ignore[arg-type]
            elif profile.lower() == "lorentzian":
                pairwise_pulse = self._lorentzian_pulse(
                    delta=center_delta,
                    width=width,
                    width_definition=width_definition,
                )  # type: ignore[arg-type]
            else:
                raise ValueError("profile must be 'gaussian' or 'lorentzian'")

            pairwise_valid = overlap_valid[:, :, None] & overlap_valid[:, None, :]
            pairwise_pulse = np.where(pairwise_valid, pairwise_pulse, 0.0)

            diagonal = np.eye(pairwise_pulse.shape[1], dtype=bool)[None, :, :]
            pairwise_pulse = np.where(diagonal, 0.0, pairwise_pulse)

            reference_amplitude = np.where(overlap_valid, amplitude_per_peak, 0.0)
            abs_reference_amplitude = np.abs(reference_amplitude)
            neighbor_height = np.sum(
                abs_reference_amplitude[:, None, :] * pairwise_pulse,
                axis=2,
            )
            own_height = np.maximum(abs_reference_amplitude, 1e-12)
            overlap_fraction = neighbor_height / own_height

            valid &= overlap_fraction <= max_peak_overlap

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
