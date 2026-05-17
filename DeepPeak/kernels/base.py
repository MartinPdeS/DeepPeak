from abc import ABC
from typing import Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from DeepPeak.peak_count import PeakCount, PoissonCount, UniformCount


RangeValue = Tuple[float, float] | float | Tuple[int, int] | int
FloatRange = Tuple[float, float]


class BaseKernel(ABC):
    """Base class for synthetic pulse kernels.

    Concrete subclasses sample per-peak parameters such as amplitude, position,
    and optionally width, then evaluate one pulse component per sampled peak on a
    shared one-dimensional ``x_values`` grid.

    Notes
    -----
    Subclasses are expected to implement ``evaluate()`` and may expose sampled
    state such as ``amplitudes``, ``positions``, and ``widths`` after evaluation.
    """

    def __repr__(self) -> str:
        field_names = tuple(getattr(self, "__dataclass_fields__", {}).keys())
        parameters = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in field_names
        )
        return f"{self.__class__.__name__}({parameters})"

    @staticmethod
    def _ensure_tuple(value: RangeValue) -> Tuple[float, float] | Tuple[int, int]:
        """If value is a scalar, return ``(value, value)``; otherwise validate a pair."""
        if isinstance(value, (int, float)):
            return (value, value)  # type: ignore[return-value]
        if len(value) != 2:
            raise ValueError("Kernel parameter ranges must contain exactly two values.")
        return value  # type: ignore[return-value]

    @classmethod
    def _normalize_range(
        cls,
        name: str,
        value: RangeValue,
        *,
        minimum: float | None = None,
        inclusive_minimum: bool = True,
    ) -> FloatRange:
        low_raw, high_raw = cls._ensure_tuple(value)
        low = float(low_raw)
        high = float(high_raw)

        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"{name} must contain only finite values.")
        if high < low:
            raise ValueError(f"{name} must satisfy low <= high.")
        if minimum is not None:
            if inclusive_minimum:
                valid = low >= minimum and high >= minimum
            else:
                valid = low > minimum and high > minimum
            if not valid:
                comparator = ">=" if inclusive_minimum else ">"
                raise ValueError(f"{name} values must be {comparator} {minimum}.")

        return (low, high)

    def _initialize_common_ranges(self, *, has_width: bool) -> None:
        self._amplitude = self._normalize_range("amplitude", self.amplitude)
        self._position = self._normalize_range("position", self.position)
        if has_width:
            self._width = self._normalize_range(
                "width",
                self.width,
                minimum=0.0,
                inclusive_minimum=False,
            )

    @staticmethod
    def _validate_x_values(x_values: NDArray) -> NDArray[np.float64]:
        x_values = np.asarray(x_values, dtype=float)
        if x_values.ndim != 1:
            raise ValueError("x_values must be a one-dimensional array.")
        return x_values

    @staticmethod
    def _validate_n_samples(n_samples: int) -> int:
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        return n_samples

    def _sample_uniform(
        self, bounds: FloatRange, size: tuple[int, int]
    ) -> NDArray[np.float64]:
        return np.random.uniform(bounds[0], bounds[1], size=size)

    def _default_plot_x_values(self, n_points: int) -> NDArray[np.float64]:
        bounds = self._plot_bounds()
        if bounds is not None:
            left, right = bounds
            return np.linspace(float(left), float(right), int(n_points), dtype=float)

        if hasattr(self, "_position"):
            position_low, position_high = self._position
        else:
            position_low, position_high = (0.0, 1.0)

        width_margin = 1.0
        if hasattr(self, "_width"):
            _, width_high = self._width
            width_margin = max(4.0 * float(width_high), 1.0)
        elif hasattr(self, "kernel"):
            kernel = np.asarray(getattr(self, "kernel"), dtype=float)
            width_margin = max(0.5 * float(kernel.size), 1.0)
        else:
            width_margin = max(float(position_high - position_low), 1.0)

        left = float(position_low) - width_margin
        right = float(position_high) + width_margin
        if right <= left:
            center = float(position_low)
            left = center - width_margin
            right = center + width_margin

        return np.linspace(left, right, int(n_points), dtype=float)

    def _plot_bounds(self) -> tuple[float, float] | None:
        return None

    def plot(
        self,
        x_values: NDArray | None = None,
        *,
        ax: Axes | None = None,
        seed: int | None = 0,
        n_points: int = 512,
        **plot_kwargs,
    ) -> Axes:
        """Plot one representative sampled peak for this kernel.

        Parameters
        ----------
        x_values : ndarray of shape (M,), optional
            Optional x-grid for the plot. If omitted, a reasonable range is inferred
            from the kernel parameters.
        ax : matplotlib.axes.Axes, optional
            Optional Matplotlib axis to draw on. If omitted, a new figure and axis
            are created.
        seed : int or None, default=0
            Optional random seed used while sampling representative parameters from
            ranged kernel definitions. Set to ``None`` to keep the current RNG state.
        n_points : int, default=512
            Number of points used when ``x_values`` is omitted.
        **plot_kwargs
            Forwarded to ``Axes.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axis used for plotting.
        """

        if x_values is None:
            x_values = self._default_plot_x_values(n_points=n_points)
        else:
            x_values = self._validate_x_values(x_values)

        if ax is None:
            _, ax = plt.subplots()

        rng_state = None
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)

        try:
            components = self.evaluate(
                x_values=x_values,
                n_samples=1,
                n_peaks=(1, 1),
                categorical_peak_count=False,
                peak_count=UniformCount(bounds=(1, 1)),
            )
        finally:
            if rng_state is not None:
                np.random.set_state(rng_state)

        signal = np.nansum(components, axis=1)[0]
        ax.plot(x_values, signal, **plot_kwargs)
        return ax

    def _prepare_common_state(
        self,
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        *,
        categorical_peak_count: bool,
        peak_count: PeakCount | None,
        peak_count_distribution: str,
        peak_count_rate: Optional[Union[float, Tuple[float, float]]],
        has_width: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None,
        NDArray[np.bool_],
        int,
    ]:
        x_values = self._validate_x_values(x_values)
        n_samples = self._validate_n_samples(n_samples)

        self.num_peaks, _, max_peaks = self._sample_num_peaks(
            n_samples=n_samples,
            n_peaks=n_peaks,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
        )

        shape = (n_samples, max_peaks)
        self.amplitudes = self._sample_uniform(self._amplitude, size=shape)
        self.positions = self._sample_uniform(self._position, size=shape)
        self.positions_for_labels = self.positions.copy()

        widths = None
        if has_width:
            self.widths = self._sample_uniform(self._width, size=shape)
            widths = self.widths[..., np.newaxis]

        active_mask = np.arange(max_peaks) < self.num_peaks[:, None]
        self.amplitudes[~active_mask] = np.nan
        self.positions[~active_mask] = np.nan
        if has_width:
            self.widths[~active_mask] = np.nan

        if categorical_peak_count:
            self.num_peaks = self._one_hot_numpy(
                self.num_peaks, max_peaks + 1, dtype=np.float32
            )

        return (
            x_values.reshape(1, 1, -1),
            self.amplitudes[..., np.newaxis],
            self.positions[..., np.newaxis],
            widths,
            active_mask,
            max_peaks,
        )

    def _state_dict(self, *names: str) -> dict[str, Any]:
        missing = [name for name in names if not hasattr(self, name)]
        if missing:
            missing_names = ", ".join(missing)
            raise AttributeError(
                f"{self.__class__.__name__} has no sampled state for {missing_names}; call evaluate() first."
            )
        return {name: getattr(self, name) for name in names}

    @staticmethod
    def _one_hot_numpy(indices: NDArray, num_classes: int, dtype=np.float32) -> NDArray:
        """
        Fast, pure-NumPy one-hot encoder.

        Parameters
        ----------
        indices: NDArray
            The indices to one-hot encode.
        num_classes: int
            The number of classes for the one-hot encoding.
        dtype: type
            The data type of the output array.
        """
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size == 0:
            return np.zeros((0, num_classes), dtype=dtype)
        if (indices < 0).any() or (indices >= num_classes).any():
            raise ValueError("indices out of range for the specified num_classes")

        out = np.zeros((indices.shape[0], num_classes), dtype=dtype)
        out[np.arange(indices.shape[0]), indices] = 1
        return out

    @staticmethod
    def _sample_num_peaks(
        n_samples: int,
        n_peaks: tuple,
        peak_count: PeakCount | None = None,
        peak_count_distribution: str = "uniform",
        peak_count_rate: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> tuple[NDArray[np.int64], int, int]:
        """Sample per-trace peak counts within inclusive bounds."""

        min_peaks, max_peaks = int(n_peaks[0]), int(n_peaks[1])
        if min_peaks < 0:
            raise ValueError("n_peaks minimum must be non-negative.")
        if max_peaks < min_peaks:
            raise ValueError("n_peaks maximum must be >= minimum.")

        if peak_count is not None:
            counts = peak_count.sample(n_samples)
            return np.asarray(counts, dtype=np.int64), min_peaks, max_peaks

        distribution = str(peak_count_distribution).strip().lower()
        if distribution == "uniform":
            counts = UniformCount(bounds=(min_peaks, max_peaks)).sample(n_samples)
        elif distribution == "poisson":
            if peak_count_rate is None:
                raise ValueError(
                    "peak_count_rate must be provided when peak_count_distribution='poisson'."
                )
            counts = PoissonCount(
                bounds=(min_peaks, max_peaks),
                rate=peak_count_rate,
            ).sample(n_samples)
        else:
            raise ValueError(
                "peak_count_distribution must be either 'uniform' or 'poisson'."
            )

        return np.asarray(counts, dtype=np.int64), min_peaks, max_peaks
