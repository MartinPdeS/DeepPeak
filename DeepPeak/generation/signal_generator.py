"""Synthetic signal dataset generation utilities."""

from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

from .dataset import DataSet
from .kernels import BaseKernel
from .noises import BaseNoise, GaussianNoise, NonstationaryGaussianNoise
from .peak_count import PeakCount


class SignalGenerator:
    """Generate synthetic 1D peak signals from kernel parameter samplers.

    Key features:
    - Supports all kernel classes derived from ``BaseKernel``
    - Returns labels marking discrete peak locations
    - Optional Gaussian noise
    - Optional instrument response, baseline offset, saturation, and quantization
    - Optional NumPy-based one-hot encoding for the number of peaks
    - Clean pulse traces are retained alongside noisy observations for
      reconstruction-model training
    """

    def __init__(
        self, sequence_length: int, x_values: Optional[np.ndarray] = None
    ) -> None:
        self.sequence_length = sequence_length
        self.x_values = (
            x_values if x_values is not None else np.arange(self.sequence_length)
        )

        assert self.x_values.ndim == 1 and len(self.x_values) == self.sequence_length, (
            "x_values must be a 1D array of length sequence_length "
            f"[{self.sequence_length}] got shape {self.x_values.shape}"
        )
        self._dataset_parts: list[DataSet] = []

    def add_to_set(self, **generate_kwargs: Any) -> DataSet:
        """Generate one batch and append it to the internal dataset buffer."""

        dataset = self.generate(**generate_kwargs)
        self._dataset_parts.append(dataset)
        return dataset

    def dataset(self) -> DataSet:
        """Return one dataset built by concatenating all buffered batches."""

        if len(self._dataset_parts) == 0:
            raise RuntimeError(
                "No generated batches are buffered. Call add_to_set() first."
            )

        return self._merge_datasets(self._dataset_parts)

    def clear(self) -> None:
        """Drop all buffered generated batches."""

        self._dataset_parts.clear()

    def generate(
        self,
        *,
        n_samples: int,
        kernel: BaseKernel,
        peak_count: PeakCount,
        seed: Optional[int] = None,
        noise_std: Optional[Union[float, Tuple[float, float]]] = None,
        noise: Optional[BaseNoise] = None,
        drift: Optional[Union[float, Tuple[float, float]]] = None,
        baseline_level: Optional[Union[float, Tuple[float, float]]] = None,
        instrument_response: Optional[np.ndarray] = None,
        saturation: Optional[Union[float, Tuple[float, float]]] = None,
        quantization_step: Optional[float] = None,
        noise_profile: Literal["constant", "linear"] = "constant",
        noise_end_scale: Union[float, Tuple[float, float]] = 1.0,
        categorical_peak_count: bool = False,
        shift_min_to_zero: bool = False,
        minimum_level: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> DataSet:
        """Generate a dataset of parametric peak signals.

        Parameters
        ----------
        peak_count : PeakCount
            Object that samples the number of peaks per trace, e.g.
            ``UniformCount(bounds=(1, 4))`` or
            ``PoissonCount(bounds=(1, 4), rate=2.0)``.
        shift_min_to_zero : bool, default=False
            If True, shift each generated trace upward by ``-min(trace)`` when
            its minimum is negative so the final per-trace minimum is exactly 0.
        minimum_level : float or tuple of float, optional
            Desired per-trace minimum after generation. A scalar applies the same
            minimum to every trace. A ``(low, high)`` tuple samples one minimum
            uniformly per trace. Cannot be combined with ``shift_min_to_zero``.
        baseline_level : float or tuple of float, optional
            Constant additive baseline sampled independently for each trace.
        instrument_response : 1D array, optional
            Finite impulse response applied to the clean signal before noise.
        saturation : float or tuple of float, optional
            Upper clipping limit, or explicit ``(lower, upper)`` clipping bounds.
        quantization_step : float, optional
            Quantization interval applied to the observed signal.
        noise_profile : {"constant", "linear"}, default="constant"
            Whether ``noise_std`` is stationary or changes linearly across time.
        noise_end_scale : float or tuple of float, default=1.0
            End-to-start standard-deviation ratio for a linear noise profile.
        """
        self.n_samples = n_samples
        n_peaks = self._ensure_tuple(peak_count.bounds)
        noise_std = self._ensure_tuple(noise_std) if noise_std is not None else None
        drift = self._ensure_tuple(drift) if drift is not None else None
        minimum_level = (
            self._ensure_tuple(minimum_level) if minimum_level is not None else None
        )
        baseline_level = (
            self._ensure_tuple(baseline_level) if baseline_level is not None else None
        )
        response = self._validate_instrument_response(instrument_response)
        saturation_bounds = self._resolve_saturation(saturation)
        if quantization_step is not None and quantization_step <= 0:
            raise ValueError("quantization_step must be greater than zero.")
        if noise_profile not in {"constant", "linear"}:
            raise ValueError("noise_profile must be 'constant' or 'linear'.")
        if noise is not None and noise_profile != "constant":
            raise ValueError(
                "noise_profile is only supported with noise_std, not a custom noise object."
            )

        if shift_min_to_zero and minimum_level is not None:
            raise ValueError(
                "shift_min_to_zero and minimum_level are mutually exclusive."
            )
        if noise is not None and noise_std is not None:
            raise ValueError("noise and noise_std are mutually exclusive.")

        rng = np.random.default_rng(seed)

        peak_components = kernel.evaluate(
            self.x_values,
            self.n_samples,
            n_peaks,
            categorical_peak_count,
            peak_count=peak_count,
            rng=rng,
        )
        clean_signals = np.nansum(peak_components, axis=1)
        signals = clean_signals.copy()

        if response is not None:
            signals = np.stack(
                [np.convolve(signal, response, mode="same") for signal in signals],
                axis=0,
            )

        labels = np.zeros((self.n_samples, self.sequence_length))
        true_positions = kernel.positions_for_labels
        diff = np.abs(true_positions[..., None] - self.x_values[None, None, :])
        peak_indices = diff.argmin(axis=-1)
        peak_indices = np.clip(peak_indices, 0, self.sequence_length - 1)

        for i in range(self.n_samples):
            labels[i, peak_indices[i, : kernel.num_peaks[i]]] = 1

        resolved_noise = noise
        if resolved_noise is None and noise_std is not None:
            resolved_noise = (
                NonstationaryGaussianNoise(
                    std=noise_std,
                    end_scale=noise_end_scale,
                )
                if noise_profile == "linear"
                else GaussianNoise(std=noise_std)
            )

        if resolved_noise is not None:
            signals = signals + resolved_noise.sample(
                signals.shape,
                x_values=np.asarray(self.x_values, dtype=float),
                rng=rng,
            )

        if drift is not None:
            drift_levels = rng.uniform(drift[0], drift[1], size=(self.n_samples, 1))
            baseline = drift_levels * np.linspace(0, 1, self.sequence_length)
            signals = signals + baseline

        if baseline_level is not None:
            baseline_levels = rng.uniform(
                baseline_level[0], baseline_level[1], size=(self.n_samples, 1)
            )
            signals = signals + baseline_levels

        if shift_min_to_zero:
            minimum_per_trace = np.min(signals, axis=1, keepdims=True)
            signals = signals - np.minimum(minimum_per_trace, 0.0)

        if minimum_level is not None:
            target_minimum = rng.uniform(
                minimum_level[0], minimum_level[1], size=(self.n_samples, 1)
            )
            minimum_per_trace = np.min(signals, axis=1, keepdims=True)
            signals = signals + (target_minimum - minimum_per_trace)

        if saturation_bounds is not None:
            signals = np.clip(signals, *saturation_bounds)
        if quantization_step is not None:
            signals = np.round(signals / quantization_step) * quantization_step

        dataset = DataSet(
            n_samples=self.n_samples,
            sequence_length=self.sequence_length,
            signals=signals,
            clean_signals=clean_signals,
            **kernel.get_kwargs(),
            labels=labels,
            x_values=self.x_values,
            num_peaks=kernel.num_peaks,
            baseline_level=baseline_level,
            instrument_response=response,
            saturation=saturation_bounds,
            quantization_step=quantization_step,
        )
        return dataset

    def _merge_datasets(self, datasets: list[DataSet]) -> DataSet:
        first_dataset = datasets[0]
        reference_x_values = np.asarray(first_dataset.x_values)
        sample_counts = [
            int(np.asarray(dataset.signals).shape[0]) for dataset in datasets
        ]
        total_samples = int(np.sum(sample_counts))

        for dataset in datasets[1:]:
            current_x_values = np.asarray(dataset.x_values)
            if current_x_values.shape != reference_x_values.shape or not np.allclose(
                current_x_values, reference_x_values
            ):
                raise ValueError(
                    "All buffered datasets must share the same x_values to be merged."
                )

        attribute_names = {
            attribute
            for dataset in datasets
            for attribute in getattr(dataset, "list_of_attributes", [])
        }

        merged_attributes: dict[str, Any] = {"x_values": reference_x_values.copy()}
        for attribute_name in sorted(attribute_names):
            if attribute_name in {"x_values", "n_samples", "sequence_length"}:
                continue

            values = [getattr(dataset, attribute_name, None) for dataset in datasets]
            present_values = [value for value in values if value is not None]
            if len(present_values) == 0:
                continue

            if self._is_sample_aligned_attribute(present_values[0], sample_counts[0]):
                merged_attributes[attribute_name] = (
                    self._merge_sample_aligned_attribute(
                        attribute_name,
                        values,
                        sample_counts,
                    )
                )
                continue

            if self._all_values_equal(present_values):
                merged_attributes[attribute_name] = present_values[0]

        return DataSet(
            n_samples=total_samples,
            sequence_length=self.sequence_length,
            **merged_attributes,
        )

    @staticmethod
    def _is_sample_aligned_attribute(value: Any, sample_count: int) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == sample_count
        )

    def _merge_sample_aligned_attribute(
        self,
        attribute_name: str,
        values: list[Any],
        sample_counts: list[int],
    ) -> np.ndarray:
        present_arrays = [np.asarray(value) for value in values if value is not None]
        template = present_arrays[0]
        template_shape = template.shape[1:]
        target_ndim = template.ndim

        for array in present_arrays[1:]:
            if array.ndim != target_ndim:
                raise ValueError(
                    f"Cannot merge attribute {attribute_name!r} with incompatible ranks: "
                    f"{template.shape} and {array.shape}."
                )

        target_shape = tuple(
            max(array.shape[axis] for array in present_arrays)
            for axis in range(1, target_ndim)
        )
        merged_dtype = np.result_type(*[array.dtype for array in present_arrays])

        if not np.issubdtype(merged_dtype, np.number):
            for array in present_arrays[1:]:
                if array.shape[1:] != template_shape:
                    raise ValueError(
                        f"Cannot merge non-numeric attribute {attribute_name!r} with incompatible shapes: "
                        f"{template.shape} and {array.shape}."
                    )
            target_shape = template_shape
            fill_dtype = merged_dtype
            fill_value = None
        else:
            fill_dtype = (
                np.result_type(merged_dtype, np.float64)
                if np.issubdtype(merged_dtype, np.integer)
                else merged_dtype
            )
            fill_value = np.nan if np.issubdtype(fill_dtype, np.floating) else 0

        merged_parts = []
        for sample_count, value in zip(sample_counts, values):
            if value is None:
                if fill_value is None:
                    raise ValueError(
                        f"Cannot fill missing non-numeric attribute {attribute_name!r} while merging datasets."
                    )
                merged_parts.append(
                    np.full((sample_count, *target_shape), fill_value, dtype=fill_dtype)
                )
                continue

            array = np.asarray(value)
            if array.ndim != target_ndim:
                raise ValueError(
                    f"Cannot merge attribute {attribute_name!r} with incompatible ranks: "
                    f"{template.shape} and {array.shape}."
                )

            if array.shape[1:] == target_shape:
                merged_parts.append(array.astype(fill_dtype, copy=False))
                continue

            if fill_value is None:
                raise ValueError(
                    f"Cannot merge non-numeric attribute {attribute_name!r} with incompatible shapes: "
                    f"{template.shape} and {array.shape}."
                )

            padded = np.full(
                (array.shape[0], *target_shape), fill_value, dtype=fill_dtype
            )
            slices = (slice(None),) + tuple(slice(0, size) for size in array.shape[1:])
            padded[slices] = array
            merged_parts.append(padded)

        return np.concatenate(merged_parts, axis=0)

    @staticmethod
    def _all_values_equal(values: list[Any]) -> bool:
        first_value = values[0]
        for value in values[1:]:
            if isinstance(first_value, np.ndarray) and isinstance(value, np.ndarray):
                if first_value.shape != value.shape or not np.array_equal(
                    first_value, value, equal_nan=True
                ):
                    return False
                continue
            if value != first_value:
                return False
        return True

    @staticmethod
    def _ensure_tuple(
        value: Tuple[float, float] | float | Tuple[int, int] | int,
    ) -> Tuple[float, float] | Tuple[int, int]:
        """If value is a scalar, return ``(v, v)``; otherwise return value."""
        if isinstance(value, (int, float)):
            return (value, value)  # type: ignore[return-value]
        return value  # type: ignore[return-value]

    @staticmethod
    def _validate_instrument_response(
        response: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if response is None:
            return None
        response = np.asarray(response, dtype=float)
        if response.ndim != 1 or response.size == 0:
            raise ValueError("instrument_response must be a non-empty 1D array.")
        if not np.all(np.isfinite(response)):
            raise ValueError("instrument_response must contain finite values.")
        if not np.any(response):
            raise ValueError("instrument_response must contain a non-zero value.")
        return response.copy()

    @staticmethod
    def _resolve_saturation(
        saturation: Optional[Union[float, Tuple[float, float]]],
    ) -> Optional[Tuple[float, float]]:
        if saturation is None:
            return None
        if isinstance(saturation, (int, float)):
            bounds = (-np.inf, float(saturation))
        else:
            if len(saturation) != 2:
                raise ValueError("saturation must be a limit or a (low, high) pair.")
            bounds = (float(saturation[0]), float(saturation[1]))
        if bounds[1] < bounds[0]:
            raise ValueError("saturation must satisfy low <= high.")
        return bounds
