"""Validated configuration objects shared by DeepPeak domains."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from .exceptions import InvalidConfigurationError


_NORMALIZATIONS = {"none", "zscore", "robust_zscore", "minmax", "min-max", "maxabs"}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


class SerializableConfig:
    """Mixin providing JSON-compatible config serialization."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to JSON-compatible values.

        Returns
        -------
        dict
            Configuration fields represented by standard Python values.
        """
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]):
        """Construct a configuration from a mapping.

        Parameters
        ----------
        values : mapping
            Configuration fields. Unknown fields are ignored.

        Returns
        -------
        SerializableConfig
            Reconstructed configuration instance.
        """
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the configuration to JSON.

        Parameters
        ----------
        path : path-like, optional
            If provided, write the JSON document to this path.
        indent : int, default=2
            Indentation level for the JSON document.

        Returns
        -------
        str
            Serialized JSON text.
        """
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_json(cls, source: str | Path):
        """Construct a configuration from JSON text or a JSON file.

        Parameters
        ----------
        source : str or path-like
            JSON document or path to a JSON document.

        Returns
        -------
        SerializableConfig
            Reconstructed configuration instance.
        """
        text = str(source)
        if isinstance(source, Path) or not text.lstrip().startswith(("{", "[")):
            try:
                path = Path(source)
                if path.exists():
                    text = path.read_text()
            except OSError:
                # A JSON document can be longer than the platform's filename limit.
                pass
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class TraceConfig(SerializableConfig):
    """Sampling and preprocessing settings for a trace workflow.

    Parameters
    ----------
    sequence_length : int, optional
        Expected number of samples per trace.
    normalization : str, default="zscore"
        Normalization strategy.
    sampling_rate_hz : float, optional
        Sampling frequency in hertz.
    """

    sequence_length: Optional[int] = None
    normalization: str = "zscore"
    sampling_rate_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sequence_length is not None and int(self.sequence_length) <= 0:
            raise InvalidConfigurationError(
                "sequence_length must be positive when provided."
            )
        normalized = str(self.normalization).strip().lower()
        if normalized not in _NORMALIZATIONS:
            choices = ", ".join(sorted(_NORMALIZATIONS))
            raise InvalidConfigurationError(f"normalization must be one of {choices}.")
        object.__setattr__(self, "normalization", normalized)
        if self.sampling_rate_hz is not None and float(self.sampling_rate_hz) <= 0:
            raise InvalidConfigurationError(
                "sampling_rate_hz must be positive when provided."
            )


@dataclass(frozen=True)
class DetectionConfig(TraceConfig):
    """Configuration shared by standard and neural detectors.

    Parameters
    ----------
    trigger : object, optional
        Trigger configuration used by the detector.
    low_pass : float, optional
        Low-pass filter cutoff.
    amplitude_sigma_samples : float, optional
        Gaussian width used for amplitude estimation.
    amplitude_cluster_radius_sigma : float, default=4.0
        Radius used to group overlapping peaks.
    amplitude_baseline : float or {"median"}, optional
        Baseline subtraction setting.
    """

    trigger: Any = None
    low_pass: Optional[float] = None
    amplitude_sigma_samples: Optional[float] = None
    amplitude_cluster_radius_sigma: float = 4.0
    amplitude_baseline: Optional[float | str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("low_pass", "amplitude_sigma_samples"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise InvalidConfigurationError(
                    f"{name} must be positive when provided."
                )
        if float(self.amplitude_cluster_radius_sigma) <= 0:
            raise InvalidConfigurationError(
                "amplitude_cluster_radius_sigma must be positive."
            )
        if isinstance(self.amplitude_baseline, str):
            if self.amplitude_baseline.strip().lower() != "median":
                raise InvalidConfigurationError(
                    'amplitude_baseline must be a number, "median", or None.'
                )
        elif self.amplitude_baseline is not None and not isinstance(
            self.amplitude_baseline, (int, float)
        ):
            raise InvalidConfigurationError(
                'amplitude_baseline must be a number, "median", or None.'
            )


@dataclass(frozen=True)
class SeriesConfig(TraceConfig):
    """Configuration shared by dilution-series workflows.

    Parameters
    ----------
    initial_concentration : float, default=1.0
        Starting concentration for the series.
    nrows : int, default=200000
        Number of rows allocated to the series.
    low_pass : float, optional
        Standard detector filter cutoff.
    cnn_low_pass : float, optional
        Neural detector filter cutoff.
    """

    initial_concentration: float = 1.0
    nrows: int = 200_000
    low_pass: Optional[float] = None
    cnn_low_pass: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.initial_concentration) <= 0:
            raise InvalidConfigurationError("initial_concentration must be positive.")
        if int(self.nrows) <= 0:
            raise InvalidConfigurationError("nrows must be positive.")
        for name in ("low_pass", "cnn_low_pass"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise InvalidConfigurationError(
                    f"{name} must be positive when provided."
                )


@dataclass(frozen=True)
class PlotConfig(SerializableConfig):
    """Non-interactive plotting defaults shared by plotting entry points.

    Parameters
    ----------
    figsize : tuple of float, default=(10.0, 4.0)
        Figure width and height in inches.
    show : bool, default=False
        Whether to display the figure.
    close : bool, default=False
        Whether to close the figure after creation.
    dpi : int, default=300
        Figure resolution.
    title : str, optional
        Optional figure title.
    """

    figsize: tuple[float, float] = (10.0, 4.0)
    show: bool = False
    close: bool = False
    dpi: int = 300
    title: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.figsize) != 2 or any(float(value) <= 0 for value in self.figsize):
            raise InvalidConfigurationError("figsize must contain two positive values.")
        if int(self.dpi) <= 0:
            raise InvalidConfigurationError("dpi must be positive.")


@dataclass(frozen=True)
class GenerationConfig(SerializableConfig):
    """Serializable acquisition effects for ``SignalGenerator``.

    Parameters
    ----------
    sequence_length : int
        Number of samples in each generated trace.
    seed : int, optional
        Random seed for reproducible generation.
    noise_std : float or tuple of float, optional
        Noise standard deviation or sampling range.
    noise_profile : {"constant", "linear"}, default="constant"
        Noise profile across each trace.
    baseline_level : float or tuple of float, optional
        Additive baseline level or sampling range.
    drift : float or tuple of float, optional
        Linear baseline drift or sampling range.
    instrument_response : tuple of float, optional
        Finite impulse response kernel.
    saturation : float or tuple of float, optional
        Upper limit or explicit clipping bounds.
    quantization_step : float, optional
        Quantization interval.
    missing_peak_probability : float, default=0.0
        Probability of dropping each generated peak.
    """

    sequence_length: int
    seed: Optional[int] = None
    noise_std: Optional[float | tuple[float, float]] = None
    noise_profile: Literal["constant", "linear"] = "constant"
    noise_end_scale: float | tuple[float, float] = 1.0
    baseline_level: Optional[float | tuple[float, float]] = None
    drift: Optional[float | tuple[float, float]] = None
    instrument_response: Optional[tuple[float, ...]] = None
    saturation: Optional[float | tuple[float, float]] = None
    quantization_step: Optional[float] = None
    missing_peak_probability: float = 0.0
    categorical_peak_count: bool = False
    shift_min_to_zero: bool = False
    minimum_level: Optional[float | tuple[float, float]] = None

    def __post_init__(self) -> None:
        if int(self.sequence_length) <= 0:
            raise InvalidConfigurationError("sequence_length must be positive.")
        if self.noise_profile not in {"constant", "linear"}:
            raise InvalidConfigurationError(
                "noise_profile must be 'constant' or 'linear'."
            )
        if not 0.0 <= float(self.missing_peak_probability) <= 1.0:
            raise InvalidConfigurationError(
                "missing_peak_probability must be between 0 and 1."
            )
        if self.quantization_step is not None and float(self.quantization_step) <= 0:
            raise InvalidConfigurationError("quantization_step must be positive.")
        if self.shift_min_to_zero and self.minimum_level is not None:
            raise InvalidConfigurationError(
                "shift_min_to_zero and minimum_level are mutually exclusive."
            )

    def signal_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``SignalGenerator.generate``.

        Returns
        -------
        dict
            Generation keyword arguments excluding ``sequence_length``.
        """

        values = self.to_dict()
        values.pop("sequence_length", None)
        return values


@dataclass(frozen=True)
class NoiseConfig(SerializableConfig):
    """Serializable description of a synthetic noise process.

    Parameters
    ----------
    kind : {"gaussian", "laplace", "nonstationary_gaussian", "correlated_gaussian"}
        Noise process to construct.
    scale : float or tuple of float, default=0.0
        Noise scale or sampling range.
    mean : float or tuple of float, default=0.0
        Noise mean or sampling range.
    end_scale : float or tuple of float, default=1.0
        End-to-start scale ratio for nonstationary noise.
    correlation_length : float, default=0.0
        Correlation length for correlated Gaussian noise.
    """

    kind: Literal[
        "gaussian", "laplace", "nonstationary_gaussian", "correlated_gaussian"
    ] = "gaussian"
    scale: float | tuple[float, float] = 0.0
    mean: float | tuple[float, float] = 0.0
    end_scale: float | tuple[float, float] = 1.0
    correlation_length: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {
            "gaussian",
            "laplace",
            "nonstationary_gaussian",
            "correlated_gaussian",
        }:
            raise InvalidConfigurationError("Unsupported noise kind.")
        if float(self.correlation_length) < 0:
            raise InvalidConfigurationError("correlation_length must be non-negative.")

    def build(self):
        """Construct the configured :class:`~DeepPeak.generation.BaseNoise`.

        Returns
        -------
        BaseNoise
            Configured noise generator.
        """

        from ..generation.noises import (
            CorrelatedGaussianNoise,
            GaussianNoise,
            LaplaceNoise,
            NonstationaryGaussianNoise,
        )

        if self.kind == "gaussian":
            return GaussianNoise(std=self.scale, mean=self.mean)
        if self.kind == "laplace":
            return LaplaceNoise(scale=self.scale, mean=self.mean)
        if self.kind == "nonstationary_gaussian":
            return NonstationaryGaussianNoise(
                std=self.scale,
                end_scale=self.end_scale,
                mean=self.mean,
            )
        return CorrelatedGaussianNoise(
            std=self.scale,
            correlation_length=self.correlation_length,
            mean=self.mean,
        )


@dataclass(frozen=True)
class ModelConfig(SerializableConfig):
    """Serializable common model settings for experiment manifests.

    Parameters
    ----------
    architecture : {"dense", "wavenet", "unet1d"}
        Neural architecture name.
    sequence_length : int
        Number of input samples.
    output_activation : str, default="linear"
        Output-layer activation.
    optimizer : str, default="adam"
        Optimizer name.
    loss : str, default="huber"
        Loss name.
    metrics : tuple of str, default=("mae",)
        Metric names.
    """

    architecture: Literal["dense", "wavenet", "unet1d"]
    sequence_length: int
    output_activation: Optional[str] = "linear"
    optimizer: str = "adam"
    loss: str = "huber"
    metrics: tuple[str, ...] = ("mae",)

    def __post_init__(self) -> None:
        if self.architecture not in {"dense", "wavenet", "unet1d"}:
            raise InvalidConfigurationError(
                "architecture must be dense, wavenet, or unet1d."
            )
        if int(self.sequence_length) <= 0:
            raise InvalidConfigurationError("sequence_length must be positive.")


@dataclass(frozen=True)
class AnalysisConfig(SerializableConfig):
    """Serializable options for one high-level pipeline analysis.

    Parameters
    ----------
    detector : {"standard", "cnn"}, default="standard"
        Detector record selected by the pipeline.
    dx : float, default=1.0
        Sample spacing.
    filename : str, default="<memory>"
        Input filename metadata.
    dilution : float, default=nan
        Optional dilution metadata.
    concentration : float, default=nan
        Optional concentration metadata.
    """

    detector: Literal["standard", "cnn"] = "standard"
    dx: float = 1.0
    filename: str = "<memory>"
    dilution: float = float("nan")
    concentration: float = float("nan")

    def __post_init__(self) -> None:
        if self.detector not in {"standard", "cnn"}:
            raise InvalidConfigurationError("detector must be 'standard' or 'cnn'.")
        if float(self.dx) <= 0:
            raise InvalidConfigurationError("dx must be positive.")


__all__ = [
    "AnalysisConfig",
    "DetectionConfig",
    "GenerationConfig",
    "ModelConfig",
    "NoiseConfig",
    "PlotConfig",
    "SeriesConfig",
    "TraceConfig",
]
