"""Lazy top-level public API for DeepPeak.

Public objects are imported only when first accessed. This keeps package and
subpackage imports lightweight: using :mod:`DeepPeak.models`, for example,
does not initialize analysis, plotting, or TensorFlow until needed.
"""

from importlib import import_module
from typing import Any

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"


_EXPORTS = {
    # Analysis
    "BasePeakTrigger": ("detection.triggers", "BasePeakTrigger"),
    "FlashDilutionSeries": ("analysis.dilution_series", "FlashDilutionSeries"),
    "HeightPeakTrigger": ("detection.triggers", "HeightPeakTrigger"),
    "NeuralTraceAnalyzer": ("analysis.wavenet_trace", "NeuralTraceAnalyzer"),
    "ProminencePeakTrigger": ("detection.triggers", "ProminencePeakTrigger"),
    "PulseShapeAnalyzer": ("analysis.pulse_shape", "PulseShapeAnalyzer"),
    "SeriesComparisonResult": ("analysis.comparison", "SeriesComparisonResult"),
    "SigmaPeakTrigger": ("detection.triggers", "SigmaPeakTrigger"),
    "StandardDilutionSeries": (
        "analysis.dilution_series",
        "StandardDilutionSeries",
    ),
    "StandardTraceAnalyzer": ("analysis.wavenet_trace", "StandardTraceAnalyzer"),
    "TraceAnalyzer": ("analysis.wavenet_trace", "TraceAnalyzer"),
    "TraceComparisonAnalyzer": ("analysis.comparison", "TraceComparisonAnalyzer"),
    "TraceComparisonResult": ("analysis.comparison", "TraceComparisonResult"),
    # Generation
    "BaseKernel": ("generation.kernels", "BaseKernel"),
    "BaseNoise": ("generation.noises", "BaseNoise"),
    "CorrelatedGaussianNoise": (
        "generation.noises",
        "CorrelatedGaussianNoise",
    ),
    "CustomKernel": ("generation.kernels", "CustomKernel"),
    "DataSet": ("generation.dataset", "DataSet"),
    "Dirac": ("generation.kernels", "Dirac"),
    "Gaussian": ("generation.kernels", "Gaussian"),
    "GaussianNoise": ("generation.noises", "GaussianNoise"),
    "LaplaceNoise": ("generation.noises", "LaplaceNoise"),
    "Lorentzian": ("generation.kernels", "Lorentzian"),
    "NegativeBinomialCount": ("generation.peak_count", "NegativeBinomialCount"),
    "NonstationaryGaussianNoise": (
        "generation.noises",
        "NonstationaryGaussianNoise",
    ),
    "PeakCount": ("generation.peak_count", "PeakCount"),
    "PoissonCount": ("generation.peak_count", "PoissonCount"),
    "SignalGenerator": ("generation.signal_generator", "SignalGenerator"),
    "Square": ("generation.kernels", "Square"),
    "TwoLobeGaussian": ("generation.kernels", "TwoLobeGaussian"),
    "UniformCount": ("generation.peak_count", "UniformCount"),
    # Core and pipeline
    "AnalysisConfig": ("core", "AnalysisConfig"),
    "DetectionConfig": ("core", "DetectionConfig"),
    "DetectionResult": ("core", "DetectionResult"),
    "GenerationConfig": ("core", "GenerationConfig"),
    "MetricResult": ("core", "MetricResult"),
    "ModelConfig": ("core", "ModelConfig"),
    "NoiseConfig": ("core", "NoiseConfig"),
    "Pipeline": ("pipeline", "Pipeline"),
    "PipelineResult": ("pipeline", "PipelineResult"),
    "PlotConfig": ("core", "PlotConfig"),
    "SeriesConfig": ("core", "SeriesConfig"),
    "SeriesResult": ("core", "SeriesResult"),
    "Trace": ("core", "Trace"),
    "TraceConfig": ("core", "TraceConfig"),
    # Optional machine-learning API (DeepPeak.models is itself lazy).
    "DenseNet": ("models", "DenseNet"),
    "ModelEvaluationResult": ("models", "ModelEvaluationResult"),
    "ShapeAwarePulseLoss": ("models", "ShapeAwarePulseLoss"),
    "SmoothBinaryCrossentropy": ("models", "SmoothBinaryCrossentropy"),
    "TrainingConfig": ("models", "TrainingConfig"),
    "UNet1D": ("models", "UNet1D"),
    "WaveNet": ("models", "WaveNet"),
    "WeightedBinaryCrossentropy": ("models", "WeightedBinaryCrossentropy"),
    "WeightedHuber": ("models", "WeightedHuber"),
    "shape_aware_pulse_loss": ("models", "shape_aware_pulse_loss"),
    "smooth_bce": ("models", "smooth_bce"),
    "weighted_bce": ("models", "weighted_bce"),
    "weighted_huber": ("models", "weighted_huber"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    """Import and cache a public object on first access."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module 'DeepPeak' has no attribute {name!r}") from error

    try:
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    except ModuleNotFoundError as error:
        if error.name in {"tensorflow", "sklearn"}:
            raise ModuleNotFoundError(
                f"{name} requires the optional DeepPeak machine-learning dependencies."
            ) from error
        raise

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eagerly defined and lazily exported public names."""

    return sorted(set(globals()) | set(__all__))
