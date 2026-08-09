"""Top-level public API for DeepPeak.

The root package exposes the lightweight analysis, dataset-generation, kernel,
and noise-model types that are expected to be stable for library users.
TensorFlow-based classifier objects are loaded lazily so importing
:mod:`DeepPeak` does not require the optional machine-learning stack.
"""

try:
    from ._version import version as __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0"

from .analysis import (
    BasePeakTrigger,
    CNNTraceAnalyzer,
    FlashDilutionSeries,
    HeightPeakTrigger,
    PulseShapeAnalyzer,
    ProminencePeakTrigger,
    SigmaPeakTrigger,
    StandardDilutionSeries,
    StandardTraceAnalyzer,
    SeriesComparisonResult,
    TraceComparisonAnalyzer,
    TraceComparisonResult,
    WaveNetTraceAnalyzer,
)
from .generation.dataset import DataSet
from .generation.kernels import (
    BaseKernel,
    CustomKernel,
    Dirac,
    Gaussian,
    Lorentzian,
    Square,
    TwoLobeGaussian,
)
from .generation.noises import BaseNoise, GaussianNoise, LaplaceNoise
from .generation.peak_count import (
    NegativeBinomialCount,
    PeakCount,
    PoissonCount,
    UniformCount,
)
from .generation import SignalGenerator
from .core import DetectionResult, MetricResult, SeriesResult, Trace


__all__ = [
    "__version__",
    "BaseKernel",
    "BaseNoise",
    "DetectionResult",
    "BasePeakTrigger",
    "CNNTraceAnalyzer",
    "CustomKernel",
    "DataSet",
    "Dirac",
    "FlashDilutionSeries",
    "Gaussian",
    "GaussianNoise",
    "HeightPeakTrigger",
    "LaplaceNoise",
    "Lorentzian",
    "MetricResult",
    "NegativeBinomialCount",
    "PeakCount",
    "PoissonCount",
    "ProminencePeakTrigger",
    "PulseShapeAnalyzer",
    "SigmaPeakTrigger",
    "SignalGenerator",
    "Square",
    "SeriesResult",
    "StandardDilutionSeries",
    "StandardTraceAnalyzer",
    "SeriesComparisonResult",
    "TraceComparisonAnalyzer",
    "TraceComparisonResult",
    "TwoLobeGaussian",
    "Trace",
    "UniformCount",
    "WaveNetTraceAnalyzer",
    "DenseNet",
    "ShapeAwarePulseLoss",
    "SmoothBinaryCrossentropy",
    "UNet1D",
    "WaveNet",
    "WeightedHuber",
    "WeightedBinaryCrossentropy",
    "shape_aware_pulse_loss",
    "smooth_bce",
    "weighted_bce",
    "weighted_huber",
]


_LAZY_NEURAL_NETWORK_EXPORTS = set(__all__) - {
    "__version__",
    "BaseKernel",
    "BaseNoise",
    "BasePeakTrigger",
    "CNNTraceAnalyzer",
    "CustomKernel",
    "DataSet",
    "Dirac",
    "FlashDilutionSeries",
    "Gaussian",
    "GaussianNoise",
    "HeightPeakTrigger",
    "LaplaceNoise",
    "Lorentzian",
    "NegativeBinomialCount",
    "PeakCount",
    "PoissonCount",
    "ProminencePeakTrigger",
    "PulseShapeAnalyzer",
    "SigmaPeakTrigger",
    "SignalGenerator",
    "Square",
    "StandardDilutionSeries",
    "StandardTraceAnalyzer",
    "TwoLobeGaussian",
    "UniformCount",
    "WaveNetTraceAnalyzer",
}


def __getattr__(name: str):
    """Load TensorFlow-backed symbols only when they are requested."""

    if name not in _LAZY_NEURAL_NETWORK_EXPORTS:
        raise AttributeError(f"module 'DeepPeak' has no attribute {name!r}")

    try:
        from . import models

        value = getattr(models, name)
    except ModuleNotFoundError as error:
        if error.name in {"tensorflow", "sklearn"}:
            raise ModuleNotFoundError(
                f"{name} requires the optional DeepPeak machine-learning dependencies."
            ) from error
        raise

    globals()[name] = value
    return value
