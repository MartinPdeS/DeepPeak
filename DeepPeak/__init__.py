"""Top-level public API for DeepPeak.

The root package exposes the lightweight analysis, dataset-generation, and
kernel types that are expected to be stable for library users. TensorFlow-based
classifier objects are loaded lazily so importing :mod:`DeepPeak` does not
require the optional machine-learning stack.
"""

try:
    from ._version import version as __version__  # noqa: F401

except ImportError:
    __version__ = "0.0.0"

from .analysis import (
    DilutionSeries,
    PeakCountSeries,
    PeakCountSeriesResult,
    PeakTrigger,
    WaveNetTraceAnalyzer,
)
from .dataset import DataSet
from .kernel import BaseKernel, Gaussian, Lorentzian
from .signal_generator import SignalGenerator

SignalDatasetGenerator = SignalGenerator

__all__ = [
    "__version__",
    "Autoencoder",
    "BaseKernel",
    "BinaryIoU",
    "DataSet",
    "DenseNet",
    "DilutionSeries",
    "Gaussian",
    "Lorentzian",
    "PeakCountSeries",
    "PeakCountSeriesResult",
    "PeakTrigger",
    "SignalGenerator",
    "WaveNet",
    "WaveNetTraceAnalyzer",
    "plot_predictions",
]

_LAZY_CLASSIFIER_EXPORTS = {
    "Autoencoder",
    "BinaryIoU",
    "DenseNet",
    "WaveNet",
    "plot_predictions",
}


def __getattr__(name: str):
    if name in _LAZY_CLASSIFIER_EXPORTS:
        try:
            from .machine_learning.classifier import __dict__ as classifier_exports
        except ModuleNotFoundError as error:
            if error.name in {"tensorflow", "sklearn", "tf_explain"}:
                raise ModuleNotFoundError(
                    f"{name} requires the optional DeepPeak[ml] dependencies."
                ) from error
            raise

        return classifier_exports[name]

    raise AttributeError(f"module 'DeepPeak' has no attribute {name!r}")
