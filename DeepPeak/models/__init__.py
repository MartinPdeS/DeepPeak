"""Public API for DeepPeak neural-network models.

TensorFlow-backed objects are imported on first access so that importing the
subpackage itself remains cheap and the public names stay in one place.
"""

from importlib import import_module
from typing import Any


__all__ = [
    "BinaryIoU",
    "DenseNet",
    "ShapeAwarePulseLoss",
    "SmoothBinaryCrossentropy",
    "UNet1D",
    "WaveNet",
    "WeightedBinaryCrossentropy",
    "WeightedHuber",
    "plot_predictions",
    "TrainingConfig",
    "shape_aware_pulse_loss",
    "smooth_bce",
    "weighted_bce",
    "weighted_huber",
]


_EXPORTS = {
    "BinaryIoU": ("metrics", "BinaryIoU"),
    "DenseNet": ("dense", "DenseNet"),
    "ShapeAwarePulseLoss": ("losses", "ShapeAwarePulseLoss"),
    "SmoothBinaryCrossentropy": ("losses", "SmoothBinaryCrossentropy"),
    "UNet1D": ("unet1d", "UNet1D"),
    "WaveNet": ("wavenet", "WaveNet"),
    "WeightedBinaryCrossentropy": ("losses", "WeightedBinaryCrossentropy"),
    "WeightedHuber": ("losses", "WeightedHuber"),
    "plot_predictions": ("plotting", "plot_predictions"),
    "TrainingConfig": ("training", "TrainingConfig"),
    "shape_aware_pulse_loss": ("losses", "shape_aware_pulse_loss"),
    "smooth_bce": ("losses", "smooth_bce"),
    "weighted_bce": ("losses", "weighted_bce"),
    "weighted_huber": ("losses", "weighted_huber"),
}


def __getattr__(name: str) -> Any:
    """Load a public neural-network symbol on demand."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module 'DeepPeak.models' has no attribute {name!r}"
        ) from error

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
    return sorted(set(globals()) | set(__all__))
