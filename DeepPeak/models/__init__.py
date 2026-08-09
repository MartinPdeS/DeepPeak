"""Public API for DeepPeak neural-network models.

TensorFlow-backed objects are imported on first access so that importing the
subpackage itself remains cheap and the public names stay in one place.
"""

from importlib import import_module
from typing import Any


__all__ = [
    "DenseNet",
    "ShapeAwarePulseLoss",
    "UNet1D",
    "WaveNet",
    "WeightedHuber",
    "shape_aware_pulse_loss",
    "weighted_huber",
]


_EXPORTS = {
    "DenseNet": ("dense", "DenseNet"),
    "ShapeAwarePulseLoss": ("losses", "ShapeAwarePulseLoss"),
    "UNet1D": ("unet1d", "UNet1D"),
    "WaveNet": ("wavenet", "WaveNet"),
    "WeightedHuber": ("losses", "WeightedHuber"),
    "shape_aware_pulse_loss": ("losses", "shape_aware_pulse_loss"),
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
