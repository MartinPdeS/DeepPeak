"""Lazy public API for TensorFlow-backed classifier components."""

from importlib import import_module

__all__ = [
    "Autoencoder",
    "BinaryIoU",
    "DenseNet",
    "ShapeAwarePulseLoss",
    "WaveNet",
    "WeightedBinaryCrossentropy",
    "WeightedHuber",
    "plot_predictions",
    "shape_aware_pulse_loss",
    "weighted_bce",
    "weighted_huber",
]

_EXPORT_TO_MODULE = {
    "Autoencoder": ".autoencoder",
    "BinaryIoU": ".metrics",
    "DenseNet": ".dense",
    "ShapeAwarePulseLoss": ".losses",
    "WaveNet": ".wavenet",
    "WeightedBinaryCrossentropy": ".losses",
    "WeightedHuber": ".losses",
    "plot_predictions": ".utils",
    "shape_aware_pulse_loss": ".losses",
    "weighted_bce": ".losses",
    "weighted_huber": ".losses",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
