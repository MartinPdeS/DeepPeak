"""Small plotting helpers for model predictions."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def plot_predictions(
    signal: np.ndarray,
    prediction: np.ndarray,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot an input trace and its reconstruction and return the axes."""

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(signal, label="signal", **kwargs)
    ax.plot(prediction, label="prediction")
    ax.legend()
    if show:
        plt.show()
    return ax
