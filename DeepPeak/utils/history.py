from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _coerce_history(history: Any) -> Mapping[str, Sequence[float]]:
    if hasattr(history, "history") and isinstance(history.history, Mapping):
        return history.history

    if isinstance(history, Mapping):
        return history

    raise TypeError(f"Unsupported history type: {type(history)!r}")


def merge_and_plot_histories(
    *histories: Any,
    plot_keys: tuple[str, ...] = ("loss", "val_loss"),
) -> tuple[dict[str, list[float]], Any]:
    """
    Merge any number of Keras History-like objects and plot selected keys.
    """
    import matplotlib.pyplot as plt

    merged: dict[str, list[float]] = {}
    normalized_histories = [_coerce_history(history) for history in histories]

    keys = {key for history in normalized_histories for key in history.keys()}
    for key in keys:
        merged[key] = []
        for history in normalized_histories:
            merged[key].extend(list(history.get(key, [])))

    figure = plt.figure()
    for key in plot_keys:
        values = merged.get(key, [])
        if not values:
            continue

        epochs = np.arange(1, len(values) + 1)
        plt.plot(epochs, values, label=key)

    plt.xlabel("Epoch")
    plt.legend()

    return merged, figure
