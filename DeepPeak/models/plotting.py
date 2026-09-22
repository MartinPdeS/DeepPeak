"""Plotting helpers for model predictions."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_predictions(
    model: Any,
    dataset: Any,
    *,
    n_samples: int = 6,
    n_columns: int = 3,
    randomize: bool = False,
    seed: int | None = None,
    show_target: bool = True,
    normalization: str = "none",
    batch_size: int = 32,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Figure:
    """Plot model predictions for several samples from a :class:`DataSet`.

    The input signal is plotted in black, the model prediction in blue, and
    the first available target (``clean_signals`` or ``labels``) in orange.
    Models accepting channel-last inputs are supported automatically when the
    dataset stores signals as a two-dimensional array.

    Parameters
    ----------
    model : object
        Object exposing ``predict(signal, batch_size=..., verbose=0)``.
    dataset : DataSet
        Dataset containing ``signals`` and optionally ``clean_signals`` or
        ``labels``.
    n_samples : int, default=6
        Maximum number of traces to plot.
    n_columns : int, default=3
        Number of subplot columns.
    randomize : bool, default=False
        Select random samples instead of the first samples.
    seed : int, optional
        Seed used when ``randomize`` is true.
    show_target : bool, default=True
        Whether to overlay the clean trace or labels when available.
    normalization : str, default="none"
        Dataset signal normalization applied before prediction and plotting.
    batch_size : int, default=32
        Batch size passed to ``model.predict``.
    figsize : tuple, optional
        Overall figure size. Defaults to 5 by 3 inches per subplot.
    show : bool, default=True
        Whether to display the figure before returning it.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the prediction panels.
    """
    signals = np.asarray(dataset.signals)
    if signals.ndim != 2:
        raise ValueError(
            "dataset.signals must have shape (n_samples, sequence_length)."
        )
    if n_samples < 1 or n_columns < 1:
        raise ValueError("n_samples and n_columns must be positive.")

    sample_count = min(int(n_samples), signals.shape[0])
    if sample_count == 0:
        raise ValueError("dataset.signals must contain at least one sample.")
    if randomize:
        indices = np.random.default_rng(seed).choice(
            signals.shape[0], size=sample_count, replace=False
        )
    else:
        indices = np.arange(sample_count)

    plotted_signals = (
        np.asarray(dataset.get_normalized_signal(normalization), dtype=float)
        if normalization != "none"
        else signals.astype(float, copy=True)
    )
    model_inputs = plotted_signals[indices]
    if model_inputs.ndim == 2:
        model_inputs = model_inputs[..., None]
    predictions = np.asarray(
        model.predict(model_inputs, batch_size=batch_size, verbose=0)
    )
    predictions = _as_traces(predictions, sample_count, "model predictions")

    x_values = np.asarray(
        getattr(dataset, "x_values", np.arange(signals.shape[1])), dtype=float
    )
    if x_values.ndim != 1 or x_values.size != signals.shape[1]:
        raise ValueError("dataset.x_values must match the signal sequence length.")

    target = None
    if show_target:
        for attribute in ("clean_signals", "labels"):
            if hasattr(dataset, attribute):
                target = _as_traces(
                    np.asarray(getattr(dataset, attribute))[indices],
                    sample_count,
                    attribute,
                )
                break

    n_rows = int(np.ceil(sample_count / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        squeeze=False,
        figsize=figsize or (5.0 * n_columns, 3.0 * n_rows),
        sharex=True,
    )

    for panel, (sample_index, axis) in enumerate(zip(indices, axes.flat)):
        axis.plot(
            x_values, plotted_signals[sample_index], color="black", label="Signal"
        )
        if target is not None:
            target_label = (
                "Clean signal" if hasattr(dataset, "clean_signals") else "Target"
            )
            axis.plot(x_values, target[panel], color="tab:orange", label=target_label)
        axis.plot(x_values, predictions[panel], color="tab:blue", label="Prediction")
        axis.set_title(f"Sample {sample_index}")

    for axis in axes.flat[sample_count:]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    figure.supxlabel("Time step [AU]")
    figure.supylabel("Amplitude [AU]")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    if show:
        plt.show()
    return figure


def _as_traces(values: np.ndarray, n_samples: int, name: str) -> np.ndarray:
    """Normalize channel-last model outputs to ``(n_samples, sequence_length)``."""
    if values.shape[0] != n_samples:
        raise ValueError(f"{name} has an unexpected number of samples.")
    if values.ndim == 3:
        if values.shape[-1] != 1:
            raise ValueError(f"{name} must have one output channel.")
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, sequence_length).")
    return values
