from typing import Callable, Optional, Union

import numpy as np
import re
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.models import Model  # type: ignore
from MPSPlots.styles import mps



def plot_conv1D(model, input_signal, layer_name):
    """
    Plot activations for a Conv1D layer.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The full model containing the Conv1D layer.
    input_signal : np.ndarray
        A single input signal of shape (1, sequence_length, 1).
    layer_name : str
        The name of the Conv1D layer to visualize.

    Returns
    -------
    None
    """
    # Create a submodel to output intermediate activations
    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get the activations for the input signal
    activations = activation_model.predict(input_signal)

    # Get shape details
    num_filters = activations.shape[-1]
    sequence_length = activations.shape[1]

    # Plot the activations
    plt.figure(figsize=(12, 8))
    for i in range(num_filters):
        plt.plot(range(sequence_length), activations[0, :, i], label=f"Filter {i}")

    plt.title(f"Activations for {layer_name}")
    plt.xlabel("Sequence Index")
    plt.ylabel("Activation Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_dense(model, input_signal, layer_name):
    """
    Plot activations for a Dense layer using a step plot.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The full model containing the Dense layer.
    input_signal : np.ndarray
        A single input signal of shape (1, input_length, 1).
    layer_name : str
        The name of the Dense layer to visualize.

    Returns
    -------
    None
    """
    # Create a submodel to output intermediate activations
    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get the activations for the input signal
    activations = activation_model.predict(input_signal)

    # Plot the activations using a step plot
    plt.figure(figsize=(12, 6))
    plt.step(range(len(activations[0])), activations[0], where="mid", color="blue", linewidth=2)
    plt.title(f"Activations for {layer_name}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_gradcam_with_signal(
    model: object,
    layer_name: str,
    signal_index: int,
    output_neuron: int,
    signals: np.ndarray,
    outputs: np.ndarray,
    input_length: int,
    max_channels: int = 10,
):
    """
    Visualize the input signal, Grad-CAM heatmaps, and model predictions.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The trained Keras model.
    layer_name : str
        Name of the layer to analyze (typically the last Conv1D layer).
    signal_index : int
        Index of the signal in the dataset to visualize.
    output_neuron : int
        Index of the model output neuron for which Grad-CAM is computed.
    signals : np.ndarray
        Array of input signals of shape (num_samples, input_length, 1).
    outputs : np.ndarray
        Array of corresponding ground-truth outputs of shape (num_samples, num_outputs).
    input_length : int
        Length of the input signals.
    max_channels : int, optional
        Maximum number of channels to visualize from the Grad-CAM heatmaps, by default 10.

    Returns
    -------
    None
    """
    # Extract the specific signal and ground-truth output
    signal = signals[signal_index : signal_index + 1]

    # Compute Grad-CAM heatmaps
    explainer = GradCAM()
    heatmaps = explainer.explain(
        validation_data=(signal, signal),
        model=model,
        layer_name=layer_name,
        class_index=output_neuron,
    )

    # Get model predictions for the signal
    predictions = model.predict(signal.reshape([1, input_length, 1])).flatten()

    # Create the figure with subplots
    num_axes = min(max_channels, heatmaps.shape[-1]) + 1  # 1 for signal, rest for heatmaps
    fig, axes = plt.subplots(
        num_axes,
        1,
        figsize=(12, 2 * num_axes),
        squeeze=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2] + [1] * (num_axes - 1)},
    )

    # Plot the signal and predictions
    axes[0].plot(signal.squeeze(), color="blue", linewidth=1.5, label="Input Signal")
    for i, pred in enumerate(predictions):
        axes[0].axvline(x=pred * input_length, color="red", linestyle="--", label=f"Predicted Peak {i + 1}")
    axes[0].set_ylabel("Signal Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Plot the Grad-CAM heatmaps for selected channels
    for i, ax in enumerate(axes[1:]):
        if i >= max_channels:
            break
        ax.imshow(heatmaps[:, :, i].T, aspect="auto", cmap="jet", extent=[0, input_length, 0, 1])
        ax.set_ylabel(f"Channel {i}")
        ax.set_yticks([])

    axes[-1].set_xlabel("Signal Index")
    plt.tight_layout()
    plt.show()

def plot_training_history(history, metrics: list, y_scale: str = 'log', show: bool = True) -> plt.Figure:
    """
    Plot training and validation performance metrics (loss and accuracy).

    Parameters
    ----------
    history : tensorflow.keras.callbacks.History
        The training history object from model.fit().
    filtering : list of str, optional
        List of wildcard patterns to filter the keys in the history dictionary. Use '*' as a wildcard.
    """
    metrics = set(metrics).intersection(set(history.history))

    with plt.style.context(mps):
        figure, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            squeeze=True,
            figsize=(8, 3)
        )

    for metric in metrics:
        values = history.history[metric]
        ax.plot(values, label=metric.replace('_', ' '))
        ax.legend(loc='upper left')
        ax.set_yscale(y_scale)

    ax.set_xlabel('Number of Epochs')
    figure.suptitle('Training History')

    plt.tight_layout()

    if show:
        plt.show()

    return figure

# def plot_training_history(*histories, filtering: list = None, y_scale: str = 'log', show: bool = True) -> plt.Figure:
#     """
#     Plot training and validation performance metrics (loss and accuracy).

#     Parameters
#     ----------
#     history : tensorflow.keras.callbacks.History
#         The training history object from model.fit().
#     filtering : list of str, optional
#         List of wildcard patterns to filter the keys in the history dictionary. Use '*' as a wildcard.
#     """
#     # Convert wildcard patterns to regex patterns
#     def wildcard_to_regex(pattern):
#         return "^" + re.escape(pattern).replace("\\*", ".*") + "$"

#     with plt.style.context(mps):
#         figure, axes = plt.subplots(
#             nrows=len(histories),
#             ncols=1,
#             sharex=True,
#             squeeze=False,
#             figsize=(8, 3 * len(histories))
#         )

#     for history in histories:
#         # Filter the history dictionary based on converted patterns
#         if filtering is not None:
#             regex_patterns = [wildcard_to_regex(pattern) for pattern in filtering]
#             history_dict = {
#                 k: v for k, v in history.history.items()
#                 if any(re.fullmatch(regex, k) for regex in regex_patterns)
#             }
#         else:
#             history_dict = history.history

#         if not history_dict:
#             print("No matching keys found for the provided filtering patterns.")
#             return

#         for ax, (key, value) in zip(axes.flatten(), history_dict.items()):
#             ax.plot(value, label=history.name if hasattr(history, 'name') else '')
#             ax.legend(loc='upper left')
#             ax.set_ylabel(key.replace('_', ' '))
#             ax.set_yscale(y_scale)

#     axes.flatten()[-1].set_xlabel('Number of Epochs')
#     figure.suptitle('Training History')

#     plt.tight_layout()

#     if show:
#         plt.show()

#     return figure


class SignalPlotter:
    """
    A class for plotting multiple 1D signals with additional overlays such as
    scatter markers, vertical lines, horizontal lines, ROI masks, and custom function-based curves.

    Key Features
    -------------
    - add_signals(...): Stores the main signals and optionally the x-axis values.
    - add_scatter(...): Accumulates scatter overlays (e.g., peak markers with optional amplitude annotations).
    - add_vline(...): Accumulates vertical line overlays.
    - add_hline(...): Accumulates horizontal line overlays.
    - add_roi(...): Accumulates ROI masks along with plotting parameters (color, label, alpha).
    - add_custom_curves(...): Accumulates custom curves (function-based) to overlay on each signal.
    - plot(...): Creates a grid of subplots, handles sample selection, and overlays all stored elements.

    Examples
    --------
    >>> plotter = SignalPlotter()
    >>> plotter.add_signals(signals_array)
    >>> plotter.add_scatter(scatter_x=positions_array, scatter_y=amplitudes_array, color='red', label='Peak', marker='o', alpha=0.8)
    >>> plotter.add_vline(x=0.5, color='magenta', label='vline', linestyle='--')
    >>> plotter.add_hline(y=0.2, color='orange', label='hline', linestyle='-.')
    >>> plotter.add_roi(roi_array, color='green', label='ROI', alpha=0.6)
    >>> plotter.add_custom_curves(curve_function=my_curve_func, label="Curve", color="blue", style="--", pos=pos_array, width=width_array, amp=amp_array)
    >>> plotter.set_title("Demo Plot")
    >>> plotter.plot(n_examples=6, n_columns=3, random_select=False)
    """

    def __init__(self):
        # Internal storage for required signals and overlays.
        self.signals = None            # 2D array of shape (n_samples, sequence_length)
        self.x_values = None           # 1D array of shape (sequence_length,), inferred if not provided

        # Accumulate overlays in lists
        self._scatter = []           # List of scatter dictionaries: keys: scatter_x, scatter_y, color, marker, label, alpha.
        self._vlines = []            # List of vertical line dictionaries: keys: x, color, linestyle, label, linewidth, alpha.
        self._hlines = []            # List of horizontal line dictionaries: keys: y, color, linestyle, label, linewidth, alpha.
        self._rois = []              # List of ROI overlay dictionaries: keys: roi_array, color, label, alpha.
        self._custom_curves = []     # List of custom curve dictionaries: keys: curve_func, label, color, style, kwargs_arrays.

        # Display toggles
        self.show_scatter = True
        self.show_vlines = True
        self.show_hlines = True
        self.show_roi = True

        # Optional figure title
        self.title = None

    def add_signals(self, signals: np.ndarray, x_values: Optional[np.ndarray] = None):
        """
        Store the main 1D signals and optionally the x-axis values.

        Parameters
        ----------
        signals : numpy.ndarray
            2D array of shape (n_samples, sequence_length) containing the signals.
        x_values : numpy.ndarray, optional
            1D array of shape (sequence_length,). If None, defaults to linspace(0,1,sequence_length).

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).
        """
        signals = np.asarray(signals)
        if signals.ndim != 2:
            raise ValueError("signals must be 2D of shape (n_samples, sequence_length).")
        self.signals = signals

        if x_values is not None:
            x_values = np.asarray(x_values)
            if x_values.shape[0] != signals.shape[1]:
                raise ValueError("x_values length must match signals.shape[1].")
            self.x_values = x_values
        else:
            seq_length = signals.shape[1]
            self.x_values = np.linspace(0, 1, seq_length)
        return self

    def add_scatter(
        self,
        scatter_x: np.ndarray,
        scatter_y: np.ndarray,
        color: str = "red",
        marker: str = "o",
        label: str = "Scatter",
        alpha: float = 0.8
    ):
        """
        Accumulate scatter overlay points (e.g., for peak markers).

        Parameters
        ----------
        scatter_x : numpy.ndarray
            2D array of shape (n_samples, n_points) for x-coordinates.
        scatter_y : numpy.ndarray
            2D array of shape (n_samples, n_points) for y-coordinates.
        color : str, optional
            Color of the scatter points. Default is "red".
        marker : str, optional
            Marker style. Default is "o".
        label : str, optional
            Legend label for these points. Default is "Scatter".
        alpha : float, optional
            Opacity of the points. Default is 0.8.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).
        """
        scatter_x = np.asarray(scatter_x)
        scatter_y = np.asarray(scatter_y)
        if scatter_x.shape != scatter_y.shape:
            raise ValueError("scatter_x and scatter_y must have the same shape.")
        self._scatter.append({
            "scatter_x": scatter_x,
            "scatter_y": scatter_y,
            "color": color,
            "marker": marker,
            "label": label,
            "alpha": alpha
        })
        return self

    def add_vline(
        self,
        x: Union[float, np.ndarray],
        color: str = "magenta",
        linestyle: str = "--",
        label: str = "vline",
        linewidth: float = 1.0,
        alpha: float = 1.0
    ):
        """
        Accumulate vertical line overlay(s).

        Parameters
        ----------
        x : float or numpy.ndarray
            x-coordinate(s) at which to draw vertical line(s). If an array is provided,
            it should be 1D and of length equal to the number of points per signal or a single value per sample.
        color : str, optional
            Color of the vertical line(s). Default is "magenta".
        linestyle : str, optional
            Linestyle for the vertical line(s) (e.g., '--', '-.', ':'). Default is "--".
        label : str, optional
            Legend label for the vertical line(s). Default is "vline".
        linewidth : float, optional
            Width of the line(s). Default is 1.0.
        alpha : float, optional
            Opacity of the line(s). Default is 1.0.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).
        """
        # Convert x to numpy array if scalar.
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.asarray(x)
        self._vlines.append({
            "x": x,
            "color": color,
            "linestyle": linestyle,
            "label": label,
            "linewidth": linewidth,
            "alpha": alpha
        })
        return self

    def add_hline(self, y: Union[float, np.ndarray],
                  color: str = "cyan",
                  linestyle: str = "-.",
                  label: str = "hline",
                  linewidth: float = 1.0,
                  alpha: float = 1.0):
        """
        Accumulate horizontal line overlay(s).

        Parameters
        ----------
        y : float or numpy.ndarray
            y-coordinate(s) at which to draw horizontal line(s). If an array is provided,
            it should be 1D and of length equal to the number of points per signal or a single value per sample.
        color : str, optional
            Color of the horizontal line(s). Default is "cyan".
        linestyle : str, optional
            Linestyle for the horizontal line(s) (e.g., '--', '-.', ':'). Default is "-.".
        label : str, optional
            Legend label for the horizontal line(s). Default is "hline".
        linewidth : float, optional
            Width of the line(s). Default is 1.0.
        alpha : float, optional
            Opacity of the line(s). Default is 1.0.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).
        """
        if np.isscalar(y):
            y = np.array([y])
        else:
            y = np.asarray(y)
        self._hlines.append({
            "y": y,
            "color": color,
            "linestyle": linestyle,
            "label": label,
            "linewidth": linewidth,
            "alpha": alpha
        })
        return self

    def configure_display(self, show_scatter: bool = True, show_vlines: bool = True, show_hlines: bool = True, show_roi: bool = True):
        """
        Configure display options for overlays.

        Parameters
        ----------
        show_scatter : bool, optional
            Whether to display scatter overlays. Default is True.
        show_vlines : bool, optional
            Whether to display vertical line overlays. Default is True.
        show_hlines : bool, optional
            Whether to display horizontal line overlays. Default is True.
        show_roi : bool, optional
            Whether to display ROI overlays. Default is True.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).
        """
        self.show_scatter = show_scatter
        self.show_vlines = show_vlines
        self.show_hlines = show_hlines
        self.show_roi = show_roi
        return self

    def set_title(self, title: str):
        """
        Set a title for the plot.

        Parameters
        ----------
        title : str
            The title text for the figure.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).
        """
        self.title = title
        return self

    def add_roi(self, roi_array: np.ndarray, color: str = 'green', label: str = 'ROI', alpha: float = 0.6):
        """
        Add an ROI mask overlay along with plotting parameters. Multiple ROI overlays
        can be added and will all be plotted.

        Parameters
        ----------
        roi_array : numpy.ndarray
            Binary (0/1) 2D array of shape (n_samples, sequence_length) representing the ROI mask.
        color : str, optional
            Color used to shade the ROI. Default is 'green'.
        label : str, optional
            Legend label for the ROI. Default is 'ROI'.
        alpha : float, optional
            Transparency of the ROI shading. Default is 0.6.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).

        Raises
        ------
        ValueError
            If roi_array is not 2D or its shape does not match the signals (if signals are set).
        """
        roi_array = np.asarray(roi_array)
        if roi_array.ndim != 2:
            raise ValueError("ROI array must be 2D of shape (n_samples, sequence_length).")
        if self.signals is not None and roi_array.shape != self.signals.shape:
            raise ValueError(f"ROI array shape must match signals shape: {roi_array.shape} vs {self.signals.shape}.")
        self._rois.append({
            "roi_array": roi_array,
            "color": color,
            "label": label,
            "alpha": alpha
        })
        return self

    def add_custom_curves(
        self,
        curve_function: Callable,
        label: str = "CustomCurves",
        color: str = "green",
        style: str = "--",
        **kwargs_arrays
    ):
        """
        Add one or more custom curves to overlay on each signal. Each keyword argument
        must be a numpy.ndarray of shape (n_samples, n_curves) representing a parameter
        for the curve function.

        Parameters
        ----------
        curve_function : callable
            A function with signature: curve_function(x_values, **param_dict) -> 1D array.
        label : str, optional
            Legend label for the curves (only the first curve per sample is labeled). Default is "CustomCurves".
        color : str, optional
            Color for the curve lines. Default is "green".
        style : str, optional
            Linestyle (e.g., "--", "-.", ":"). Default is "--".
        **kwargs_arrays : dict
            Keyword arguments where each value is a 2D array of shape (n_samples, n_curves).

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).

        Raises
        ------
        ValueError
            If the provided keyword arrays do not have matching shapes.
        """
        shapes = [arr.shape for arr in kwargs_arrays.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"All keyword arrays must have the same shape. Got shapes: {shapes}")
        if self.signals is not None:
            n_samples = self.signals.shape[0]
            for name, arr in kwargs_arrays.items():
                if arr.shape[0] != n_samples:
                    raise ValueError(f"{name}.shape[0] = {arr.shape[0]} != n_samples={n_samples}")
        self._custom_curves.append({
            "curve_func": curve_function,
            "label": label,
            "color": color,
            "style": style,
            "kwargs_arrays": kwargs_arrays
        })
        return self

    def plot(
        self,
        sample_indices: Optional[list] = None,
        n_examples: int = 4,
        n_columns: int = 2,
        random_select: bool = False
    ):
        """
        Plot the signals with all overlays (ROI, scatter, vertical/horizontal lines, custom curves).

        Parameters
        ----------
        sample_indices : list of int, optional
            List of signal indices to plot. If None, a subset of n_examples is auto-selected.
        n_examples : int, optional
            Number of signals to plot if sample_indices is not provided. Default is 4.
        n_columns : int, optional
            Number of columns in the subplot grid. Default is 2.
        random_select : bool, optional
            If True and sample_indices is None, randomly select n_examples signals.

        Returns
        -------
        self : SignalPlotter
            The current instance (for method chaining).

        Raises
        ------
        ValueError
            If no signals have been added.
        """
        if self.signals is None:
            raise ValueError("No signals to plot. Please call add_signals(...) first.")

        n_samples, sequence_length = self.signals.shape
        x_vals = self.x_values

        # Determine sample indices
        if sample_indices is None:
            if random_select:
                sample_indices = np.random.choice(n_samples, size=min(n_examples, n_samples), replace=False)
            else:
                sample_indices = np.arange(min(n_examples, n_samples))
        else:
            sample_indices = sample_indices[:n_examples]

        n_actual = len(sample_indices)
        n_rows = int(np.ceil(n_actual / n_columns))

        with plt.style.context(mps):
            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_columns,
                figsize=(5 * n_columns, 4 * n_rows),
                squeeze=False,
                sharex=True,
                sharey=True
            )

        # Iterate over selected samples
        for i, idx in enumerate(sample_indices):
            row = i // n_columns
            col = i % n_columns
            ax = axes[row, col]

            # Plot the main signal
            ax.plot(x_vals, self.signals[idx], label=f"Signal #{idx}", color='blue')

            # Plot ROI overlays
            if self._rois and self.show_roi:
                for roi_info in self._rois:
                    roi_mask = roi_info["roi_array"][idx]
                    ax.fill_between(
                        x_vals,
                        0,
                        1,
                        where=(roi_mask > 0),
                        color=roi_info["color"],
                        alpha=roi_info.get("alpha", 0.6),
                        transform=ax.get_xaxis_transform(),
                        label=roi_info["label"]
                    )

            # Plot scatter overlays
            if self._scatter and self.show_scatter:
                for scatter_info in self._scatter:
                    scatter_x = scatter_info["scatter_x"][idx]
                    scatter_y = scatter_info["scatter_y"][idx]
                    ax.scatter(
                        scatter_x, scatter_y,
                        color=scatter_info["color"],
                        marker=scatter_info["marker"],
                        alpha=scatter_info["alpha"],
                        label=scatter_info["label"]
                    )

            # Plot vertical line overlays
            if self._vlines and self.show_vlines:
                for vline_info in self._vlines:
                    # vline_info["x"] can be an array; iterate over values
                    ax.vlines(
                        x=vline_info["x"][idx],
                        color=vline_info["color"],
                        linestyle=vline_info["linestyle"],
                        linewidth=vline_info["linewidth"],
                        alpha=vline_info["alpha"],
                        label=vline_info["label"],
                        transform=ax.get_xaxis_transform(),
                        ymin=0, ymax=1
                    )
            # Plot horizontal line overlays
            if self._hlines and self.show_hlines:
                for hline_info in self._hlines:
                    ax.hlines(
                        y=hline_info["y"][idx],
                        color=hline_info["color"],
                        linestyle=hline_info["linestyle"],
                        linewidth=hline_info["linewidth"],
                        alpha=hline_info["alpha"],
                        label=hline_info["label"],
                        transform=ax.get_yaxis_transform(),
                        xmin=0, xmax=1
                    )

            # Plot custom curves overlays
            for curve_info in self._custom_curves:
                curve_func = curve_info["curve_func"]
                label = curve_info["label"]
                color = curve_info["color"]
                style = curve_info["style"]
                kwargs_arrays = curve_info["kwargs_arrays"]
                # Assume each keyword array has shape (n_samples, n_curves)
                n_curves_sample = list(kwargs_arrays.values())[0].shape[1]
                for j in range(n_curves_sample):
                    param_dict = {k: v[idx, j] for k, v in kwargs_arrays.items()}
                    y_curve = curve_func(x_vals, **param_dict)
                    curve_label = label if j == 0 else None
                    ax.plot(x_vals, y_curve, style, color=color, label=curve_label)

            ax.set_title(f"Signal #{idx}")
            ax.legend()

        total_axes = n_rows * n_columns
        if n_actual < total_axes:
            for j in range(n_actual, total_axes):
                row_j = j // n_columns
                col_j = j % n_columns
                axes[row_j, col_j].axis("off")

        if self.title:
            fig.suptitle(self.title, fontsize=14)
        plt.tight_layout()
        plt.show()
        return self
