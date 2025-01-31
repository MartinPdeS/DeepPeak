import numpy as np
import re
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.models import Model  # type: ignore
from MPSPlots.styles import mps
# from DeepPeak.utils.ROI import get_peak_rois_for_sample


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

def plot_training_history(*histories, filtering: list = None, y_scale: str = 'log'):
    """
    Plot training and validation performance metrics (loss and accuracy).

    Parameters
    ----------
    history : tensorflow.keras.callbacks.History
        The training history object from model.fit().
    filtering : list of str, optional
        List of wildcard patterns to filter the keys in the history dictionary. Use '*' as a wildcard.
    """
    # Convert wildcard patterns to regex patterns
    def wildcard_to_regex(pattern):
        return "^" + re.escape(pattern).replace("\\*", ".*") + "$"

    with plt.style.context(mps):
        figure, axes = plt.subplots(
            nrows=len(histories),
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(8, 3 * len(histories))
        )

    for history in histories:
        # Filter the history dictionary based on converted patterns
        if filtering is not None:
            regex_patterns = [wildcard_to_regex(pattern) for pattern in filtering]
            history_dict = {
                k: v for k, v in history.history.items()
                if any(re.fullmatch(regex, k) for regex in regex_patterns)
            }
        else:
            history_dict = history.history

        if not history_dict:
            print("No matching keys found for the provided filtering patterns.")
            return

        for ax, (key, value) in zip(axes.flatten(), history_dict.items()):
            ax.plot(value, label=history.params['epochs'])
            ax.legend(loc='upper left')
            ax.set_ylabel(key.replace('_', ' '))
            ax.set_yscale(y_scale)

    axes.flatten()[-1].set_xlabel('Number of Epochs')
    figure.suptitle('Training History')

    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

class SignalPlotter:
    """
    A class for plotting multiple 1D signals, including optional
    peak positions, amplitudes, ROI masks, and custom function-based curves.

    Key Features:
    -------------
    - add_signals(...): Store main signals (2D).
    - add_positions(...)/add_amplitudes(...): For peak scatter.
    - add_roi(...): For binary ROI shading.
    - add_custom_curves(...): Plot multiple curves per sample using a callable
      that takes kwargs, each of which is a 2D array of shape (n_samples, n_curves).
    - plot(...): Creates subplots, handles sample selection (manual, random),
      and overlays everything.

    Example for custom curves:
    --------------------------
        def gaussian_curve(x, pos, width, amp):
            return amp * np.exp(-0.5 * ((x - pos)/width)**2)

        # Suppose:
        #   positions.shape = widths.shape = amps.shape = (n_samples, n_curves)
        # Then for each sample i, each j in range(n_curves),
        # we call gaussian_curve(x, pos=positions[i,j], width=..., amp=...).

        plotter.add_custom_curves(
            curve_function=gaussian_curve,
            label="Gaussians",
            color="magenta",
            style="--",
            positions=positions_array,
            widths=widths_array,
            amp=amplitudes_array_for_curve
        )
    """

    def __init__(self):
        # Internal storage
        self.signals = None            # shape (n_samples, seq_length)
        self.x_values = None           # shape (seq_length,) or None => infer
        self.positions = None          # shape (n_samples, n_peaks)
        self.amplitudes = None         # shape (n_samples, n_peaks)
        self.roi = None                # shape (n_samples, seq_length), binary (optional)

        # Plot toggles
        self.show_peaks = True
        self.show_amplitudes = True
        self.show_roi = True

        # Optional figure title
        self.title = None

        # Custom curves: a list of dicts, each describing how to plot them
        #   {
        #       "curve_func": <callable>,
        #       "label": str,
        #       "color": str,
        #       "style": str,
        #       "kwargs_arrays": {
        #           "param1": <array shape (n_samples, n_curves)>,
        #           "param2": <...>,
        #           ...
        #       }
        #   }
        self._custom_curves = []

    ###########################################################################
    # "Add" methods (fluent interface)
    ###########################################################################

    def add_signals(self, signals: np.ndarray, x_values: np.ndarray = None):
        """
        Store the 1D signals (shape (n_samples, sequence_length)) and optionally
        the x_values array (shape (sequence_length,)).

        If x_values is None, it defaults to linspace(0,1,...).
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

    def add_positions(self, positions: np.ndarray):
        """
        Store the peak positions array (n_samples, n_peaks).
        Used for scattering points if show_peaks=True.
        """
        self.positions = np.asarray(positions)
        return self

    def add_amplitudes(self, amplitudes: np.ndarray):
        """
        Store the peak amplitudes array (n_samples, n_peaks).
        Used for scattering (pos, amp) if show_peaks=True.
        """
        self.amplitudes = np.asarray(amplitudes)
        return self

    def add_roi(self, roi_array: np.ndarray, show_roi: bool = True):
        """
        Store a binary (0/1) ROI array of shape (n_samples, sequence_length).
        The ROI is shaded if show_roi=True.
        """
        roi_array = np.asarray(roi_array)
        if roi_array.ndim != 2:
            raise ValueError("ROI array must be 2D of shape (n_samples, sequence_length).")
        if self.signals is not None and roi_array.shape != self.signals.shape:
            raise ValueError(
                "ROI array shape must match signals shape: "
                f"{roi_array.shape} vs {self.signals.shape}."
            )
        self.roi = roi_array
        self.show_roi = show_roi
        return self

    def configure_display(self, show_peaks: bool = True, show_amplitudes: bool = True, show_roi: bool = True):
        """
        Quick method to enable/disable certain display features.
        """
        self.show_peaks = show_peaks
        self.show_amplitudes = show_amplitudes
        self.show_roi = show_roi
        return self

    def set_title(self, title: str):
        """
        Set a title for the final plot(s).
        """
        self.title = title
        return self

    def add_custom_curves(
        self,
        curve_function: callable,
        label: str = "CustomCurves",
        color: str = "green",
        style: str = "--",
        **kwargs_arrays
    ):
        """
        Adds a function-based set of curves to be plotted for each sample, possibly multiple curves per sample.

        Parameters
        ----------
        curve_function : callable
            A function with signature: curve_function(x_values, **param_dict) -> 1D array.
        label : str
            Plot label for the legend.
        color : str
            Plot color for the curves.
        style : str
            Linestyle, e.g. '--' or '-.' or ':'.
        **kwargs_arrays : dict of np.ndarray
            Each array must have shape (n_samples, n_curves), so for sample i,
            we have 'n_curves' sets of parameter values. We'll iterate over each
            curve index j for that sample.

        Example
        -------
        def gaussian_curve(x, pos, width, amp):
            return amp * np.exp(-0.5 * ((x - pos)/width)**2)

        # positions.shape = widths.shape = amps.shape = (n_samples, n_curves)
        # usage:
        # plotter.add_custom_curves(
        #     gaussian_curve, label="Gaussians", color="magenta", style="--",
        #     pos=positions, width=widths, amp=amps
        # )
        """
        # Basic shape checks
        # All arrays in kwargs_arrays should have the same shape => (n_samples, n_curves).
        shapes = [arr.shape for arr in kwargs_arrays.values()]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"All keyword arrays must have the same shape. Got shapes: {shapes}"
            )

        # Optionally, check that shape[0] == n_samples (if signals is known)
        if self.signals is not None:
            n_samples = self.signals.shape[0]
            for name, arr in kwargs_arrays.items():
                if arr.shape[0] != n_samples:
                    raise ValueError(
                        f"{name}.shape[0] = {arr.shape[0]} != n_samples={n_samples}"
                    )

        self._custom_curves.append({
            "curve_func": curve_function,
            "label": label,
            "color": color,
            "style": style,
            "kwargs_arrays": kwargs_arrays
        })
        return self

    ###########################################################################
    # The main plot method
    ###########################################################################

    def plot(
        self,
        sample_indices: list = None,
        n_examples: int = 4,
        n_columns: int = 2,
        random_select: bool = False
    ):
        """
        Display multiple signals in a grid of subplots. Optionally,
        show peaks (positions, amplitudes) and ROI or custom curves.

        Parameters
        ----------
        sample_indices : list of int, optional
            Which signal indices to plot. If None, we auto-select.
        n_examples : int, optional
            How many total signals to plot (if sample_indices is None).
        n_columns : int, optional
            How many columns in the subplot grid.
        random_select : bool, optional
            If True, and sample_indices is None, randomly select signals to plot.
        """
        if self.signals is None:
            raise ValueError("No signals to plot. Please call add_signals(...) first.")

        n_samples, sequence_length = self.signals.shape
        x_vals = self.x_values

        # 1) Decide which samples to plot
        if sample_indices is None:
            if random_select:
                sample_indices = np.random.choice(n_samples, size=min(n_examples, n_samples), replace=False)
            else:
                sample_indices = np.arange(min(n_examples, n_samples))
        else:
            sample_indices = sample_indices[:n_examples]

        # 2) Create subplots
        n_actual = len(sample_indices)
        n_rows = int(np.ceil(n_actual / n_columns))
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(5*n_columns, 4*n_rows), squeeze=False)

        # 3) Iterate over chosen samples
        for i, idx in enumerate(sample_indices):
            row = i // n_columns
            col = i % n_columns
            ax = axes[row, col]

            # Plot the main signal
            ax.plot(x_vals, self.signals[idx], label=f"Signal #{idx}", color='blue')

            # If we have a binary ROI array, shade it where ROI=1
            if (self.roi is not None) and self.show_roi:
                roi_mask = self.roi[idx]
                ax.fill_between(
                    x_vals,
                    0,
                    1,
                    where=(roi_mask > 0),
                    color='green',
                    alpha=0.2,
                    transform=ax.get_xaxis_transform(),
                    label="ROI"
                )

            # Optionally plot peaks
            if self.show_peaks and (self.positions is not None) and (self.amplitudes is not None):
                for (px, py) in zip(self.positions[idx], self.amplitudes[idx]):
                    if py <= 0:
                        continue  # skip "inactive" peaks
                    ax.scatter(px, py, color='red', s=40, alpha=0.8, marker='o')
                    if self.show_amplitudes:
                        ax.text(px, py, f"{py:.2f}", fontsize=8, ha='left', va='bottom', color='red')

            # Plot custom curves
            for curve_info in self._custom_curves:
                curve_func = curve_info["curve_func"]
                label = curve_info["label"]
                color = curve_info["color"]
                style = curve_info["style"]
                kwargs_arrays = curve_info["kwargs_arrays"]

                # All arrays in kwargs_arrays have shape (n_samples, n_curves).
                # We'll plot n_curves curves for this sample i.
                # For each j, gather param_j = {k: v[i, j] for (k,v) in kwargs_arrays.items()}
                shapes = [
                    arr.shape for arr in kwargs_arrays.values()
                ]
                # e.g. (n_samples, n_curves) => shape[1] is n_curves
                n_curves_sample = shapes[0][1]  # second dimension

                for j in range(n_curves_sample):
                    # Build dict of param => scalar value for sample i, curve j
                    param_dict = {
                        k: v[idx, j] for k, v in kwargs_arrays.items()
                    }
                    y_curve = curve_func(x_vals, **param_dict)
                    # Label only the first curve
                    curve_label = label if j == 0 else None
                    ax.plot(
                        x_vals,
                        y_curve,
                        style,
                        color=color,
                        label=curve_label
                    )

            ax.set_xlabel("x-values")
            ax.set_ylabel("Signal amplitude")
            ax.set_title(f"Signal #{idx}")
            ax.legend()

        # Hide any unused subplots
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
