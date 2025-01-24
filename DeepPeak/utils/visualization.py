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

def plot_training_history(history, filtering: list = None, y_scale: str = 'log'):
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

    with plt.style.context(mps):
        figure, axes = plt.subplots(
            nrows=len(history_dict),
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(8, 3 * len(history_dict))
        )

    for ax, (key, value) in zip(axes.flatten(), history_dict.items()):
        ax.plot(value, label=key.replace('_', ' '))
        ax.legend(loc='upper left')
        ax.set_ylabel(key.replace('_', ' '))
        ax.set_yscale(y_scale)

    axes.flatten()[-1].set_xlabel('Number of Epochs')
    figure.suptitle('Training History')

    plt.tight_layout()
    plt.show()



def plot_gaussian_components(ax, x_values, positions, amplitudes, widths):
    """
    Plot individual Gaussian components on the provided axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot the Gaussians on.
    x_values : np.ndarray
        The x-values for plotting the Gaussian curves.
    positions : np.ndarray
        The positions of the Gaussian peaks.
    amplitudes : np.ndarray
        The amplitudes of the Gaussian peaks.
    widths : np.ndarray
        The widths of the Gaussian peaks.
    """
    for pos, amp, width in zip(positions, amplitudes, widths):
        if pos != 0:  # Ignore placeholder positions
            gaussian = amp * np.exp(-((x_values - pos)**2) / (2 * width**2))
            ax.plot(x_values, gaussian, linestyle='--', color='green', linewidth=1, label='True Gaussian')

def visualize_validation_cases(model, validation_data, sequence_length: int, num_examples: int = 5, n_columns: int = 1, unit_size: tuple = (3.5, 2.5), normalize_x: bool = True):
    """
    Visualize validation cases by comparing true and predicted values.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The trained Keras model.
    validation_data : dict
        Dictionary containing validation data and labels:
        {'x': input signals, 'peak_count': ground truth peak counts, 'positions': peak positions, 'widths': peak widths, 'amplitudes': peak amplitudes}.
    sequence_length : int
        Length of the input signal.
    num_examples : int, optional
        Number of validation cases to visualize. Default is 5.
    n_columns : int, optional
        Number of columns in the plot layout. Default is 1.
    """

    n_rows = int(np.ceil(num_examples / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, sharex=True, sharey=True, figsize=(unit_size[0] * n_columns, unit_size[1] * n_rows))
    axes = np.atleast_1d(axes).ravel()

    indices = np.random.choice(len(validation_data['x']), num_examples, replace=False)

    if normalize_x:
        x_values = np.linspace(0, 1, sequence_length)
    else:
        x_values = np.arange(sequence_length)  # Shared x-axis for all sequences

    for ax, idx in zip(axes, indices):
        input_signal = validation_data['x'][idx, :, 0]
        true_positions = validation_data['positions'][idx]
        true_amplitudes = validation_data['amplitudes'][idx]
        true_widths = validation_data['widths'][idx]
        peak_count = np.argmax(validation_data['peak_count'][idx])

        # Predict peak positions
        reshaped_signal = input_signal.reshape(1, sequence_length, 1)


        # Plot input signal
        ax.plot(x_values, input_signal, label='Input Signal', color='blue')

        # Plot true Gaussian components
        plot_gaussian_components(ax, x_values, true_positions, true_amplitudes, true_widths)

        # Plot predicted positions as vertical lines

        attributes_set = set(model.output_names)
        if attributes_set == {'positions',}:
            predictions = np.asarray(model.predict(reshaped_signal, verbose=0))[0]
            for position in predictions:
                ax.axvline(position, linestyle='dotted', color='red', linewidth=1, label='Predicted Position')

        elif attributes_set == {'positions', 'amplitudes'}:
            positions, amplitudes = np.asarray(model.predict(reshaped_signal, verbose=0))
            for position, amplitude in zip(positions, amplitudes):
                ax.scatter(position, amplitude, color='red', label='Predicted Peak')

        # Formatting
        ax.set_title(f"Validation Case {idx} \n peak count: {peak_count}")
        ax.set_xlabel("Signal Index")
        ax.set_ylabel("Amplitude")
        ax.legend()

    # Remove duplicate legend entries
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()

def plot_dataset(
    signals: np.ndarray,
    amplitudes: np.ndarray = None,
    positions: np.ndarray = None,
    widths: np.ndarray = None,
    x_values: np.ndarray = None,
    num_samples: int = 5,
    title: str = "Dataset Visualization"):
    """
    Plots a subset of sequences from the generated dataset.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape (sample_count, sequence_length, 1) containing sequences with peaks.
    amplitudes : np.ndarray, optional
        Array of shape (sample_count, max_peaks) containing amplitudes of each peak.
    positions : np.ndarray, optional
        Array of shape (sample_count, max_peaks) containing positions of peaks.
    widths : np.ndarray, optional
        Array of shape (sample_count, max_peaks) containing widths of peaks.
    x_values : np.ndarray, optional
        The x-axis values, either normalized (0 to 1) or integer indices.
    num_samples : int, optional
        Number of samples to plot. Default is 5.
    title : str, optional
        Title of the plot. Default is "Dataset Visualization".

    """
    if len(signals.shape) != 3 or signals.shape[2] != 1:
        raise ValueError("Signals must be a 3D array with shape (sample_count, sequence_length, 1).")

    sample_count, sequence_length, _ = signals.shape
    num_samples = min(num_samples, sample_count)

    # Use default x_values if not provided
    if x_values is None:
        x_values = np.linspace(0, 1, sequence_length)

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 2), sharex=True)
    if num_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        signal = signals[i, :, 0]
        ax.plot(x_values, signal, label="Signal", linewidth=1.5)

        if positions is not None and amplitudes is not None:
            for pos, amp in zip(positions[i], amplitudes[i]):
                if not np.isnan(pos):
                    ax.plot(
                        [x_values[int(pos)]] * 2,
                        [0, amp],
                        label=f"Peak at {pos:.1f}",
                        linestyle="--",
                        alpha=0.7
                    )

        if widths is not None and positions is not None:
            for pos, width in zip(positions[i], widths[i]):
                if not np.isnan(pos):
                    start = max(0, int(pos - width // 2))
                    end = min(sequence_length, int(pos + width // 2))
                    ax.axvspan(
                        x_values[start],
                        x_values[end - 1],
                        color="red",
                        alpha=0.2,
                        label="Peak Width" if i == 0 else None
                    )

        ax.set_ylabel(f"Sample {i + 1}")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("X-axis")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
