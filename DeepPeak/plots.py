import numpy as np
import re
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.models import Model  # type: ignore
from MPSPlots.styles import mps
import matplotlib.pyplot as plt
import numpy as np


def get_gaussian(amplitude, x, center, width):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def plot_dense(model, input_signal, layer_name):
    """
    Plot activations for a Dense layer.

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

    # Plot the activations
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(activations[0])), activations[0], color="blue")
    plt.title(f"Activations for {layer_name}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Value")
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
    ground_truth_output = outputs[signal_index : signal_index + 1]

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

def plot_training_history(history, filtering: list = None):
    """
    Plot training and validation performance metrics (loss and accuracy).

    Parameters
    ----------
    history : tensorflow.keras.callbacks.History
        The training history object from model.fit().
    filtering : list of str, optional
        List of wildcard patterns to filter the keys in the history dictionary. Use '*' as a wildcard.
    """
    plt.close('all')

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

    axes.flatten()[-1].set_xlabel('Number of Epochs')
    figure.suptitle('Training History')
    plt.tight_layout()
    plt.show()



def visualize_validation_cases(
    model,
    validation_data,
    input_length: int,
    num_examples: int = 5,
    n_columns: int = 1,
    ax_size: tuple = (3, 3)
):
    """
    Visualize validation cases by comparing true and predicted values.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The trained Keras model.
    validation_data : dict
        Dictionary containing validation data and labels:
        {'x': input signals, 'peak_count': ground truth peak counts, 'positions': peak positions, 'widths': peak widths, 'amplitudes': peak amplitudes}.
    input_length : int
        Length of the input signal.
    num_examples : int, optional
        Number of validation cases to visualize. Default is 5.
    n_columns : int, optional
        Number of columns in the plot layout. Default is 1.
    """
    # Close any previous plots
    plt.close('all')

    # Determine rows and columns for subplots
    n_rows = int(np.ceil(num_examples / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(ax_size[0] * n_columns, ax_size[1] * n_rows))
    axes = np.atleast_1d(axes).ravel()

    # Randomly sample indices to visualize
    indices = np.random.choice(len(validation_data['x']), num_examples, replace=False)
    x_values = np.arange(input_length)

    for ax, idx in zip(axes, indices):
        # Extract input signal and true labels
        input_signal = validation_data['x'][idx, :, 0]
        true_num_peaks = np.argmax(validation_data['peak_count'][idx])

        true_positions = validation_data['positions'][idx][:true_num_peaks]
        true_widths = validation_data['widths'][idx][:true_num_peaks]
        true_amplitudes = validation_data['amplitudes'][idx][:true_num_peaks]

        # Predict peak positions
        reshaped_signal = input_signal.reshape(1, input_length, 1)
        predictions = model.predict(reshaped_signal, verbose=0)[0]

        # Plot input signal
        ax.plot(input_signal, label='Input Signal', color='blue')

        # Plot true peaks as dashed lines
        for pos, amp, width in zip(true_positions, true_amplitudes, true_widths):
            gaussian_curve = amp * np.exp(-((x_values - pos)**2) / (2 * width**2))
            ax.plot(x_values, gaussian_curve, linestyle='--', color='green', linewidth=1, label='True Peak' if 'True Peak' not in ax.get_legend_handles_labels()[1] else None)

        # Plot predicted peaks as dotted lines
        for pred_pos in predictions:
            ax.axvline(pred_pos, linestyle='dotted', color='red', linewidth=1, label='Predicted Peak' if 'Predicted Peak' not in ax.get_legend_handles_labels()[1] else None)

        # Formatting
        ax.set_title(f"Validation Case {idx} -- {true_num_peaks} peaks")
        ax.set_xlabel("Signal Index")
        ax.set_ylabel("Amplitude")
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
