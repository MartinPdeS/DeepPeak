import numpy as np
from DeepPeak.signals import generate_gaussian_dataset
from DeepPeak.utils.training_utils import dataset_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def plot_predictions(signals, labels, predictions, sample_count=5):
    """
    Plot predictions with 'fill_between' for the prediction probabilities.

    Parameters:
    ----------
    signals : numpy.ndarray
        Input signals of shape (n_samples, sequence_length, 1).
    labels : numpy.ndarray
        Ground truth labels of shape (n_samples, sequence_length, 1).
    predictions : numpy.ndarray
        Model predictions of shape (n_samples, sequence_length, 1).
    sample_count : int
        Number of samples to visualize.
    """
    figure, axes = plt.subplots(nrows=sample_count, ncols=1, figsize=(10, 3 * sample_count), squeeze=False, sharex=True)

    for i, ax in enumerate(axes.flatten()):


        signal = signals[i].squeeze()
        label = labels[i].squeeze()
        prediction = predictions[i].squeeze()

        # Plot the signal
        ax.plot(signal, label='Signal', color='blue')

        # Fill between for predictions
        twin = ax.twinx()
        twin.fill_between(
            np.arange(len(prediction)),
            y1=0,
            y2=prediction,
            color='red',
            alpha=0.3,
            label='Prediction'
        )
        twin.legend(loc='lower right')

        # Plot the ground truth
        # ax.plot(label, label='Ground Truth', color='green', linestyle='--')


        twin.fill_between(
            np.arange(len(prediction)),
            y1=0,
            y2=1,
            color='green',
            where=label == 1,
            alpha=0.3,
            label='Prediction'
        )


        ax.set_title(f"Sample {i + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right')
        ax.grid()

    plt.tight_layout()

    plt.show()


def build_peak_detection_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Bottleneck
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)

    # Decoder
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)

    # Output Layer
    outputs = layers.Conv1D(1, kernel_size=1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model



# Dataset generation parameters
sample_count = 6000
sequence_length = 128
peak_count = (1, 4)
amplitude_range = (1, 150)
center_range = (0.1, 0.9)
width_range = 0.04
noise_std = 0.1
normalize = False
normalize_x = True

# Generate the dataset
signals, amplitudes, peak_counts, positions, widths, x_values, labels = generate_gaussian_dataset(
    sample_count=sample_count,
    sequence_length=sequence_length,
    peak_count=peak_count,
    amplitude_range=amplitude_range,
    center_range=center_range,
    width_range=width_range,
    noise_std=noise_std,
    normalize=normalize,
    normalize_x=normalize_x,
    nan_values=0,
    sort_peak='position',
    categorical_peak_count=True,
    probability_range=(0.7, 0.7)
)


# Train-test split
dataset = dataset_split(
    signals=signals,
    positions=positions,
    amplitudes=amplitudes,
    peak_counts=peak_counts,
    labels=labels,
    widths=widths,
    test_size=0.2,
    random_state=None,
)

# Build the model
model = build_peak_detection_model((sequence_length, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    dataset['train']['signals'], dataset['train']['labels'],
    validation_data=(dataset['test']['signals'], dataset['test']['labels']),
    epochs=30,
    batch_size=32
)
predictions = model.predict(dataset['test']['signals'])

plot_predictions(dataset['test']['signals'], dataset['test']['labels'], predictions, sample_count=10)