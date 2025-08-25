
from tensorflow.keras import layers, models  # type: ignore


import tensorflow as tf
from tensorflow.keras import layers, models

# def get_position_model(input_length: int, max_peak_count: int) -> models.Model:
#     """
#     Builds a 1D CNN model that outputs 'max_peak_count' positions (indices) where peaks occur.

#     Args:
#         input_length (int): The length of the 1D input signal.
#         max_peak_count (int): The maximum number of peaks you want to detect.

#     Returns:
#         models.Model: A Keras model that outputs peak positions in a vector of length `max_peak_count`.
#     """
#     input_layer = layers.Input(shape=(input_length, 1))

#     # First convolution
#     x = layers.Conv1D(16, 7, activation='relu', padding='same')(input_layer)
#     x = layers.MaxPooling1D(2)(x)

#     # Second convolution
#     x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)

#     # Third convolution
#     x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
#     x = layers.MaxPooling1D(5)(x)

#     # Fourth convolution
#     x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)
#     x = layers.MaxPooling1D(5)(x)

#     # Flatten the spatial dimension
#     x = layers.Flatten()(x)
#     x = layers.Dropout(0.2)(x)  # helps prevent overfitting

#     # Dense layers for regression
#     x = layers.Dense(128, activation='relu')(x)

#     # Output layer: regress the positions of up to `max_peak_count` peaks
#     # Using ReLU here, but you might use another activation (or none) depending on how you encode positions.
#     output = layers.Dense(max_peak_count, activation='relu', name="positions")(x)

#     return models.Model(inputs=input_layer, outputs=output)

def get_position_model(sequence_length: int, max_peak_count: int) -> models.Model:
    input_layer = layers.Input(shape=(sequence_length, 1))

    x = layers.Conv1D(8, 3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(5)(x)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)

    x = layers.MaxPooling1D(5)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu')(x)

    x = layers.Dense(max_peak_count, activation='sigmoid', name="positions")(x)

    return models.Model(inputs=input_layer, outputs=x)