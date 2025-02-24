
from DeepPeak.utils.visualization import visualize_validation_cases
import numpy as np
from DeepPeak.signals import generate_square_dataset
from DeepPeak.utils.visualization import plot_dataset  # Replace 'your_module' with the actual module name
# from DeepPeak import models
# from DeepPeak.signals import generate_gaussian_dataset
# from DeepPeak.utils.visualization import plot_training_history, visualize_validation_cases
# from DeepPeak.utils.training_utils import dataset_split
# from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
# from tensorflow.keras.models import load_model  # type: ignore
# from DeepPeak.utils.visualization import plot_dataset
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow import keras


# def build_position_model(input_shape):
#     """
#     Builds a model that predicts a single scalar: the position of the peak.
#     """
#     inputs = tf.keras.Input(shape=input_shape)

#     # --- Encoder ---
#     x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
#     x = layers.MaxPooling1D(pool_size=2)(x)
#     x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
#     x = layers.MaxPooling1D(pool_size=2)(x)

#     # Bottleneck (high-level features)
#     x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)

#     # --- Flatten or Global Pool ---
#     x = layers.GlobalAveragePooling1D()(x)  # shape: (batch_size, 128)

#     # --- Final Dense for Peak Position ---
#     outputs = layers.Dense(4, activation=None)(x)  # Regress a single position

#     model = models.Model(inputs=inputs, outputs={'positions': outputs, 'amplitudes': outputs})
#     return model


# # Dataset generation parameters
# sample_count = 1000
# sequence_length = 128
# peak_count = (1, 4)
# amplitude_range = (1, 150)
# center_range = (0.1, 0.9)
# width_range = 0.04
# noise_std = 0.1
# normalize = False
# normalize_x = True

# # Generate the dataset
# signals, amplitudes, peak_counts, positions, widths, x_values, labels = generate_gaussian_dataset(
#     sample_count=sample_count,
#     sequence_length=sequence_length,
#     peak_count=peak_count,
#     amplitude_range=amplitude_range,
#     center_range=center_range,
#     width_range=width_range,
#     noise_std=noise_std,
#     normalize=normalize,
#     normalize_x=normalize_x,
#     nan_values=0,
#     sort_peak='position',
#     categorical_peak_count=True,
#     probability_range=(0.7, 0.7)
# )


# # Train-test split
# dataset = dataset_split(
#     signals=signals,
#     positions=positions,
#     amplitudes=amplitudes,
#     peak_counts=peak_counts,
#     labels=labels,
#     widths=widths,
#     test_size=0.2,
#     random_state=None,
# )

# # Build the model
# model = build_position_model((sequence_length, 1))
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='mse')

# # Train the model
# history = model.fit(
#     dataset['train']['signals'], dataset['train']['positions'],
#     validation_data=(dataset['test']['signals'], dataset['test']['positions']),
#     epochs=10,
#     batch_size=32
# )


# visualize_validation_cases(
#     model=model,
#     validation_data=dataset['test'],
#     model_type='gaussian',
#     sequence_length=128,
#     num_examples=5,
#     n_columns=5,
#     unit_size=(3.5, 2.5),
#     normalize_x=True
# )