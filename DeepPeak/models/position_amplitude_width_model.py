from tensorflow.keras import layers, models  # type: ignore

def get_position_amplitude_width_model(input_length: int, max_peak_count: int):
    input_layer = layers.Input(shape=(input_length, 1))
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    positions = layers.Dense(max_peak_count, activation='linear', name="positions")(x)
    amplitudes = layers.Dense(max_peak_count, activation='linear', name="amplitudes")(x)
    widths = layers.Dense(max_peak_count, activation='linear', name="widths")(x)
    return models.Model(inputs=input_layer, outputs=[positions, amplitudes, widths])
