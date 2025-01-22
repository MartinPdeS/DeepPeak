
def simple(input_length: int, max_peak_count: int) -> models.Model:
    input_layer = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(16, 7, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)

    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(5)(x)

    x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)

    x = layers.MaxPooling1D(5)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu')(x)

    x = layers.Dense(max_peak_count, activation='relu', name="peak_positions")(x)

    return models.Model(inputs=input_layer, outputs=x)