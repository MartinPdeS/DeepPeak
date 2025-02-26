from tensorflow.keras import layers, models # type: ignore

def get_count_model(input_length: int):
    input_layer = layers.Input(shape=(input_length, 1))
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name="peak_count")(x)
    return models.Model(inputs=input_layer, outputs=output)
