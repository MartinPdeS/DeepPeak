from tensorflow.keras import layers, models # type: ignore

class NameSpace():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)




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

def advanced(input_length: int, max_peak_count: int):
    input_layer = layers.Input(shape=(input_length, 1))

    # Convolutional block 1
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Convolutional block 2
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Multi-output predictions
    x = layers.Dense(max_peak_count, activation='linear', name="peak_positions")(x)

    return models.Model(inputs=input_layer, outputs=x)


def _get_advanced_peak_model(input_length: int, max_peak_count: int):
    input_layer = layers.Input(shape=(input_length, 1))

    # Convolutional block 1
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Convolutional block 2
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Convolutional block 3
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Multi-output predictions
    peak_positions_output = layers.Dense(max_peak_count, activation='sigmoid', name="peak_positions")(x)  # Predict normalized positions


    return models.Model(inputs=input_layer, outputs=peak_positions_output)
