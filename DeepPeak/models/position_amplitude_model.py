from tensorflow.keras import layers, models  # type: ignore

def get_position_amplitude_model(input_length: int, max_peak_count: int) -> models.Model:
    """
    Build a deep learning model for predicting peak positions and amplitudes.

    Parameters
    ----------
    input_length : int
        Length of the input signal.
    max_peak_count : int
        Maximum number of peaks the model will predict.

    Returns
    -------
    tensorflow.keras.Model
        The compiled Keras model for peak position and amplitude prediction.
    """
    # Input layer
    input_layer = layers.Input(shape=(input_length, 1), name="input_signal")

    # Convolutional blocks
    x = layers.Conv1D(16, kernel_size=7, activation='relu', padding='same', name="conv1d_block1")(input_layer)
    x = layers.MaxPooling1D(pool_size=10, name="pooling_block1")(x)

    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', name="conv1d_block2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pooling_block2")(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', name="conv1d_block3")(x)
    x = layers.MaxPooling1D(pool_size=5, name="pooling_block3")(x)

    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same', name="conv1d_block4")(x)
    # x = layers.MaxPooling1D(pool_size=10, name="pooling_block4")(x)

    # Flatten and dense layers
    x = layers.Flatten(name="flatten_layer")(x)
    x = layers.Dropout(rate=0.2, name="dropout_layer")(x)
    x = layers.Dense(128, activation='relu', name="dense_layer")(x)

    x = layers.Dense(32, activation='relu', name="dense_layer-2")(x)

    # Separate outputs
    position_output = layers.Dense(max_peak_count, activation='relu', name="positions")(x)
    amplitude_output = layers.Dense(max_peak_count, activation='linear', name="amplitudes")(x)

    # Build model
    model = models.Model(inputs=input_layer, outputs=[position_output, amplitude_output], name="position_amplitude_model")

    return model
