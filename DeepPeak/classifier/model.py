import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

def build_auto_encoder(sequence_length: int, dropout_rate: float = 0.3) -> models.Model:
    """
    Build a 1D convolutional autoencoder-like model for predicting an ROI (Region Of Interest) mask.

    The network consists of an encoder-decoder structure with several Conv1D, MaxPooling,
    and UpSampling layers, concluding with a final Conv1D (kernel_size=1) layer using a
    sigmoid activation to produce the ROI mask. Dropout layers are included in both the
    encoder and bottleneck parts of the network for regularization.

    Parameters
    ----------
    sequence_length : int
        Length of the input 1D sequence (time series or spatial data).
    dropout_rate : float, optional
        The probability of dropping units during training in dropout layers.
        Default value is 0.3.

    Returns
    -------
    model : tensorflow.keras.models.Model
        A compiled Keras model with the following structure:
          - Input: shape (sequence_length, 1)
          - Encoder: Conv1D layers (32, 64, 128 filters), Dropout, MaxPooling
          - Decoder: UpSampling, Conv1D layers mirroring the encoder
          - Output 'ROI': 1D mask with shape (sequence_length, 1) and sigmoid activation

    Notes
    -----
    The model is compiled with:
      - Optimizer: 'adam'
      - Loss: 'binary_crossentropy' for the 'ROI' output
      - Metrics: ['accuracy']

    Example
    -------
    >>> model = build_ROI_model(sequence_length=128, dropout_rate=0.3)
    >>> model.summary()  # Inspect the constructed layers
    """
    inputs = tf.keras.Input(shape=(sequence_length, 1))

    # Encoder with Dropout
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Bottleneck
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Decoder
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)

    # Output Layer
    ROI = layers.Conv1D(1, kernel_size=1, activation='sigmoid', name='ROI')(x)

    model = models.Model(inputs, outputs={'ROI': ROI})
    model.compile(optimizer='adam', loss={'ROI': 'binary_crossentropy'}, metrics=['accuracy'])

    return model


def build_dense_layers(sequence_length: int) -> tf.keras.Model:
    """
    Builds a dense 1D convolutional neural network for peak detection.

    This model consists of three 1D convolutional layers with increasing filter sizes.
    It has two outputs:
    - `peak_output`: A per-time-step binary classification output (1 if peak, 0 otherwise).
    - `count_output`: A regression output predicting the total number of peaks in the sequence.

    Parameters
    ----------
    sequence_length : int
        The length of the input sequences.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model for peak detection with two outputs.

    Notes
    -----
    - The model uses ReLU activations in the convolutional layers.
    - The `peak_output` uses a sigmoid activation for binary classification.
    - The `count_output` uses a dense layer with ReLU activation to ensure non-negative peak count predictions.
    """
    input_shape = (sequence_length, 1)
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional layers (WaveNet-like)
    x = layers.Conv1D(32, kernel_size=3, dilation_rate=1, activation="relu", padding="same")(inputs)
    x = layers.Conv1D(64, kernel_size=3, dilation_rate=2, activation="relu", padding="same")(x)
    x = layers.Conv1D(128, kernel_size=3, dilation_rate=4, activation="relu", padding="same")(x)

    # Output for per-time-step classification (1 if peak, 0 if not)
    peak_output = layers.Conv1D(1, kernel_size=1, activation="sigmoid", padding="same", name="peak_output")(x)

    # Auxiliary output: Predict the total number of peaks (Regression)
    count_output = layers.GlobalAveragePooling1D()(x)
    count_output = layers.Dense(1, activation="relu", name="count_output")(count_output)  # Ensures non-negative count

    # Create model with two outputs
    model = models.Model(inputs, [peak_output, count_output])
    model.compile(
        optimizer="adam",
        loss={"peak_output": "binary_crossentropy", "count_output": "mean_squared_error"},  # MSE for peak count regression
        metrics={"peak_output": "accuracy", "count_output": "mae"}
    )

    return model


def build_wavenet(sequence_length: int, num_filters: int = 64, num_dilation_layers: int = 6) -> tf.keras.Model:
    """
    Builds a WaveNet-style deep learning model for peak detection in time-series data.

    This model consists of multiple dilated convolutional layers, following the WaveNet architecture.
    It applies increasing dilation rates to capture both local and long-range dependencies in the input sequence.

    Parameters
    ----------
    sequence_length : int
        The length of the input sequences.
    num_filters : int, optional
        The number of filters in each convolutional layer (default is 64).
    num_dilation_layers : int, optional
        The number of dilated convolutional layers (default is 6).

    Returns
    -------
    tf.keras.Model
        A compiled Keras model for peak detection.

    Notes
    -----
    - Uses dilated causal convolutions to preserve sequence ordering.
    - Applies residual connections for better gradient flow.
    - Skip connections are aggregated before passing to the final output layer.
    - The model predicts a per-time-step binary classification (1 for peak, 0 otherwise).
    """
    input_shape = (sequence_length, 1)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    skip_connections = []

    for i in range(num_dilation_layers):
        dilation_rate = 2 ** i  # Exponential dilation
        x = layers.Conv1D(num_filters, kernel_size=3, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)

        # Residual connection
        res = layers.Conv1D(num_filters, kernel_size=1, padding="same")(x)
        x = layers.Add()([x, res])

        # Skip connection
        skip_connections.append(layers.Conv1D(num_filters, kernel_size=1, padding="same")(x))

    # Combine all skip connections
    x = layers.Add()(skip_connections)
    x = layers.ReLU()(x)

    # Output: 1D Conv to produce per-step binary classification
    outputs = layers.Conv1D(1, kernel_size=1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model