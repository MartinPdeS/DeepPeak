import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

def build_ROI_model(sequence_length: int, dropout_rate: float = 0.3) -> models.Model:
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


def filter_predictions(
        model: models.Model,
        signals: np.ndarray,
        n_samples: int = 30,
        threshold: float = 0.9,
        std_threshold: float = 0.1) -> np.ndarray:
    """
    Estimate a binarized ROI mask using Monte Carlo dropout sampling.

    This function repeatedly calls the model in training mode to obtain multiple
    stochastic predictions via dropout. It then computes the average (mean) and standard
    deviation across these predictions, applies a threshold to create a binary mask,
    and optionally returns the standard deviation as a measure of uncertainty.

    Parameters
    ----------
    model : tensorflow.keras.models.Model
        A trained Keras model with dropout layers. Must output a dictionary containing
        the key 'ROI'.
    signals : np.ndarray
        Input tensor of shape (batch_size, sequence_length, 1). The function
        repeatedly passes this data through the model in training mode.
    n_samples : int, optional
        Number of forward passes through the model. Default is 30.
    threshold : float, optional
        Probability threshold to binarize the mean prediction. Default is 0.9.

    Returns
    -------
    mean_prediction : np.ndarray
        Binarized prediction (0 or 1) based on the threshold. The shape will match
        (batch_size, sequence_length, 1). Values below the threshold are set to 0,
        above or equal to threshold are set to 1.

    Notes
    -----
    - Internally, the function also computes the standard deviation (std) across
      predictions. You can modify the code to return this `uncertainty` if needed.
    - This approach exploits dropout at inference time to estimate model uncertainty.

    Examples
    --------
    >>> model = build_ROI_model(128, dropout_rate=0.3)
    >>> # Assume model is trained...
    >>> test_data = np.random.rand(5, 128, 1)
    >>> filtered_mask = filter_predictions(model, test_data, n_samples=10, threshold=0.8)
    >>> print(filtered_mask.shape)
    (5, 128, 1)
    """
    predictions = np.array(
        [model(signals, training=True)['ROI'].numpy() for _ in range(n_samples)]
    )

    mean_prediction = predictions.mean(axis=0)

    uncertainty = predictions.std(axis=0)

    mean_prediction[mean_prediction < threshold] = 0
    mean_prediction[mean_prediction >= threshold] = 1

    std_mask = np.zeros_like(uncertainty)
    std_mask[uncertainty < std_threshold] = 1

    mean_prediction *= std_mask

    return mean_prediction.squeeze(), uncertainty.squeeze()

def mc_dropout_prediction(
        model: models.Model,
        signals: np.ndarray,
        num_samples: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Monte Carlo (MC) dropout to estimate the mean and uncertainty of ROI predictions.

    The function runs multiple stochastic forward passes (with dropout active) and
    returns the mean prediction and standard deviation across these passes.

    Parameters
    ----------
    model : tensorflow.keras.models.Model
        A trained Keras model that outputs {'ROI': ...} and contains dropout layers.
    signals : np.ndarray
        The input data to predict on, with shape (batch_size, sequence_length, 1).
    num_samples : int, optional
        Number of forward passes through the network. Default is 30.

    Returns
    -------
    mean_prediction : np.ndarray
        The mean of the ROI predictions across `num_samples` forward passes.
        Shape (batch_size, sequence_length, 1).
    uncertainty : np.ndarray
        The standard deviation (std) across the multiple predictions, same shape as
        `mean_prediction`.

    Notes
    -----
    - Dropout is forced to be active by calling the model with `training=True`.
    - The `uncertainty` metric is a simple std; you could use alternative uncertainty
      measures.

    Example
    -------
    >>> model = build_ROI_model(sequence_length=128, dropout_rate=0.3)
    >>> # Assume model is trained
    >>> input_data = np.random.rand(10, 128, 1)
    >>> mean_pred, std_pred = mc_dropout_prediction(model, input_data, num_samples=50)
    >>> print(mean_pred.shape, std_pred.shape)
    (10, 128, 1) (10, 128, 1)
    """
    predictions = np.array([
        model(signals, training=True)['ROI'].numpy() for _ in range(num_samples)
    ])
    mean_prediction = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_prediction, uncertainty
