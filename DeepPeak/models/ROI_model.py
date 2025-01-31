import tensorflow as tf
from tensorflow.keras import layers, models  #type: ignore

def build_ROI_model(sequence_length):
    inputs = tf.keras.Input(shape=(sequence_length, 1))
    down_up_scaling = 2
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
    ROI = layers.Conv1D(1, kernel_size=1, activation='sigmoid', name='ROI')(x)

    model = models.Model(inputs, outputs={'ROI': ROI})

    model.compile(optimizer='adam', loss={'ROI': 'binary_crossentropy'}, metrics=['accuracy'])

    return model