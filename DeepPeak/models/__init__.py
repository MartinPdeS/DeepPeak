from .position_model import get_position_model
from .count_model import get_count_model
from .position_amplitude_model import get_position_amplitude_model
from .position_amplitude_width_model import get_position_amplitude_width_model
from .ROI_model import build_ROI_model
from tensorflow import keras
import tensorflow as tf

@keras.utils.register_keras_serializable()
def permutation_invariant_loss(y_true, y_pred):
    # y_true and y_pred shape: (batch_size, max_peak_count)
    # Sort each row
    y_true_sorted = tf.sort(y_true, axis=1)
    y_pred_sorted = tf.sort(y_pred, axis=1)

    # Compute MSE between sorted arrays
    return tf.reduce_mean(tf.square(y_true_sorted - y_pred_sorted))