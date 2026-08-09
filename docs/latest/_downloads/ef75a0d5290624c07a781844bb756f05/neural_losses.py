"""DeepPeak neural losses and configurable training."""

# %%
import numpy as np
import tensorflow as tf

from DeepPeak.models import (
    ShapeAwarePulseLoss,
    SmoothBinaryCrossentropy,
    TrainingConfig,
    WeightedBinaryCrossentropy,
    WeightedHuber,
)

# %%
# The reconstruction losses accept targets and predictions with either
# ``(batch, length)`` or ``(batch, length, 1)`` shapes.
y_true = tf.constant(np.random.default_rng(42).random((4, 128, 1)), dtype=tf.float32)
y_pred = tf.constant(np.random.default_rng(43).random((4, 128, 1)), dtype=tf.float32)

losses = {
    "weighted huber": WeightedHuber(delta=0.25, alpha=2.0),
    "shape aware pulse": ShapeAwarePulseLoss(
        amplitude_weight=1.0,
        shape_weight=0.25,
        smoothness_weight=0.05,
    ),
    "weighted BCE": WeightedBinaryCrossentropy(alpha=2.0),
    "smooth BCE": SmoothBinaryCrossentropy(alpha=2.0, smoothness_weight=0.02),
}

values = {name: float(loss(y_true, y_pred).numpy()) for name, loss in losses.items()}
print(values)

# %%
# Pass the same configuration to DenseNet, WaveNet, or UNet1D:
#
#     model.fit(signals, clean_signals, config=TrainingConfig(
#         epochs=50, validation_split=0.2, patience=6,
#     ))
config = TrainingConfig(epochs=20, validation_split=0.2, patience=5)
print(config)
