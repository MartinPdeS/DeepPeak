import tensorflow as tf

# ---------- Generic metaclass factory ----------

class KerasMetricFactoryMeta(type):
    """
    Metaclass that makes a class act as a factory for a Keras Metric.

    Any class using this metaclass must define:
      - metric_cls: the tf.keras.metrics.Metric subclass to instantiate
      - name (optional): default metric name if none is provided at call
      - default_kwargs (optional): dict of default kwargs for `metric_cls`
      - make_name(kwargs) (optional): function to create a dynamic default name
    """
    def __call__(cls, *args, **kwargs):
        metric_cls = getattr(cls, "metric_cls", None)
        if metric_cls is None:
            raise TypeError(f"{cls.__name__} must define `metric_cls`.")

        defaults = (getattr(cls, "default_kwargs", {}) or {}).copy()
        merged = {**defaults, **kwargs}

        # dynamic name support
        if "name" in merged:
            metric_name = merged.pop("name")
        elif hasattr(cls, "make_name"):
            metric_name = cls.make_name(merged)  # type: ignore[attr-defined]
        else:
            metric_name = getattr(cls, "name", cls.__name__)

        # Most tf.keras metrics ignore *args; pass merged kwargs
        return metric_cls(name=metric_name, **merged)

# ---------- Metric factory wrappers (call to get a metric instance) ----------

class BinaryIoU(metaclass=KerasMetricFactoryMeta):
    """Binary IoU on thresholded predictions."""
    metric_cls = tf.keras.metrics.BinaryIoU
    name = "BinaryIoU"
    default_kwargs = {"target_class_ids": [1], "threshold": 0.5}

class BinaryAccuracy(metaclass=KerasMetricFactoryMeta):
    """Element-wise binary accuracy with threshold."""
    metric_cls = tf.keras.metrics.BinaryAccuracy
    name = "BinaryAccuracy"
    default_kwargs = {"threshold": 0.5}

class Precision(metaclass=KerasMetricFactoryMeta):
    """Binary precision at threshold."""
    metric_cls = tf.keras.metrics.Precision
    name = "Precision"
    default_kwargs = {"thresholds": 0.5}

class Recall(metaclass=KerasMetricFactoryMeta):
    """Binary recall at threshold."""
    metric_cls = tf.keras.metrics.Recall
    name = "Recall"
    default_kwargs = {"thresholds": 0.5}

class AUROC(metaclass=KerasMetricFactoryMeta):
    """Area under ROC curve (threshold-free)."""
    metric_cls = tf.keras.metrics.AUC
    name = "AUC-ROC"
    default_kwargs = {"curve": "ROC", "from_logits": False}

class AUCPR(metaclass=KerasMetricFactoryMeta):
    """Area under Precision–Recall curve (threshold-free)."""
    metric_cls = tf.keras.metrics.AUC
    name = "AUC-PR"
    default_kwargs = {"curve": "PR", "from_logits": False}

class MeanIoU(metaclass=KerasMetricFactoryMeta):
    """Mean IoU for multiclass segmentation; pass num_classes=N."""
    metric_cls = tf.keras.metrics.MeanIoU
    name = "MeanIoU"
    default_kwargs = {}  # require num_classes at call time

class SparseTopKAcc(metaclass=KerasMetricFactoryMeta):
    """Top-K accuracy for sparse labels (multiclass classification)."""
    metric_cls = tf.keras.metrics.SparseTopKCategoricalAccuracy
    default_kwargs = {"k": 5}
    @staticmethod
    def make_name(kwargs):
        return f"Top{kwargs.get('k', 5)}Acc"
