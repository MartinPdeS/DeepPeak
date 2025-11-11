import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import models  # type: ignore
import matplotlib.pyplot as plt
from MPSPlots import helper


def compute_segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """
    Compute segmentation metrics between the predicted and ground truth ROI masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary ROI mask. Expected shape is (n_samples, sequence_length) (or any shape,
        as long as it matches true_mask).
    true_mask : np.ndarray
        Ground truth binary ROI mask. Must have the same shape as pred_mask.

    Returns
    -------
    metrics : dict
        Dictionary with the following keys:
          - "precision": The precision (positive predictive value).
          - "recall": The recall (sensitivity).
          - "f1_score": The harmonic mean of precision and recall.
          - "iou": Intersection-over-Union metric.
          - "dice": Dice coefficient.
    """
    # Flatten the arrays so that each pixel is a sample
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    # Compute precision, recall, and F1 score.
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average="binary"
    )

    # Compute Intersection over Union (IoU)
    intersection = np.logical_and(true_flat, pred_flat).sum()
    union = np.logical_or(true_flat, pred_flat).sum()
    iou = intersection / union if union > 0 else 0.0

    # Compute Dice coefficient: (2 * Intersection) / (Total area of both masks)
    dice = (
        (2 * intersection) / (true_flat.sum() + pred_flat.sum())
        if (true_flat.sum() + pred_flat.sum()) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "dice": dice,
    }


def roi_containment_metric(roi_pred: np.ndarray, roi_gt: np.ndarray) -> dict:
    """
    Compute the ROI containment metric, which quantifies the fraction of the predicted
    ROI that lies within the ground truth ROI.

    This metric is useful when you wish to evaluate whether the predicted ROI regions
    are completely within the ground truth regions. It is defined as:

        containment_ratio = (# predicted ROI pixels that are within ground truth) / (# predicted ROI pixels)

    A complementary misfit ratio is defined as:

        misfit_ratio = 1 - containment_ratio

    Parameters
    ----------
    roi_pred : numpy.ndarray
        A binary (0/1) numpy array of predicted ROI mask. Shape can be (H, W) or any
        shape as long as it matches roi_gt.
    roi_gt : numpy.ndarray
        A binary (0/1) numpy array of the ground truth ROI mask, of the same shape as roi_pred.

    Returns
    -------
    metrics : dict
        A dictionary containing:
            - 'containment_ratio': float
                  The fraction of predicted ROI pixels that are within the ground truth ROI.
            - 'misfit_ratio': float
                  The complement of the containment ratio (i.e., fraction of predicted ROI pixels
                  that are outside the ground truth ROI).
            - 'n_pred': int
                  The total number of predicted ROI pixels.
            - 'n_intersection': int
                  The number of predicted ROI pixels that lie within the ground truth ROI.

    Notes
    -----
    - If there are no predicted ROI pixels (i.e. roi_pred.sum() == 0), the function returns
      a containment ratio of 0 and misfit ratio of 1.
    - This metric only considers the predicted ROI pixels. It does not penalize missed ROI pixels
      in the ground truth (for that you might combine with recall-based metrics or IoU).

    Examples
    --------
    >>> roi_pred = np.array([[0, 1, 1], [0, 1, 0]])
    >>> roi_gt   = np.array([[0, 1, 0], [0, 1, 0]])
    >>> metrics = roi_containment_metric(roi_pred, roi_gt)
    >>> print(metrics['containment_ratio'])
    0.6666666666666666
    """
    # Ensure inputs are binary arrays
    roi_pred = (roi_pred > 0).astype(np.uint8)
    roi_gt = (roi_gt > 0).astype(np.uint8)

    # Total number of predicted ROI pixels
    n_pred = roi_pred.sum()

    if n_pred == 0:
        # If no ROI is predicted, we define containment as 0 (or you could define it as 1 by convention)
        return {
            "containment_ratio": 0.0,
            "misfit_ratio": 1.0,
            "n_pred": 0,
            "n_intersection": 0,
        }

    # Intersection of predicted ROI with ground truth ROI
    intersection = np.logical_and(roi_pred, roi_gt).astype(np.uint8)
    n_intersection = intersection.sum()

    # Compute metrics
    containment_ratio = n_intersection / n_pred
    misfit_ratio = 1 - containment_ratio

    return {
        "containment_ratio": containment_ratio,
        "misfit_ratio": misfit_ratio,
        "n_pred": int(n_pred),
        "n_intersection": int(n_intersection),
    }


def mc_dropout_prediction(
    model: models.Model, signals: np.ndarray, num_samples: int = 30
) -> tuple[np.ndarray, np.ndarray]:
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
    predictions = np.array(
        [model(signals, training=True)["ROI"].numpy() for _ in range(num_samples)]
    )
    mean_prediction = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_prediction, uncertainty


def filter_predictions(
    model: models.Model,
    signals: np.ndarray,
    n_samples: int = 30,
    threshold: float = 0.9,
    std_threshold: float = 0.1,
) -> np.ndarray:
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
        [model(signals, training=True)["ROI"].numpy() for _ in range(n_samples)]
    )

    mean_prediction = predictions.mean(axis=0)

    uncertainty = predictions.std(axis=0)

    mean_prediction[mean_prediction < threshold] = 0
    mean_prediction[mean_prediction >= threshold] = 1

    std_mask = np.zeros_like(uncertainty)
    std_mask[uncertainty < std_threshold] = 1

    mean_prediction *= std_mask

    return mean_prediction.squeeze(), uncertainty.squeeze()


@helper.post_mpl_plot
def plot_predictions(
    classifier,
    signals,
    positions,
    x_values,
    threshold: float,
    number_of_columns=1,
    number_of_samples: int = 3,
    randomize_signal: bool = False,
    show_prediction_curve: bool = True,
) -> plt.Figure:
    """
    Plot the predicted Regions of Interest (ROIs) for several sample signals.

    Parameters
    ----------
    dataset : Dataset
        Dataset object providing `signals` and corresponding `x_values`.
    threshold : float
        Probability threshold above which a region is classified as ROI.
    number_of_samples : int, default=3
        Number of signals to visualize.
    randomize_signal : bool, default=False
        If True, randomly select signals from the dataset instead of taking
        the first N samples.
    number_of_columns : int, default=1
        Number of columns in the subplot grid.
    show_prediction_curve : bool, default=True
        If True, overlay the predicted probability curve on the signal plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure instance.
    """
    num_signals = signals.shape[0]
    sample_indices = (
        np.random.choice(num_signals, size=number_of_samples, replace=False)
        if randomize_signal
        else np.arange(min(number_of_samples, num_signals))
    )

    nrows = int(np.ceil(len(sample_indices) / number_of_columns))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=number_of_columns,
        figsize=(8 * number_of_columns, 3 * nrows),
        squeeze=False,
    )

    for idx, ax in zip(sample_indices, axes.flatten()):
        signal = signals[idx, :]
        prediction = classifier.predict(signal[None, :]).squeeze()

        ax.plot(x_values, signal, color="black", linewidth=1)
        ax.fill_between(
            x_values,
            0,
            1,
            where=prediction > threshold,
            transform=ax.get_xaxis_transform(),
            color="lightblue",
            alpha=0.6,
            label="Predicted ROI",
        )

        ax.vlines(
            x=positions[idx],
            ymin=0,
            ymax=1,
            color="black",
            linewidth=1,
            linestyle="--",
            label="True Peak",
            transform=ax.get_xaxis_transform(),
        )

        if show_prediction_curve:
            twin_ax = ax.twinx()
            twin_ax.plot(
                x_values,
                prediction,
                color="red",
                linestyle="--",
                label="Predicted Probability",
            )
            twin_ax.legend(loc="upper left")
            twin_ax.set_ylim([0, 1])
            twin_ax.grid(False, which="both")

        ax.set_title(f"Sample {idx}")
        ax.legend(loc="upper left")
        twin_ax.legend(loc="upper right")

    fig.suptitle(f"Predicted ROI")
    fig.supxlabel("Time step [AU]", y=0)
    fig.supylabel("Signal [AU]", x=0)

    return fig
