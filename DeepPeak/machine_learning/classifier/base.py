import re
from typing import Iterable, Optional, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from MPSPlots import helper
import tempfile

from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore

HistoryLike = Union[tf.keras.callbacks.History, dict]


class BaseClassifier:
    def save(self, path: str) -> None:
        """Save the compiled model (architecture + weights)."""
        self._ensure_built()
        self.model.save(path)

    def load_weights(self, path: str) -> None:
        """Load weights into a built model."""
        self._ensure_built()
        self.model.load_weights(path)

    def _ensure_built(self) -> None:
        if self.model is None:
            self.build()

    @staticmethod
    def _coerce_history(h: Optional[HistoryLike]) -> Optional[dict]:
        if h is None:
            return None
        if isinstance(h, tf.keras.callbacks.History):
            return h.history
        if isinstance(h, dict):
            return h
        raise TypeError(f"Unsupported history type: {type(h)}")

    def summary(self, *args, **kwargs) -> None:
        """Print the model summary."""
        self._ensure_built()
        self.model.summary(*args, **kwargs)

    def predict(
        self,
        signal: np.ndarray,
        *,
        batch_size: int = 32,
        verbose: int = 0,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict per-timestep probabilities; optionally return a binary mask if `threshold` is set.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        batch_size : int
            Batch size for prediction.
        verbose : int
            Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        threshold : float, optional
            Threshold for binary mask.

        Returns
        -------
        np.ndarray
            Predicted probabilities or binary mask.
        """
        self._ensure_built()
        p = self.model.predict(signal, batch_size=batch_size, verbose=verbose)
        if threshold is not None:
            return (p >= float(threshold)).astype(np.float32)
        return p

    def evaluate(self, x: np.ndarray, y: np.ndarray, *, batch_size: int = 32, verbose: int = 0) -> dict:
        """
        Evaluate the model; returns a dict of metric -> value.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        y : np.ndarray
            Target data.
        batch_size : int
            Batch size for evaluation.
        verbose : int
            Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        """
        self._ensure_built()
        results = self.model.evaluate(x, y, batch_size=batch_size, verbose=verbose, return_dict=True)
        return results

    def receptive_field(self) -> int:
        """
        Receptive field in time steps for the dilated stack (causal).

        For dilation rates d_i = 2^i and kernel size K:
            RF = 1 + sum_i (K - 1) * d_i
        """
        rf = 1 + sum((self.kernel_size - 1) * (2**i) for i in range(self.num_dilation_layers))
        return rf

    def fit(
        self,
        *args,
        use_best_weights: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> tf.keras.callbacks.History:
        """
        Fit the WaveNet model to training data, optionally restoring the best
        weights (based on validation performance) or keeping the latest ones.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed directly to `tf.keras.Model.fit`.
        use_best_weights : bool, default=True
            If True, restores the model weights from the best training iteration
            according to the monitored metric. If False, keeps the final weights.
        monitor : str, default="val_loss"
            Metric to monitor for selecting the best model weights. Common values
            include "val_loss", "val_accuracy", or a custom metric name.
        mode : {"min", "max"}, default="min"
            Whether the monitored metric should be minimized ("min") or maximized ("max").
        checkpoint_path : str or None, default=None
            Optional path to save model checkpoints. If None, a temporary file is
            used internally and discarded after training.

        Returns
        -------
        tf.keras.callbacks.History
            The training history returned by `model.fit`.

        Notes
        -----
        - The model must be built before fitting (see `build()`).
        - If `use_best_weights=True`, the best weights are automatically restored
        after training completes.
        """

        if self.model is None:
            raise RuntimeError("The model must be built before fitting. Call `build()` first.")

        # If user didn’t provide a path, use a temporary checkpoint
        if checkpoint_path is None:
            tmp_dir = tempfile.mkdtemp(prefix="wavenet_ckpt_")
            checkpoint_path = os.path.join(tmp_dir, "best.weights.h5")

        callbacks = kwargs.pop("callbacks", [])

        # Add ModelCheckpoint callback if best-weight tracking is requested
        if use_best_weights:
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
            callbacks.append(checkpoint_callback)

        # Fit the model
        history = self.model.fit(*args, callbacks=callbacks, **kwargs)

        # Restore best weights if requested
        if use_best_weights and os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
            print(f"Restored best model weights from {checkpoint_path}")

        self.history = history.history
        return history

    # --------------------------------------------------------------------- #
    # Plotting
    # --------------------------------------------------------------------- #
    @helper.post_mpl_plot
    def plot_model_history(self, filter_pattern: str = None) -> None:
        """
        Plot training/validation curves from a History or dict-like object.

        Accepts:
        - `tf.keras.callbacks.History` (uses `.history`)
        - `dict` with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'

        Parameters
        ----------
        filter_pattern : str
            Optional regex pattern to filter the metrics to plot.
        """
        hist = self.history

        if hist is None:
            raise ValueError("No history available. Train the model or pass a History/dict to plot.")

        parameters = list(hist.keys())

        filter_pattern = r".*" if filter_pattern is None else filter_pattern

        if filter_pattern is not None:
            pattern = re.compile(filter_pattern)
            parameters = [s for s in parameters if pattern.match(s)]

        if len(parameters) == 0:
            print(f"No matching parameters found. List of parameters: {list(hist.keys())}")
            return

        figure, axes = plt.subplots(nrows=len(parameters), ncols=1, figsize=(8, 3 * len(parameters)), squeeze=False)

        for ax, parameter in zip(axes.flatten(), parameters):
            ax.plot(hist[parameter], label=parameter)
            ax.legend()
            ax.set_title(parameter)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(parameter)

        return figure

    @helper.post_mpl_plot
    def plot_prediction(
        self,
        dataset,
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
        import numpy as np
        import matplotlib.pyplot as plt

        num_signals = dataset.signals.shape[0]
        sample_indices = np.random.choice(num_signals, size=number_of_samples, replace=False) if randomize_signal else np.arange(min(number_of_samples, num_signals))

        nrows = int(np.ceil(len(sample_indices) / number_of_columns))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=number_of_columns,
            figsize=(8 * number_of_columns, 3 * nrows),
            squeeze=False,
        )

        for idx, ax in zip(sample_indices, axes.flatten()):
            signal = dataset.signals[idx, :]
            prediction = self.predict(signal[None, :]).squeeze()

            ax.plot(dataset.x_values, signal, color="black", linewidth=1)
            ax.fill_between(
                dataset.x_values,
                0,
                1,
                where=prediction > threshold,
                transform=ax.get_xaxis_transform(),
                color="lightblue",
                alpha=0.6,
                label="Predicted ROI",
            )

            ax.vlines(
                x=dataset.positions[idx],
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
                twin_ax.plot(dataset.x_values, prediction, color="red", linestyle="--", label="Predicted Probability")
                twin_ax.legend(loc="upper left")
                twin_ax.set_ylim([0, 1])
                twin_ax.grid(False, which="both")

            ax.set_title(f"Predicted ROI (Sample {idx})")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")

        return fig
