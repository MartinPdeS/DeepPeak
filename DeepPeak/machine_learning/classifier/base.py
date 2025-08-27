from typing import Optional, Union, Iterable
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DeepPeak.helper import mpl_plot

HistoryLike = Union[tf.keras.callbacks.History, dict]

class BaseClassifier():
    @mpl_plot
    def plot_prediction(self, figure: plt.Figure, ax: plt.Axes, signal: np.ndarray, threshold: float) -> None:
        """
        Plot the predicted region of interest (ROI) for a given signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal to predict the ROI.
        threshold : float
            Threshold for determining the ROI.
        """
        region_of_interest = self.predict(signal.reshape([1, signal.size])).squeeze()
        x_values = np.arange(signal.size)

        ax.plot(x_values, signal.squeeze(), color='black')

        ax.fill_between(
            x=x_values,
            y1=0,
            y2=1,
            transform=ax.get_xaxis_transform(),
            where=region_of_interest > threshold,
            color='lightblue',
            alpha=0.6,
            label='Predicted ROI'
        )

        ax.set_title("Predicted Region of Interest")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")

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

    def predict(self, signal: np.ndarray, *, batch_size: int = 32, verbose: int = 0, threshold: Optional[float] = None) -> np.ndarray:
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
        rf = 1 + sum((self.kernel_size - 1) * (2 ** i) for i in range(self.num_dilation_layers))
        return rf

    def fit(self, x: np.ndarray, y: np.ndarray, *,
        batch_size: int = 32,
        epochs: int = 20,
        validation_split: float = 0.2,
        callbacks: Optional[Iterable[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1,
        shuffle: bool = True) -> tf.keras.callbacks.History:
        """
        Train the model and store the History in `history_`.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        y : np.ndarray
            Target data.
        batch_size : int
            Batch size for training.
        epochs : int
            Number of epochs to train.
        validation_split : float
            Fraction of the training data to use for validation.
        callbacks : Optional[Iterable[tf.keras.callbacks.Callback]]
            List of callbacks to apply during training.
        verbose : int
            Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        shuffle : bool
            Whether to shuffle the training data.
        """
        self._ensure_built()
        history = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=list(callbacks) if callbacks is not None else None,
            verbose=verbose,
            shuffle=shuffle,
        )
        self.history_ = history
        return history

    # --------------------------------------------------------------------- #
    # Plotting
    # --------------------------------------------------------------------- #
    def plot_model_history(self, history: Optional[HistoryLike] = None) -> None:
        """
        Plot training/validation curves from a History or dict-like object.

        Accepts:
        - `tf.keras.callbacks.History` (uses `.history`)
        - `dict` with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'

        Parameters
        ----------
        history : Optional[HistoryLike]
            Training history to plot.
        """
        hist = self._coerce_history(history or self.history_)
        if hist is None:
            raise ValueError("No history available. Train the model or pass a History/dict to plot.")

        # Determine which keys are present
        keys = list(hist.keys())
        loss_keys = [k for k in ("loss", "val_loss") if k in hist]
        acc_keys = [k for k in ("accuracy", "val_accuracy", "acc", "val_acc") if k in hist]

        nrows = 1 + (1 if acc_keys else 0)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(7, 4 * nrows))
        if nrows == 1:
            axes = [axes]

        # Loss
        ax = axes[0]
        for k in loss_keys:
            ax.plot(hist[k], label=k)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        # Accuracy (if available)
        if nrows > 1:
            ax = axes[1]
            for k in acc_keys:
                ax.plot(hist[k], label=k)
            ax.set_title("Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()

        plt.tight_layout()
        plt.show()

