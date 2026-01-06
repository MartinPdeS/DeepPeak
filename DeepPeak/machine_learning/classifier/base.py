import re
from typing import Iterable, Optional, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from MPSPlots import helper
import tempfile

from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from DeepPeak.utils import merge_and_plot_histories

HistoryLike = Union[tf.keras.callbacks.History, dict]


class BaseClassifier:
    histories = []

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

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, *, batch_size: int = 32, verbose: int = 0
    ) -> dict:
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
        results = self.model.evaluate(
            x, y, batch_size=batch_size, verbose=verbose, return_dict=True
        )
        return results

    def receptive_field(self) -> int:
        """
        Receptive field in time steps for the dilated stack (causal).

        For dilation rates d_i = 2^i and kernel size K:
            RF = 1 + sum_i (K - 1) * d_i
        """
        rf = 1 + sum(
            (self.kernel_size - 1) * (2**i) for i in range(self.num_dilation_layers)
        )
        return rf

    def fit(self, *args, **kwargs) -> tf.keras.callbacks.History:
        """
        Fit the WaveNet model to training data.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed directly to `tf.keras.Model.fit`.

        Returns
        -------
        tf.keras.callbacks.History
            The training history returned by `model.fit`.

        Notes
        -----
        - The model must be built before fitting (see `build()`).
        """
        self._ensure_built()
        history = self.model.fit(*args, **kwargs)
        self.histories.append(history)
        return history

    # def fit(
    #     self,
    #     *args,
    #     use_best_weights: bool = True,
    #     monitor: str = "val_loss",
    #     mode: str = "min",
    #     checkpoint_path: str | None = None,
    #     **kwargs,
    # ) -> tf.keras.callbacks.History:
    #     """
    #     Fit the WaveNet model to training data, optionally restoring the best
    #     weights (based on validation performance) or keeping the latest ones.

    #     Parameters
    #     ----------
    #     *args, **kwargs
    #         Arguments passed directly to `tf.keras.Model.fit`.
    #     use_best_weights : bool, default=True
    #         If True, restores the model weights from the best training iteration
    #         according to the monitored metric. If False, keeps the final weights.
    #     monitor : str, default="val_loss"
    #         Metric to monitor for selecting the best model weights. Common values
    #         include "val_loss", "val_accuracy", or a custom metric name.
    #     mode : {"min", "max"}, default="min"
    #         Whether the monitored metric should be minimized ("min") or maximized ("max").
    #     checkpoint_path : str or None, default=None
    #         Optional path to save model checkpoints. If None, a temporary file is
    #         used internally and discarded after training.

    #     Returns
    #     -------
    #     tf.keras.callbacks.History
    #         The training history returned by `model.fit`.

    #     Notes
    #     -----
    #     - The model must be built before fitting (see `build()`).
    #     - If `use_best_weights=True`, the best weights are automatically restored
    #     after training completes.
    #     """

    #     if self.model is None:
    #         raise RuntimeError(
    #             "The model must be built before fitting. Call `build()` first."
    #         )

    #     # If user didn’t provide a path, use a temporary checkpoint
    #     if checkpoint_path is None:
    #         tmp_dir = tempfile.mkdtemp(prefix="wavenet_ckpt_")
    #         checkpoint_path = os.path.join(tmp_dir, "best.weights.h5")

    #     callbacks = kwargs.pop("callbacks", [])

    #     # Add ModelCheckpoint callback if best-weight tracking is requested
    #     if use_best_weights:
    #         checkpoint_callback = ModelCheckpoint(
    #             filepath=checkpoint_path,
    #             monitor=monitor,
    #             mode=mode,
    #             save_best_only=True,
    #             save_weights_only=True,
    #             verbose=1,
    #         )
    #         callbacks.append(checkpoint_callback)

    #     # Fit the model
    #     history = self.model.fit(*args, callbacks=callbacks, **kwargs)

    #     # Restore best weights if requested
    #     if use_best_weights and os.path.exists(checkpoint_path):
    #         self.model.load_weights(checkpoint_path)
    #         print(f"Restored best model weights from {checkpoint_path}")

    #     self.histories.append(history)

    # --------------------------------------------------------------------- #
    # Plotting
    # --------------------------------------------------------------------- #
    # @helper.post_mpl_plot
    # def plot_model_history(
    #     self,
    #     show_split: str = "both",  # {"train","val","both"}
    #     kind: str = "all",  # {"all","loss","metrics"}
    #     filter_regex: (
    #         str | None
    #     ) = None,  # e.g. r"acc|auc|f1" matches these metric base names
    #     legend_loc: str = "best",
    # ) -> None:
    #     """
    #     Plot training/validation curves from a Keras History (or dict-like) cleanly.

    #     Behavior
    #     --------
    #     - Groups *all losses* in one axis and *all metrics* in another axis.
    #     - You can show only train, only validation, or both.
    #     - You can restrict to loss-only or metrics-only.
    #     - `filter_regex` filters by *base* metric name (e.g., "accuracy" matches both
    #     "accuracy" and "val_accuracy").

    #     Parameters
    #     ----------
    #     show_split : {"train","val","both"}
    #         Which split(s) to show.
    #     kind : {"all","loss","metrics"}
    #         What to plot.
    #     filter_regex : str or None
    #         Regex applied to base metric names (without the 'val_' prefix).
    #         Example: r"acc|auc" to include only accuracy and AUC families.
    #     legend_loc : str
    #         Matplotlib legend location.

    #     Notes
    #     -----
    #     - Accepts `tf.keras.callbacks.History` or `dict` with keys like:
    #     'loss', 'val_loss', 'accuracy', 'val_accuracy', 'auc', 'val_auc', etc.
    #     """
    #     merged, figure = merge_and_plot_histories(*self.histories, plot_keys=None)

    #     return figure

    #     # ---- 1) Pull history dict ----
    #     raw_hist = getattr(self.history, "history", None)
    #     hist: dict = raw_hist if isinstance(raw_hist, dict) else self.history
    #     if not isinstance(hist, dict) or not hist:
    #         raise ValueError(
    #             "No history available. Pass a Keras History or a dict of lists."
    #         )

    #     # ---- 2) Normalize keys into base name + split ----
    #     # Map: base_name -> {"train": series or None, "val": series or None}
    #     # Treat any key starting with "val_" as validation; others as train.
    #     series_map: dict[str, dict[str, list[float] | None]] = {}
    #     for key, values in hist.items():
    #         if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
    #             continue
    #         is_val = key.startswith("val_")
    #         base = key[4:] if is_val else key
    #         series_map.setdefault(base, {"train": None, "val": None})
    #         series_map[base]["val" if is_val else "train"] = list(values)

    #     # ---- 3) Split into losses vs metrics by name heuristic ----
    #     # Everything with base == "loss" or base endswith("_loss") is treated as loss.
    #     losses = {
    #         b: s for b, s in series_map.items() if (b == "loss" or b.endswith("_loss"))
    #     }
    #     metrics = {b: s for b, s in series_map.items() if b not in losses}

    #     # ---- 4) Apply base-name regex filter if provided ----
    #     if filter_regex:
    #         pat = re.compile(filter_regex)
    #         losses = {b: s for b, s in losses.items() if pat.search(b)}
    #         metrics = {b: s for b, s in metrics.items() if pat.search(b)}

    #     # ---- 5) Decide which panels to show ----
    #     show_losses = kind in ("all", "loss") and len(losses) > 0
    #     show_metrics = kind in ("all", "metrics") and len(metrics) > 0
    #     if not show_losses and not show_metrics:
    #         # Provide a helpful message with available keys
    #         available = ", ".join(sorted(series_map.keys()))
    #         raise ValueError(
    #             f"Nothing to plot with the current filters. Available: {available}"
    #         )

    #     n_panels = int(show_losses) + int(show_metrics)
    #     fig, axes = plt.subplots(
    #         n_panels, 1, figsize=(9, 3.6 * n_panels), squeeze=False
    #     )
    #     axes = axes.flatten()

    #     # Helper to plot a group (losses or metrics) on one axis
    #     def _plot_group(
    #         ax, group: dict[str, dict[str, Iterable[float] | None]], title: str
    #     ):
    #         drawn = False
    #         # Determine epoch range from the longest series we find
    #         max_len = 0
    #         for base, splits in group.items():
    #             for split_name, series in splits.items():
    #                 if series is not None:
    #                     max_len = max(max_len, len(series))
    #         epochs = np.arange(1, max_len + 1)

    #         for base, splits in sorted(group.items()):
    #             # Train
    #             if show_split in ("train", "both") and splits["train"] is not None:
    #                 y = np.asarray(splits["train"], dtype=float)
    #                 ax.plot(epochs[: len(y)], y, label=f"{base} (train)")
    #                 drawn = True
    #             # Val
    #             if show_split in ("val", "both") and splits["val"] is not None:
    #                 y = np.asarray(splits["val"], dtype=float)
    #                 # To help distinguish validation visually, use a dashed line
    #                 ax.plot(epochs[: len(y)], y, linestyle="--", label=f"{base} (val)")
    #                 drawn = True

    #         ax.set_title(title)
    #         ax.set_xlabel("Epoch")
    #         ax.grid(True, linewidth=0.5, alpha=0.6)
    #         if drawn:
    #             ax.legend(loc=legend_loc, ncol=2)
    #         else:
    #             ax.text(
    #                 0.5,
    #                 0.5,
    #                 "No series to display",
    #                 ha="center",
    #                 va="center",
    #                 transform=ax.transAxes,
    #             )

    #     # ---- 6) Render panels ----
    #     panel_idx = 0
    #     if show_losses:
    #         _plot_group(axes[panel_idx], losses, "Losses")
    #         panel_idx += 1
    #     if show_metrics:
    #         _plot_group(axes[panel_idx], metrics, "Metrics")

    #     return fig

    @helper.post_mpl_plot
    def plot_model_history(
        self,
        show_split: str = "both",  # {"train","val","both"}
        kind: str = "all",  # {"all","loss","metrics"}
        filter_regex: str | None = None,
    ):
        """
        Plot training and validation curves from one or many Keras History objects.

        Behavior
        --------
        - Merges any number of histories (self.histories if present, else self.history).
        - Groups losses in one panel and metrics in another.
        - Supports plotting train, val, or both.
        - Supports filtering by regex on base names (without 'val_').

        Parameters
        ----------
        show_split : {"train","val","both"}
            Which split(s) to show.
        kind : {"all","loss","metrics"}
            What to plot.
        filter_regex : str or None
            Regex applied to base metric names (without the 'val_' prefix).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if show_split not in {"train", "val", "both"}:
            raise ValueError("show_split must be one of: 'train', 'val', 'both'")
        if kind not in {"all", "loss", "metrics"}:
            raise ValueError("kind must be one of: 'all', 'loss', 'metrics'")

        histories = []
        if hasattr(self, "histories") and self.histories is not None:
            histories = list(self.histories)
        elif hasattr(self, "history") and self.history is not None:
            histories = [self.history]

        if len(histories) == 0:
            raise ValueError("No history available on this object.")

        merged: dict[str, list[float]] = {}
        all_keys: set[str] = set()

        for history in histories:
            hist_dict = getattr(history, "history", history)
            if not isinstance(hist_dict, dict):
                raise ValueError(
                    "Each history must be a Keras History or a dict of lists."
                )
            all_keys |= set(hist_dict.keys())

        for key in all_keys:
            merged[key] = []
            for history in histories:
                hist_dict = getattr(history, "history", history)
                values = hist_dict.get(key, [])
                if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
                    merged[key].extend([float(v) for v in values])

        if not merged:
            raise ValueError("Merged history is empty.")

        series_map: dict[str, dict[str, list[float] | None]] = {}
        for key, values in merged.items():
            if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
                continue
            is_val = key.startswith("val_")
            base = key[4:] if is_val else key
            series_map.setdefault(base, {"train": None, "val": None})
            series_map[base]["val" if is_val else "train"] = list(values)

        if not series_map:
            raise ValueError("No plottable series found in history.")

        losses = {
            b: s for b, s in series_map.items() if (b == "loss" or b.endswith("_loss"))
        }
        metrics = {b: s for b, s in series_map.items() if b not in losses}

        if filter_regex:
            pattern = re.compile(filter_regex)
            losses = {b: s for b, s in losses.items() if pattern.search(b)}
            metrics = {b: s for b, s in metrics.items() if pattern.search(b)}

        show_losses = (kind in {"all", "loss"}) and (len(losses) > 0)
        show_metrics = (kind in {"all", "metrics"}) and (len(metrics) > 0)

        if not show_losses and not show_metrics:
            available = ", ".join(sorted(series_map.keys()))
            raise ValueError(
                f"Nothing to plot with current filters. Available: {available}"
            )

        n_panels = int(show_losses) + int(show_metrics)
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(9, 3.6 * n_panels), squeeze=False
        )
        axes = axes.flatten()

        def plot_group(
            ax, group: dict[str, dict[str, list[float] | None]], title: str
        ) -> None:
            max_len = 0
            for splits in group.values():
                for series in splits.values():
                    if series is not None:
                        max_len = max(max_len, len(series))

            epochs = np.arange(1, max_len + 1)
            drawn_any = False

            for base in sorted(group.keys()):
                splits = group[base]

                if show_split in {"train", "both"} and splits["train"] is not None:
                    y = np.asarray(splits["train"], dtype=float)
                    ax.plot(epochs[: len(y)], y, label=f"{base} (train)")
                    drawn_any = True

                if show_split in {"val", "both"} and splits["val"] is not None:
                    y = np.asarray(splits["val"], dtype=float)
                    ax.plot(epochs[: len(y)], y, linestyle="--", label=f"{base} (val)")
                    drawn_any = True

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, linewidth=0.5, alpha=0.6)
            if drawn_any:
                ax.legend(ncol=2)

        panel_index = 0
        if show_losses:
            plot_group(axes[panel_index], losses, "Losses")
            panel_index += 1
        if show_metrics:
            plot_group(axes[panel_index], metrics, "Metrics")

        return fig
