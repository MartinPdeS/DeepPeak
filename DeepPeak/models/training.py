"""Typed training options for DeepPeak neural models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf


@dataclass(frozen=True)
class TrainingConfig:
    """Reusable options for a supervised model training run.

    The callback options add early stopping, learning-rate scheduling,
    deterministic seeding, and optional best-model checkpointing. Validation
    callbacks are enabled for either ``validation_split`` or explicit
    ``validation_data`` passed to ``fit``.

    Examples
    --------
    >>> config = TrainingConfig(epochs=50, patience=6, verbose=0)
    >>> model.fit(x_train, y_train, config=config)

    Parameters
    ----------
    epochs : int, default=1
        Maximum number of training epochs.
    batch_size : int, default=32
        Number of samples per optimization step.
    validation_split : float, default=0.0
        Fraction of training data reserved for validation.
    patience : int, default=10
        Number of unimproved epochs before adaptive callbacks act.
    monitor : str, default="val_loss"
        History metric monitored by validation callbacks.
    seed : int, optional
        Random seed applied before fitting.
    checkpoint_path : path-like, optional
        Destination for an optional best-model checkpoint.

    Notes
    -----
    The remaining fields control callback behavior and are passed to Keras as
    documented by their names.
    """

    epochs: int = 1
    batch_size: int = 32
    validation_split: float = 0.0
    patience: int = 10
    monitor: str = "val_loss"
    use_early_stopping: bool = True
    restore_best_weights: bool = True
    reduce_lr_on_plateau: bool = True
    learning_rate_factor: float = 0.5
    min_learning_rate: float = 1e-6
    min_delta: float = 0.0
    monitor_mode: Literal["auto", "min", "max"] = "auto"
    seed: int | None = None
    checkpoint_path: str | Path | None = None
    save_best_only: bool = True
    save_weights_only: bool = True
    verbose: int = 1
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if not 0.0 < self.learning_rate_factor < 1.0:
            raise ValueError("learning_rate_factor must be in (0, 1)")
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.monitor_mode not in {"auto", "min", "max"}:
            raise ValueError("monitor_mode must be 'auto', 'min', or 'max'")
        if self.checkpoint_path is not None and not str(self.checkpoint_path):
            raise ValueError("checkpoint_path must not be empty")

    def callbacks(
        self, *, validation_available: bool = False
    ) -> list[tf.keras.callbacks.Callback]:
        """Build fresh callbacks for this training run.

        Parameters
        ----------
        validation_available : bool, default=False
            Whether explicit ``validation_data`` was supplied to ``fit``.

        Returns
        -------
        list of tensorflow.keras.callbacks.Callback
            Newly constructed callbacks for this run.
        """

        callbacks: list[tf.keras.callbacks.Callback] = []
        has_validation = validation_available or self.validation_split > 0.0
        mode = self.monitor_mode
        if mode == "auto":
            mode = (
                "max"
                if any(
                    token in self.monitor.lower()
                    for token in (
                        "acc",
                        "auc",
                        "f1",
                        "iou",
                        "dice",
                        "precision",
                        "recall",
                    )
                )
                else "min"
            )
        if self.use_early_stopping and has_validation:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    patience=self.patience,
                    min_delta=self.min_delta,
                    mode=mode,
                    restore_best_weights=self.restore_best_weights,
                )
            )
        if self.reduce_lr_on_plateau and has_validation:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=self.monitor,
                    factor=self.learning_rate_factor,
                    patience=max(1, self.patience // 2),
                    min_delta=self.min_delta,
                    mode=mode,
                    min_lr=self.min_learning_rate,
                )
            )
        checkpoint_has_monitor = has_validation or not self.monitor.startswith("val_")
        if self.checkpoint_path is not None and checkpoint_has_monitor:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.checkpoint_path),
                    monitor=self.monitor,
                    mode=mode,
                    save_best_only=self.save_best_only,
                    save_weights_only=self.save_weights_only,
                    verbose=1 if self.verbose > 1 else 0,
                )
            )
        return callbacks

    def fit_kwargs(self) -> dict[str, Any]:
        """Return the Keras ``fit`` keyword arguments represented by config.

        Returns
        -------
        dict
            Keyword arguments compatible with ``Model.fit``.
        """

        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "verbose": self.verbose,
            "shuffle": self.shuffle,
        }
