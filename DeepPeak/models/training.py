"""Typed training options for DeepPeak neural models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass(frozen=True)
class TrainingConfig:
    """Reusable options for a supervised model training run.

    The callback options add early stopping and learning-rate scheduling for
    validation runs while allowing callers to provide additional callbacks.

    Examples
    --------
    >>> config = TrainingConfig(epochs=50, patience=6, verbose=0)
    >>> model.fit(x_train, y_train, config=config)
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

    def callbacks(self) -> list[tf.keras.callbacks.Callback]:
        """Build fresh callbacks for this training run."""

        callbacks: list[tf.keras.callbacks.Callback] = []
        if self.use_early_stopping and self.validation_split > 0.0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    patience=self.patience,
                    restore_best_weights=self.restore_best_weights,
                )
            )
        if self.reduce_lr_on_plateau and self.validation_split > 0.0:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=self.monitor,
                    factor=self.learning_rate_factor,
                    patience=max(1, self.patience // 2),
                    min_lr=self.min_learning_rate,
                )
            )
        return callbacks

    def fit_kwargs(self) -> dict[str, Any]:
        """Return the Keras ``fit`` keyword arguments represented by config."""

        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "verbose": self.verbose,
            "shuffle": self.shuffle,
        }
