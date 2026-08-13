"""Results returned by the unified model evaluation workflow."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelEvaluationResult:
    """Container for metrics and optional arrays produced by evaluation.

    Parameters
    ----------
    metrics : dict[str, float]
        Named scalar metrics returned by the model.
    inputs : numpy.ndarray, optional
        Model inputs, conventionally shaped ``(n_samples, sequence_length,
        n_channels)``.
    targets : numpy.ndarray, optional
        Ground-truth model targets.
    predictions : numpy.ndarray, optional
        Model predictions with the same shape as ``targets``.
    """

    metrics: dict[str, float]
    inputs: np.ndarray | None = None
    targets: np.ndarray | None = None
    predictions: np.ndarray | None = None

    def __getitem__(self, name: str) -> float:
        """Return a metric by name.

        Parameters
        ----------
        name : str
            Metric key.

        Returns
        -------
        float
            Stored metric value.

        Raises
        ------
        KeyError
            If ``name`` is not present.
        """
        return self.metrics[name]

    def get(self, name: str, default: float | None = None) -> float | None:
        """Return one metric, or ``default`` when it is absent.

        Parameters
        ----------
        name : str
            Metric key.
        default : float, optional
            Value returned when ``name`` is not present.

        Returns
        -------
        float or None
            Metric value or the supplied default.
        """
        return self.metrics.get(name, default)

    @property
    def residuals(self) -> np.ndarray | None:
        """Return ``predictions - targets`` when both arrays are available.

        Returns
        -------
        numpy.ndarray or None
            Residual array, or ``None`` when evaluation did not retain arrays.
        """
        if self.predictions is None or self.targets is None:
            return None
        return self.predictions - self.targets

    def summary(self) -> dict[str, float]:
        """Return a shallow copy of the scalar evaluation metrics.

        Returns
        -------
        dict[str, float]
            Copy of :attr:`metrics`.
        """
        return dict(self.metrics)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Serialize metrics and optionally include evaluation arrays.

        Parameters
        ----------
        include_arrays : bool, default=False
            Include ``inputs``, ``targets``, and ``predictions`` in the result.

        Returns
        -------
        dict[str, Any]
            Dictionary containing metrics and, when requested, retained arrays.
        """
        result: dict[str, Any] = {"metrics": self.summary()}
        if include_arrays:
            result.update(
                inputs=self.inputs,
                targets=self.targets,
                predictions=self.predictions,
            )
        return result
