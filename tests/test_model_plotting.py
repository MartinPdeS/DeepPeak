import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from DeepPeak.generation import DataSet
from DeepPeak.models.plotting import plot_predictions


class FakeModel:
    def predict(self, values, *, batch_size, verbose):
        assert values.ndim == 3
        assert values.shape[-1] == 1
        assert batch_size == 4
        assert verbose == 0
        return values * 2.0


def test_plot_predictions_builds_requested_grid_and_overlays_target():
    dataset = DataSet(
        signals=np.ones((5, 8)),
        clean_signals=np.full((5, 8), 0.5),
        x_values=np.arange(8),
    )

    figure = plot_predictions(
        FakeModel(), dataset, n_samples=5, n_columns=3, batch_size=4, show=False
    )

    assert len(figure.axes) == 6
    assert sum(axis.get_visible() for axis in figure.axes) == 5
    assert len(figure.legends) == 1
    assert {line.get_label() for line in figure.axes[0].lines} == {
        "Signal",
        "Clean signal",
        "Prediction",
    }


def test_plot_predictions_rejects_invalid_layout():
    dataset = DataSet(signals=np.ones((2, 8)))

    with pytest.raises(ValueError, match="must be positive"):
        plot_predictions(FakeModel(), dataset, n_samples=0, show=False)
