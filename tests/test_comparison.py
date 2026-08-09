import numpy as np

from DeepPeak import (
    HeightPeakTrigger,
    Trace,
    TraceComparisonAnalyzer,
)


class IdentityDeconvolver:
    """Small model double exposing the public prediction contract."""

    def predict(self, signal):
        return np.asarray(signal)


def _trace():
    x = np.arange(128, dtype=float)
    signal = np.exp(-0.5 * ((x - 30.0) / 2.0) ** 2)
    signal += np.exp(-0.5 * ((x - 90.0) / 3.0) ** 2)
    return Trace(signal=signal, dx=0.1)


def test_comparison_can_run_without_a_deconvolver():
    result = TraceComparisonAnalyzer(
        HeightPeakTrigger(height=0.3),
    ).compare(_trace())

    assert result.deconvolved is None
    assert result.standard.peaks.tolist() == [30, 90]


def test_comparison_exposes_arrival_amplitude_and_width_distributions():
    result = TraceComparisonAnalyzer(
        HeightPeakTrigger(height=0.3),
        deconvolver=IdentityDeconvolver(),
    ).compare(_trace())

    assert result.deconvolved is not None
    assert result.distribution("arrival", "standard").tolist() == [3.0, 9.0]
    assert result.distribution("arrival", "deconvolved").tolist() == [3.0, 9.0]
    assert result.distribution("amplitude", "standard").size == 2
    assert result.distribution("amplitude", "deconvolved").size == 2
    assert result.distribution("width", "standard").size == 2
    assert set(result.compare()) == {"arrival", "amplitude", "width"}
    assert result.compare_distribution("arrival")["wasserstein_distance"] == 0.0


def test_series_comparison_aggregates_the_same_distributions():
    analyzer = TraceComparisonAnalyzer(
        HeightPeakTrigger(height=0.3),
        deconvolver=IdentityDeconvolver(),
    )
    result = analyzer.compare_many([_trace(), _trace()])

    assert len(result) == 2
    assert result.distribution("arrival", "standard").size == 4
    assert result.summary("deconvolved")["amplitude"]["count"] == 4
    assert result.compare()["width"]["count_difference"] == 0.0
