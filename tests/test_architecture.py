import numpy as np
import pytest

from DeepPeak.core import (
    DetectionConfig,
    DetectionResult,
    MetricResult,
    PlotConfig,
    SeriesConfig,
    SeriesResult,
    Trace,
    TraceConfig,
)


def test_core_result_objects_validate_and_serialize():
    trace = Trace(signal=[1.0, 2.0, 3.0], dx=0.5, metadata={"source": "test"})
    detection = DetectionResult(
        peaks=np.array([1]),
        properties={"prominences": np.array([2.0])},
        detection_kwargs={"height": 1.5},
        amplitudes=np.array([2.0]),
    )
    metric = MetricResult("mean_amplitude", np.array(2.0), units="V")
    series = SeriesResult(records=[detection], metadata={"count": 1})

    assert trace.n_samples == 3
    assert trace.duration == pytest.approx(1.5)
    assert trace.to_dict()["signal"] == [1.0, 2.0, 3.0]
    assert detection.peak_count == 1
    assert detection.to_dict()["peaks"] == [1]
    assert DetectionResult.from_dict(detection.to_dict()).peak_count == 1
    assert metric.to_dict()["values"] == pytest.approx(2.0)
    assert MetricResult.from_dict(metric.to_dict()).name == "mean_amplitude"
    assert len(series) == 1
    assert series.to_dict()["metadata"] == {"count": 1}
    assert len(SeriesResult.from_dict(series.to_dict())) == 1


def test_legacy_trace_and_series_results_adapt_to_core_objects():
    from pathlib import Path

    from DeepPeak.analysis.metrics import PeakCountSeriesResult, TraceRecord
    from DeepPeak.plotting import standard_detection

    empty = DetectionResult(peaks=np.array([], dtype=int))
    record = TraceRecord(
        filename=Path("trace.csv"),
        dilution=1.0,
        concentration=2.0,
        dx=0.5,
        signal=np.array([0.0, 1.0]),
        standard=empty,
        prediction=np.array([]),
        cnn=empty,
    )
    result = PeakCountSeriesResult(
        dilution=np.array([1.0]),
        concentration=np.array([2.0]),
        standard_particle_count=np.array([0]),
        standard_particle_flow=np.array([0.0]),
        cnn_particle_count=np.array([0]),
        cnn_particle_flow=np.array([0.0]),
        water_record=None,
        records=[record],
    )

    assert record.to_trace().n_samples == 2
    assert record.trace.n_samples == 2
    assert record.to_dict()["standard"]["peak_count"] == 0
    assert len(result.to_series_result()) == 1
    figure = standard_detection(record, config=PlotConfig(close=True))
    assert figure is not None


def test_domain_namespaces_expose_the_migrated_entry_points():
    from DeepPeak.detection import HeightPeakTrigger
    from DeepPeak.generation import Gaussian, SignalGenerator
    from DeepPeak.io import CsvTrace
    from DeepPeak.models import __all__ as model_exports
    from DeepPeak.plotting import (
        PlotConfig as PublicPlotConfig,
        plot_detection_comparison,
    )

    assert HeightPeakTrigger.__name__ == "HeightPeakTrigger"
    assert Gaussian.__name__ == "Gaussian"
    assert SignalGenerator.__name__ == "SignalGenerator"
    assert CsvTrace.__name__ == "CsvTrace"
    assert "WaveNet" in model_exports
    assert callable(plot_detection_comparison)
    assert PublicPlotConfig is PlotConfig
    assert Gaussian.__module__.startswith("DeepPeak.generation")


def test_typed_configuration_objects_validate_and_are_usable():
    from DeepPeak.analysis import StandardTraceAnalyzer
    from DeepPeak.detection import HeightPeakTrigger

    trace_config = TraceConfig(
        sequence_length=4, normalization="zscore", sampling_rate_hz=2.0
    )
    assert trace_config.sequence_length == 4
    assert trace_config.normalization == "zscore"

    detection_config = DetectionConfig(
        sequence_length=4,
        trigger=HeightPeakTrigger(height=0.5),
    )
    analyzer = StandardTraceAnalyzer(config=detection_config)
    result = analyzer.detect(Trace(signal=[0.0, 1.0, 0.0, 0.0], dx=0.5))
    assert result.peak_count == 1

    series_config = SeriesConfig(initial_concentration=2.0, nrows=10)
    assert series_config.initial_concentration == 2.0
    assert series_config.nrows == 10
    assert PlotConfig(show=False, close=True).close is True

    with pytest.raises(ValueError, match="normalization"):
        TraceConfig(normalization="unknown")
    with pytest.raises(ValueError, match="sequence_length"):
        TraceConfig(sequence_length=0)
