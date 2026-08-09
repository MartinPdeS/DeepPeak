"""Regression tests for analysis boundaries and domain-specific errors."""

from pathlib import Path

import numpy as np
import pytest

from DeepPeak.analysis import StandardDilutionSeries, StandardTraceAnalyzer
from DeepPeak.analysis.metrics import PeakCountSeriesResult, TraceRecord
from DeepPeak.analysis.series_calculations import estimate_expected_particle_flow
from DeepPeak.core import (
    AnalysisStateError,
    DetectionConfig,
    InvalidConfigurationError,
    InvalidDetectorError,
    MissingDetectorError,
    Trace,
)
from DeepPeak.detection import HeightPeakTrigger


def _record(dilution: float, peak_count: int) -> TraceRecord:
    from DeepPeak.core import DetectionResult

    detection = DetectionResult(peaks=np.arange(peak_count, dtype=int))
    return TraceRecord(
        filename=Path(f"trace-{dilution}.csv"),
        dilution=dilution,
        concentration=1.0 / dilution,
        dx=1.0,
        signal=np.zeros(max(peak_count, 1)),
        standard=detection,
        prediction=np.array([]),
        cnn=DetectionResult(peaks=np.array([], dtype=int)),
    )


def test_configuration_errors_are_specific_and_backward_compatible():
    with pytest.raises(InvalidConfigurationError) as error:
        DetectionConfig(sequence_length=0)

    assert isinstance(error.value, ValueError)
    assert "sequence_length" in str(error.value)


def test_detector_selection_errors_are_specific():
    analyzer = StandardTraceAnalyzer(
        config=DetectionConfig(trigger=HeightPeakTrigger(height=0.5), sequence_length=1)
    )

    with pytest.raises(InvalidDetectorError):
        analyzer.detect(Trace(signal=[0.0], dx=1.0), detector="cnn")


def test_missing_detector_error_preserves_typeerror_contract(tmp_path):
    filename = tmp_path / "trace.csv"
    filename.write_text("placeholder")

    with pytest.raises(MissingDetectorError) as error:
        StandardDilutionSeries(files=[(filename, 1.0)])

    assert isinstance(error.value, TypeError)


def test_analysis_state_error_is_raised_before_a_series_run(tmp_path):
    filename = tmp_path / "trace.csv"
    filename.write_text("placeholder")
    series = StandardDilutionSeries(
        files=[(filename, 1.0)],
        trigger=HeightPeakTrigger(height=0.5),
        nrows=4,
    )

    with pytest.raises(AnalysisStateError, match="Call run"):
        series.get_last_result()


def test_extracted_expected_flow_calculation_handles_multiple_references():
    result = PeakCountSeriesResult(
        dilution=np.array([1.0, 2.0, 4.0]),
        concentration=np.array([1.0, 0.5, 0.25]),
        standard_particle_count=np.array([8, 4, 2]),
        standard_particle_flow=np.array([8.0, 4.0, 2.0]),
        cnn_particle_count=np.zeros(3),
        cnn_particle_flow=np.zeros(3),
        water_record=None,
        records=[_record(1.0, 8), _record(2.0, 4), _record(4.0, 2)],
    )

    expected = estimate_expected_particle_flow(
        result,
        index=2,
        reference_indices=[0, 1],
        use_water_baseline=False,
    )

    assert expected == pytest.approx(1.0)
