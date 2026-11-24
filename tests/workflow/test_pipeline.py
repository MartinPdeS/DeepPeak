import numpy as np

from DeepPeak import (
    Gaussian,
    GenerationConfig,
    HeightPeakTrigger,
    Pipeline,
    SignalGenerator,
    UniformCount,
)
from DeepPeak.analysis import StandardTraceAnalyzer


def test_pipeline_returns_canonical_detection_result():
    analyzer = StandardTraceAnalyzer(
        std_trigger=HeightPeakTrigger(height=0.5),
        sequence_length=5,
    )
    result = Pipeline(detector=analyzer).run([0.0, 0.0, 1.0, 0.0, 0.0])

    assert result.detection.peak_count == 1
    assert result.detection.peaks.tolist() == [2]
    assert result.trace.n_samples == 5


def test_pipeline_generates_and_analyzes_dataset():
    generator = SignalGenerator(sequence_length=32)
    analyzer = StandardTraceAnalyzer(
        std_trigger=HeightPeakTrigger(height=0.2),
        sequence_length=32,
    )
    pipeline = Pipeline(
        generator=generator,
        detector=analyzer,
        generation_config=GenerationConfig(
            sequence_length=32,
            noise_std=0.01,
            baseline_level=(0.0, 0.1),
            seed=12,
        ),
    )
    results = pipeline.generate_and_run(
        n_samples=3,
        kernel=Gaussian(amplitude=(1.0, 1.0), position=(16.0, 16.0), width=2.0),
        peak_count=UniformCount(bounds=(1, 1)),
    )

    assert len(results) == 3
    assert all(result.dataset is not None for result in results)


def test_pipeline_accepts_custom_detector_without_detector_keyword():
    class Detector:
        def detect(self, trace):
            from DeepPeak.core import DetectionResult

            return DetectionResult(peaks=np.array([0]))

    result = Pipeline(detector=Detector()).run(np.array([1.0, 0.0]))
    assert result.detection.peaks.tolist() == [0]
