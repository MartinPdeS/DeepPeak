"""High-level composition of generation, detection, and analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .core.config import AnalysisConfig, GenerationConfig
from .core.types import DetectionResult, Trace
from .generation.dataset import DataSet
from .generation.peak_count import PeakCount
from .generation.signal_generator import SignalGenerator
from .generation.kernels.base import BaseKernel


@dataclass
class PipelineResult:
    """Canonical output of one pipeline analysis.

    Parameters
    ----------
    trace : Trace
        Input trace analyzed by the pipeline.
    detection : DetectionResult
        Canonical peak-detection result.
    record : object, optional
        Full analyzer-specific record, when an analyzer was used.
    dataset : DataSet, optional
        Source generated dataset, when returned by ``generate_and_run``.
    metadata : dict, optional
        Per-result metadata.
    """

    trace: Trace
    detection: DetectionResult
    record: Any = None
    dataset: Optional[DataSet] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Compose a generator, detector, and optional trace analyzer.

    The detector may be any object exposing ``detect(trace)``. An analyzer may
    expose ``analyze_processed_signal`` (as the built-in trace analyzers do) or
    be a callable accepting a :class:`Trace`. This keeps the orchestration layer
    independent of TensorFlow and compatible with custom detectors.

    Parameters
    ----------
    generator : SignalGenerator, optional
        Generator used by :meth:`generate`.
    detector : object, optional
        Detector exposing ``detect``.
    analyzer : object or callable, optional
        Analyzer exposing ``analyze_processed_signal`` or accepting a
        :class:`Trace`.
    generation_config : GenerationConfig, optional
        Default generation settings.
    analysis_config : AnalysisConfig, optional
        Default analysis settings.

    Examples
    --------
    >>> from DeepPeak import HeightPeakTrigger, StandardTraceAnalyzer
    >>> detector = StandardTraceAnalyzer(
    ...     std_trigger=HeightPeakTrigger(height=0.5), sequence_length=3
    ... )
    >>> pipeline = Pipeline(detector=detector)
    >>> result = pipeline.run([0.0, 1.0, 0.0])
    """

    def __init__(
        self,
        *,
        generator: SignalGenerator | None = None,
        detector: Any | None = None,
        analyzer: Any | Callable[[Trace], Any] | None = None,
        generation_config: GenerationConfig | None = None,
        analysis_config: AnalysisConfig | None = None,
    ) -> None:
        """Create a pipeline from generation and analysis components.

        Parameters
        ----------
        generator : SignalGenerator, optional
            Generator used by :meth:`generate`.
        detector : object, optional
            Detector exposing ``detect``.
        analyzer : object or callable, optional
            Analyzer exposing ``analyze_processed_signal`` or accepting a
            :class:`Trace`.
        generation_config : GenerationConfig, optional
            Default generation settings.
        analysis_config : AnalysisConfig, optional
            Default analysis settings.

        Raises
        ------
        ValueError
            If neither a detector nor an analyzer is provided.
        """
        if detector is None and analyzer is None:
            raise ValueError("Pipeline requires a detector or analyzer.")
        self.generator = generator
        self.detector = detector
        self.analyzer = analyzer
        self.generation_config = generation_config
        self.analysis_config = analysis_config or AnalysisConfig()

    def run(
        self,
        signal: Trace | np.ndarray | list[float],
        *,
        dx: float | None = None,
        filename: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Analyze one trace and return a stable :class:`PipelineResult`.

        Parameters
        ----------
        signal : Trace or array-like
            Trace samples to analyze.
        dx : float, optional
            Sample spacing for array-like input.
        filename : path-like, optional
            Source filename for array-like input.
        metadata : dict, optional
            Metadata attached to the input trace and result.

        Returns
        -------
        PipelineResult
            Trace, detection result, and optional analyzer record.
        """

        config = self.analysis_config
        if isinstance(signal, Trace):
            if dx is not None or filename is not None:
                raise ValueError("dx and filename cannot override an existing Trace.")
            trace = signal
        else:
            trace = Trace(
                signal=np.asarray(signal, dtype=float),
                dx=config.dx if dx is None else dx,
                filename=config.filename if filename is None else filename,
                metadata=metadata or {},
            )

        record = None
        if self.analyzer is not None:
            if hasattr(self.analyzer, "analyze_processed_signal"):
                record = self.analyzer.analyze_processed_signal(
                    trace.signal,
                    dx=trace.dx,
                    filename=trace.filename or config.filename,
                    dilution=config.dilution,
                    concentration=config.concentration,
                )
            elif callable(self.analyzer):
                record = self.analyzer(trace)

        if record is not None:
            detection = self._detection_from_record(record, config.detector)
        else:
            if self.detector is None:
                raise ValueError(
                    "A detector is required when analyzer returns no record."
                )
            try:
                detection = self.detector.detect(trace, detector=config.detector)
            except TypeError:
                detection = self.detector.detect(trace)

        return PipelineResult(
            trace=trace,
            detection=detection,
            record=record,
            metadata=dict(metadata or {}),
        )

    def generate(
        self,
        *,
        n_samples: int,
        kernel: BaseKernel,
        peak_count: PeakCount,
        config: GenerationConfig | None = None,
        **overrides: Any,
    ) -> DataSet:
        """Generate a dataset using configured acquisition conditions.

        Parameters
        ----------
        n_samples : int
            Number of traces to generate.
        kernel : BaseKernel
            Pulse kernel used for synthetic traces.
        peak_count : PeakCount
            Distribution used to sample peak counts.
        config : GenerationConfig, optional
            Generation settings overriding the pipeline default.
        **overrides : object
            Generation keyword arguments overriding ``config``.

        Returns
        -------
        DataSet
            Generated noisy signals and clean targets.
        """

        if self.generator is None:
            raise ValueError("Pipeline.generate requires a SignalGenerator.")
        config = config or self.generation_config
        kwargs = {} if config is None else config.signal_kwargs()
        kwargs.update(overrides)
        if (
            config is not None
            and config.sequence_length != self.generator.sequence_length
        ):
            raise ValueError(
                "GenerationConfig sequence_length does not match the generator."
            )
        return self.generator.generate(
            n_samples=n_samples,
            kernel=kernel,
            peak_count=peak_count,
            **kwargs,
        )

    def run_dataset(self, dataset: DataSet) -> list[PipelineResult]:
        """Analyze every observed signal in a generated dataset.

        Parameters
        ----------
        dataset : DataSet
            Dataset whose observed signals should be analyzed.

        Returns
        -------
        list of PipelineResult
            One result for each observed signal.
        """

        results = []
        for index, signal in enumerate(np.asarray(dataset.signals)):
            results.append(
                self.run(
                    signal,
                    dx=(
                        float(np.diff(dataset.x_values)[0])
                        if len(dataset.x_values) > 1
                        else 1.0
                    ),
                    metadata={"dataset_index": index},
                )
            )
        return results

    def generate_and_run(self, **kwargs: Any) -> list[PipelineResult]:
        """Generate a dataset and immediately analyze all generated signals.

        Parameters
        ----------
        **kwargs : object
            Arguments forwarded to :meth:`generate`.

        Returns
        -------
        list of PipelineResult
            Results linked to the generated dataset.
        """

        dataset = self.generate(**kwargs)
        results = self.run_dataset(dataset)
        for result in results:
            result.dataset = dataset
        return results

    @staticmethod
    def _detection_from_record(record: Any, detector: str) -> DetectionResult:
        attribute = "cnn" if detector == "cnn" else "standard"
        detection = getattr(record, attribute, None)
        if not isinstance(detection, DetectionResult):
            raise TypeError(
                f"Analyzer record must expose a DetectionResult as {attribute!r}."
            )
        return detection


__all__ = ["Pipeline", "PipelineResult"]
