API reference
=============

This document provides the API documentation for the DeepPeak package.


Analysis API
------------

.. autoclass:: DeepPeak.analysis.StandardDilutionSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.FlashDilutionSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.WaveNetTraceAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.StandardTraceAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.TraceComparisonAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.TraceComparisonResult
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.analysis.SeriesComparisonResult
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.analysis.BasePeakTrigger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.NoiseAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.analysis.PulseShapeAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Core result API
---------------

.. autoclass:: DeepPeak.core.Trace
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.DetectionResult
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.MetricResult
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.SeriesResult
   :members:
   :undoc-members:

Analysis errors
---------------

DeepPeak exposes specific exceptions while retaining compatibility with the
corresponding built-in exception types.

.. autoclass:: DeepPeak.core.InvalidConfigurationError
   :members:

.. autoclass:: DeepPeak.core.InvalidDetectorError
   :members:

.. autoclass:: DeepPeak.core.MissingDetectorError
   :members:

.. autoclass:: DeepPeak.core.AnalysisStateError
   :members:

.. autoclass:: DeepPeak.core.AnalysisInputError
   :members:

Typed configuration API
-----------------------

Configuration objects provide validated, reusable settings for trace
processing, detection, dilution-series workflows, and plotting.

.. autoclass:: DeepPeak.core.TraceConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.DetectionConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.SeriesConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.PlotConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.AnalysisConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.GenerationConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.ModelConfig
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.core.NoiseConfig
   :members:
   :undoc-members:

Pipeline API
------------

.. autoclass:: DeepPeak.Pipeline
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.PipelineResult
   :members:
   :undoc-members:

Plotting API
------------

Plotting is kept in ``DeepPeak.plotting`` and returns Matplotlib figures.

.. autofunction:: DeepPeak.plotting.standard_detection

.. autofunction:: DeepPeak.plotting.wavenet_detection

.. autofunction:: DeepPeak.plotting.standard_detection_with_histogram

.. autofunction:: DeepPeak.plotting.wavenet_detection_with_histogram

Neural-network API
------------------

All neural models reconstruct clean pulse traces. They accept a
``TrainingConfig`` through ``fit(config=...)`` for reproducible training with
optional early stopping, learning-rate scheduling, deterministic seeds, and
best-model checkpointing.

.. autoclass:: DeepPeak.models.WaveNet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.models.DenseNet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.models.UNet1D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.models.WeightedBinaryCrossentropy
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.models.SmoothBinaryCrossentropy
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.models.WeightedHuber
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.models.ShapeAwarePulseLoss
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.models.BinaryIoU
   :members:
   :undoc-members:

.. autoclass:: DeepPeak.models.TrainingConfig
   :members:
   :undoc-members:

.. autofunction:: DeepPeak.models.weighted_bce

.. autofunction:: DeepPeak.models.smooth_bce

.. autofunction:: DeepPeak.models.weighted_huber

.. autofunction:: DeepPeak.models.shape_aware_pulse_loss

.. autofunction:: DeepPeak.models.plot_predictions

Signal-generation API
----------------------

``SignalGenerator`` supports optional acquisition effects while keeping the
underlying ``clean_signals`` target available: constant or linearly varying
noise, baseline offsets and drift, finite instrument responses, saturation,
and quantization.

.. autoclass:: DeepPeak.SignalGenerator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.Gaussian
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.NonstationaryGaussianNoise
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DeepPeak.CorrelatedGaussianNoise
   :members:
   :undoc-members:
   :show-inheritance:
