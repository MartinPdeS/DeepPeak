DeepPeak usage guide
====================

Quickstart: generate a signal
-----------------------------

The generation API can create reproducible training and evaluation data:

.. code-block:: python

   from DeepPeak import Gaussian, SignalGenerator, UniformCount

   generator = SignalGenerator(sequence_length=1_000)
   dataset = generator.generate(
       n_samples=128,
       kernel=Gaussian(
           amplitude=(1.0, 10.0),
           position=(0.1, 0.9),
           width=(0.02, 0.05),
       ),
       peak_count=UniformCount(bounds=(1, 4)),
       seed=42,
       noise_std=0.05,
   )

   signals = dataset.signals

Analysis quickstart
-------------------

For standard dilution-series analysis, provide trace files as
``(filename, dilution)`` pairs and run the configured workflow:

.. code-block:: python

   from DeepPeak.analysis import HeightPeakTrigger, StandardDilutionSeries

   series = StandardDilutionSeries(
       folder="path/to/traces",
       files=[
           ("path/to/traces/trace_1.csv", 1.0),
           ("path/to/traces/trace_2.csv", 10.0),
       ],
       trigger=HeightPeakTrigger(height=0.05, hysteresis=0.03),
       initial_concentration=1.0,
       nrows=100_000,
   )

   result = series.run()
   series.plot.standard_detection(index=0)

   series.poisson.plot.expected_histogram(
       index=0,
       base_index=0,
       detector="standard",
       x_axis="time",
   )

   series.amplitude.plot.histogram(index=0, detector="standard")
   series.width.plot.histogram(index=0, detector="standard", x_axis="time")

For reusable settings, prefer the typed configuration objects exposed by
``DeepPeak.core``. They validate values at construction time and can be passed
to analyzers, dilution-series workflows, and plotting helpers:

.. code-block:: python

   import numpy as np

   from DeepPeak.core import DetectionConfig, PlotConfig, Trace
   from DeepPeak.detection import HeightPeakTrigger
   from DeepPeak.analysis import StandardTraceAnalyzer

   detection_config = DetectionConfig(
       sequence_length=1_000,
       normalization="zscore",
       trigger=HeightPeakTrigger(height=0.05),
   )
   analyzer = StandardTraceAnalyzer(config=detection_config)
   signal = np.zeros(1_000)
   signal[500] = 1.0
   detection = analyzer.detect(Trace(signal=signal, dx=1.0))
   record = analyzer.analyze_processed_signal(signal, dx=1.0)

   figure = record.plot_standard_detection(
       config=PlotConfig(show=False, close=False, dpi=150),
   )

The plotting API always returns figures. Set ``show=False`` for scripts and
tests, and set ``close=True`` when a figure should be closed automatically after
it is created.

The detector-specific classes ``StandardDilutionSeries`` and
``FlashDilutionSeries`` are also available when a workflow should expose only
one detector mode. Use ``FlashDilutionSeries`` with a trained neural model for
CNN-based workflows.

Direct versus deconvolved comparison
------------------------------------

The neural model is an optional deconvolution stage. The same peak-detection
concept can therefore be evaluated directly on the raw trace and after neural
deconvolution:

.. code-block:: text

   raw trace ────────────────> detector ──> direct result
       │
       └─ optional CNN/WaveNet/U-Net ──> detector ──> deconvolved result

Use ``TraceComparisonAnalyzer`` to compare the two branches. The result
provides arrival-time, amplitude, and width distributions, together with
summary statistics and distribution differences. Set ``deconvolver=None`` to
run only the direct branch.

.. code-block:: python

   from DeepPeak.analysis import TraceComparisonAnalyzer
   from DeepPeak.core import Trace
   from DeepPeak.detection import HeightPeakTrigger

   comparison = TraceComparisonAnalyzer(
       standard_trigger=HeightPeakTrigger(height=0.05),
       deconvolver=trained_model,  # optional CNN, WaveNet, or U-Net wrapper
   ).compare(Trace(signal=signal, dx=1e-9))

   arrival_comparison = comparison.compare_distribution("arrival")
   amplitude_comparison = comparison.compare_distribution("amplitude")
   width_comparison = comparison.compare_distribution("width")

The same comparison can be run over multiple traces with ``compare_many``.
This makes it possible to quantify changes in event counts, time-of-arrival
distributions, retrieved amplitudes, widths, and distribution distances.

For this workflow, the neural model must be trained as a reconstruction model:
its target should be a clean or deconvolved pulse trace. For example, a
WaveNet model should use a linear output head and a regression loss when it is
trained to predict pulse amplitudes.

Architecture
------------

DeepPeak is being organized around clear domain boundaries:

.. code-block:: text

   DeepPeak/
   ├── core/          shared types, protocols, configuration, exceptions
   ├── generation/    synthetic signals, kernels, noise, datasets
   ├── detection/     classical and neural detection algorithms
   ├── models/        trainable neural-network architectures and losses
   ├── analysis/      trace and dilution-series workflows
   ├── metrics/       numerical diagnostics and distribution summaries
   ├── plotting/      visualization of traces, detections, and metrics
   └── io/            trace loading and result serialization

The intended dependency direction is:

.. code-block:: text

   core
     ↓
   generation / detection / models
     ↓
   analysis
     ↓
   metrics / plotting / io

This separation keeps numerical analysis independent from plotting and keeps
TensorFlow-specific code isolated from the core signal-processing API. The
domain namespaces are the supported public API for new code.

Public API guide
----------------

Use these namespaces when writing new code:

``DeepPeak.analysis``
   Trace analyzers, dilution-series workflows, triggers, and analysis results.

``DeepPeak.generation``
   Synthetic datasets, kernels, noise models, and peak-count models.

``DeepPeak.detection``
   Detection algorithms, triggers, and the common detection-result type.

``DeepPeak.models``
   DenseNet, WaveNet, UNet1D, neural losses, and model utilities.

``DeepPeak.metrics``
   Detection, amplitude, width, arrival-time, and series metrics.

``DeepPeak.plotting`` and ``DeepPeak.io``
   Figure helpers and trace/file loading utilities.

``DeepPeak.core``
   Stable ``Trace``, ``DetectionResult``, ``MetricResult``, and ``SeriesResult``
   objects, plus typed ``TraceConfig``, ``DetectionConfig``, ``SeriesConfig``,
   and ``PlotConfig`` settings.

The root ``DeepPeak`` namespace exposes the most common user-facing types for
interactive work and notebooks.
