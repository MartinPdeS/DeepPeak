Getting started
================

DeepPeak workflows are built from four kinds of objects:

* ``Trace`` stores one signal and its sampling information.
* detector analyzers produce ``DetectionResult`` or a ``TraceRecord``;
* metric functions produce ``MetricResult`` and series summaries;
* plotting functions consume results and return Matplotlib figures.

This separation makes it possible to run numerical pipelines on a server and
decide later whether, where, and how to render a figure.

The smallest complete workflow
------------------------------

The following example generates a trace, detects peaks, and plots the result.
The ``PlotConfig`` explicitly disables display, which is the recommended
default for scripts, CI, and documentation builds.

.. code-block:: python

   import numpy as np

   from DeepPeak.core import DetectionConfig, PlotConfig, Trace
   from DeepPeak.detection import HeightPeakTrigger
   from DeepPeak.analysis import StandardTraceAnalyzer
   from DeepPeak.plotting import standard_detection

   signal = np.zeros(500)
   signal[[100, 300]] = [1.0, 0.8]
   trace = Trace(signal=signal, dx=1e-6)

   config = DetectionConfig(
       trigger=HeightPeakTrigger(height=0.5),
       sequence_length=500,
   )
   analyzer = StandardTraceAnalyzer(config=config)
   detection = analyzer.detect(trace)
   record = analyzer.analyze_processed_signal(signal, dx=trace.dx)

   figure = standard_detection(
       record,
       config=PlotConfig(show=False, close=False, dpi=150),
   )
   figure.savefig("detection.png")

``detect`` is the uniform detector-level API. ``analyze_processed_signal`` is
useful when you also need the richer ``TraceRecord`` consumed by the plotting
and metric layers.

Configuration objects
---------------------

Use frozen, validated configuration objects instead of repeating long keyword
argument lists:

``TraceConfig``
   Common sequence length, normalization, and sampling-rate settings.

``DetectionConfig``
   Trace settings plus detector trigger and CNN amplitude-recovery settings.

``SeriesConfig``
   Trace settings plus dilution-series concentration, row count, and filter
   settings.

``PlotConfig``
   Figure size, resolution, title, and non-interactive display behavior.

Invalid values fail early, for example ``TraceConfig(sequence_length=0)``
raises ``ValueError``. Configuration objects are immutable, so create a new
configuration when a workflow needs different settings.

Choosing an API layer
---------------------

For a single signal, use ``Trace`` and one of the trace analyzers. For repeated
measurements, use ``StandardDilutionSeries`` or ``FlashDilutionSeries`` with a
``SeriesConfig``. For visualization, import functions from
``DeepPeak.plotting`` rather than coupling new analysis code to plotting
methods on legacy record objects.

See the :doc:`gallery/index` for runnable examples and the :doc:`code` page for
the complete API reference.

Comparing direct and neural-deconvolved detection
-------------------------------------------------

The neural stage is optional. Direct detection can be used as the baseline,
while a trained CNN, WaveNet, or U-Net can produce a reconstructed signal for
the second branch:

.. code-block:: python

   from DeepPeak.analysis import TraceComparisonAnalyzer
   from DeepPeak.core import Trace
   from DeepPeak.detection import HeightPeakTrigger

   comparison_analyzer = TraceComparisonAnalyzer(
       standard_trigger=HeightPeakTrigger(height=0.05),
       deconvolver=trained_deconvolution_model,  # or None
       sequence_length=1_000,
   )
   comparison = comparison_analyzer.compare(trace)

   direct_arrivals = comparison.distribution("arrival", "standard")
   deconvolved_arrivals = comparison.distribution("arrival", "deconvolved")
   amplitude_change = comparison.compare_distribution("amplitude")
   width_change = comparison.compare_distribution("width")

The comparison object uses the same detector-result contract in both branches.
It reports means, medians, quantiles, count differences, Wasserstein distance,
and Kolmogorov-Smirnov statistics for arrival-time, amplitude, and width
distributions. With ``deconvolver=None``, the direct branch remains available
without requiring a neural model.

The reconstruction model should be trained against clean pulse traces. For a
WaveNet-style model, use a linear output head and a regression loss:

.. code-block:: python

   from DeepPeak.models import WaveNet

   model = WaveNet(
       sequence_length=1_000,
       output_activation="linear",
       loss="huber",
   )
   model.build()
   model.fit(noisy_signals, clean_signals)

For multiple traces, use ``compare_many`` and aggregate the distributions:

.. code-block:: python

   series_comparison = comparison_analyzer.compare_many(traces)
   arrival_summary = series_comparison.summary("deconvolved")["arrival"]
   all_improvements = series_comparison.compare()
