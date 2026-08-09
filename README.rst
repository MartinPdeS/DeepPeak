|logo|

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Badge
     - Status
   * - Python versions
     - |python|
   * - Documentation
     - |docs|
   * - Continuous integration
     - |ci/cd|
   * - Test coverage
     - |coverage|
   * - Google Colab
     - |colab|
   * - PyPI package
     - |PyPI|
   * - PyPI downloads
     - |PyPI_download|
   * - Anaconda package
     - |anaconda|
   * - Anaconda downloads
     - |anaconda_download|
   * - Latest Anaconda release
     - |anaconda_date|


DeepPeak
========

DeepPeak is a Python package for generating, detecting, and analyzing peaks in
one-dimensional signals. Its central workflow is to compare direct peak
detection with optional neural deconvolution followed by peak detection. It
also provides classical signal-processing methods, trainable neural-network
models, synthetic signal generation, and dilution-series analysis.

It is designed for researchers and engineers working with pulse-like traces,
event streams, and other sparse one-dimensional signals.

Key Features
------------
- **Classical peak detection**: Height, sigma, prominence, zero-crossing, and
  non-maximum-suppression methods.
- **Optional neural deconvolution**: CNN, WaveNet, and 1D U-Net models can
  reconstruct a cleaner pulse signal before peak detection.
- **Synthetic data generation**: Gaussian, Lorentzian, square, Dirac, custom,
  and two-lobe kernels with configurable noise and peak-count models.
- **Trace analysis**: Arrival-time, amplitude, width, pulse-shape, noise, and
  dead-time analysis.
- **Direct-versus-deconvolved evaluation**: Compare event counts,
  time-of-arrival, amplitude, and width distributions on the same traces.
- **Dilution-series workflows**: Standard and flash dilution-series analysis
  with detector-specific metrics and plots.
- **Plotting and diagnostics**: Figures are returned as Matplotlib objects so
  they can be customized, saved, or embedded in notebooks.

Installation
------------

Install the released package from PyPI:

.. code-block:: console

   pip install DeepPeak

The classical signal-processing and generation APIs do not require TensorFlow.
Install the optional neural-network stack when using the models package:

.. code-block:: console

   pip install "DeepPeak[ml]"

For development, install the repository and its test/documentation tools in
your preferred virtual environment.

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
-------------------------------------

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

Documentation
-------------

The full API reference, theory notes, and executable examples are available at
`the DeepPeak documentation <https://martinpdes.github.io/DeepPeak/>`_.

Contact
-------
For questions or contributions, contact `martin.poinsinet.de.sivry@gmail.com <mailto:martin.poinsinet.de.sivry@gmail.com>`_.

.. |python| image:: https://img.shields.io/pypi/pyversions/deeppeak.svg
    :alt: Python
    :target: https://www.python.org/
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Google Colab
    :target: https://colab.research.google.com/github/MartinPdeS/DeepPeak/blob/master/notebook.ipynb
.. |docs| image:: https://github.com/martinpdes/deeppeak/actions/workflows/deploy_documentation.yml/badge.svg
    :target: https://martinpdes.github.io/DeepPeak/
    :alt: Documentation Status
.. |PyPI| image:: https://badge.fury.io/py/DeepPeak.svg
    :alt: PyPI version
    :target: https://badge.fury.io/py/DeepPeak
.. |PyPI_download| image:: https://img.shields.io/pypi/dm/DeepPeak?style=plastic&label=PyPI%20downloads&labelColor=hex&color=hex
    :alt: PyPI downloads
    :target: https://pypistats.org/packages/deeppeak
.. |coverage| image:: https://raw.githubusercontent.com/MartinPdeS/DeepPeak/python-coverage-comment-action-data/badge.svg
    :alt: Unittest coverage
    :target: https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html
.. |ci/cd| image:: https://github.com/martinpdes/deeppeak/actions/workflows/deploy_coverage.yml/badge.svg
    :alt: Unittest Status
.. |anaconda| image:: https://anaconda.org/martinpdes/deeppeak/badges/version.svg
    :alt: Anaconda version
    :target: https://anaconda.org/martinpdes/deeppeak
.. |anaconda_download| image:: https://anaconda.org/martinpdes/deeppeak/badges/downloads.svg
    :alt: Anaconda downloads
    :target: https://anaconda.org/martinpdes/deeppeak
.. |anaconda_date| image:: https://anaconda.org/martinpdes/deeppeak/badges/latest_release_relative_date.svg
    :alt: Latest release date
    :target: https://anaconda.org/martinpdes/deeppeak
.. |logo| image:: https://github.com/MartinPdeS/DeepPeak/raw/master/docs/images/logo.svg
    :alt: DeepPeak logo
