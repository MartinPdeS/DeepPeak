Migration guide
===============

DeepPeak's public API is organized by domain. New code should use the
namespaces below; they are stable entry points for application code and
notebooks.

===============================  =============================================
Purpose                          Preferred namespace
===============================  =============================================
Shared result/config objects     ``DeepPeak.core``
Synthetic data and kernels       ``DeepPeak.generation``
Peak detection                   ``DeepPeak.detection``
Neural architectures             ``DeepPeak.models``
Analysis workflows               ``DeepPeak.analysis``
Numerical metrics                ``DeepPeak.metrics``
Figures                          ``DeepPeak.plotting``
Trace loading and serialization  ``DeepPeak.io``
===============================  =============================================

Common API changes
------------------

* Use ``StandardDilutionSeries`` for classical detection and
  ``FlashDilutionSeries`` for CNN-based workflows.
* Use ``HeightPeakTrigger``, ``SigmaPeakTrigger``, or
  ``ProminencePeakTrigger`` from ``DeepPeak.detection``.
* Use ``Trace``, ``DetectionResult``, ``MetricResult``, and ``SeriesResult``
  from ``DeepPeak.core`` when passing data between pipeline stages.
* Use ``DeepPeak.plotting.standard_detection`` and related functions for new
  plotting code. Plotting functions return figures and never need to display
  them automatically.
* Pass ``DetectionConfig``, ``SeriesConfig``, and ``PlotConfig`` when settings
  are reused across runs.

The old internal module layout is no longer the preferred import path. Replace
imports from modules such as ``DeepPeak.algorithms``, ``DeepPeak.kernels``, and
``DeepPeak.machine_learning`` with the domain namespaces above.

Headless plotting
-----------------

Plots are created with ``show=False`` by default. In automated environments,
use an ``Agg`` backend before importing ``matplotlib.pyplot``:

.. code-block:: python

   import matplotlib

   matplotlib.use("Agg")

   from DeepPeak.core import PlotConfig

   plot_config = PlotConfig(show=False, close=True)

``close=True`` closes the returned figure after rendering. This is useful in
large loops and tests where retaining every figure would consume memory.
