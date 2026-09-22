Testing and reproducibility
===========================

Run the test suite from the repository root with the project's virtual
environment:

.. code-block:: console

   MPLBACKEND=Agg MPLCONFIGDIR=/tmp/deeppeak-mpl python -m pytest -q

The ``Agg`` backend is non-interactive, so tests that exercise plotting do not
open windows. The test fixture also closes figures between tests. Use the same
pattern for downstream CI jobs and notebooks executed without a display.

Tests are grouped by capability:

* ``test_architecture.py`` checks stable result objects, namespaces, and typed
  configurations;
* ``test_peak_count_analysis.py`` covers analyzers, dilution series, metrics,
  and plotting accessors;
* ``test_noise_analysis.py`` and ``test_pulse_shape_analysis.py`` cover trace
  diagnostics;
* ``test_classifiers.py`` covers optional TensorFlow model behavior.

For a fast local check, exclude tests requiring the optional ML stack:

.. code-block:: console

   MPLBACKEND=Agg python -m pytest -q -m "not ml and not slow"

Run the focused neural architecture, metric, and serialization checks with:

.. code-block:: console

   TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -m ml

Full multi-epoch training belongs in reproducible benchmark jobs, not the unit
suite. Coverage is measured across both lanes; new changes must not reduce the
reported percentage, and the project target is 80% branch coverage.

Examples in the Sphinx-Gallery are executable documentation. They should
return figures rather than call ``plt.show()`` so that the same examples work
in documentation builds, notebooks, and headless CI.

Docstring style
---------------

DeepPeak uses NumPy-style docstrings. Public callables should document inputs
and outputs with underlined sections:

.. code-block:: text

   Parameters
   ----------
   signal : array-like
       Input signal samples.

   Returns
   -------
   ndarray
       Processed signal.

Use ``Raises``, ``Examples``, and ``Notes`` sections in the same format when
they add useful information. Google-style ``Args:`` and ``Returns:`` blocks
are not used. Sphinx is configured with NumPy parsing enabled and Google
parsing disabled.
