
Examples
========

Welcome to the DeepPeak examples gallery! This directory contains executable
examples demonstrating the public DeepPeak API for signal processing, peak
detection, analysis, and visualization. Examples are designed to run with a
non-interactive Matplotlib backend and return figures instead of requiring a
user to close plot windows.


DeepPeak is a Python library designed for generating, detecting, and analyzing peaks in 1D signals. These examples showcase the library's main features:

- **Signal Generation**: Create synthetic datasets with controllable noise and peak characteristics
- **Classical Detection**: Detect peaks with trigger and suppression algorithms
- **Neural deconvolution**: Train models to reconstruct clean pulse traces
- **Direct-versus-deconvolved comparison**: Compare arrival, amplitude, and width distributions
- **Stable Results**: Pass trace, detection, and metric objects between pipeline stages
- **Trace Analysis**: Characterize noise and pulse shapes
- **Peak Detection Algorithms**: Apply traditional and ML-enhanced peak detection methods
- **Visualization Tools**: Plot and analyze results with built-in visualization utilities

Suggested order
---------------

#. Start with ``core_result_objects.py`` to learn the stable data contracts.
#. Continue with ``classical_detection_pipeline.py`` for a complete detector
   workflow.
#. Use ``data_generation.py`` and ``data_generation_custom_kernel.py`` to
   create controlled training signals.
#. Explore ``noise_and_pulse_analysis.py`` and ``amplitude_retrieval.py`` for
   downstream trace diagnostics.
#. Use the classifier examples only when TensorFlow and a trained model are
   required.
#. Read the direct-versus-deconvolved comparison section in ``getting_started``
   before training a model for reconstruction. The ROI-classification gallery
   examples use binary masks and are a different objective.
