:orphan:

DeepPeak Examples
=================

Welcome to the DeepPeak examples gallery! This directory contains comprehensive examples demonstrating the key capabilities of the DeepPeak library for signal processing and peak detection using deep learning.

Overview
--------

DeepPeak is a Python library designed for detecting and analyzing peaks in 1D signals using machine learning approaches. These examples showcase the library's main features:

- **Signal Generation**: Create synthetic datasets with controllable noise and peak characteristics
- **Machine Learning Classifiers**: Train neural networks to detect regions of interest
- **Peak Detection Algorithms**: Apply traditional and ML-enhanced peak detection methods
- **Visualization Tools**: Plot and analyze results with built-in visualization utilities

Examples Description
--------------------

**Data Generation and Visualization**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`data_generation`
    Learn how to generate synthetic signals with Gaussian pulses and visualize them with region-of-interest masks. This is the foundation for understanding how DeepPeak handles signal data.

**Machine Learning Classifiers**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`classifier_dense`
    Demonstrates training a DenseNet classifier to identify regions of interest in noisy signals containing Gaussian peaks. Shows the complete workflow from data preparation to model evaluation.

:doc:`classifier_autoencoder`
    Explores using autoencoder architectures for unsupervised feature learning and anomaly detection in signal data.

:doc:`classifier_wavenet`
    Showcases the WaveNet classifier for temporal pattern recognition in 1D signals, leveraging dilated convolutions for multi-scale feature extraction.

**Peak Detection Algorithms**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`non_maximum_suppression`
    Implements classical non-maximum suppression for peak detection, providing a baseline comparison for machine learning approaches.

Getting Started
---------------

To run these examples, you'll need to install DeepPeak and its dependencies:

.. code-block:: bash

    pip install DeepPeak

Each example is self-contained and can be run independently. They are designed to be:

- **Educational**: Clear explanations of concepts and methods
- **Reproducible**: Fixed random seeds and documented parameters
- **Interactive**: Rich visualizations to understand the data and results
- **Practical**: Real-world applicable techniques and best practices

Example Structure
-----------------

Each example follows a consistent structure:

1. **Introduction**: Brief overview of the demonstrated technique
2. **Data Preparation**: Signal generation or loading
3. **Method Application**: Algorithm implementation and training
4. **Results Visualization**: Plotting and analysis of outputs
5. **Discussion**: Interpretation of results and next steps

Usage Tips
----------

**For Beginners**
    Start with the ``data_generation`` example to understand the data format, then proceed to ``classifier_dense`` for a complete machine learning workflow.

**For Researchers**
    The examples demonstrate reproducible research practices with proper documentation, visualization, and evaluation metrics.

**For Developers**
    Each example showcases different aspects of the DeepPeak API, serving as practical documentation for library usage.

Requirements
------------

The examples require:

- Python ≥ 3.8
- DeepPeak library
- NumPy, Matplotlib for basic functionality
- TensorFlow/Keras for machine learning examples
- Additional dependencies are automatically installed with DeepPeak

Running Examples
---------------

You can run examples in several ways:

**Direct execution**:

.. code-block:: bash

    python classifier_dense.py

**Jupyter notebook**:
    Convert any example to a notebook for interactive exploration:

.. code-block:: bash

    jupytext --to notebook classifier_dense.py

**Sphinx-Gallery**:
    These examples are designed for Sphinx-Gallery and will automatically generate beautiful documentation with embedded plots.

Contributing
------------

Found an issue or want to add a new example? Contributions are welcome! Please:

1. Follow the existing example structure and style
2. Include clear documentation and comments
3. Add appropriate visualizations
4. Test your example thoroughly
5. Submit a pull request

For questions or suggestions, please open an issue on the `DeepPeak GitHub repository <https://github.com/MartinPdeS/DeepPeak>`_.

Next Steps
----------

After exploring these examples, you might want to:

- Adapt the code for your own signal data
- Experiment with different neural network architectures
- Combine multiple detection methods for robust peak detection
- Explore advanced topics in the main DeepPeak documentation

Happy peak detecting!



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use DeepPeak&#x27;s DenseNet classifier to identify regions of interest (ROIs) in synthetic 1D signals containing Gaussian peaks.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_classifier_autoencoder_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_classifier_autoencoder.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DenseNet Classifier: Detecting Regions of Interest in Synthetic Signals</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use DeepPeak&#x27;s DenseNet classifier to identify regions of interest (ROIs) in synthetic 1D signals containing Gaussian peaks.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_classifier_dense_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_classifier_dense.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DenseNet Classifier: Detecting Regions of Interest in Synthetic Signals</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use DeepPeak&#x27;s DenseNet classifier to identify regions of interest (ROIs) in synthetic 1D signals containing Gaussian peaks.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_classifier_wavenet_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_classifier_wavenet.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DenseNet Classifier: Detecting Regions of Interest in Synthetic Signals</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to:   1. Generate synthetic signals with up to 3 Gaussian pulses.   2. Compute a Region of Interest (ROI) mask based on pulse positions.   3. Visualize signals with peak positions, amplitudes, and the ROI mask.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_data_generation_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_data_generation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Generating and Visualizing Signal Data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the use of the NonMaximumSuppression class to detect Gaussian pulses in a one-dimensional signal. It generates a synthetic dataset of Gaussian pulses, applies the non-maximum suppression algorithm, and plots the results.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_non_maximum_suppression_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_non_maximum_suppression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Non-Maximum Suppression for Gaussian Pulse Detection</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /gallery/classifier_autoencoder
   /gallery/classifier_dense
   /gallery/classifier_wavenet
   /gallery/data_generation
   /gallery/non_maximum_suppression



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
