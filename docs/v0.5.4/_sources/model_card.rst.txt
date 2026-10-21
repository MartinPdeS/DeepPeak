Model cards
===========

Every distributed model should include a model card containing:

* architecture and DeepPeak/TensorFlow versions;
* intended use and out-of-scope use;
* training and validation dataset generation, seeds, and splits;
* preprocessing and output threshold;
* benchmark results for overlap, noise, baseline drift, and domain shift;
* calibration error and uncertainty intervals;
* known limitations, especially the pulse shapes and sampling rates not seen
  during training;
* artifact checksum and license.

DeepPeak does not currently distribute pretrained weights in its wheel. A
locally trained model must not be described as generally validated until it
has been evaluated on held-out experimental data representative of its use.
