Generate synthetic traces
=========================

Create a seeded dataset so experiments can be reproduced exactly:

.. code-block:: python

   from DeepPeak import Gaussian, GaussianNoise, SignalGenerator, UniformCount

   dataset = SignalGenerator(sequence_length=512).generate(
       n_samples=256,
       kernel=Gaussian(amplitude=(0.8, 1.2), position=(0.05, 0.95), width=(0.01, 0.04)),
       peak_count=UniformCount(bounds=(1, 5)),
       noise=GaussianNoise(std=0.05),
       seed=42,
   )

   inputs = dataset.to_model_inputs(normalization="zscore")
   targets = dataset.targets("clean")

Keep the seed and generation configuration with every experiment. Use
``DeepPeak.benchmarking.make_benchmark_dataset`` when comparing detectors;
that dataset adds controlled overlap, noise, drift, and domain-shift levels.
