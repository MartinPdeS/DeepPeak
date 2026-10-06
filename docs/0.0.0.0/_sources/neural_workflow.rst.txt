Neural model workflow
=====================

DeepPeak datasets and neural models share a small, explicit workflow. A
dataset validates its common shapes, exposes channel-last model arrays, and
can be split reproducibly. Models provide ``fit_dataset`` and
``evaluate_dataset`` convenience methods.

.. code-block:: python

   from DeepPeak import Gaussian, SignalGenerator, UniformCount
   from DeepPeak.models import TrainingConfig, WaveNet, plot_predictions

   generator = SignalGenerator(sequence_length=256)
   dataset = generator.generate(
       n_samples=128,
       kernel=Gaussian(
           amplitude=(1.0, 2.0),
           position=(20.0, 236.0),
           width=(8.0, 16.0),
       ),
       peak_count=UniformCount(bounds=(1, 3)),
       seed=42,
       noise_std=0.05,
   )

   train, test = dataset.train_test_split(test_size=0.2, seed=42)
   model = WaveNet(
       sequence_length=dataset.sequence_length,
       num_filters=16,
       num_dilation_layers=4,
       kernel_size=3,
       output_activation="sigmoid",
       loss="binary_crossentropy",
   )
   model.fit_dataset(train, target="labels", config=TrainingConfig(epochs=3))
   result = model.evaluate_dataset(test, target="labels")
   print(result.summary())
   plot_predictions(model, test, n_samples=6, n_columns=3)

``to_model_inputs`` returns arrays shaped ``(n_samples, sequence_length, 1)``
and ``targets`` selects labels by default, falling back to clean traces when
labels are unavailable. ``ModelEvaluationResult`` stores scalar metrics and,
by default, the inputs, targets, and predictions used to calculate them.
