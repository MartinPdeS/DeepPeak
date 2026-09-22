Train and evaluate WaveNet
==========================

Install the optional stack with ``pip install "DeepPeak[ml]"``. Then build a
small model and train directly from a generated ``DataSet``:

.. code-block:: python

   from DeepPeak.models import TrainingConfig, WaveNet

   model = WaveNet(
       sequence_length=512,
       num_filters=32,
       num_dilation_layers=6,
       loss="huber",
       metrics=("mae",),
       seed=42,
   )
   model.build()
   model.fit_dataset(
       training_dataset,
       target="clean",
       normalization="zscore",
       config=TrainingConfig(epochs=30, batch_size=32, validation_split=0.2),
       verbose=0,
   )
   evaluation = model.evaluate_dataset(test_dataset, target="clean")
   print(evaluation.summary())
   model.model.save("artifacts/wavenet.keras")

Run the saved model through ``tools/run_benchmarks.py`` before reporting it.
Record the data seed, package version, model configuration, stopping epoch,
and benchmark CSV with the trained artifact.
