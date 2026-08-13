DeepPeak model weights
======================

Place trusted, trained model artifacts in the subdirectory matching their
architecture:

.. code-block:: text

   weights/
   ├── unet1d/
   │   └── default-smooth-bce/
   │       ├── config.json
   │       ├── .weights.h5
   │       └── history.json
   ├── wavenet/
   │   ├── default-smooth-bce/
   │   ├── base-smooth-bce/
   │   └── regression-shape-aware/
   │       ├── config.json
   │       ├── .weights.h5
   │       └── history.json
   ├── densenet/
   │   └── clean-trace-v1.keras
   └── other/

The directory-based layout is produced by ``UNet1D.save()`` and
``WaveNet.save()`` and loaded with the corresponding ``.load()`` method:

.. code-block:: python

   from DeepPeak.directories import weights_path
   from DeepPeak.models import UNet1D, WaveNet

   unet_path = weights_path / "unet1d" / "default-smooth-bce"
   model = UNet1D.load(unet_path)

   wavenet_path = weights_path / "wavenet" / "default-smooth-bce"
   model = WaveNet.load(wavenet_path)

For Keras files such as DenseNet artifacts, use the model's standard save and
load APIs with a path under ``weights/densenet``.

The imported artifacts currently use these descriptive names:

* ``wavenet/default-smooth-bce`` — sigmoid WaveNet with smooth BCE;
* ``wavenet/base-smooth-bce`` — earlier WaveNet smooth-BCE checkpoint;
* ``wavenet/regression-shape-aware`` — linear WaveNet with shape-aware loss;
* ``unet1d/default-smooth-bce`` — U-Net with smooth BCE.

Weight files can be large and are intentionally not required for installing
DeepPeak. Every checked-in artifact should include a short manifest describing
the model architecture, sequence length, training data version,
preprocessing, loss, and validation metrics.
