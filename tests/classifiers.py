from unittest.mock import patch

import numpy as np
import pytest

from DeepPeak.machine_learning.classifier import Autoencoder, DenseNet, WaveNet
from DeepPeak.signals import Kernel, SignalDatasetGenerator

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200


architectures = [DenseNet, WaveNet, Autoencoder]


@pytest.fixture
def dataset():
    generator = SignalDatasetGenerator(n_samples=600, sequence_length=SEQUENCE_LENGTH)

    dataset = generator.generate(
        signal_type=Kernel.GAUSSIAN,
        n_peaks=(1, NUM_PEAKS),
        amplitude=(1, 20),
        position=(0.1, 0.9),
        width=(0.03, 0.05),
        noise_std=0.1,
        categorical_peak_count=False,
        compute_region_of_interest=True,
    )

    return dataset


@pytest.mark.parametrize("architecture", architectures)
@patch("matplotlib.pyplot.show")
def test_architecture(patch, dataset, architecture):
    model = architecture(
        sequence_length=200,
    )
    model.build()
    model.summary()

    history = model.fit(
        dataset.signals,
        dataset.region_of_interest,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
    )

    model.plot_model_history()

    model.predict(signal=dataset.signals[0:1, :], threshold=0.4)


if __name__ == "__main__":
    pytest.main(["-W error", __file__])
