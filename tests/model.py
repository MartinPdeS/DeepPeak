import pytest
from DeepPeak.models import get_position_model
from DeepPeak.signals import generate_signal_dataset
from DeepPeak.utils.training_utils import dataset_split
from unittest.mock import patch

def test_deeppeak_workflow():
    # Generate dataset
    signals, _, _, positions, _, _ = generate_signal_dataset(
        n_samples=300,
        sequence_length=200,
        n_peaks=(0, 2),
        amplitude=(1, 5),
        position=(0, 1),
        width=(0.04, 0.04),
        noise_std=0.1,
        categorical_peak_count=True
    )

    # Initialize the model
    model = get_position_model(sequence_length=200, max_peak_count=2)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(
        signals,
        positions,
        epochs=10,  # Reduced epochs for testing
        batch_size=32,
        validation_split=0.2
    )

    # Assert the history has keys for loss and metrics
    assert 'loss' in history.history, "Training history does not contain 'loss'."
    assert 'val_loss' in history.history, "Training history does not contain 'val_loss'."


if __name__ == "__main__":
    pytest.main(["-W error", __file__])

