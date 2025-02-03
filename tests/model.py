import pytest
from DeepPeak.models import get_position_model
from DeepPeak.signals import generate_gaussian_dataset
from DeepPeak.utils.training_utils import dataset_split
from unittest.mock import patch

def test_deeppeak_workflow():
    # Generate dataset
    slices, amplitudes, peak_count, positions, widths, x_values = generate_gaussian_dataset(
        n_samples=300,
        sequence_length=200,
        n_peaks=(0, 2),
        amplitude=5,
        position=(0, 199),
        width=4,
        noise_std=0.1,
        categorical_peak_count=True
    )

    # Split the dataset
    dataset = dataset_split(
        x=slices,
        positions=positions,
        peak_count=peak_count,
        amplitudes=amplitudes,
        widths=widths,
        test_size=0.2,
        random_state=42,
    )

    # Initialize the model
    model = get_position_model(sequence_length=200, max_peak_count=2)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(
        dataset['train']['x'],
        dataset['train']['positions'],
        epochs=10,  # Reduced epochs for testing
        batch_size=32,
        validation_split=0.2
    )

    # Assert the history has keys for loss and metrics
    assert 'loss' in history.history, "Training history does not contain 'loss'."
    assert 'val_loss' in history.history, "Training history does not contain 'val_loss'."


if __name__ == "__main__":
    pytest.main(["-W error", __file__])

