import pytest
import numpy as np
from DeepPeak.DeepPeak.models import get_simple_model
from DeepPeak.data_generation import generate_gaussian_dataset
from DeepPeak.plots import plot_training_history, visualize_validation_cases
from DeepPeak.utils import dataset_split

def test_deeppeak_workflow():
    # Generate dataset
    slices, amplitudes, peak_count, positions, widths = generate_gaussian_dataset(
        sample_count=300,
        sequence_length=200,
        peak_count=2,
        amplitude_range=5,
        center_range=(0, 199),
        width_range=4,
        noise_std=0.1,
        normalize=False,
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
        max_number_of_peaks=2
    )

    # Initialize the model
    model = get_simple_model(
        input_length=200,
        max_peak_count=2
    )

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

    # Test visualization of validation cases (non-blocking)
    try:
        visualize_validation_cases(
            model=model,
            validation_data=dataset['test'],
            input_length=200,
            num_examples=5,
            n_columns=2
        )
    except Exception as e:
        pytest.fail(f"Visualization of validation cases failed: {e}")

if __name__ == "__main__":
    pytest.main(["-W error", __file__])

