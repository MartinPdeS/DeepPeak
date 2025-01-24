from DeepPeak import models
from DeepPeak.data.data_generation import generate_gaussian_dataset
from DeepPeak.utils.visualization import plot_training_history, visualize_validation_cases
from DeepPeak.utils.training_utils import dataset_split
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

# Define the callback to save the best model
checkpoint_callback = ModelCheckpoint(
    filepath="best_model.keras",          # Path to save the best model
    monitor="val_loss",                # Metric to monitor (e.g., "val_loss" or "val_accuracy")
    save_best_only=True,               # Save only the best model
    save_weights_only=False,           # Save the entire model (set True to save only weights)
    mode="min",                        # "min" for loss (smaller is better), "max" for accuracy
    verbose=1                          # Display a message when saving the model
)

# Call the function
slices, amplitudes, peak_count, positions, widths, index = generate_gaussian_dataset(
    sample_count=4000,
    sequence_length=200,
    peak_count=(1, 3),
    amplitude_range=(3, 3),
    center_range=(0.1, 0.9),
    width_range=0.02,
    noise_std=0.1,
    normalize=True,
    normalize_x=True,
    categorical_peak_count=True
)

dataset = dataset_split(
    x=slices,
    positions=positions,
    amplitudes=amplitudes,
    peak_count=peak_count,
    widths=widths,
    test_size=0.2,
    random_state=None,
)

model = models.get_position_model(
    input_length=200,
    max_peak_count=3
)

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

history = model.fit(
    dataset['train']['x'],
    dataset['train']['positions'],
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint_callback]
)

model = load_model("best_model.keras")

# plot_training_history(history, filtering=['*loss*'])

visualize_validation_cases(
    model=model,
    validation_data=dataset['test'],
    sequence_length=200,
    num_examples=12,
    n_columns=3
)