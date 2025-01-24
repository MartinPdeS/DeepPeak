import numpy as np
from DeepPeak.data import generate_square_peak_dataset
from DeepPeak.utils.visualization import plot_dataset  # Replace 'your_module' with the actual module name

# Parameters for dataset generation
sample_count = 10
sequence_length = 128
peak_count = (2, 5)
amplitude_range = (1, 5)
center_range = (0, 127)
width_range = (5, 20)
noise_std = 0.1
normalize = True
normalize_x = True

# Generate a square peak dataset
signals, amplitudes, peak_counts, positions, widths, x_values = generate_square_peak_dataset(
    sample_count=sample_count,
    sequence_length=sequence_length,
    peak_count=peak_count,
    amplitude_range=amplitude_range,
    center_range=center_range,
    width_range=width_range,
    noise_std=noise_std,
    normalize=normalize,
    normalize_x=normalize_x,
    nan_values=0,
    sort_peak='position',
    categorical_peak_count=False
)

# Plot the dataset
plot_dataset(
    signals=signals,
    amplitudes=amplitudes,
    positions=positions,
    widths=widths,
    x_values=x_values,
    num_samples=5,
    title="Example of Square Peak Dataset"
)
