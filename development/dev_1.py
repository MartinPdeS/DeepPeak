import numpy as np
import matplotlib.pyplot as plt
from DeepPeak.data import generate_square_dataset  # Replace with actual module name

def showcase_plot_normalized_widths():
    # Parameters for dataset generation
    sample_count = 3
    sequence_length = 128
    peak_count = (1, 4)
    amplitude_range = (1, 1)
    center_range = (0, 1)
    width_range = 0.1
    normalize_x = True

    # Generate the dataset
    signals, amplitudes, peak_counts, positions, widths, x_values = generate_square_dataset(
        sample_count=sample_count,
        sequence_length=sequence_length,
        peak_count=peak_count,
        amplitude_range=amplitude_range,
        center_range=center_range,
        width_range=width_range,
        noise_std=0.0,
        normalize_x=normalize_x,
        normalize=True,
        nan_values=0,
        sort_peak='position',
        categorical_peak_count=False
    )

    # Plot the result
    plt.figure(figsize=(10, 5))
    signal = signals[0, :, 0]
    plt.plot(x_values, signal, label="Signal", linewidth=2)

    for pos, width, amp in zip(positions[0], widths[0], amplitudes[0]):
        if not np.isnan(pos):
            start = max(0, pos - width / 2)
            end = min(1, pos + width / 2)
            plt.axvspan(start, end, color="red", alpha=0.3, label="Peak Width")
            plt.scatter([pos], [amp], color="blue", label="Peak Center", zorder=3)

    plt.title("Normalized Widths with Normalized X-Values")
    plt.xlabel("X-axis (normalized)")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Showcase the plot
showcase_plot_normalized_widths()
