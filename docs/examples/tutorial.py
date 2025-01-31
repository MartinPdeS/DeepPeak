import numpy as np
from DeepPeak.signals import generate_gaussian_dataset
from DeepPeak.utils.visualization import plot_training_history, SignalPlotter
from DeepPeak.models import build_ROI_model
from DeepPeak.utils.ROI import compute_rois_from_signals

from tensorflow import keras


NUM_PEAKS = 3
SEQUENCE_LENGTH = 200
sample_count = 6000

signals, amplitudes, positions, widths, x_values, num_peaks = generate_gaussian_dataset(
    n_samples=sample_count,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)


ROI = compute_rois_from_signals(
    signals=signals, positions=positions, width_in_pixels=3, amplitudes=amplitudes
)

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_positions(positions)
plotter.add_amplitudes(amplitudes)
plotter.add_roi(ROI)
plotter.set_title("Demo: Signals + Peaks + ROI")
plotter.plot(n_examples=6, n_columns=3, random_select=False)

roi_model = build_ROI_model(SEQUENCE_LENGTH)

# Train the model
history = roi_model.fit(
    signals, ROI,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)

plot_training_history(history, filtering=['*loss*'])

ROI_prediction = roi_model.predict(signals, verbose=0)['ROI'][:, :, 0]
ROI_prediction[ROI_prediction<0.5] = 0
ROI_prediction[ROI_prediction>0.5] = 1


plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_positions(positions)
plotter.add_amplitudes(amplitudes)
plotter.add_roi(ROI_prediction)


def multi_gauss(x, pos, width, amp):
    return amp * np.exp(-0.5 * ((x - pos) / width)**2)

plotter.add_custom_curves(
    curve_function=multi_gauss,
    label="Multi-Gauss",
    color="black",
    style="--",
    pos=positions,
    width=widths,
    amp=amplitudes
)

plotter.set_title("Demo: Signals + Peaks + ROI")
plotter.plot(n_examples=6, n_columns=3, random_select=True)