from DeepPeak.classifier.model import build_ROI_model
from DeepPeak.classifier.utils import compute_rois_from_signals, filter_predictions
from DeepPeak.signals import generate_signal_dataset
from DeepPeak.visualization import SignalPlotter, plot_training_history

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

signals, labels, amplitudes, positions, widths, x_values, num_peaks = generate_signal_dataset(
    n_samples=6000,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)

ROI = compute_rois_from_signals(signals=signals, positions=positions, width_in_pixels=3, amplitudes=amplitudes)

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_vline(positions)
plotter.add_hline(amplitudes)
plotter.add_roi(ROI)
plotter.set_title("Demo: Signals + Peaks + ROI")
_ = plotter.plot(n_examples=6, n_columns=3, random_select=False)

roi_model = build_ROI_model(SEQUENCE_LENGTH)

history = roi_model.fit(signals, ROI, validation_split=0.2, epochs=20, batch_size=32)

_ = plot_training_history(history, filtering=["*loss*"])

signals, _, amplitudes, positions, _, _, _ = generate_signal_dataset(
    n_samples=100,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)

predictions, uncertainty = filter_predictions(signals=signals, model=roi_model, n_samples=30, threshold=0.9)

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_vline(positions)
plotter.add_roi(predictions)

plotter.set_title("Demo: Signals + Peaks + ROI")
_ = plotter.plot(n_examples=6, n_columns=3, random_select=True)
