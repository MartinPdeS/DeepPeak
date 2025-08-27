import tensorflow as tf

from DeepPeak.classifier.utils import filter_predictions, find_middle_indices
from DeepPeak.directories import weights_path
from DeepPeak.signals import generate_signal_dataset
from DeepPeak.visualization import SignalPlotter

NUM_PEAKS = 3
SEQUENCE_LENGTH = 128

model_path = weights_path / "ROI_128.keras"
roi_model = tf.keras.models.load_model(model_path)


training = generate_signal_dataset(
    n_samples=100,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)

predictions, uncertainty = filter_predictions(signals=training.signals, model=roi_model, n_samples=30, threshold=0.9)

indices = find_middle_indices(ROIs=predictions, pad_width=5, fill_value=0) / 200


# %%
# Compare Predicted ROI with Original Signals
# -------------------------------------------
#
# We overlay the predicted ROI mask on the signals, and also draw the
# individual Gaussians using a custom curve function.

plotter = SignalPlotter()
plotter.add_signals(training.signals)
plotter.add_scatter(training.positions, training.amplitudes)
plotter.add_roi(predictions)

plotter.set_title("Demo: Signals + Peaks + ROI")
_ = plotter.plot(n_examples=15, n_columns=5, random_select=True)
