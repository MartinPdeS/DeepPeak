
from DeepPeak.signals import generate_signal_dataset
from DeepPeak.utils.visualization import SignalPlotter
from DeepPeak.models import filter_predictions
import tensorflow as tf
from DeepPeak.directories import weights_path
from DeepPeak.utils.ROI import find_middle_indices

NUM_PEAKS = 3
SEQUENCE_LENGTH = 200

model_path = weights_path / 'ROI_Model.keras'
roi_model = tf.keras.models.load_model(model_path)


signals, amplitudes, positions, _, _, _ = generate_signal_dataset(
    n_samples=100,
    sequence_length=SEQUENCE_LENGTH,
    n_peaks=(1, NUM_PEAKS),
    amplitude=(1, 20),
    position=(0.1, 0.9),
    width=(0.03, 0.05),
    noise_std=0.1,
    categorical_peak_count=False,
)

predictions, uncertainty = filter_predictions(
    signals=signals,
    model=roi_model,
    n_samples=30,
    threshold=0.9
)

indices = find_middle_indices(
    ROIs=predictions,
    pad_width=5,
    fill_value=0
) / 200


# %%
# Compare Predicted ROI with Original Signals
# -------------------------------------------
#
# We overlay the predicted ROI mask on the signals, and also draw the
# individual Gaussians using a custom curve function.

plotter = SignalPlotter()
plotter.add_signals(signals)
plotter.add_scatter(positions, amplitudes)
plotter.add_vline(indices)
plotter.add_roi(predictions)

plotter.set_title("Demo: Signals + Peaks + ROI")
_ = plotter.plot(n_examples=6, n_columns=3, random_select=True)
