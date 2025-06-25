from tensorflow.keras.utils import to_categorical  # type: ignore
from itertools import islice
import sklearn.model_selection as sk
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib import style as mps

def batched(iterable, n: int):  # Function is present in itertools for python 3.12+
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def dataset_split(test_size: float, random_state: float, **kwargs) -> dict:
    values = list(kwargs.values())

    splitted = sk.train_test_split(*values, test_size=test_size, random_state=random_state)

    output = {
        'train': dict(), 'test': dict()
    }

    for (k, v), (train_data, test_data) in zip(kwargs.items(), batched(splitted, 2)):
        output['train'][k] = train_data
        output['test'][k] = test_data

    return output


def filter_with_wavelet_transform(signals: np.ndarray, low_boundary: int, high_boundary: int, kernel: str = 'mexh') -> tuple[np.ndarray, np.ndarray]:
    """
    Efficient filtering of multiple signals using CWT, with minimal looping.

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_signals, n_samples).
    low_boundary : int
        Lower bound of the scale range.
    high_boundary : int
        Upper bound of the scale range.
    kernel : str, optional
        Name of the wavelet kernel. Default is 'mexh'.

    Returns
    -------
    filtered_signals : np.ndarray
        Filtered signals of shape (n_signals, n_samples).
    all_coeffs : np.ndarray
        All CWT coefficients of shape (n_signals, n_scales, n_samples).
    """
    signals = np.atleast_2d(signals)
    n_signals, n_samples = signals.shape

    # Define scales and mask
    all_scales = np.arange(1, 100)
    scale_mask = (all_scales >= low_boundary) & (all_scales <= high_boundary)
    n_selected_scales = np.sum(scale_mask)

    # Perform CWT individually (CWT must be done per-signal due to PyWavelets)
    coeffs_list = [pywt.cwt(signals[i], all_scales, kernel)[0] for i in range(n_signals)]
    coeffs = np.stack(coeffs_list, axis=0)  # Shape: (n_signals, n_scales, n_samples)

    # Apply mask and reconstruct using broadcasting
    filtered_coeffs = coeffs * scale_mask[:, np.newaxis]  # scale_mask: (n_scales, 1)
    filtered_signals = np.sum(filtered_coeffs, axis=1) / np.sqrt(n_selected_scales) / abs(high_boundary - low_boundary)

    return filtered_signals, coeffs


class PulseDeconvolver:
    """
    Class for recovering true amplitudes of overlapping Gaussian pulses
    from measured signals and known peak positions.

    Parameters
    ----------
    width : float
        Known Gaussian width (in same domain as x-axis, e.g., normalized to [0, 1]).
    sequence_length : int
        Number of samples in each signal (used to generate x-axis).
    """

    def __init__(self, width: float, sequence_length: int):
        self.width = width
        self.sequence_length = sequence_length
        self.x_values = np.linspace(0, 1, sequence_length)

    def _build_design_matrix(self, centers):
        """
        Build Gaussian design matrix for a given set of centers.
        """
        return np.exp(-0.5 * ((self.x_values[:, None] - centers[None, :]) / self.width) ** 2)

    def deconvolve(self, signals: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Recover true amplitudes from multiple signals.

        Parameters
        ----------
        signals : np.ndarray of shape (N, L)
            Batch of signals, each of length L.
        centers : np.ndarray of shape (N, P)
            For each signal, P known peak positions in normalized [0, 1] coordinates.

        Returns
        -------
        amplitudes : np.ndarray of shape (N, P)
            Recovered amplitudes for each signal.
        """
        if signals.shape[0] != centers.shape[0]:
            raise ValueError("signals and centers must have same batch dimension")

        batch_size, signal_length = signals.shape
        num_peaks = centers.shape[1]

        amplitudes = np.zeros((batch_size, num_peaks))

        for i in range(batch_size):
            G = self._build_design_matrix(centers[i])
            a, *_ = np.linalg.lstsq(G, signals[i], rcond=None)
            amplitudes[i] = a

        return amplitudes

    def plot(self, data_set: object) -> None:

        estimated_amplitudes = self.deconvolve(
            signals=data_set.signals,
            centers=data_set.positions
        )

        n_plot = data_set.signals.shape[0]

        with plt.style.context(mps):
            _, axes = plt.subplots(ncols=1, nrows=n_plot, figsize=(8, 4 * n_plot), squeeze=False)

        for index in range(n_plot):
            ax = axes[index, 0]

            ax.plot(data_set.x_values, data_set.signals[index], color='C0', linewidth=2, label='Raw signal')

            for a, p, w in zip(data_set.amplitudes[index], data_set.positions[index], data_set.widths[index]):
                y = a * np.exp(-(data_set.x_values - p)**2 / (2 * w**2))
                ax.plot(data_set.x_values, y, linestyle='--', linewidth=1, color='black', label='Individual pulses')

                ax.axvline(p, color='green', label='measurement position')

            ax.scatter(x=data_set.positions[index], y=estimated_amplitudes[index], color='red', s=60, zorder=10, label='Evaluated amplitudes')

            ax.legend()

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Removes duplicates
            ax.legend(by_label.values(), by_label.keys())



        ax.set_ylabel('Amplitude of the signal [Normalized]')
        ax.set_xlabel('Time [Normalized]')

        plt.tight_layout()
        plt.show()


