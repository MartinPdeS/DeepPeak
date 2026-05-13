from typing import Any

import numpy as np


class PulseDeconvolver:
    """
    Recover Gaussian pulse amplitudes from overlapping signals with known centers.
    """

    def __init__(self, width: float, sequence_length: int):
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("width must be a finite positive number.")
        if int(sequence_length) <= 0:
            raise ValueError("sequence_length must be a strictly positive integer.")

        self.width = float(width)
        self.sequence_length = int(sequence_length)
        self.x_values = np.linspace(0.0, 1.0, self.sequence_length)

    def _build_design_matrix(self, centers: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers, dtype=float)
        return np.exp(
            -0.5 * ((self.x_values[:, None] - centers[None, :]) / self.width) ** 2
        )

    def deconvolve(self, signals: np.ndarray, centers: np.ndarray) -> np.ndarray:
        signals = np.asarray(signals, dtype=float)
        centers = np.asarray(centers, dtype=float)

        if signals.ndim != 2:
            raise ValueError(
                "signals must be a 2D array of shape (batch, sequence_length)."
            )
        if centers.ndim != 2:
            raise ValueError(
                "centers must be a 2D array of shape (batch, number_of_peaks)."
            )
        if signals.shape[0] != centers.shape[0]:
            raise ValueError("signals and centers must have the same batch dimension.")
        if signals.shape[1] != self.sequence_length:
            raise ValueError(
                f"signals must have sequence length {self.sequence_length}, got {signals.shape[1]}."
            )

        batch_size, _ = signals.shape
        number_of_peaks = centers.shape[1]
        amplitudes = np.zeros((batch_size, number_of_peaks), dtype=float)

        for index in range(batch_size):
            design_matrix = self._build_design_matrix(centers[index])
            amplitudes[index], *_ = np.linalg.lstsq(
                design_matrix, signals[index], rcond=None
            )

        return amplitudes

    def plot(self, data_set: Any) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import style as mps

        estimated_amplitudes = self.deconvolve(
            signals=data_set.signals,
            centers=data_set.positions,
        )

        number_of_plots = data_set.signals.shape[0]

        with plt.style.context(mps):
            _, axes = plt.subplots(
                ncols=1,
                nrows=number_of_plots,
                figsize=(8, 4 * number_of_plots),
                squeeze=False,
            )

        for index in range(number_of_plots):
            ax = axes[index, 0]

            ax.plot(
                data_set.x_values,
                data_set.signals[index],
                color="C0",
                linewidth=2,
                label="Raw signal",
            )

            for amplitude, position, width in zip(
                data_set.amplitudes[index],
                data_set.positions[index],
                data_set.widths[index],
            ):
                y_values = amplitude * np.exp(
                    -((data_set.x_values - position) ** 2) / (2 * width**2)
                )
                ax.plot(
                    data_set.x_values,
                    y_values,
                    linestyle="--",
                    linewidth=1,
                    color="black",
                    label="Individual pulses",
                )
                ax.axvline(position, color="green", label="Measurement position")

            ax.scatter(
                x=data_set.positions[index],
                y=estimated_amplitudes[index],
                color="red",
                s=60,
                zorder=10,
                label="Evaluated amplitudes",
            )

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.set_ylabel("Amplitude of the signal [Normalized]")

        axes[-1, 0].set_xlabel("Time [Normalized]")
        plt.tight_layout()
        plt.show()
