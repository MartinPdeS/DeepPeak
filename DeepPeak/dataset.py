import matplotlib.pyplot as plt
from MPSPlots import helper
import numpy as np


class DataSet:
    """
    A simple container class for datasets.

    This class dynamically sets attributes based on the provided keyword arguments,
    allowing for flexible storage of various dataset components.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to be set as attributes of the instance.
    """

    list_of_attributes = None

    def __init__(self, **kwargs):
        self.list_of_attributes = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.list_of_attributes.append(key)

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ", ".join(f"{key}" for key in self.list_of_attributes)
        return f"{class_name}({attributes})"

    @helper.post_mpl_plot
    def plot(
        self,
        number_of_samples: int | None = 3,
        number_of_columns: int = 1,
        randomize_signal: bool = False,
    ):
        """
        Plot the predicted Regions of Interest (ROIs) for several sample signals.

        Parameters
        ----------
        number_of_samples : int, default=3
            Number of signals to visualize.
        randomize_signal : bool, default=False
            If True, randomly select signals from the dataset instead of taking
            the first N samples.
        number_of_columns : int, default=1
            Number of columns in the subplot grid.
        """
        sample_count = self.signals.shape[0]

        if number_of_samples is None:
            number_of_samples = sample_count

        # Select which samples to display
        if randomize_signal:
            indices = np.random.choice(sample_count, size=number_of_samples, replace=False)
        else:
            indices = np.arange(min(number_of_samples, sample_count))

        number_of_rows = int(np.ceil(len(indices) / number_of_columns))

        figure, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(8 * number_of_columns, 3 * number_of_rows),
            squeeze=False,
        )

        for plot_index, ax in zip(indices, axes.flatten()):
            signal = self.signals[plot_index]

            # Plot signal
            ax.plot(self.x_values, signal, label="signal", color="black")

            # Highlight predicted region of interest

            handles, labels = ax.get_legend_handles_labels()

            if self.region_of_interest is not None:
                roi_patch = ax.fill_between(
                    self.x_values,
                    y1=0,
                    y2=1,
                    where=(self.region_of_interest[plot_index] != 0),
                    color="lightblue",
                    alpha=1.0,
                    transform=ax.get_xaxis_transform(),
                )

                handles.append(roi_patch)
                labels.append("Predicted ROI")

            # Build legend (consistent with your existing plotting logic)
            by_label = {}
            for h, l in zip(handles, labels):
                if l and not l.startswith("_") and l not in by_label:
                    by_label[l] = h

            ax.legend(by_label.values(), by_label.keys())
            ax.set_xlabel("Time step")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Predicted ROI (Sample {plot_index})")

        return figure
