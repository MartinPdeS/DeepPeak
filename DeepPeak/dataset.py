import numpy as np
import matplotlib.pyplot as plt
from MPSPlots.styles import mps as plot_style

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
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot(self, ax: plt.Axes = None, show: bool = True):
        sample_count = self.signals.shape[0]
        with plt.style.context(plot_style):
            if ax is None:
                figure, axes = plt.subplots(nrows=sample_count, ncols=1, figsize=(8, 3 * sample_count), squeeze=False)
            else:
                axes = np.array([[ax]])

            for idx, ax in enumerate(axes.flatten()):
                ax.plot(self.x_values, self.signals[idx], label='signal')
                ax.vlines(self.positions[idx], ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color='red', linestyle='--', linewidth=1, label='positions')

                if self.region_of_interest is not None:
                    roi_patch = ax.fill_between(self.x_values, y1=0, y2=1,
                        where=(self.region_of_interest[idx] != 0),
                        color='lightblue',
                        alpha=1.0,
                        transform=ax.get_xaxis_transform(),
                    )

                # build legend (add custom entry for ROI if present)
                handles, labels = ax.get_legend_handles_labels()
                if self.region_of_interest is not None:
                    handles.append(roi_patch)
                    labels.append("ROI")

                by_label = {}
                for h, l in zip(handles, labels):
                    if l and not l.startswith("_") and l not in by_label:
                        by_label[l] = h
                ax.legend(by_label.values(), by_label.keys())

        if show:
            plt.show()