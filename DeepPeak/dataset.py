import matplotlib.pyplot as plt
from MPSPlots import helper


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
    def plot(self, number_of_samples: int | None = None):
        sample_count = self.signals.shape[0]

        if number_of_samples is None:
            number_of_samples = sample_count

        figure, axes = plt.subplots(
            nrows=number_of_samples,
            ncols=1,
            figsize=(8, 3 * number_of_samples),
            squeeze=False,
        )

        for idx, ax in enumerate(axes.flatten()):
            ax.plot(self.x_values, self.signals[idx], label="signal")
            ax.vlines(
                self.positions[idx],
                ymin=0,
                ymax=1,
                transform=ax.get_xaxis_transform(),
                color="red",
                linestyle="--",
                linewidth=1,
                label="positions",
            )

            if self.region_of_interest is not None:
                roi_patch = ax.fill_between(
                    self.x_values,
                    y1=0,
                    y2=1,
                    where=(self.region_of_interest[idx] != 0),
                    color="lightblue",
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

        return figure
