# -*- coding: utf-8 -*-
from functools import wraps

import matplotlib.pyplot as plt
from MPSPlots.styles import mps as plot_style


def mpl_plot(function):
    @wraps(function)
    def wrapper(self, show: bool = True, save_as: str = None, **kwargs):
        """
        Decorator to set the plot style and handle figure saving and showing.
        This decorator applies a specific plot style, saves the figure if a filename is provided,
        and shows the plot if requested.

        Parameters
        ----------
        function : callable
            The plotting function to be wrapped.
        show : bool, optional
            Whether to display the plot (default: True).
        save_as : str, optional
            If provided, the figure is saved to this filename.
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        plt.Figure
            The figure containing the plot.
        """
        # Create plot
        with plt.style.context(plot_style):
            figure, ax = plt.subplots(1, 1, figsize=(12, 5))

        with plt.style.context(plot_style):
            function(self, figure=figure, ax=ax, **kwargs)

            figure.tight_layout()

            if save_as is not None:
                plt.savefig(save_as)
                print(f"Figure saved as {save_as}")

            if show:
                plt.show()

        return figure

    return wrapper
