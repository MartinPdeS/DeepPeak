"""Trace-level analysis result models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from ...utils.helper import get_throughput_label
from .detection import PeakDetectionResult


@dataclass
class TraceRecord:
    """Canonical analysis bundle for one processed trace."""

    filename: Path
    dilution: float
    concentration: float
    dx: float
    signal: np.ndarray
    standard: PeakDetectionResult
    prediction: np.ndarray
    cnn: PeakDetectionResult

    @property
    def processed_signal(self) -> np.ndarray:
        """Return the processed signal stored in the record."""

        return self.signal

    @property
    def wavenet_prediction(self) -> np.ndarray:
        """Return the WaveNet prediction stored in the record."""

        return self.prediction

    @property
    def delta_x(self) -> float:
        """Return the total duration covered by the processed signal."""

        return float(self.dx * self.signal.size)

    @property
    def standard_particle_flow(self) -> float:
        """Return the standard detector particle flow."""

        return float(self.standard.peaks.size / self.delta_x)

    @property
    def cnn_particle_flow(self) -> float:
        """Return the WaveNet detector particle flow."""

        return float(self.cnn.peak_count / self.delta_x)

    def plot_standard_detection(
        self,
        x_axis: str = "sample",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        figsize: tuple[float, float] = (10.0, 4.0),
        line_width: float = 0.8,
        marker_size: float = 18.0,
        threshold_line_width: float = 0.8,
        show_threshold: bool = True,
        show_peaks: bool = True,
        show_legend: bool = True,
        show_grid: bool = False,
        show_title: bool = True,
        title: Optional[str] = None,
        signal_color: str = "C0",
        marker_color: str = "black",
        threshold_color: str = "black",
        rasterize_dense_artists: bool = True,
        ax: Optional[Any] = None,
        show: bool = False,
        close: bool = False,
        save_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot the processed trace with its standard-detector annotations.

        Parameters
        ----------
        x_axis : {"sample", "time"}, default="sample"
            Coordinate system used for the x-axis.
        xlim, ylim : tuple of float, optional
            Optional axis bounds.
        figsize : tuple of float, default=(10.0, 4.0)
            Figure size used when a new axis is created.
        line_width, marker_size, threshold_line_width : float
            Styling parameters for lines and markers.
        show_threshold, show_peaks, show_legend, show_grid, show_title : bool
            Toggles controlling displayed plot elements.
        title : str, optional
            Explicit title override.
        signal_color, marker_color, threshold_color : str
            Matplotlib colors used for the plotted artists.
        rasterize_dense_artists : bool, default=True
            Rasterize dense line artists when exporting vector graphics.
        ax : matplotlib.axes.Axes, optional
            Existing axis on which to draw the trace.
        show, close : bool, default=False
            Whether to display or close the figure.
        save_path : str or pathlib.Path, optional
            Optional output file path.
        dpi : int, default=300
            Raster resolution used when saving.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the standard detector trace view.
        """

        from ..trace_plots import get_detection_threshold

        signal = np.asarray(self.signal).ravel()
        if signal.size == 0:
            raise ValueError("Cannot plot an empty signal.")

        if x_axis == "sample":
            x_values = np.arange(signal.size, dtype=float)
            x_label = "Sample index"
        elif x_axis == "time":
            x_values = np.arange(signal.size, dtype=float) * float(self.dx)
            x_label = "Time"
        else:
            raise ValueError('x_axis must be either "sample" or "time".')

        standard_peak_indices = np.asarray(self.standard.peaks, dtype=int)
        standard_peak_indices = standard_peak_indices[
            (standard_peak_indices >= 0) & (standard_peak_indices < signal.size)
        ]
        standard_threshold = get_detection_threshold(self.standard)

        if ax is not None:
            figure, axis = ax.figure, ax
            created_figure = False
        else:
            figure, axis = plt.subplots(figsize=figsize)
            created_figure = True

        axis.plot(
            x_values,
            signal,
            color=signal_color,
            linewidth=line_width,
            label="Signal",
            zorder=1,
            rasterized=rasterize_dense_artists,
        )

        if show_threshold and standard_threshold is not None:
            axis.axhline(
                standard_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=threshold_line_width,
                label="Threshold",
                zorder=2,
            )

        if show_peaks and standard_peak_indices.size > 0:
            if x_axis == "sample":
                standard_peak_x_values = standard_peak_indices.astype(float)
            else:
                standard_peak_x_values = standard_peak_indices.astype(float) * float(
                    self.dx
                )
            axis.scatter(
                standard_peak_x_values,
                signal[standard_peak_indices],
                color=marker_color,
                edgecolors=marker_color,
                s=marker_size,
                marker="o",
                label=f"Peaks ({standard_peak_indices.size})",
                zorder=10,
                rasterized=rasterize_dense_artists,
            )

        figure.subplots_adjust(
            left=0.10, right=0.97, bottom=0.14, top=0.88 if show_title else 0.95
        )
        if xlim is not None:
            axis.set_xlim(xlim)
        if ylim is not None:
            axis.set_ylim(ylim)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Signal amplitude")
        if show_title and title is not None:
            axis.set_title(title)
        if show_grid:
            axis.grid(True, which="both", alpha=0.2, zorder=0)
        if show_legend:
            legend = axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            legend.set_zorder(1000)
        if created_figure:
            figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(figure)
        return figure

    def plot_wavenet_detection(
        self,
        x_axis: str = "sample",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        figsize: tuple[float, float] = (10.0, 4.0),
        line_width: float = 0.8,
        marker_size: float = 18.0,
        threshold_line_width: float = 0.8,
        show_signal: bool = True,
        show_prediction: bool = True,
        show_cnn_signal_peaks: bool = False,
        show_cnn_recovered_signal_peaks: bool = False,
        show_cnn_reconstructed_trace: bool = False,
        show_cnn_prediction_peaks: bool = True,
        show_cnn_threshold: bool = True,
        show_legend: bool = True,
        show_grid: bool = False,
        show_title: bool = True,
        title: Optional[str] = None,
        signal_color: str = "C0",
        prediction_color: str = "C1",
        reconstructed_trace_color: str = "C3",
        marker_color: str = "black",
        threshold_color: str = "black",
        rasterize_dense_artists: bool = True,
        ax: Optional[Any] = None,
        show: bool = False,
        close: bool = False,
        save_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot the trace together with its WaveNet prediction and CNN peaks.

        Parameters
        ----------
        x_axis : {"sample", "time"}, default="sample"
            Coordinate system used for the x-axis.
        xlim, ylim : tuple of float, optional
            Optional axis bounds.
        figsize : tuple of float, default=(10.0, 4.0)
            Figure size used when a new axis is created.
        line_width, marker_size, threshold_line_width : float
            Styling parameters for lines and markers.
        show_signal, show_prediction, show_cnn_signal_peaks,
        show_cnn_recovered_signal_peaks, show_cnn_reconstructed_trace,
        show_cnn_prediction_peaks, show_cnn_threshold : bool
            Toggles controlling displayed plot elements. By default, CNN detections
            are shown on the prediction rather than projected onto the signal.
        show_legend, show_grid, show_title : bool
            Axes presentation toggles.
        title : str, optional
            Explicit title override.
        signal_color, prediction_color, reconstructed_trace_color,
        marker_color, threshold_color : str
            Matplotlib colors used for the plotted artists.
        rasterize_dense_artists : bool, default=True
            Rasterize dense line artists when exporting vector graphics.
        ax : matplotlib.axes.Axes, optional
            Existing axis on which to draw the trace.
        show, close : bool, default=False
            Whether to display or close the figure.
        save_path : str or pathlib.Path, optional
            Optional output file path.
        dpi : int, default=300
            Raster resolution used when saving.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the WaveNet detector trace view.
        """

        from ..trace_plots import get_detection_threshold, reconstruct_gaussian_trace

        signal = np.asarray(self.signal).ravel()
        prediction = np.asarray(self.prediction).ravel()
        if signal.size == 0:
            raise ValueError("Cannot plot an empty signal.")
        if prediction.size == 0:
            raise ValueError("Cannot plot an empty prediction.")

        if x_axis == "sample":
            x_values = np.arange(signal.size, dtype=float)
            prediction_x_values = np.arange(prediction.size, dtype=float)
            x_label = "Sample index"
        elif x_axis == "time":
            x_values = np.arange(signal.size, dtype=float) * float(self.dx)
            prediction_x_values = np.arange(prediction.size, dtype=float) * float(
                self.dx
            )
            x_label = "Time"
        else:
            raise ValueError('x_axis must be either "sample" or "time".')

        cnn_peak_indices_for_signal = np.asarray(self.cnn.peaks, dtype=int)
        cnn_peak_indices_for_signal = cnn_peak_indices_for_signal[
            (cnn_peak_indices_for_signal >= 0)
            & (cnn_peak_indices_for_signal < signal.size)
        ]
        cnn_peak_indices_for_prediction = np.asarray(self.cnn.peaks, dtype=int)
        cnn_peak_indices_for_prediction = cnn_peak_indices_for_prediction[
            (cnn_peak_indices_for_prediction >= 0)
            & (cnn_peak_indices_for_prediction < prediction.size)
        ]
        cnn_signal_peak_values = signal[cnn_peak_indices_for_signal]
        cnn_signal_peak_label = (
            f"CNN positions on signal ({cnn_peak_indices_for_signal.size})"
        )
        cnn_recovered_amplitudes = getattr(self.cnn, "amplitudes", None)
        reconstructed_trace = None
        if (
            (show_cnn_recovered_signal_peaks or show_cnn_reconstructed_trace)
            and cnn_recovered_amplitudes is not None
            and cnn_peak_indices_for_signal.size > 0
        ):
            cnn_recovered_amplitudes = np.asarray(
                cnn_recovered_amplitudes, dtype=float
            ).ravel()
            if (
                cnn_recovered_amplitudes.size
                == np.asarray(self.cnn.peaks, dtype=int).size
            ):
                valid_signal_mask = (np.asarray(self.cnn.peaks, dtype=int) >= 0) & (
                    np.asarray(self.cnn.peaks, dtype=int) < signal.size
                )
                recovered_signal_peak_values = cnn_recovered_amplitudes[
                    valid_signal_mask
                ]
                finite_recovered_mask = np.isfinite(recovered_signal_peak_values)
                cnn_peak_indices_for_signal = cnn_peak_indices_for_signal[
                    finite_recovered_mask
                ]
                recovered_baseline = float(
                    getattr(self.cnn, "properties", {}).get("recovered_baseline", 0.0)
                )
                cnn_signal_peak_values = (
                    recovered_signal_peak_values[finite_recovered_mask]
                    + recovered_baseline
                )
                cnn_signal_peak_label = f"CNN recovered amplitudes on signal ({cnn_peak_indices_for_signal.size})"
                if show_cnn_reconstructed_trace:
                    sigma_samples = getattr(self.cnn, "properties", {}).get(
                        "recovered_sigma_samples"
                    )
                    if sigma_samples is not None and np.isfinite(float(sigma_samples)):
                        reconstructed_trace = reconstruct_gaussian_trace(
                            number_of_samples=signal.size,
                            peak_indices=cnn_peak_indices_for_signal,
                            amplitudes=cnn_signal_peak_values - recovered_baseline,
                            sigma_samples=float(sigma_samples),
                            baseline=recovered_baseline,
                        )
        cnn_threshold = get_detection_threshold(self.cnn)

        if ax is not None:
            figure, axis = ax.figure, ax
            created_figure = False
        else:
            figure, axis = plt.subplots(figsize=figsize)
            created_figure = True
        if show_signal:
            axis.plot(
                x_values,
                signal,
                color=signal_color,
                linewidth=line_width,
                label="Signal",
                zorder=1,
                rasterized=rasterize_dense_artists,
            )
        if show_prediction:
            axis.plot(
                prediction_x_values,
                prediction,
                color=prediction_color,
                linewidth=line_width,
                label="Prediction",
                zorder=2,
                rasterized=rasterize_dense_artists,
            )
        if reconstructed_trace is not None:
            axis.plot(
                x_values,
                reconstructed_trace,
                color=reconstructed_trace_color,
                linewidth=line_width,
                linestyle="-.",
                label="CNN reconstructed trace",
                zorder=3,
                rasterized=rasterize_dense_artists,
            )
        if show_cnn_threshold and cnn_threshold is not None:
            axis.axhline(
                cnn_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=threshold_line_width,
                label="CNN threshold",
                zorder=4,
            )
        if (
            show_cnn_signal_peaks or show_cnn_recovered_signal_peaks
        ) and cnn_peak_indices_for_signal.size > 0:
            if x_axis == "sample":
                cnn_signal_peak_x_values = cnn_peak_indices_for_signal.astype(float)
            else:
                cnn_signal_peak_x_values = cnn_peak_indices_for_signal.astype(
                    float
                ) * float(self.dx)
            axis.scatter(
                cnn_signal_peak_x_values,
                cnn_signal_peak_values,
                color=marker_color,
                s=marker_size,
                marker="x",
                label=cnn_signal_peak_label,
                zorder=20,
                rasterized=rasterize_dense_artists,
            )
        if show_cnn_prediction_peaks and cnn_peak_indices_for_prediction.size > 0:
            if x_axis == "sample":
                cnn_prediction_peak_x_values = cnn_peak_indices_for_prediction.astype(
                    float
                )
            else:
                cnn_prediction_peak_x_values = cnn_peak_indices_for_prediction.astype(
                    float
                ) * float(self.dx)
            axis.scatter(
                cnn_prediction_peak_x_values,
                prediction[cnn_peak_indices_for_prediction],
                color=marker_color,
                edgecolors=marker_color,
                s=marker_size,
                marker="o",
                label=f"CNN peaks on prediction ({cnn_peak_indices_for_prediction.size})",
                zorder=21,
                rasterized=rasterize_dense_artists,
            )

        figure.subplots_adjust(
            left=0.10, right=0.97, bottom=0.14, top=0.88 if show_title else 0.95
        )
        if xlim is not None:
            axis.set_xlim(xlim)
        if ylim is not None:
            axis.set_ylim(ylim)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Signal / prediction amplitude")
        if show_title and title is not None:
            axis.set_title(title)
        if show_grid:
            axis.grid(True, which="both", alpha=0.2, zorder=0)
        if show_legend:
            legend = axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            legend.set_zorder(1000)
        if created_figure:
            figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(figure)
        return figure

    def plot_standard_detection_with_histogram(self, **plot_kwargs) -> Any:
        """Plot the standard detector trace with a right-side amplitude histogram.

        This view is designed for notebook inspection of one analyzed trace.
        The left axis uses :meth:`plot_standard_detection` to render the
        processed signal, detected standard peaks, and optional threshold line.
        The right axis shows a horizontal histogram of the amplitudes measured
        directly on ``self.signal`` at ``self.standard.peaks``.

        Parameters
        ----------
        x_axis : {"sample", "time"}, default="sample"
            Coordinate system used for the trace axis.
        xlim, ylim : tuple of float, optional
            Optional axis limits for the trace panel. When ``ylim`` is provided,
            the same vertical span is also applied to the histogram so both
            panels remain aligned.
        figsize : tuple of float, default=(12.0, 4.5)
            Figure size used when new axes are created.
        line_width, marker_size, threshold_line_width : float
            Styling parameters forwarded to :meth:`plot_standard_detection`.
        histogram_bins : int, default=20
            Default number of histogram bins.
        bins : int or sequence, optional
            Matplotlib-style alias for histogram bin selection. When provided,
            it overrides ``histogram_bins``.
        histogram_color, histogram_edgecolor, histogram_line_width
            Histogram bar styling.
        histogram_title : str, default="Amplitude histogram"
            Title applied to the histogram axis.
        histogram_xlim : tuple of float, optional
            Optional x-limits applied to the histogram axis.
        histogram_reference_lines : sequence of float, optional
            Horizontal reference lines drawn across the histogram axis.
        histogram_reference_line_kwargs : dict, optional
            Styling forwarded to each histogram reference line.
        show_threshold, show_peaks, show_legend, show_grid, show_title : bool
            Display toggles for the trace and histogram panels.
        title : str, optional
            Title applied to the trace axis. If omitted and ``show_title`` is
            true, ``"Standard detection"`` is used.
        expected_particle_flow : float, optional
            Expected throughput value shown in the figure suptitle when
            ``show_throughput`` is true.
        show_throughput : bool, default=False
            Whether to add a throughput suptitle.
        throughput_template : str, default="Throughput: {label} / second"
            Format string used for the throughput suptitle. Receives
            ``label`` and ``flow``.
        throughput_label_formatter : callable, optional
            Callable converting ``expected_particle_flow`` into a compact label.
        suptitle_kwargs : dict, optional
            Keyword arguments forwarded to ``figure.suptitle``.
        show_inset : bool, default=False
            Whether to add a zoomed inset trace axis.
        inset_width, inset_height, inset_loc, inset_borderpad
            Inset placement parameters forwarded to ``inset_axes``.
        inset_xlim, inset_ylim : tuple of float, optional
            Axis limits applied to the inset.
        inset_show_ticks : bool, default=False
            Whether to show ticks and axis labels on the inset.
        inset_show_grid : bool, default=False
            Whether to enable the grid on the inset.
        inset_plot_kwargs : dict, optional
            Additional keyword arguments forwarded to the inset trace plot.
        mark_inset : bool, default=True
            Whether to draw connectors between the main trace and the inset.
        mark_inset_kwargs : dict, optional
            Keyword arguments forwarded to ``mark_inset``.
        subplots_adjust : dict, optional
            Layout arguments forwarded to ``figure.subplots_adjust``.
        signal_color, marker_color, threshold_color : str
            Colors forwarded to :meth:`plot_standard_detection`.
        rasterize_dense_artists : bool, default=True
            Rasterize dense line artists when exporting vector graphics.
        axes : tuple(matplotlib.axes.Axes, matplotlib.axes.Axes), optional
            Existing ``(trace_axis, histogram_axis)`` pair. When omitted, a new
            figure with two columns is created.
        show, close : bool, default=False
            Whether to display or close the resulting figure.
        save_path : str or pathlib.Path, optional
            Optional path used to save the figure.
        dpi : int, default=300
            Raster resolution used when saving.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the standard trace panel and amplitude histogram.
        """

        x_axis = plot_kwargs.pop("x_axis", "sample")
        xlim = plot_kwargs.pop("xlim", None)
        ylim = plot_kwargs.pop("ylim", None)
        figsize = plot_kwargs.pop("figsize", (12.0, 4.5))
        line_width = plot_kwargs.pop("line_width", 0.8)
        marker_size = plot_kwargs.pop("marker_size", 18.0)
        threshold_line_width = plot_kwargs.pop("threshold_line_width", 0.8)
        histogram_bins = plot_kwargs.pop("histogram_bins", 20)
        bins = plot_kwargs.pop("bins", None)
        histogram_color = plot_kwargs.pop("histogram_color", "0.75")
        histogram_edgecolor = plot_kwargs.pop("histogram_edgecolor", "black")
        histogram_line_width = plot_kwargs.pop("histogram_line_width", 0.8)
        histogram_title = plot_kwargs.pop("histogram_title", "Amplitude histogram")
        histogram_xlim = plot_kwargs.pop("histogram_xlim", None)
        histogram_reference_lines = plot_kwargs.pop("histogram_reference_lines", None)
        histogram_reference_line_kwargs = dict(
            plot_kwargs.pop(
                "histogram_reference_line_kwargs",
                {"color": "black", "linestyle": "--", "linewidth": 2.0},
            )
        )
        show_threshold = plot_kwargs.pop("show_threshold", True)
        show_peaks = plot_kwargs.pop("show_peaks", True)
        show_legend = plot_kwargs.pop("show_legend", True)
        show_grid = plot_kwargs.pop("show_grid", False)
        show_title = plot_kwargs.pop("show_title", True)
        title = plot_kwargs.pop("title", None)
        expected_particle_flow = plot_kwargs.pop("expected_particle_flow", None)
        show_throughput = plot_kwargs.pop("show_throughput", False)
        throughput_template = plot_kwargs.pop(
            "throughput_template", "Throughput: {label} / second"
        )
        throughput_label_formatter = plot_kwargs.pop(
            "throughput_label_formatter", get_throughput_label
        )
        suptitle_kwargs = dict(plot_kwargs.pop("suptitle_kwargs", {}))
        show_inset = plot_kwargs.pop("show_inset", False)
        inset_width = plot_kwargs.pop("inset_width", "52%")
        inset_height = plot_kwargs.pop("inset_height", "50%")
        inset_loc = plot_kwargs.pop("inset_loc", "upper right")
        inset_borderpad = plot_kwargs.pop("inset_borderpad", 2.2)
        inset_xlim = plot_kwargs.pop("inset_xlim", None)
        inset_ylim = plot_kwargs.pop("inset_ylim", None)
        inset_show_ticks = plot_kwargs.pop("inset_show_ticks", False)
        inset_show_grid = plot_kwargs.pop("inset_show_grid", False)
        inset_plot_kwargs = dict(plot_kwargs.pop("inset_plot_kwargs", {}))
        mark_inset_enabled = plot_kwargs.pop("mark_inset", True)
        mark_inset_kwargs = dict(
            plot_kwargs.pop(
                "mark_inset_kwargs",
                {"loc1": 2, "loc2": 4, "fc": "none", "ec": "0.05", "lw": 2.0},
            )
        )
        subplots_adjust = plot_kwargs.pop(
            "subplots_adjust",
            {
                "left": 0.10,
                "right": 0.97,
                "bottom": 0.14,
                "top": 0.88 if show_title else 0.95,
                "wspace": 0.05,
            },
        )
        signal_color = plot_kwargs.pop("signal_color", "C0")
        marker_color = plot_kwargs.pop("marker_color", "black")
        threshold_color = plot_kwargs.pop("threshold_color", "black")
        rasterize_dense_artists = plot_kwargs.pop("rasterize_dense_artists", True)
        axes = plot_kwargs.pop("axes", None)
        show = plot_kwargs.pop("show", False)
        close = plot_kwargs.pop("close", False)
        save_path = plot_kwargs.pop("save_path", None)
        dpi = plot_kwargs.pop("dpi", 300)
        if plot_kwargs:
            unexpected_arguments = ", ".join(sorted(plot_kwargs))
            raise TypeError(
                f"plot_standard_detection_with_histogram() got unexpected keyword argument(s): {unexpected_arguments}"
            )

        signal = np.asarray(self.signal, dtype=float).ravel()
        if signal.size == 0:
            raise ValueError("Cannot plot an empty signal.")

        standard_peak_indices = np.asarray(self.standard.peaks, dtype=int)
        standard_peak_indices = standard_peak_indices[
            (standard_peak_indices >= 0) & (standard_peak_indices < signal.size)
        ]
        standard_amplitudes = signal[standard_peak_indices]
        standard_amplitudes = standard_amplitudes[np.isfinite(standard_amplitudes)]
        histogram_bin_spec = histogram_bins if bins is None else bins

        if axes is not None:
            trace_axis, histogram_axis = axes
            figure = trace_axis.figure
            created_figure = False
        else:
            figure, (trace_axis, histogram_axis) = plt.subplots(
                ncols=2,
                figsize=figsize,
                sharey=True,
                gridspec_kw={"width_ratios": [4.0, 1.2], "wspace": 0.05},
            )
            created_figure = True

        self.plot_standard_detection(
            x_axis=x_axis,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            line_width=line_width,
            marker_size=marker_size,
            threshold_line_width=threshold_line_width,
            show_threshold=show_threshold,
            show_peaks=show_peaks,
            show_legend=show_legend,
            show_grid=show_grid,
            show_title=False,
            title=None,
            signal_color=signal_color,
            marker_color=marker_color,
            threshold_color=threshold_color,
            rasterize_dense_artists=rasterize_dense_artists,
            ax=trace_axis,
            show=False,
            close=False,
            save_path=None,
            dpi=dpi,
        )

        if standard_amplitudes.size > 0:
            histogram_axis.hist(
                standard_amplitudes,
                bins=histogram_bin_spec,
                orientation="horizontal",
                color=histogram_color,
                edgecolor=histogram_edgecolor,
                linewidth=histogram_line_width,
            )
        else:
            histogram_axis.text(
                0.5,
                0.5,
                "No peaks",
                ha="center",
                va="center",
                transform=histogram_axis.transAxes,
            )

        if ylim is not None:
            histogram_axis.set_ylim(ylim)
        if histogram_xlim is not None:
            histogram_axis.set_xlim(histogram_xlim)
        if histogram_reference_lines is not None:
            for value in histogram_reference_lines:
                histogram_axis.axhline(value, **histogram_reference_line_kwargs)
        histogram_axis.set_xlabel("Peak count")
        histogram_axis.set_title(histogram_title)
        if show_grid:
            histogram_axis.grid(True, which="both", axis="x", alpha=0.2, zorder=0)
        else:
            histogram_axis.grid(False, axis="x")
        histogram_axis.tick_params(axis="y", labelleft=False)

        if show_title:
            trace_axis.set_title(title or "Standard detection")
        else:
            trace_axis.set_title("")

        if show_inset:
            inset_axis = inset_axes(
                trace_axis,
                width=inset_width,
                height=inset_height,
                loc=inset_loc,
                borderpad=inset_borderpad,
            )
            inset_kwargs = {
                "x_axis": inset_plot_kwargs.pop("x_axis", x_axis),
                "xlim": inset_plot_kwargs.pop("xlim", inset_xlim),
                "ylim": inset_plot_kwargs.pop(
                    "ylim", inset_ylim if inset_ylim is not None else ylim
                ),
                "figsize": inset_plot_kwargs.pop("figsize", figsize),
                "line_width": inset_plot_kwargs.pop("line_width", line_width),
                "marker_size": inset_plot_kwargs.pop("marker_size", marker_size),
                "threshold_line_width": inset_plot_kwargs.pop(
                    "threshold_line_width", threshold_line_width
                ),
                "show_threshold": inset_plot_kwargs.pop(
                    "show_threshold", show_threshold
                ),
                "show_peaks": inset_plot_kwargs.pop("show_peaks", show_peaks),
                "show_legend": inset_plot_kwargs.pop("show_legend", show_legend),
                "show_grid": inset_plot_kwargs.pop("show_grid", inset_show_grid),
                "show_title": inset_plot_kwargs.pop("show_title", False),
                "title": inset_plot_kwargs.pop("title", None),
                "signal_color": inset_plot_kwargs.pop("signal_color", signal_color),
                "marker_color": inset_plot_kwargs.pop("marker_color", marker_color),
                "threshold_color": inset_plot_kwargs.pop(
                    "threshold_color", threshold_color
                ),
                "rasterize_dense_artists": inset_plot_kwargs.pop(
                    "rasterize_dense_artists", rasterize_dense_artists
                ),
            }
            inset_kwargs.update(inset_plot_kwargs)
            self.plot_standard_detection(
                **inset_kwargs,
                ax=inset_axis,
                show=False,
                close=False,
                save_path=None,
                dpi=dpi,
            )
            if not inset_show_ticks:
                inset_axis.set_xticks([])
                inset_axis.set_yticks([])
                inset_axis.set_xlabel("")
                inset_axis.set_ylabel("")
            if mark_inset_enabled:
                mark_inset(trace_axis, inset_axis, **mark_inset_kwargs)

        if show_throughput and expected_particle_flow is not None:
            throughput_label = throughput_label_formatter(expected_particle_flow)
            default_suptitle_kwargs = {"x": 0.35, "y": 0.98}
            default_suptitle_kwargs.update(suptitle_kwargs)
            figure.suptitle(
                throughput_template.format(
                    flow=expected_particle_flow,
                    label=throughput_label,
                ),
                **default_suptitle_kwargs,
            )

        figure.subplots_adjust(**subplots_adjust)
        if save_path is not None:
            figure.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(figure)
        return figure

    def plot_wavenet_detection_with_histogram(self, **plot_kwargs) -> Any:
        """Plot the CNN detector trace with a right-side amplitude histogram.

        This view is designed for notebook inspection of one analyzed trace.
        The left axis uses :meth:`plot_wavenet_detection` to render the signal,
        prediction, optional reconstructed trace, and CNN detections. The right
        axis shows a horizontal histogram of CNN amplitudes. Recovered
        amplitudes stored on ``self.cnn.amplitudes`` are preferred when they are
        available and aligned with the detected peaks; otherwise the histogram
        falls back to the processed signal values at ``self.cnn.peaks``.

        Parameters
        ----------
        x_axis : {"sample", "time"}, default="sample"
            Coordinate system used for the trace axis.
        xlim, ylim : tuple of float, optional
            Optional axis limits for the trace panel. When ``ylim`` is provided,
            the same vertical span is also applied to the histogram so both
            panels remain aligned.
        figsize : tuple of float, default=(12.0, 4.5)
            Figure size used when new axes are created.
        line_width, marker_size, threshold_line_width : float
            Styling parameters forwarded to :meth:`plot_wavenet_detection`.
        histogram_bins : int, default=20
            Default number of histogram bins.
        bins : int or sequence, optional
            Matplotlib-style alias for histogram bin selection. When provided,
            it overrides ``histogram_bins``.
        histogram_color, histogram_edgecolor, histogram_line_width
            Histogram bar styling.
        histogram_title : str, default="Amplitude histogram"
            Title applied to the histogram axis.
        histogram_xlim : tuple of float, optional
            Optional x-limits applied to the histogram axis.
        histogram_reference_lines : sequence of float, optional
            Horizontal reference lines drawn across the histogram axis.
        histogram_reference_line_kwargs : dict, optional
            Styling forwarded to each histogram reference line.
        show_signal, show_prediction, show_cnn_signal_peaks,
        show_cnn_recovered_signal_peaks, show_cnn_reconstructed_trace,
        show_cnn_prediction_peaks, show_cnn_threshold : bool
            Display toggles forwarded to :meth:`plot_wavenet_detection`.
        show_legend, show_grid, show_title : bool
            Display toggles for the trace and histogram panels.
        title : str, optional
            Title applied to the trace axis. If omitted and ``show_title`` is
            true, ``"CNN detection"`` is used.
        expected_particle_flow : float, optional
            Expected throughput value shown in the figure suptitle when
            ``show_throughput`` is true.
        show_throughput : bool, default=False
            Whether to add a throughput suptitle.
        throughput_template : str, default="Throughput: {label} / second"
            Format string used for the throughput suptitle. Receives
            ``label`` and ``flow``.
        throughput_label_formatter : callable, optional
            Callable converting ``expected_particle_flow`` into a compact label.
        suptitle_kwargs : dict, optional
            Keyword arguments forwarded to ``figure.suptitle``.
        show_inset : bool, default=False
            Whether to add a zoomed inset trace axis.
        inset_width, inset_height, inset_loc, inset_borderpad
            Inset placement parameters forwarded to ``inset_axes``.
        inset_xlim, inset_ylim : tuple of float, optional
            Axis limits applied to the inset.
        inset_show_ticks : bool, default=False
            Whether to show ticks and axis labels on the inset.
        inset_show_grid : bool, default=False
            Whether to enable the grid on the inset.
        inset_plot_kwargs : dict, optional
            Additional keyword arguments forwarded to the inset trace plot.
        mark_inset : bool, default=True
            Whether to draw connectors between the main trace and the inset.
        mark_inset_kwargs : dict, optional
            Keyword arguments forwarded to ``mark_inset``.
        subplots_adjust : dict, optional
            Layout arguments forwarded to ``figure.subplots_adjust``.
        signal_color, prediction_color, reconstructed_trace_color,
        marker_color, threshold_color : str
            Colors forwarded to :meth:`plot_wavenet_detection`.
        rasterize_dense_artists : bool, default=True
            Rasterize dense line artists when exporting vector graphics.
        axes : tuple(matplotlib.axes.Axes, matplotlib.axes.Axes), optional
            Existing ``(trace_axis, histogram_axis)`` pair. When omitted, a new
            figure with two columns is created.
        show, close : bool, default=False
            Whether to display or close the resulting figure.
        save_path : str or pathlib.Path, optional
            Optional path used to save the figure.
        dpi : int, default=300
            Raster resolution used when saving.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the CNN trace panel and amplitude histogram.
        """

        x_axis = plot_kwargs.pop("x_axis", "sample")
        xlim = plot_kwargs.pop("xlim", None)
        ylim = plot_kwargs.pop("ylim", None)
        figsize = plot_kwargs.pop("figsize", (12.0, 4.5))
        line_width = plot_kwargs.pop("line_width", 0.8)
        marker_size = plot_kwargs.pop("marker_size", 18.0)
        threshold_line_width = plot_kwargs.pop("threshold_line_width", 0.8)
        histogram_bins = plot_kwargs.pop("histogram_bins", 20)
        bins = plot_kwargs.pop("bins", None)
        histogram_color = plot_kwargs.pop("histogram_color", "0.75")
        histogram_edgecolor = plot_kwargs.pop("histogram_edgecolor", "black")
        histogram_line_width = plot_kwargs.pop("histogram_line_width", 0.8)
        histogram_title = plot_kwargs.pop("histogram_title", "Amplitude histogram")
        histogram_xlim = plot_kwargs.pop("histogram_xlim", None)
        histogram_reference_lines = plot_kwargs.pop("histogram_reference_lines", None)
        histogram_reference_line_kwargs = dict(
            plot_kwargs.pop(
                "histogram_reference_line_kwargs",
                {"color": "black", "linestyle": "--", "linewidth": 2.0},
            )
        )
        show_signal = plot_kwargs.pop("show_signal", True)
        show_prediction = plot_kwargs.pop("show_prediction", True)
        show_cnn_signal_peaks = plot_kwargs.pop("show_cnn_signal_peaks", False)
        show_cnn_recovered_signal_peaks = plot_kwargs.pop(
            "show_cnn_recovered_signal_peaks", False
        )
        show_cnn_reconstructed_trace = plot_kwargs.pop(
            "show_cnn_reconstructed_trace", False
        )
        show_cnn_prediction_peaks = plot_kwargs.pop("show_cnn_prediction_peaks", True)
        show_cnn_threshold = plot_kwargs.pop("show_cnn_threshold", True)
        show_legend = plot_kwargs.pop("show_legend", True)
        show_grid = plot_kwargs.pop("show_grid", False)
        show_title = plot_kwargs.pop("show_title", True)
        title = plot_kwargs.pop("title", None)
        expected_particle_flow = plot_kwargs.pop("expected_particle_flow", None)
        show_throughput = plot_kwargs.pop("show_throughput", False)
        throughput_template = plot_kwargs.pop(
            "throughput_template", "Throughput: {label} / second"
        )
        throughput_label_formatter = plot_kwargs.pop(
            "throughput_label_formatter", get_throughput_label
        )
        suptitle_kwargs = dict(plot_kwargs.pop("suptitle_kwargs", {}))
        show_inset = plot_kwargs.pop("show_inset", False)
        inset_width = plot_kwargs.pop("inset_width", "52%")
        inset_height = plot_kwargs.pop("inset_height", "50%")
        inset_loc = plot_kwargs.pop("inset_loc", "upper right")
        inset_borderpad = plot_kwargs.pop("inset_borderpad", 2.2)
        inset_xlim = plot_kwargs.pop("inset_xlim", None)
        inset_ylim = plot_kwargs.pop("inset_ylim", None)
        inset_show_ticks = plot_kwargs.pop("inset_show_ticks", False)
        inset_show_grid = plot_kwargs.pop("inset_show_grid", False)
        inset_plot_kwargs = dict(plot_kwargs.pop("inset_plot_kwargs", {}))
        mark_inset_enabled = plot_kwargs.pop("mark_inset", True)
        mark_inset_kwargs = dict(
            plot_kwargs.pop(
                "mark_inset_kwargs",
                {"loc1": 2, "loc2": 4, "fc": "none", "ec": "0.05", "lw": 2.0},
            )
        )
        subplots_adjust = plot_kwargs.pop(
            "subplots_adjust",
            {
                "left": 0.10,
                "right": 0.97,
                "bottom": 0.14,
                "top": 0.88 if show_title else 0.95,
                "wspace": 0.05,
            },
        )
        signal_color = plot_kwargs.pop("signal_color", "C0")
        prediction_color = plot_kwargs.pop("prediction_color", "C1")
        reconstructed_trace_color = plot_kwargs.pop("reconstructed_trace_color", "C3")
        marker_color = plot_kwargs.pop("marker_color", "black")
        threshold_color = plot_kwargs.pop("threshold_color", "black")
        rasterize_dense_artists = plot_kwargs.pop("rasterize_dense_artists", True)
        axes = plot_kwargs.pop("axes", None)
        show = plot_kwargs.pop("show", False)
        close = plot_kwargs.pop("close", False)
        save_path = plot_kwargs.pop("save_path", None)
        dpi = plot_kwargs.pop("dpi", 300)
        if plot_kwargs:
            unexpected_arguments = ", ".join(sorted(plot_kwargs))
            raise TypeError(
                f"plot_wavenet_detection_with_histogram() got unexpected keyword argument(s): {unexpected_arguments}"
            )

        signal = np.asarray(self.signal, dtype=float).ravel()
        prediction = np.asarray(self.prediction, dtype=float).ravel()
        if signal.size == 0:
            raise ValueError("Cannot plot an empty signal.")
        if prediction.size == 0:
            raise ValueError("Cannot plot an empty prediction.")

        cnn_peak_indices = np.asarray(self.cnn.peaks, dtype=int)
        valid_peak_mask = (cnn_peak_indices >= 0) & (cnn_peak_indices < signal.size)
        cnn_peak_indices = cnn_peak_indices[valid_peak_mask]
        cnn_amplitudes = signal[cnn_peak_indices]
        cnn_amplitudes = cnn_amplitudes[np.isfinite(cnn_amplitudes)]
        histogram_bin_spec = histogram_bins if bins is None else bins

        recovered_amplitudes = getattr(self.cnn, "amplitudes", None)
        if recovered_amplitudes is not None:
            recovered_amplitudes = np.asarray(recovered_amplitudes, dtype=float).ravel()
            original_peak_indices = np.asarray(self.cnn.peaks, dtype=int)
            if recovered_amplitudes.size == original_peak_indices.size:
                recovered_baseline = float(
                    getattr(self.cnn, "properties", {}).get("recovered_baseline", 0.0)
                )
                recovered_amplitudes = recovered_amplitudes[valid_peak_mask]
                recovered_amplitudes = recovered_amplitudes[
                    np.isfinite(recovered_amplitudes)
                ]
                if recovered_amplitudes.size > 0:
                    cnn_amplitudes = recovered_amplitudes + recovered_baseline

        if axes is not None:
            trace_axis, histogram_axis = axes
            figure = trace_axis.figure
            created_figure = False
        else:
            figure, (trace_axis, histogram_axis) = plt.subplots(
                ncols=2,
                figsize=figsize,
                sharey=True,
                gridspec_kw={"width_ratios": [4.0, 1.2], "wspace": 0.05},
            )
            created_figure = True

        self.plot_wavenet_detection(
            x_axis=x_axis,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            line_width=line_width,
            marker_size=marker_size,
            threshold_line_width=threshold_line_width,
            show_signal=show_signal,
            show_prediction=show_prediction,
            show_cnn_signal_peaks=show_cnn_signal_peaks,
            show_cnn_recovered_signal_peaks=show_cnn_recovered_signal_peaks,
            show_cnn_reconstructed_trace=show_cnn_reconstructed_trace,
            show_cnn_prediction_peaks=show_cnn_prediction_peaks,
            show_cnn_threshold=show_cnn_threshold,
            show_legend=show_legend,
            show_grid=show_grid,
            show_title=False,
            title=None,
            signal_color=signal_color,
            prediction_color=prediction_color,
            reconstructed_trace_color=reconstructed_trace_color,
            marker_color=marker_color,
            threshold_color=threshold_color,
            rasterize_dense_artists=rasterize_dense_artists,
            ax=trace_axis,
            show=False,
            close=False,
            save_path=None,
            dpi=dpi,
        )

        if cnn_amplitudes.size > 0:
            histogram_axis.hist(
                cnn_amplitudes,
                bins=histogram_bin_spec,
                orientation="horizontal",
                color=histogram_color,
                edgecolor=histogram_edgecolor,
                linewidth=histogram_line_width,
            )
        else:
            histogram_axis.text(
                0.5,
                0.5,
                "No peaks",
                ha="center",
                va="center",
                transform=histogram_axis.transAxes,
            )

        if ylim is not None:
            histogram_axis.set_ylim(ylim)
        if histogram_xlim is not None:
            histogram_axis.set_xlim(histogram_xlim)
        if histogram_reference_lines is not None:
            for value in histogram_reference_lines:
                histogram_axis.axhline(value, **histogram_reference_line_kwargs)
        histogram_axis.set_xlabel("Peak count")
        histogram_axis.set_title(histogram_title)
        if show_grid:
            histogram_axis.grid(True, which="both", axis="x", alpha=0.2, zorder=0)
        else:
            histogram_axis.grid(False, axis="x")
        histogram_axis.tick_params(axis="y", labelleft=False)

        if show_title:
            trace_axis.set_title(title or "CNN detection")
        else:
            trace_axis.set_title("")

        if show_inset:
            inset_axis = inset_axes(
                trace_axis,
                width=inset_width,
                height=inset_height,
                loc=inset_loc,
                borderpad=inset_borderpad,
            )
            inset_kwargs = {
                "x_axis": inset_plot_kwargs.pop("x_axis", x_axis),
                "xlim": inset_plot_kwargs.pop("xlim", inset_xlim),
                "ylim": inset_plot_kwargs.pop(
                    "ylim", inset_ylim if inset_ylim is not None else ylim
                ),
                "figsize": inset_plot_kwargs.pop("figsize", figsize),
                "line_width": inset_plot_kwargs.pop("line_width", line_width),
                "marker_size": inset_plot_kwargs.pop("marker_size", marker_size),
                "threshold_line_width": inset_plot_kwargs.pop(
                    "threshold_line_width", threshold_line_width
                ),
                "show_signal": inset_plot_kwargs.pop("show_signal", show_signal),
                "show_prediction": inset_plot_kwargs.pop(
                    "show_prediction", show_prediction
                ),
                "show_cnn_signal_peaks": inset_plot_kwargs.pop(
                    "show_cnn_signal_peaks", show_cnn_signal_peaks
                ),
                "show_cnn_recovered_signal_peaks": inset_plot_kwargs.pop(
                    "show_cnn_recovered_signal_peaks",
                    show_cnn_recovered_signal_peaks,
                ),
                "show_cnn_reconstructed_trace": inset_plot_kwargs.pop(
                    "show_cnn_reconstructed_trace",
                    show_cnn_reconstructed_trace,
                ),
                "show_cnn_prediction_peaks": inset_plot_kwargs.pop(
                    "show_cnn_prediction_peaks", show_cnn_prediction_peaks
                ),
                "show_cnn_threshold": inset_plot_kwargs.pop(
                    "show_cnn_threshold", show_cnn_threshold
                ),
                "show_legend": inset_plot_kwargs.pop("show_legend", show_legend),
                "show_grid": inset_plot_kwargs.pop("show_grid", inset_show_grid),
                "show_title": inset_plot_kwargs.pop("show_title", False),
                "title": inset_plot_kwargs.pop("title", None),
                "signal_color": inset_plot_kwargs.pop("signal_color", signal_color),
                "prediction_color": inset_plot_kwargs.pop(
                    "prediction_color", prediction_color
                ),
                "reconstructed_trace_color": inset_plot_kwargs.pop(
                    "reconstructed_trace_color", reconstructed_trace_color
                ),
                "marker_color": inset_plot_kwargs.pop("marker_color", marker_color),
                "threshold_color": inset_plot_kwargs.pop(
                    "threshold_color", threshold_color
                ),
                "rasterize_dense_artists": inset_plot_kwargs.pop(
                    "rasterize_dense_artists", rasterize_dense_artists
                ),
            }
            inset_kwargs.update(inset_plot_kwargs)
            self.plot_wavenet_detection(
                **inset_kwargs,
                ax=inset_axis,
                show=False,
                close=False,
                save_path=None,
                dpi=dpi,
            )
            if not inset_show_ticks:
                inset_axis.set_xticks([])
                inset_axis.set_yticks([])
                inset_axis.set_xlabel("")
                inset_axis.set_ylabel("")
            if mark_inset_enabled:
                mark_inset(trace_axis, inset_axis, **mark_inset_kwargs)

        if show_throughput and expected_particle_flow is not None:
            throughput_label = throughput_label_formatter(expected_particle_flow)
            default_suptitle_kwargs = {"x": 0.35, "y": 0.98}
            default_suptitle_kwargs.update(suptitle_kwargs)
            figure.suptitle(
                throughput_template.format(
                    flow=expected_particle_flow,
                    label=throughput_label,
                ),
                **default_suptitle_kwargs,
            )

        figure.subplots_adjust(**subplots_adjust)
        if save_path is not None:
            figure.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(figure)
        return figure
