"""Dead-time correction for Poisson peak detection.

When peaks arrive as a Poisson process with rate λ but the detector cannot
resolve peaks closer than τ (minimum resolvable spacing), the observed rate is
systematically lower than the true rate:

    λ_observed = λ_true × exp(-λ_true × τ)

This module provides the forward model and its inverse so that notebook and
analysis code can estimate how many peaks are missed and recover the true
particle flow from measured data.
"""

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.special import lambertw


def expected_observed_flow(
    lambda_true: Union[float, np.ndarray],
    tau: float,
) -> Union[float, np.ndarray]:
    """Compute the expected observed flow given a dead-time constraint.

    For a Poisson process with true arrival rate λ_true and a detector that
    cannot resolve peaks closer than τ, the observable rate is:

        λ_observed = λ_true × exp(-λ_true × τ)

    Parameters
    ----------
    lambda_true : float or numpy.ndarray
        True particle flow rate (events per unit).
    tau : float
        Minimum resolvable distance between peaks, in the same units as
        lambda_true (e.g. samples or seconds).

    Returns
    -------
    float or numpy.ndarray
        Expected measured flow rate at the detector.
    """
    lambda_true = np.asarray(lambda_true, dtype=float)
    return lambda_true * np.exp(-lambda_true * float(tau))


def correct_observed_flow(
    lambda_obs: Union[float, np.ndarray],
    tau: float,
) -> Union[float, np.ndarray]:
    """Recover the true Poisson rate from the observed rate with dead time.

    Solves the inverse of :func:`expected_observed_flow` via the Lambert W
    function:

        λ_true = -W(-λ_observed × τ) / τ

    Falls back to the first-order approximation ``λ_obs / (1 - λ_obs × τ)``
    when the argument is small enough that the Lambert W correction is
    negligible (< 1 ppm error).

    Parameters
    ----------
    lambda_obs : float or numpy.ndarray
        Observed particle flow rate measured by the detector.
    tau : float
        Minimum resolvable distance between peaks, in the same units as
        lambda_obs.

    Returns
    -------
    float or numpy.ndarray
        Estimated true particle flow rate.

    Raises
    ------
    ValueError
        If any element of lambda_obs exceeds the theoretical maximum
        observable rate 1 / (e × τ).
    """
    lambda_obs = np.asarray(lambda_obs, dtype=float)
    tau = float(tau)

    max_observable = 1.0 / (np.e * tau)
    if np.any(lambda_obs > max_observable):
        raise ValueError(
            f"lambda_obs contains values above the theoretical maximum "
            f"observable rate 1/(e×τ) ≈ {max_observable:.6g}. "
            "The dead-time model cannot be inverted above this rate."
        )

    scalar_input = lambda_obs.ndim == 0
    lambda_obs = np.atleast_1d(lambda_obs)
    result = np.empty_like(lambda_obs)

    small = lambda_obs * tau < 1e-4
    result[small] = lambda_obs[small] / (1.0 - lambda_obs[small] * tau)

    if np.any(~small):
        arg = -lambda_obs[~small] * tau
        result[~small] = -lambertw(arg, k=0).real / tau

    return float(result[0]) if scalar_input else result


def fraction_missed(
    lambda_true: Union[float, np.ndarray],
    tau: float,
) -> Union[float, np.ndarray]:
    """Fraction of true peaks that are missed due to dead-time shadowing.

    Parameters
    ----------
    lambda_true : float or numpy.ndarray
        True particle flow rate.
    tau : float
        Minimum resolvable distance between peaks.

    Returns
    -------
    float or numpy.ndarray
        Value in [0, 1) giving the fraction of peaks undetected.
    """
    lambda_true = np.asarray(lambda_true, dtype=float)
    return 1.0 - np.exp(-lambda_true * float(tau))


def throughput_tick_formatter(value: float, _position: int) -> str:
    """Matplotlib tick formatter that abbreviates large throughput values.

    Maps raw numeric axis values to human-readable strings using SI-style
    suffixes (k, M, B) so that throughput axes remain legible at high counts.

    Intended for use with ``matplotlib.ticker.FuncFormatter``:

        axis.xaxis.set_major_formatter(FuncFormatter(throughput_tick_formatter))
    """
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def plot_dead_time_saturation(
    tau_values: Union[float, Sequence[float]],
    max_flow: Optional[float] = None,
    n_points: int = 500,
    figsize: tuple[float, float] = (6.0, 6.0),
    ax: Optional[plt.Axes] = None,
    show_ideal_line: bool = True,
    line_width: float = 2.0,
    show: bool = True,
    close: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Overlay dead-time saturation curves on an expected-vs-measured flow plot.

    For each value of τ in ``tau_values``, draws the theoretical relationship

        λ_observed = λ_true × exp(-λ_true × τ)

    on axes whose x-axis is the expected (true) flow and y-axis is the measured
    (observed) flow.  This matches the layout of
    ``DilutionSeries.plot.measured_vs_expected_particle_flows``.

    Parameters
    ----------
    tau_values : float or sequence of float
        Minimum resolvable peak spacing for each curve, in the same units as
        the flow axis (events per second, per sample, etc.).
    max_flow : float, optional
        Upper bound of the x-axis.  When *ax* is provided and *max_flow* is
        ``None``, the current x-axis upper limit is used automatically.
    n_points : int, default=500
        Number of points per curve.
    figsize : tuple, default=(6.0, 6.0)
        Figure size in inches.  Ignored when *ax* is provided.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  If ``None``, a new figure and axes are created.
    show_ideal_line : bool, default=True
        Draw the y = x reference line.  Set to ``False`` when the host plot
        already contains an ideal line.
    line_width : float, default=2.0
        Line width for the saturation curves.
    show : bool, default=True
        Call ``plt.show()``.  Ignored when *ax* is provided.
    close : bool, default=False
        Call ``plt.close()`` after rendering.  Ignored when *ax* is provided.
    save_path : str, optional
        File path to save the figure.
    dpi : int, default=300
        Resolution for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if np.isscalar(tau_values):
        tau_values = [tau_values]

    if ax is not None:
        figure, axis = ax.figure, ax
        created_figure = False
        if max_flow is None:
            max_flow = float(axis.get_xlim()[1])
    else:
        if max_flow is None:
            raise ValueError("max_flow must be provided when ax is None.")
        figure, axis = plt.subplots(figsize=figsize)
        created_figure = True

    lambda_true = np.linspace(0, max_flow, n_points)

    if show_ideal_line:
        axis.plot(
            lambda_true,
            lambda_true,
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label="Ideal (no dead time)",
        )

    for i, tau in enumerate(tau_values):
        lambda_obs = expected_observed_flow(lambda_true, tau)
        label = f"τ = {tau:.3g}"
        axis.plot(
            lambda_true, lambda_obs, linewidth=line_width, color=f"C{i}", label=label
        )

    axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )

    if created_figure:
        axis.set_xlabel("Expected particle flow", fontsize=12)
        axis.set_ylabel("Measured particle flow", fontsize=12)
        axis.set_title("Dead-time saturation effect", fontsize=13, fontweight="bold")
        axis.set_xlim(0, max_flow)
        axis.set_ylim(0, max_flow)
        axis.grid(True, which="both", alpha=0.2, zorder=0)
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        if close:
            plt.close(figure)

    return figure
