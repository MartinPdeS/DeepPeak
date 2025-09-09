import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from MPSPlots import helper

from DeepPeak.algorithms.base import BaseAmplitudeSolver
import matplotlib.pyplot as plt


@dataclass
class BatchedResults:
    """
    Container for batched solver results.

    Attributes
    ----------
    last_centers_ : ndarray, shape (B, A) or None
        Last input centers.
    last_matrix_ : ndarray, shape (B, A, A) or None
        Last computed Gram/response matrix.
    last_amplitudes_ : ndarray, shape (B, A) or None
        Last output amplitudes.
    """

    centers: NDArray[np.float64]
    matrix: NDArray[np.float64]
    amplitudes: NDArray[np.float64]

    @helper.post_mpl_plot
    def compare_plot(self, true_amplitudes: NDArray[np.float64], ncols: int = 2, max_plots: int = 6) -> plt.Figure:
        """
        Compare the mesured and true amplitudes in a grid of bar plots.

        Parameters
        ----------
        true_amplitudes : ndarray, shape (A,)
            Ground truth amplitudes.
        ncols : int, optional
            Number of columns in the plot grid (default is 2).
        max_plots : int, optional
            Maximum number of batch items to plot (default is 6).

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure containing the plots.
        """
        num_figure = min(self.amplitudes.shape[0], max_plots)
        figure, ax = plt.subplots(nrows=num_figure, ncols=1)

        for i in range(num_figure):
            ax[i].bar(np.arange(len(true_amplitudes[i])) - 0.2, true_amplitudes[i], width=0.4, label="True", color="C0")
            ax[i].bar(np.arange(len(true_amplitudes[i])) + 0.2, self.amplitudes[i], width=0.4, label="Measured", color="C1")

            ax[i].set(ylabel="Amplitude")
            ax[i].set_xticklabels([])

            ax[i].legend()

        return figure


class ClosedFormSolver(BaseAmplitudeSolver):
    r"""
    Closed-form amplitude solver for A ≤ 3 peaks with equal width $\sigma$.

    It builds the Gram matrix $H$ from centers and applies explicit formulas:
      - A=1: $a = m$
      - A=2: $a = \frac{1}{1-\rho^2} \begin{pmatrix} 1 & -\rho \\ -\rho & 1 \end{pmatrix} m$
      - A=3: uses analytic inverse of the 3x3 correlation matrix

    Parameters
    ----------
    sigma : float
        Common Gaussian standard deviation.
    eps : float, optional
        Small positive guard for denominators.

    Notes
    -----
    Inputs can be batched: centers and matched_responses accept (A,) or (B, A).
    """

    def __init__(self, sigma: float, *, eps: float = 1e-12) -> None:
        self.sigma = float(sigma)
        self.eps = float(eps)

        # last results
        self.last_centers_: NDArray[np.float64] | None = None
        self.last_matched_: NDArray[np.float64] | None = None
        self.last_gram_: NDArray[np.float64] | None = None
        self.last_amplitudes_: NDArray[np.float64] | None = None

    def run(self, centers: NDArray[np.float64], center_samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Solve ``G a = y_c`` in closed-form (A <= 3).

        Parameters
        ----------
        centers : ndarray, shape (A,) or (B, A)
            Peak centers μ.
        center_samples : ndarray, shape (A,) or (B, A)
            Measured **signal values at the same centers** (i.e., y(μ_k)), possibly batched.

        Returns
        -------
        amplitudes : ndarray, shape (A,) or (B, A)
            Recovered amplitudes a.
        """
        C, squeeze = self._coerce_to_2d(centers)
        center_samples, _ = self._coerce_to_2d(center_samples)
        if center_samples.shape != C.shape:
            raise ValueError("centers and center_samples must have the same shape.")

        number_of_estimated_peaks = C.shape[1]

        G = self._response_matrix_from_centers(C, self.sigma)

        match number_of_estimated_peaks:
            case 1:
                Ahat = center_samples.copy()

            case 2:
                r12 = G[:, 0, 1]
                det = (1.0 - r12**2).clip(min=self.eps)
                a1 = (center_samples[:, 0] - r12 * center_samples[:, 1]) / det
                a2 = (center_samples[:, 1] - r12 * center_samples[:, 0]) / det
                Ahat = np.stack([a1, a2], axis=1)

            case 3:
                r12, r13, r23 = G[:, 0, 1], G[:, 0, 2], G[:, 1, 2]
                det = (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2).clip(min=self.eps)

                inv00 = 1 - r23**2
                inv11 = 1 - r13**2
                inv22 = 1 - r12**2
                inv01 = r13 * r23 - r12
                inv02 = r12 * r23 - r13
                inv12 = r12 * r13 - r23

                Ginv = (
                    np.stack(
                        [
                            np.stack([inv00, inv01, inv02], axis=1),
                            np.stack([inv01, inv11, inv12], axis=1),
                            np.stack([inv02, inv12, inv22], axis=1),
                        ],
                        axis=1,
                    )
                    / det[:, None, None]
                )

                Ahat = np.einsum("bij,bj->bi", Ginv, center_samples)

            case 4:
                # Partition G and y_c:
                # G = [[A3 (3x3), b (3x1)],
                #      [b^T      , d     ]]
                # y = [y3 (3,), y4]
                #
                # Closed-form via Schur complement:
                # a4   = (y4 - b^T A3^{-1} y3) / (d - b^T A3^{-1} b)
                # a1:3 = A3^{-1} (y3 - b a4)

                # --- Invert the 3x3 top-left block A3 explicitly (unit-diagonal correlation form) ---
                r12 = G[:, 0, 1]
                r13 = G[:, 0, 2]
                r23 = G[:, 1, 2]

                det3 = (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2).clip(min=self.eps)

                inv00 = 1 - r23**2
                inv11 = 1 - r13**2
                inv22 = 1 - r12**2
                inv01 = r13 * r23 - r12
                inv02 = r12 * r23 - r13
                inv12 = r12 * r13 - r23

                # --- Extract b, d, and y parts explicitly ---
                b0, b1, b2 = G[:, 0, 3], G[:, 1, 3], G[:, 2, 3]
                d = G[:, 3, 3]  # typically 1.0 for peak-normalized Gaussians

                y0, y1, y2 = center_samples[:, 0], center_samples[:, 1], center_samples[:, 2]
                y4 = center_samples[:, 3]

                # --- Compute A3^{-1} y3 (call it Ay) ---
                Ay0 = (inv00 * y0 + inv01 * y1 + inv02 * y2) / det3
                Ay1 = (inv01 * y0 + inv11 * y1 + inv12 * y2) / det3
                Ay2 = (inv02 * y0 + inv12 * y1 + inv22 * y2) / det3

                # --- Compute A3^{-1} b (call it Ab) ---
                Ab0 = (inv00 * b0 + inv01 * b1 + inv02 * b2) / det3
                Ab1 = (inv01 * b0 + inv11 * b1 + inv12 * b2) / det3
                Ab2 = (inv02 * b0 + inv12 * b1 + inv22 * b2) / det3

                # --- Schur complement s = d - b^T A3^{-1} b (guarded) ---
                s = (d - (b0 * Ab0 + b1 * Ab1 + b2 * Ab2)).clip(min=self.eps)

                # --- a4 = (y4 - b^T A3^{-1} y3) / s ---
                a4 = (y4 - (b0 * Ay0 + b1 * Ay1 + b2 * Ay2)) / s

                # --- a1:3 = A3^{-1}(y3 - b a4) = Ay - Ab * a4 ---
                a0 = Ay0 - Ab0 * a4
                a1 = Ay1 - Ab1 * a4
                a2 = Ay2 - Ab2 * a4

                Ahat = np.stack([a0, a1, a2, a4], axis=1)

            case _:
                raise ValueError(f"Number of estimated peaks must be between 1 and 4, got {number_of_estimated_peaks}. Higher numbers haven't been implemented.")

        # cache for plotting
        self.last_centers_ = C
        self.last_matrix_ = G
        self.last_amplitudes_ = Ahat
        return Ahat[0] if squeeze else Ahat

    def run_batch(self, centers: NDArray[np.float64], center_samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Batched closed-form solve for ``G a = y_c`` with A ∈ {1,2,3,4}.

        Parameters
        ----------
        centers : ndarray, shape (B, A)
            Peak centers μ for each batch item.
        center_samples : ndarray, shape (B, A)
            Measured signal values at the same centers (i.e., y(μ_k)), per batch item.

        Returns
        -------
        amplitudes : ndarray, shape (B, A)
            Recovered amplitudes a for each batch item.

        Notes
        -----
        - All rows in the batch must share the same number of peaks A.
        - If your dataset mixes different A, split the batch by A and call this method per group.
        """
        centers = np.asarray(centers, dtype=float)
        center_samples = np.asarray(center_samples, dtype=float)
        centers = np.nan_to_num(centers, nan=0.0)
        center_samples = np.nan_to_num(center_samples, nan=0.0)

        if centers.ndim != 2 or center_samples.ndim != 2:
            raise ValueError("centers and center_samples must both be 2D: (B, A).")
        if centers.shape != center_samples.shape:
            raise ValueError("centers and center_samples must have the same shape (B, A).")

        number_of_estimated_peaks = centers.shape[1]

        # Response (Gram) matrix per batch item
        G = self._response_matrix_from_centers(centers, self.sigma)  # shape (B, A, A)

        match number_of_estimated_peaks:
            case 1:
                Ahat = center_samples.copy()

            case 2:
                r12 = G[:, 0, 1]
                det = (1.0 - r12**2).clip(min=self.eps)
                a1 = (center_samples[:, 0] - r12 * center_samples[:, 1]) / det
                a2 = (center_samples[:, 1] - r12 * center_samples[:, 0]) / det
                Ahat = np.stack([a1, a2], axis=1)

            case 3:
                r12, r13, r23 = G[:, 0, 1], G[:, 0, 2], G[:, 1, 2]
                det = (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2).clip(min=self.eps)

                inv00 = 1 - r23**2
                inv11 = 1 - r13**2
                inv22 = 1 - r12**2
                inv01 = r13 * r23 - r12
                inv02 = r12 * r23 - r13
                inv12 = r12 * r13 - r23

                Ginv = (
                    np.stack(
                        [
                            np.stack([inv00, inv01, inv02], axis=1),
                            np.stack([inv01, inv11, inv12], axis=1),
                            np.stack([inv02, inv12, inv22], axis=1),
                        ],
                        axis=1,
                    )
                    / det[:, None, None]
                )

                Ahat = np.einsum("bij,bj->bi", Ginv, center_samples)

            case 4:
                # Top-left 3x3 inversion (unit-diagonal correlation form), then Schur complement
                r12 = G[:, 0, 1]
                r13 = G[:, 0, 2]
                r23 = G[:, 1, 2]

                det3 = (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2).clip(min=self.eps)

                inv00 = 1 - r23**2
                inv11 = 1 - r13**2
                inv22 = 1 - r12**2
                inv01 = r13 * r23 - r12
                inv02 = r12 * r23 - r13
                inv12 = r12 * r13 - r23

                # Partition b, d, and y = [y0,y1,y2,y4]
                b0, b1, b2 = G[:, 0, 3], G[:, 1, 3], G[:, 2, 3]
                d = G[:, 3, 3]  # typically 1.0

                y0, y1, y2 = center_samples[:, 0], center_samples[:, 1], center_samples[:, 2]
                y4 = center_samples[:, 3]

                # A3^{-1} * y3
                Ay0 = (inv00 * y0 + inv01 * y1 + inv02 * y2) / det3
                Ay1 = (inv01 * y0 + inv11 * y1 + inv12 * y2) / det3
                Ay2 = (inv02 * y0 + inv12 * y1 + inv22 * y2) / det3

                # A3^{-1} * b
                Ab0 = (inv00 * b0 + inv01 * b1 + inv02 * b2) / det3
                Ab1 = (inv01 * b0 + inv11 * b1 + inv12 * b2) / det3
                Ab2 = (inv02 * b0 + inv12 * b1 + inv22 * b2) / det3

                # Schur complement
                s = (d - (b0 * Ab0 + b1 * Ab1 + b2 * Ab2)).clip(min=self.eps)

                # a4 and a0:2
                a4 = (y4 - (b0 * Ay0 + b1 * Ay1 + b2 * Ay2)) / s
                a0 = Ay0 - Ab0 * a4
                a1 = Ay1 - Ab1 * a4
                a2 = Ay2 - Ab2 * a4

                Ahat = np.stack([a0, a1, a2, a4], axis=1)

            case _:
                raise ValueError(f"Number of estimated peaks must be between 1 and 4, got {number_of_estimated_peaks}.")

        return BatchedResults(centers=centers, matrix=G, amplitudes=Ahat)
