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

    # ------------------------- main entry -------------------------
    def run(
        self,
        centers: NDArray[np.float64],
        center_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve G a = y_c in closed-form for A ∈ {1,2,3,4,5}.
        """
        centers, squeeze = self._coerce_to_2d(centers)
        Yc, _ = self._coerce_to_2d(center_samples)
        if Yc.shape != centers.shape:
            raise ValueError("centers and center_samples must have the same shape.")

        A = centers.shape[1]
        G = self._response_matrix_from_centers(centers, self.sigma)

        if A == 1:
            Ahat = self._solve_order1(Yc)
        elif A == 2:
            Ahat = self._solve_order2(G, Yc)
        elif A == 3:
            Ahat = self._solve_order3(G, Yc)
        elif A == 4:
            Ahat = self._solve_order4(G, Yc)
        elif A == 5:
            Ahat = self._solve_order5(G, Yc)
        else:
            raise ValueError(f"Number of estimated peaks must be 1 - 5, got {A}. Higher orders not implemented.")

        # cache for plotting
        self.last_centers_ = centers
        self.last_matrix_ = G
        self.last_amplitudes_ = Ahat
        return Ahat[0] if squeeze else Ahat

    # ------------------------- sub-solvers -------------------------
    def _solve_order1(self, Yc: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Closed-form solver for A=1.

        Parameters
        ----------
        Yc : ndarray, shape (B, 1)
            Measured signal values at the centers, per batch item.
        """
        return Yc.copy()

    def _solve_order2(self, G: NDArray[np.float64], Yc: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Closed-form solver for A=2 using analytic inverse of the 2x2 correlation matrix.

        Parameters
        ----------
        G : ndarray, shape (B, 2, 2)
            Response (Gram) matrix per batch item.
        Yc : ndarray, shape (B, 2)
            Measured signal values at the centers, per batch item.
        """
        r12 = G[:, 0, 1]
        det = (1.0 - r12**2).clip(min=self.eps)
        a1 = (Yc[:, 0] - r12 * Yc[:, 1]) / det
        a2 = (Yc[:, 1] - r12 * Yc[:, 0]) / det
        return np.stack([a1, a2], axis=1)

    def _solve_order3(self, G: NDArray[np.float64], Yc: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Closed-form solver for A=3 using analytic inverse of the 3x3 correlation matrix.

        Parameters
        ----------
        G : ndarray, shape (B, 3, 3)
            Response (Gram) matrix per batch item.
        Yc : ndarray, shape (B, 3)
            Measured signal values at the centers, per batch item.

        Returns
        -------
        amplitudes : ndarray, shape (B, 3)
            Recovered amplitudes a for each batch item.
        """
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

        return np.einsum("bij,bj->bi", Ginv, Yc)

    def _solve_order4(self, G: NDArray[np.float64], Yc: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Closed-form solver for A=4.
        Uses block partitioning:
            G = [[A3, b],
                [b^T, d]]
        then Schur complement. A3 is the top-left 3x3 block.

        Parameters
        ----------
        G : ndarray, shape (B, 4, 4)
            Response (Gram) matrix per batch item.
        Yc : ndarray, shape (B, 4)
            Measured signal values at the centers, per batch item.

        Returns
        -------
        amplitudes : ndarray, shape (B, 4)
            Recovered amplitudes a for each batch item.
        """
        # block partition
        A3_inv, det3 = self._invert3(G[:, :3, :3])

        b = G[:, :3, 3]
        d = G[:, 3, 3]
        y3 = Yc[:, :3]
        y4 = Yc[:, 3]

        Ay = np.einsum("bij,bj->bi", A3_inv, y3)
        Ab = np.einsum("bij,bj->bi", A3_inv, b)
        s = (d - np.einsum("bi,bi->b", b, Ab)).clip(min=self.eps)

        a4 = (y4 - np.einsum("bi,bi->b", b, Ay)) / s
        a3 = Ay - Ab * a4[:, None]
        return np.concatenate([a3, a4[:, None]], axis=1)

    def _solve_order5(self, G: NDArray[np.float64], Yc: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Closed-form solver for A=5.
        Uses block partitioning:
            G = [[A4, b],
                [b^T, d]]
        then Schur complement. A4 is the top-left 4x4 block.

        Parameters
        ----------
        G : ndarray, shape (B, 5, 5)
            Response (Gram) matrix per batch item.
        Yc : ndarray, shape (B, 5)
            Measured signal values at the centers, per batch item.

        Returns
        -------
        amplitudes : ndarray, shape (B, 5)
            Recovered amplitudes a for each batch item.
        """
        # Partition: top-left A4 (4x4), b (4,), d (scalar)
        A4 = G[:, :4, :4]
        b = G[:, :4, 4]
        d = G[:, 4, 4]
        y4 = Yc[:, :4]
        y5 = Yc[:, 4]

        # Explicit inversion of A4 via cofactors
        A4_inv, det4 = self._invert4(A4)

        Ay = np.einsum("bij,bj->bi", A4_inv, y4)
        Ab = np.einsum("bij,bj->bi", A4_inv, b)
        s = (d - np.einsum("bi,bi->b", b, Ab)).clip(min=self.eps)

        a5 = (y5 - np.einsum("bi,bi->b", b, Ay)) / s
        a4 = Ay - Ab * a5[:, None]
        return np.concatenate([a4, a5[:, None]], axis=1)

    # ------------------------------------------------------------------
    # Helpers for explicit inverses
    # ------------------------------------------------------------------
    def _invert3(self, A3):
        """
        Explicit inverse of 3x3 correlation-like matrix (batched).
        Returns (inv, det).

        Parameters
        ----------
        A3 : ndarray, shape (B, 3, 3)
            Batched 3x3 symmetric correlation matrices.
        """
        r12, r13, r23 = A3[:, 0, 1], A3[:, 0, 2], A3[:, 1, 2]
        det = (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2).clip(min=self.eps)

        inv00 = 1 - r23**2
        inv11 = 1 - r13**2
        inv22 = 1 - r12**2
        inv01 = r13 * r23 - r12
        inv02 = r12 * r23 - r13
        inv12 = r12 * r13 - r23

        inv = (
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

        return inv, det

    def _invert4(self, A4):
        """
        Explicit inverse of a 4x4 symmetric correlation matrix (batched).
        Returns (inv, det).
        Uses 3x3 minors via _invert3.

        Parameters
        ----------
        A4 : ndarray, shape (B, 4, 4)
            Batched 4x4 symmetric correlation matrices.
        """
        B = []
        dets = []
        for drop_row in range(4):
            row_blocks = []
            for drop_col in range(4):
                # minor is 3x3 (remove row, col)
                mask = [i for i in range(4) if i != drop_row]
                sub = A4[:, mask][:, :, [j for j in range(4) if j != drop_col]]
                inv3, det3 = self._invert3(sub)
                dets.append(det3)
                # cofactor = det(minor)*(-1)^(i+j)
                row_blocks.append(((det3) * ((-1) ** (drop_row + drop_col)))[:, None])
            B.append(np.concatenate(row_blocks, axis=1))
        adj = np.stack(B, axis=1)  # (B,4,4)
        det4 = np.einsum("bii->b", A4 * adj) / 4.0  # crude det estimate
        inv = adj / det4[:, None, None]
        return inv, det4

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
