from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from DeepPeak.algorithms.amplitude.base import BaseAmplitudeSolver


# =============================================================================
# Solver 1: Closed-form inverse for A ∈ {1,2,3} (fastest)
# =============================================================================
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

    def run(self, centers: NDArray[np.float64], matched_responses: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Solve H a = m in closed-form.

        Parameters
        ----------
        centers : ndarray, shape (A,) or (B, A)
            Peak centers μ.
        matched_responses : ndarray, shape (A,) or (B, A)
            Matched-filter responses m sampled at the centers.

        Returns
        -------
        amplitudes : ndarray, shape (A,) or (B, A)
            Recovered amplitudes a.
        """
        C, squeeze = self._coerce_to_2d(centers)
        M, _ = self._coerce_to_2d(matched_responses)
        if M.shape != C.shape:
            raise ValueError("centers and matched_responses must have the same shape.")

        B, A = C.shape
        if A not in (1, 2, 3):
            raise ValueError(f"A must be 1, 2, or 3; got A={A}")

        H = self._gram_from_centers(C, self.sigma)

        if A == 1:
            Ahat = M.copy()

        elif A == 2:
            r12 = H[:, 0, 1]
            det = (1.0 - r12**2).clip(min=self.eps)
            a1 = (M[:, 0] - r12 * M[:, 1]) / det
            a2 = (M[:, 1] - r12 * M[:, 0]) / det
            Ahat = np.stack([a1, a2], axis=1)

        else:  # A == 3
            r12, r13, r23 = H[:, 0, 1], H[:, 0, 2], H[:, 1, 2]
            det = (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2).clip(min=self.eps)

            inv00 = 1 - r23**2
            inv11 = 1 - r13**2
            inv22 = 1 - r12**2
            inv01 = r13 * r23 - r12
            inv02 = r12 * r23 - r13
            inv12 = r12 * r13 - r23

            Hinv = (
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
            Ahat = np.einsum("bij,bj->bi", Hinv, M)

        # cache
        self.last_centers_ = C
        self.last_matched_ = M
        self.last_gram_ = H
        self.last_amplitudes_ = Ahat

        return Ahat[0] if squeeze else Ahat

    def plot(
        self,
        sample_index: int = 0,
        true_amplitudes: NDArray[np.float64] | None = None,
        ax: tuple[plt.Axes, plt.Axes] | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot the Gram matrix heatmap and the recovered amplitudes (bar plot) for one sample.

        Parameters
        ----------
        sample_index : int, optional
            Which batch item to display.
        true_amplitudes : ndarray, optional, shape (A,)
            If provided, overlay ground-truth amplitudes.
        ax : tuple(Axes, Axes), optional
            (ax_heatmap, ax_bars) to draw on; if None, new figure is created.
        show : bool
            Whether to call plt.show().
        """
        if self.last_gram_ is None or self.last_amplitudes_ is None:
            raise RuntimeError("Run the solver first, then call plot().")

        H = self.last_gram_[sample_index]
        a = self.last_amplitudes_[sample_index]
        cond = np.linalg.cond(H)

        if ax is None:
            fig, (axH, axB) = plt.subplots(1, 2, figsize=(9, 3.6))
        else:
            axH, axB = ax

        im = axH.imshow(H, vmin=0, vmax=1, cmap="viridis")
        axH.set_title(f"Gram matrix (cond ≈ {cond:.2e})")
        axH.set_xticks(range(H.shape[0]))
        axH.set_yticks(range(H.shape[0]))
        plt.colorbar(im, ax=axH, fraction=0.046, pad=0.04)

        idx = np.arange(a.shape[0])
        axB.bar(idx - 0.15, a, width=0.3, label="estimated")
        if true_amplitudes is not None:
            axB.bar(idx + 0.15, true_amplitudes, width=0.3, label="true", alpha=0.7)
        axB.set_title("Amplitudes")
        axB.set_xticks(idx)
        axB.legend()

        if show and ax is None:
            plt.tight_layout()
            plt.show()
