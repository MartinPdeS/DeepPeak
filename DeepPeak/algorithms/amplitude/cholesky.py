from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from DeepPeak.algorithms.amplitude.base import BaseAmplitudeSolver


# =============================================================================
# Solver 2: Batched Cholesky (robust; supports small regularization)
# =============================================================================
class CholeskySolver(BaseAmplitudeSolver):
    r"""
    Cholesky-based amplitude solver for A ≤ 3 peaks with equal width $\sigma$.

    Solves (H + λ I) a = m with λ ≥ 0 via batched Cholesky and triangular solves.

    Parameters
    ----------
    sigma : float
        Common Gaussian standard deviation.
    regularization : float, optional
        Tikhonov parameter λ (default 0.0). Set small λ when centers are nearly coincident.
    """

    def __init__(self, sigma: float, *, regularization: float = 0.0) -> None:
        self.sigma = float(sigma)
        self.regularization = float(regularization)

        self.last_centers_: NDArray[np.float64] | None = None
        self.last_matched_: NDArray[np.float64] | None = None
        self.last_gram_: NDArray[np.float64] | None = None
        self.last_amplitudes_: NDArray[np.float64] | None = None

    def run(
        self,
        centers: NDArray[np.float64],
        matched_responses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve (H + λ I) a = m.

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
        if self.regularization != 0.0:
            H = H + self.regularization * np.eye(A)[None, :, :]

        # Batched Cholesky + triangular solves (vectorized)
        L = np.linalg.cholesky(H)  # (B, A, A)
        y = np.linalg.solve(L, M[..., None])  # forward solve: (B, A, 1)
        a = np.linalg.solve(np.swapaxes(L, -1, -2), y)[..., 0]  # backward solve

        self.last_centers_ = C
        self.last_matched_ = M
        self.last_gram_ = H
        self.last_amplitudes_ = a

        return a[0] if squeeze else a

    def plot(
        self,
        sample_index: int = 0,
        true_amplitudes: NDArray[np.float64] | None = None,
        ax: tuple[plt.Axes, plt.Axes] | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot the Gram matrix heatmap and the recovered amplitudes (bar plot) for one sample.
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
        axH.set_title(f"Gram matrix (cond ≈ {cond:.2e})  λ={self.regularization:g}")
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
