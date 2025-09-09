import numpy as np
from numpy.typing import NDArray

from DeepPeak.algorithms.base import BaseAmplitudeSolver


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

    def run(self, centers: NDArray[np.float64], matched_responses: NDArray[np.float64]) -> NDArray[np.float64]:
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
