import numpy as np
from numpy.typing import NDArray


class BaseAmplitudeSolver:

    @staticmethod
    def _gram_from_centers(centers_2d: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
        """
        Vectorized Gram matrix from centers for a unit-energy Gaussian template:

            H_{ij} = \rho(|μ_i - μ_j|),  with  \rho(Δ) = exp(-Δ^2 / (4 σ^2)),  and H_{ii} = 1.

        Parameters
        ----------
        centers_2d : ndarray, shape (B, A)
            Peak centers per batch item.
        sigma : float
            Common Gaussian standard deviation.

        Returns
        -------
        H : ndarray, shape (B, A, A)
            Batched Gram matrices.
        """
        d = centers_2d[..., :, None] - centers_2d[..., None, :]
        H = np.exp(-0.25 * (d / sigma) ** 2)
        idx = np.arange(H.shape[-1])
        H[..., idx, idx] = 1.0
        return H

    @staticmethod
    def _coerce_to_2d(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
        """Ensure (B, A) shape; remember whether to squeeze back."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x[None, :], True
        if x.ndim == 2:
            return x, False
        raise ValueError("Input must have shape (A,) or (B, A).")
