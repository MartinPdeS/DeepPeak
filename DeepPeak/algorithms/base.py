import numpy as np
from numpy.typing import NDArray
from MPSPlots import helper
import matplotlib.pyplot as plt


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
    def _coerce_to_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Ensure input is 2D array (B, A). Return (x2d, was_1d).

        Parameters
        ----------
        x : ndarray, shape (A,) or (B, A)
            Input array.

        Returns
        -------
        x2d : ndarray, shape (B, A)
            2D version of input.
        was_1d : bool
            True if input was originally 1D.
        """
        x = np.asarray(x, dtype=float)
        original_1d = x.ndim == 1
        if x.ndim not in (1, 2):
            raise ValueError("Input must have shape (A,) or (B, A).")
        x2 = np.atleast_2d(x)  # (1, A) if original was 1D, unchanged if 2D
        return x2, original_1d

    # --- Amplitudes-only plot ----------------------------------------------------
    @helper.post_mpl_plot
    def plot(self, true_amplitudes: NDArray[np.float64], sample_index: int = 0) -> plt.Figure:
        """
        Plot recovered amplitudes for one sample (bar plot only).

        Handles length mismatches (estimated vs. true) by padding the shorter vector
        with NaNs so bars stay aligned. NaN bars are not drawn by Matplotlib.

        Parameters
        ----------
        sample_index : int
            Which batch item to display.
        true_amplitudes : ndarray, optional, shape (A,)
            If provided, overlay ground-truth amplitudes.
        """
        if self.last_amplitudes_ is None:
            raise RuntimeError("Run the solver first, then call plot().")

        a_est = self.last_amplitudes_[sample_index]

        true_amplitudes = np.sort(true_amplitudes)
        a_est = np.sort(a_est)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.6))

        A_est = int(a_est.shape[0])
        if true_amplitudes is not None:
            a_true = np.asarray(true_amplitudes, dtype=float)
            A_true = int(a_true.shape[0])
        else:
            a_true = None
            A_true = 0

        A = max(A_est, A_true) or 1
        idx = np.arange(A)

        # Pad with NaNs (Matplotlib skips NaN bars)
        est_plot = np.full(A, np.nan, dtype=float)
        est_plot[:A_est] = a_est

        if a_true is not None:
            true_plot = np.full(A, np.nan, dtype=float)
            true_plot[:A_true] = a_true
        else:
            true_plot = None

        # Use centers as x tick labels if available
        if getattr(self, "last_centers_", None) is not None:
            centers = self.last_centers_[sample_index]
            labels = [f"μ{j}={centers[j]:.3f}" if j < centers.shape[0] else f"μ{j}" for j in range(A)]
        else:
            labels = [str(j) for j in range(A)]

        # Bars
        ax.bar(idx - 0.17, est_plot, width=0.34, label="estimated")
        if true_plot is not None:
            ax.bar(idx + 0.17, true_plot, width=0.34, label="true", alpha=0.7)

            # Outline extras where one side has no counterpart
            extra_est = ~np.isnan(est_plot) & np.isnan(true_plot)
            extra_true = np.isnan(est_plot) & ~np.isnan(true_plot)
            if extra_est.any():
                ax.bar(idx[extra_est] - 0.17, est_plot[extra_est], width=0.34, fill=False, edgecolor="C0", linewidth=1.5, label="_nolegend_")
            if extra_true.any():
                ax.bar(idx[extra_true] + 0.17, true_plot[extra_true], width=0.34, fill=False, edgecolor="C1", linewidth=1.5, label="_nolegend_")

            ax.set_title(f"Amplitudes (est={A_est}, true={A_true})")
        else:
            ax.set_title(f"Amplitudes (est={A_est})")

        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, A - 0.4)
        ax.legend()

        return fig

    # --- Separate Gram heatmap ---------------------------------------------------
    @helper.post_mpl_plot
    def plot_gram(self, sample_index: int = 0) -> plt.Figure:
        """
        Plot the Gram matrix heatmap for one sample.

        Parameters
        ----------
        sample_index : int
            Which batch item to display.
        """
        if self.last_gram_ is None:
            raise RuntimeError("No Gram matrix cached. Run the solver first.")

        H = self.last_gram_[sample_index]
        figure, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))

        im = ax.imshow(H, vmin=0, vmax=1, cmap="viridis")
        title = "Gram matrix"
        cond = np.linalg.cond(H)
        title += f" (cond ≈ {cond:.2e})"

        if hasattr(self, "regularization"):
            title += f"  λ={getattr(self, 'regularization'):.3g}"

        ax.set_title(title)
        ax.set_xticks(range(H.shape[0]))
        ax.set_yticks(range(H.shape[0]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return figure

    @staticmethod
    def _response_matrix_from_centers(centers_2d: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
        r"""
        Build the batched *response* matrix G for equal-width Gaussian peaks sampled at centers.

        For peaks all sharing σ and centers μ, the signal sampled at those centers obeys
            y_c = G a,
        where
            G_{ij} = exp( - (μ_i - μ_j)^2 / (2 σ^2) ),  and  G_{ii} = 1.

        Parameters
        ----------
        centers_2d : ndarray, shape (B, A)
            Peak centers (batched). A is the number of peaks (A ≤ 3).
        sigma : float
            Common Gaussian standard deviation.

        Returns
        -------
        G : ndarray, shape (B, A, A)
            Batched response matrices.
        """
        d = centers_2d[..., :, None] - centers_2d[..., None, :]  # (B, A, A)
        G = np.exp(-0.5 * (d / float(sigma)) ** 2)
        # enforce exact ones on the diagonal
        idx = np.arange(G.shape[-1])
        G[..., idx, idx] = 1.0
        return G
