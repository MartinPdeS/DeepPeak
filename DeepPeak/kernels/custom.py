from typing import Optional
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray


from .base import BaseKernel
from DeepPeak.peak_count import PeakCount


@dataclass(repr=False)
class CustomKernel(BaseKernel):
    """
    Pulse model based on a user-supplied sampled kernel shape.

    Parameters
    ----------
    kernel : ndarray of shape (K,)
        One-dimensional sampled pulse shape. The supplied support is normalized to
        ``[0, 1]`` internally and later stretched onto the evaluation grid.
    amplitude : float or tuple[float, float]
        Amplitude or inclusive sampling range for scaling the kernel.
    position : float or tuple[float, float]
        Center position or inclusive sampling range for placing kernel copies.

    Notes
    -----
    The stored sampled kernel is linearly interpolated onto the target ``x_values``
    grid for each sampled center.
    """

    kernel: NDArray
    amplitude: float
    position: float

    def __post_init__(self):
        self.kernel = np.asarray(self.kernel, dtype=float)
        if self.kernel.ndim != 1:
            raise ValueError("kernel must be a one dimensional array")
        if self.kernel.size == 0:
            raise ValueError("kernel must contain at least one sample")

        self._initialize_common_ranges(has_width=False)

        # Normalize kernel length for later interpolation
        self.kernel_x = np.linspace(0, 1, self.kernel.size)

    def get_kwargs(self) -> dict:
        return self._state_dict("kernel", "amplitudes", "positions")

    def evaluate(
        self,
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
        peak_count: PeakCount | None = None,
        peak_count_distribution: str = "uniform",
        peak_count_rate: Optional[float] = None,
    ) -> NDArray:
        """
        Evaluate the custom kernel at random positions and amplitudes.

        Parameters
        ----------
        x_values : NDArray
            One-dimensional evaluation grid.
        n_samples : int
            Number of signals to generate.
        n_peaks : tuple[int, int]
            Inclusive lower and upper bounds for the number of active peaks.
        categorical_peak_count : bool, default=False
            If ``True``, encode the sampled peak count as one-hot values.
        peak_count : PeakCount or None, optional
            Optional peak-count sampler overriding the legacy distribution arguments.
        peak_count_distribution : {"uniform", "poisson"}, default="uniform"
            Legacy peak-count distribution used when ``peak_count`` is not provided.
        peak_count_rate : float or tuple[float, float], optional
            Legacy Poisson rate configuration used when
            ``peak_count_distribution='poisson'``.

        Returns
        -------
        NDArray
            Evaluated kernel components with shape ``(n_samples, max_peaks, M)``,
            where ``M = len(x_values)``. Inactive peaks are NaN-masked.
        """
        x_, amp_, pos_, _, _, _ = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            has_width=False,
        )

        # Evaluate the kernel
        y = self._kernel(x_values=x_, amplitudes=amp_, centers=pos_)

        return y

    def _kernel(
        self,
        x_values: NDArray,
        amplitudes: NDArray,
        centers: NDArray,
    ) -> NDArray:
        """
        Evaluate the user kernel at each center without truncation.

        Parameters
        ----------
        x_values : NDArray
            Input x-values with shape ``(1, 1, M)``.
        amplitudes : NDArray
            Pulse amplitudes with shape ``(n_samples, max_peaks, 1)``.
        centers : NDArray
            Pulse centers with shape ``(n_samples, max_peaks, 1)``.

        Returns
        -------
        NDArray
            Interpolated kernel values with shape ``(n_samples, max_peaks, M)``.

        Notes
        -----
        The stored kernel support is mapped from ``[0, 1]`` to a physical width
        inferred from the spacing of ``x_values`` and the kernel sample count.
        """
        n_samples, max_peaks = amplitudes.shape[0], amplitudes.shape[1]
        M = x_values.shape[-1]

        # True kernel support in x coordinates
        # Kernel width equals the median dx times kernel length
        # This preserves your recovered kernel exactly
        x_grid = x_values[0, 0, :]
        dx = float(np.median(np.diff(x_grid)))
        kernel_width = dx * self.kernel.size

        # Kernel coordinate in real x-space
        kernel_support_x = np.linspace(
            -0.5 * kernel_width, 0.5 * kernel_width, self.kernel.size
        )

        # Output
        y = np.zeros((n_samples, max_peaks, M), dtype=float)

        for i in range(n_samples):
            for j in range(max_peaks):

                A = amplitudes[i, j, 0]
                x0 = centers[i, j, 0]

                # Inactive peaks
                if np.isnan(A) or np.isnan(x0):
                    y[i, j, :] = np.nan
                    continue

                # Shift kernel to center x0
                shifted_support = kernel_support_x + x0

                # Interpolate without truncation
                vals = np.interp(
                    x_grid, shifted_support, self.kernel, left=0.0, right=0.0
                )

                y[i, j, :] = A * vals

        return y
