from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray


from .base import BaseKernel
from ..peak_count import PeakCount


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
    center_shift: Union[float, Tuple[float, float]] = 0.0
    width_scale: Union[float, Tuple[float, float]] = 1.0
    left_width_scale: Union[float, Tuple[float, float]] = 1.0
    right_width_scale: Union[float, Tuple[float, float]] = 1.0

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        amplitude: float,
        position: float,
        *,
        center_shift: Union[float, Tuple[float, float]] = 0.0,
        width_scale: Union[float, Tuple[float, float]] = 1.0,
        left_width_scale: Union[float, Tuple[float, float]] = 1.0,
        right_width_scale: Union[float, Tuple[float, float]] = 1.0,
        npz_key: Optional[str] = None,
        csv_column: Union[int, str] = 0,
        csv_kwargs: Optional[dict] = None,
    ) -> "CustomKernel":
        """
        Load a kernel from a ``.npy``, ``.npz``, or ``.csv`` file.

        Parameters
        ----------
        path : str or Path
            Path to the kernel file. The format is inferred from the suffix
            (``.npy``, ``.npz``, or ``.csv`` / ``.txt``).
        amplitude : float or tuple[float, float]
            Passed directly to :class:`CustomKernel`.
        position : float or tuple[float, float]
            Passed directly to :class:`CustomKernel`.
        center_shift : float or tuple[float, float], default 0.0
            Offset applied to the kernel center relative to ``position``, in the
            same units as ``x_values``. Use a range to sample randomly per peak.
        width_scale : float or tuple[float, float], default 1.0
            Scale factor applied to the kernel width. ``1.0`` keeps the original
            width; ``2.0`` stretches it; ``0.5`` compresses it. Use a range to
            sample randomly per peak.
        left_width_scale : float or tuple[float, float], default 1.0
            Extra width scale applied only to the left half of the kernel.
            Values other than ``1.0`` introduce controlled asymmetry.
        right_width_scale : float or tuple[float, float], default 1.0
            Extra width scale applied only to the right half of the kernel.
            Values other than ``1.0`` introduce controlled asymmetry.
        npz_key : str or None, optional
            Key to extract from an ``.npz`` archive. If *None*, the first key
            in the archive is used.
        csv_column : int or str, default 0
            Column index (int) or header name (str) to use when loading a CSV.
            Ignored for ``.npy`` / ``.npz`` files.
        csv_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`numpy.genfromtxt` when
            reading a CSV.

        Returns
        -------
        CustomKernel
            A new instance with the loaded kernel array.

        Raises
        ------
        ValueError
            If the file extension is not recognised or the loaded array is not
            one-dimensional.
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Kernel file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".npy":
            kernel = np.load(path)

        elif suffix == ".npz":
            archive = np.load(path)
            key = npz_key if npz_key is not None else next(iter(archive))
            kernel = archive[key]

        elif suffix in {".csv", ".txt"}:
            kw = dict(delimiter=",")
            if csv_kwargs:
                kw.update(csv_kwargs)
            raw = np.genfromtxt(path, **kw)
            if raw.ndim == 1:
                kernel = raw
            else:
                if isinstance(csv_column, str):
                    names = np.genfromtxt(
                        path, delimiter=",", names=True, max_rows=1
                    ).dtype.names
                    col_idx = list(names).index(csv_column)
                else:
                    col_idx = int(csv_column)
                kernel = raw[:, col_idx]

        else:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. "
                "Expected one of: .npy, .npz, .csv, .txt"
            )

        kernel = np.asarray(kernel, dtype=float).ravel()
        return cls(
            kernel=kernel,
            amplitude=amplitude,
            position=position,
            center_shift=center_shift,
            width_scale=width_scale,
            left_width_scale=left_width_scale,
            right_width_scale=right_width_scale,
        )

    def __post_init__(self):
        self.kernel = np.asarray(self.kernel, dtype=float)
        if self.kernel.ndim != 1:
            raise ValueError("kernel must be a one dimensional array")
        if self.kernel.size == 0:
            raise ValueError("kernel must contain at least one sample")

        self._initialize_common_ranges(has_width=False)
        self._center_shift = self._normalize_range("center_shift", self.center_shift)
        self._width_scale = self._normalize_range(
            "width_scale", self.width_scale, minimum=0.0, inclusive_minimum=False
        )
        self._left_width_scale = self._normalize_range(
            "left_width_scale",
            self.left_width_scale,
            minimum=0.0,
            inclusive_minimum=False,
        )
        self._right_width_scale = self._normalize_range(
            "right_width_scale",
            self.right_width_scale,
            minimum=0.0,
            inclusive_minimum=False,
        )

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
        rng: np.random.Generator | None = None,
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
        rng = np.random.default_rng() if rng is None else rng
        x_, amp_, pos_, _, active_mask, max_peaks = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            rng=rng,
            has_width=False,
        )

        shape = (n_samples, max_peaks)
        self.center_shifts = self._sample_uniform(
            self._center_shift, size=shape, rng=rng
        )
        self.width_scales = self._sample_uniform(self._width_scale, size=shape, rng=rng)
        self.left_width_scales = self._sample_uniform(
            self._left_width_scale, size=shape, rng=rng
        )
        self.right_width_scales = self._sample_uniform(
            self._right_width_scale, size=shape, rng=rng
        )
        self.center_shifts[~active_mask] = np.nan
        self.width_scales[~active_mask] = np.nan
        self.left_width_scales[~active_mask] = np.nan
        self.right_width_scales[~active_mask] = np.nan

        y = self._kernel(
            x_values=x_,
            amplitudes=amp_,
            centers=pos_,
            center_shifts=self.center_shifts[..., np.newaxis],
            width_scales=self.width_scales[..., np.newaxis],
            left_width_scales=self.left_width_scales[..., np.newaxis],
            right_width_scales=self.right_width_scales[..., np.newaxis],
        )

        return y

    def _kernel(
        self,
        x_values: NDArray,
        amplitudes: NDArray,
        centers: NDArray,
        center_shifts: NDArray,
        width_scales: NDArray,
        left_width_scales: NDArray,
        right_width_scales: NDArray,
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
        center_shifts : NDArray
            Per-peak offset applied to the kernel center, same units as
            ``x_values``. Shape ``(n_samples, max_peaks, 1)``.
        width_scales : NDArray
            Per-peak scale factor applied to the kernel width.
            ``1.0`` keeps the original width, ``2.0`` doubles it (stretches),
            ``0.5`` halves it (compresses). Shape ``(n_samples, max_peaks, 1)``.
        left_width_scales : NDArray
            Extra scale factor applied only to the left half of the kernel.
            Shape ``(n_samples, max_peaks, 1)``.
        right_width_scales : NDArray
            Extra scale factor applied only to the right half of the kernel.
            Shape ``(n_samples, max_peaks, 1)``.

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

        x_grid = x_values[0, 0, :]
        dx = float(np.median(np.diff(x_grid)))
        base_kernel_width = dx * self.kernel.size

        y = np.zeros((n_samples, max_peaks, M), dtype=float)

        for i in range(n_samples):
            for j in range(max_peaks):

                A = amplitudes[i, j, 0]
                x0 = centers[i, j, 0]
                shift = center_shifts[i, j, 0]
                scale = width_scales[i, j, 0]
                left_scale = left_width_scales[i, j, 0]
                right_scale = right_width_scales[i, j, 0]

                if np.isnan(A) or np.isnan(x0):
                    y[i, j, :] = np.nan
                    continue

                normalized_support = np.linspace(-0.5, 0.5, self.kernel.size)
                kernel_support_x = np.empty(self.kernel.size, dtype=float)
                left_mask = normalized_support < 0.0
                right_mask = normalized_support > 0.0
                kernel_support_x[left_mask] = (
                    normalized_support[left_mask]
                    * base_kernel_width
                    * scale
                    * left_scale
                )
                kernel_support_x[right_mask] = (
                    normalized_support[right_mask]
                    * base_kernel_width
                    * scale
                    * right_scale
                )
                kernel_support_x[~(left_mask | right_mask)] = 0.0

                # Apply center shift then place at x0
                shifted_support = kernel_support_x + x0 + shift

                vals = np.interp(
                    x_grid, shifted_support, self.kernel, left=0.0, right=0.0
                )

                y[i, j, :] = A * vals

        return y


@dataclass(repr=False)
class CustomKernels(BaseKernel):
    """
    Pulse model using a **library** of sampled kernel shapes.

    At each peak a kernel is drawn at random from the library.  Different peaks
    in the same window (and across windows) can therefore have different shapes,
    improving the diversity of the synthetic training set.

    Parameters
    ----------
    kernel_set : ndarray of shape ``(N, K)``
        Library of N kernels, each K samples long.  All kernels must share the
        same length K.
    amplitude : float or tuple[float, float]
        Peak amplitude or inclusive sampling range.
    position : float or tuple[float, float]
        Center position or inclusive sampling range.
    center_shift : float or tuple[float, float], default 0.0
        Offset applied to the kernel center, in the same units as ``x_values``.
    width_scale : float or tuple[float, float], default 1.0
        Global width scale applied symmetrically to both halves.
    left_width_scale : float or tuple[float, float], default 1.0
        Extra scale applied only to the left half of the kernel.
    right_width_scale : float or tuple[float, float], default 1.0
        Extra scale applied only to the right half of the kernel.
    """

    kernel_set: NDArray
    amplitude: float
    position: float
    center_shift: Union[float, Tuple[float, float]] = 0.0
    width_scale: Union[float, Tuple[float, float]] = 1.0
    left_width_scale: Union[float, Tuple[float, float]] = 1.0
    right_width_scale: Union[float, Tuple[float, float]] = 1.0

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        amplitude: float,
        position: float,
        *,
        center_shift: Union[float, Tuple[float, float]] = 0.0,
        width_scale: Union[float, Tuple[float, float]] = 1.0,
        left_width_scale: Union[float, Tuple[float, float]] = 1.0,
        right_width_scale: Union[float, Tuple[float, float]] = 1.0,
        npz_key: Optional[str] = None,
    ) -> "CustomKernels":
        """
        Load a kernel library from a ``.npy`` or ``.npz`` file.

        The file must contain a 2-D array of shape ``(N, K)``.  A 1-D array is
        also accepted and is treated as a library with a single kernel.

        Parameters
        ----------
        path : str or Path
            Path to the file.
        npz_key : str or None, optional
            Key to extract from an ``.npz`` archive.  Defaults to the first key.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Kernel file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".npy":
            kernel_set = np.load(path)
        elif suffix == ".npz":
            archive = np.load(path)
            key = npz_key if npz_key is not None else next(iter(archive))
            kernel_set = archive[key]
        else:
            raise ValueError(f"Unsupported extension {suffix!r}. Expected .npy or .npz")

        return cls(
            kernel_set=kernel_set,
            amplitude=amplitude,
            position=position,
            center_shift=center_shift,
            width_scale=width_scale,
            left_width_scale=left_width_scale,
            right_width_scale=right_width_scale,
        )

    def __post_init__(self):
        self.kernel_set = np.asarray(self.kernel_set, dtype=float)
        if self.kernel_set.ndim == 1:
            self.kernel_set = self.kernel_set[np.newaxis, :]
        if self.kernel_set.ndim != 2:
            raise ValueError("kernel_set must be a 1-D or 2-D array")
        if self.kernel_set.shape[1] == 0:
            raise ValueError("Kernels must contain at least one sample")

        self._n_kernels = self.kernel_set.shape[0]

        self._initialize_common_ranges(has_width=False)
        self._center_shift = self._normalize_range("center_shift", self.center_shift)
        self._width_scale = self._normalize_range(
            "width_scale", self.width_scale, minimum=0.0, inclusive_minimum=False
        )
        self._left_width_scale = self._normalize_range(
            "left_width_scale",
            self.left_width_scale,
            minimum=0.0,
            inclusive_minimum=False,
        )
        self._right_width_scale = self._normalize_range(
            "right_width_scale",
            self.right_width_scale,
            minimum=0.0,
            inclusive_minimum=False,
        )

    def get_kwargs(self) -> dict:
        return self._state_dict("kernel_set", "amplitudes", "positions")

    def evaluate(
        self,
        x_values: NDArray,
        n_samples: int,
        n_peaks: tuple,
        categorical_peak_count: bool = False,
        peak_count: PeakCount | None = None,
        peak_count_distribution: str = "uniform",
        peak_count_rate: Optional[float] = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray:
        rng = np.random.default_rng() if rng is None else rng
        x_, amp_, pos_, _, active_mask, max_peaks = self._prepare_common_state(
            x_values=x_values,
            n_samples=n_samples,
            n_peaks=n_peaks,
            categorical_peak_count=categorical_peak_count,
            peak_count=peak_count,
            peak_count_distribution=peak_count_distribution,
            peak_count_rate=peak_count_rate,
            rng=rng,
            has_width=False,
        )

        shape = (n_samples, max_peaks)
        self.center_shifts = self._sample_uniform(
            self._center_shift, size=shape, rng=rng
        )
        self.width_scales = self._sample_uniform(self._width_scale, size=shape, rng=rng)
        self.left_width_scales = self._sample_uniform(
            self._left_width_scale, size=shape, rng=rng
        )
        self.right_width_scales = self._sample_uniform(
            self._right_width_scale, size=shape, rng=rng
        )
        self.center_shifts[~active_mask] = np.nan
        self.width_scales[~active_mask] = np.nan
        self.left_width_scales[~active_mask] = np.nan
        self.right_width_scales[~active_mask] = np.nan

        # Draw one random kernel index per peak; inactive peaks are ignored later
        kernel_indices = rng.integers(0, self._n_kernels, size=shape)
        kernel_indices[~active_mask] = 0

        y = self._kernel(
            x_values=x_,
            amplitudes=amp_,
            centers=pos_,
            center_shifts=self.center_shifts[..., np.newaxis],
            width_scales=self.width_scales[..., np.newaxis],
            left_width_scales=self.left_width_scales[..., np.newaxis],
            right_width_scales=self.right_width_scales[..., np.newaxis],
            kernel_indices=kernel_indices,
        )

        return y

    def _kernel(
        self,
        x_values: NDArray,
        amplitudes: NDArray,
        centers: NDArray,
        center_shifts: NDArray,
        width_scales: NDArray,
        left_width_scales: NDArray,
        right_width_scales: NDArray,
        kernel_indices: NDArray,
    ) -> NDArray:
        n_samples, max_peaks = amplitudes.shape[0], amplitudes.shape[1]
        M = x_values.shape[-1]
        x_grid = x_values[0, 0, :]
        dx = float(np.median(np.diff(x_grid)))

        K = self.kernel_set.shape[1]
        base_kernel_width = dx * K
        normalized_support = np.linspace(-0.5, 0.5, K)
        left_mask = normalized_support < 0.0
        right_mask = normalized_support > 0.0

        y = np.zeros((n_samples, max_peaks, M), dtype=float)

        for i in range(n_samples):
            for j in range(max_peaks):

                A = amplitudes[i, j, 0]
                x0 = centers[i, j, 0]

                if np.isnan(A) or np.isnan(x0):
                    y[i, j, :] = np.nan
                    continue

                shift = center_shifts[i, j, 0]
                scale = width_scales[i, j, 0]
                left_scale = left_width_scales[i, j, 0]
                right_scale = right_width_scales[i, j, 0]
                kernel = self.kernel_set[kernel_indices[i, j]]

                kernel_support_x = np.empty(K, dtype=float)
                kernel_support_x[left_mask] = (
                    normalized_support[left_mask]
                    * base_kernel_width
                    * scale
                    * left_scale
                )
                kernel_support_x[right_mask] = (
                    normalized_support[right_mask]
                    * base_kernel_width
                    * scale
                    * right_scale
                )
                kernel_support_x[~(left_mask | right_mask)] = 0.0

                shifted_support = kernel_support_x + x0 + shift
                vals = np.interp(x_grid, shifted_support, kernel, left=0.0, right=0.0)
                y[i, j, :] = A * vals

        return y
