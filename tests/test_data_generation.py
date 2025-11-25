# tests/test_data_generation.py
from __future__ import annotations

import numpy as np
import pytest

# Headless plotting not required here; we do not call .plot()

# Import your project pieces (skip the suite if not installed in the env)
DP_alg = pytest.importorskip("DeepPeak.algorithms")
DP_sig = pytest.importorskip("DeepPeak.signals")
DP_ker = pytest.importorskip("DeepPeak.kernel")

# Aliases
SignalDatasetGenerator = DP_sig.SignalDatasetGenerator

# Kernels that should exist in DeepPeak.kernel
Gaussian = getattr(DP_ker, "Gaussian")
Lorentzian = getattr(DP_ker, "Lorentzian")
Square = getattr(DP_ker, "Square")
Dirac = getattr(DP_ker, "Dirac")

# ----------------------
# Global test parameters
# ----------------------
SEED = 20250919
N = 400  # sequence length
DT = 1.0 / N


def grid01(n: int) -> np.ndarray:
    """Uniform grid [0, 1) with n samples."""
    return np.linspace(0.0, 1.0, n, endpoint=False)


# ----------
# Utilities
# ----------
def _active_mask_from_cube(Y: np.ndarray) -> np.ndarray:
    """
    Given an array shaped (n_samples, max_peaks, N), return boolean mask
    of shape (n_samples, max_peaks) marking which peaks are active
    (i.e., not all-NaN across time).
    """
    assert Y.ndim == 3
    return ~np.all(np.isnan(Y), axis=-1)


def _assert_in_range(arr: np.ndarray, lo: float, hi: float) -> None:
    arr = np.asarray(arr, dtype=float)
    # Only check finite (skip NaNs from inactive peaks)
    m = np.isfinite(arr)
    if np.any(m):
        assert np.nanmin(arr[m]) >= lo - 1e-12
        assert np.nanmax(arr[m]) <= hi + 1e-12


def _sum_components(Y: np.ndarray) -> np.ndarray:
    """
    Collapse (n_samples, max_peaks, N) to (n_samples, N)
    by summing across peaks, ignoring NaNs.
    """
    return np.nansum(Y, axis=1)


# ==========================================================
#                  KERNELS — SHAPE & RANGE
# ==========================================================
@pytest.mark.parametrize(
    "Kernel, has_width",
    [
        (Gaussian, True),
        (Lorentzian, True),  # width is gamma (HWHM)
        (Square, True),
    ],
)
def test_kernel_shape_mask_and_ranges(Kernel, has_width):
    np.random.seed(SEED)
    x = grid01(N)
    n_samples = 6
    n_peaks = (1, 3)

    # Parameter ranges
    amp_rng = (10.0, 50.0)
    pos_rng = (0.2, 0.8)
    wid_rng = (0.01, 0.05) if has_width else None

    if has_width:
        kernel = Kernel(amplitude=amp_rng, position=pos_rng, width=wid_rng)
    else:
        kernel = Kernel(amplitude=amp_rng, position=pos_rng)

    Y = kernel.evaluate(
        x_values=x,
        n_samples=n_samples,
        n_peaks=n_peaks,
        categorical_peak_count=False,
    )

    # Expect (n_samples, max_peaks, N)
    assert isinstance(Y, np.ndarray)
    assert Y.shape == (n_samples, n_peaks[1], N)

    # Inactive peaks: full-NaN across time
    active_mask = _active_mask_from_cube(Y)
    active_counts = np.sum(active_mask, axis=1)  # per-sample
    # num_peaks attribute should exist
    assert hasattr(kernel, "num_peaks")
    # If evaluate didn't convert to one-hot, it's 1D ints; both are acceptable
    # We just check that active_counts are within the configured bounds.
    assert np.all((active_counts >= n_peaks[0]) & (active_counts <= n_peaks[1]))

    # The parameter arrays should exist and use NaN to mark inactive peaks
    for name in ("amplitudes", "positions"):
        assert hasattr(kernel, name), f"{Kernel.__name__} missing {name}"
        arr = getattr(kernel, name)
        assert arr.shape[:2] == (n_samples, n_peaks[1])
        # inactive peaks should be NaN in parameter arrays (except *_for_labels)
        assert np.all(np.isnan(arr[~active_mask]))

    if has_width:
        assert hasattr(kernel, "widths")
        assert np.all(np.isnan(kernel.widths[~active_mask]))

    # Values sampled in expected ranges (for active entries)
    _assert_in_range(kernel.amplitudes, *amp_rng)
    _assert_in_range(kernel.positions, *pos_rng)
    if has_width:
        _assert_in_range(kernel.widths, *wid_rng)


def test_kernel_categorical_peak_count_one_hot():
    np.random.seed(SEED)
    x = grid01(N)
    n_samples = 7
    n_peaks = (0, 3)

    ker = Gaussian(amplitude=(1, 2), position=(0.1, 0.9), width=(0.02, 0.03))
    _ = ker.evaluate(
        x_values=x, n_samples=n_samples, n_peaks=n_peaks, categorical_peak_count=True
    )

    assert hasattr(ker, "num_peaks")
    num_peaks = ker.num_peaks
    assert num_peaks.shape == (n_samples, n_peaks[1] + 1)
    # one-hot rows
    row_sums = np.sum(num_peaks, axis=1)
    assert np.allclose(row_sums, 1.0)


# ==========================================================
#        KERNELS — DETERMINISTIC CENTER VALUES/PLACEMENT
# ==========================================================
def test_gaussian_center_value_matches_amplitude():
    """
    If we place the Gaussian center exactly on a grid sample and use a fixed amplitude/width,
    the value at that sample should equal the amplitude (exp(0) = 1).
    """
    x = grid01(N)
    idx = 100
    pos_exact = x[idx]
    amp = 123.0
    sigma = 0.03

    ker = Gaussian(
        amplitude=(amp, amp), position=(pos_exact, pos_exact), width=(sigma, sigma)
    )
    Y = ker.evaluate(
        x_values=x, n_samples=1, n_peaks=(1, 1), categorical_peak_count=False
    )
    y = Y[0, 0]  # (N,)
    assert np.isfinite(y[idx])
    assert y[idx] == pytest.approx(amp, rel=1e-12, abs=1e-12)


def test_lorentzian_center_value_matches_amplitude():
    """
    L(x) = A * γ^2 / ((x - x0)^2 + γ^2); at x = x0 → A.
    """
    x = grid01(N)
    idx = 250
    pos_exact = x[idx]
    amp = 77.0
    gamma = 0.02

    ker = Lorentzian(
        amplitude=(amp, amp), position=(pos_exact, pos_exact), width=(gamma, gamma)
    )
    Y = ker.evaluate(
        x_values=x, n_samples=1, n_peaks=(1, 1), categorical_peak_count=False
    )
    y = Y[0, 0]
    assert y[idx] == pytest.approx(amp, rel=1e-12, abs=1e-12)


def test_square_area_and_edges_inclusive():
    """
    Square pulse S(x) = A on [x0 - w/2, x0 + w/2], inclusive.
    Count of samples inside the interval times A approximates the area.
    """
    x = grid01(N)
    idx = 120
    pos_exact = x[idx]
    amp = 5.0
    width = 0.1  # should cover about width/DT samples

    ker = Square(
        amplitude=(amp, amp), position=(pos_exact, pos_exact), width=(width, width)
    )
    Y = ker.evaluate(
        x_values=x, n_samples=1, n_peaks=(1, 1), categorical_peak_count=False
    )
    y = Y[0, 0]

    left = pos_exact - 0.5 * width
    right = pos_exact + 0.5 * width
    in_rect = (x >= left) & (x <= right)  # inclusive per implementation
    # Values inside should equal A; outside 0
    assert np.allclose(y[in_rect], amp)
    assert np.allclose(y[~in_rect], 0.0)
    # Approximate area check
    area_est = np.sum(y) * DT
    assert area_est == pytest.approx(amp * width, rel=0.02)


def test_dirac_impulse_placed_at_nearest_sample_on_uniform_grid():
    x = grid01(N)
    dt = float(x[1] - x[0])

    # Choose a position halfway between two samples -> rounds to nearest
    idx = 200
    mid_pos = (x[idx] + x[idx + 1]) * 0.5
    amp = 42.0

    ker = Dirac(amplitude=(amp, amp), position=(mid_pos, mid_pos))
    Y = ker.evaluate(
        x_values=x, n_samples=1, n_peaks=(1, 1), categorical_peak_count=False
    )
    y = Y[0, 0]

    # Nearest index calculation should match the implementation
    nearest = int(round((mid_pos - x[0]) / dt))
    assert (
        np.count_nonzero(np.isfinite(y)) == y.size
    )  # Dirac returns zeros elsewhere (not NaNs for active)
    assert np.isclose(y[nearest], amp)
    # Other samples should be 0
    assert np.allclose(np.delete(y, nearest), 0.0)


# ==========================================================
#          GENERATOR — BASIC INTEGRATION & NOISE EFFECT
# ==========================================================
def test_signal_dataset_generator_shapes_and_counts():
    np.random.seed(SEED)
    gen = SignalDatasetGenerator(sequence_length=N)
    kernel = Gaussian(amplitude=(10, 300), position=(0.3, 0.7), width=0.02)

    ds = gen.generate(
        n_samples=10,
        kernel=kernel,
        n_peaks=(3, 3),
        noise_std=0.0,
        categorical_peak_count=False,
    )
    assert ds is not None

    sig = ds.signals
    assert (
        sig is not None
    ), "Dataset should expose signals-like ndarray via a common attribute"

    # Accept either (B, P, N) or (B, N)
    assert sig.ndim in (2, 3)
    assert sig.shape[0] == 10
    if sig.ndim == 3:
        assert sig.shape[-1] == N


def test_generator_noise_increases_variance():
    np.random.seed(SEED)
    gen = SignalDatasetGenerator(sequence_length=N)
    kernel = Gaussian(amplitude=(10, 300), position=(0.3, 0.7), width=0.02)

    ds_clean = gen.generate(
        n_samples=8,
        kernel=kernel,
        n_peaks=(1, 3),
        noise_std=0.0,
        categorical_peak_count=False,
    )
    np.random.seed(SEED)  # reset for fair comparison of underlying pulses
    ds_noisy = gen.generate(
        n_samples=8,
        kernel=kernel,
        n_peaks=(1, 3),
        noise_std=0.1,
        categorical_peak_count=False,
    )

    # Pull signals and collapse to (B, N) if needed
    s_clean = ds_clean.signals
    s_noisy = ds_noisy.signals
    assert s_clean is not None and s_noisy is not None

    if s_clean.ndim == 3:
        s_clean = _sum_components(s_clean)
    if s_noisy.ndim == 3:
        s_noisy = _sum_components(s_noisy)

    # Per-sample variance should on average increase with added noise
    var_clean = np.var(s_clean, axis=1)
    var_noisy = np.var(s_noisy, axis=1)
    # Compare means across batch (tolerant)
    assert np.mean(var_noisy) > np.mean(var_clean)


def test_generator_reproducibility_with_seed():
    """
    Setting numpy's global seed should make successive generate(...) calls reproducible.
    (This assumes generator/kernels use NumPy RNG; if not, adjust your generator to accept a seed.)
    """
    gen = SignalDatasetGenerator(sequence_length=N)
    kernel = Gaussian(amplitude=(5, 5), position=(0.25, 0.75), width=0.02)

    np.random.seed(SEED)
    ds1 = gen.generate(
        n_samples=5,
        kernel=kernel,
        n_peaks=(2, 3),
        noise_std=0.0,
        categorical_peak_count=False,
    )
    s1 = ds1.signals

    np.random.seed(SEED)
    ds2 = gen.generate(
        n_samples=5,
        kernel=kernel,
        n_peaks=(2, 3),
        noise_std=0.0,
        categorical_peak_count=False,
    )
    s2 = ds2.signals

    assert s1 is not None and s2 is not None
    assert s1.shape == s2.shape
    assert np.allclose(s1, s2, equal_nan=True)


def test_generator_time_axis_if_present():
    """
    If the dataset exposes a time axis, verify it is 1D and monotonic increasing
    with the configured sequence length.
    """
    np.random.seed(SEED)
    gen = SignalDatasetGenerator(sequence_length=N)
    kernel = Gaussian(amplitude=(10, 20), position=(0.35, 0.65), width=0.02)

    ds = gen.generate(
        n_samples=30,
        kernel=kernel,
        n_peaks=(1, 1),
        noise_std=0.0,
        categorical_peak_count=False,
    )

    t = ds.x_values
    assert t.ndim == 1 and t.size == N
    dt = np.diff(t)
    assert np.all(dt > 0)
    assert np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12)


if __name__ == "__main__":
    pytest.main(["-W error", __file__])
