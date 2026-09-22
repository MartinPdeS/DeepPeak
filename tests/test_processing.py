import numpy as np
import pytest

from DeepPeak.processing import high_pass_filter, low_pass_filter, normalize_signal


def test_frequency_filters_separate_low_and_high_components():
    time = np.arange(256) / 256
    low = np.sin(2 * np.pi * 4 * time)
    high = 0.5 * np.sin(2 * np.pi * 60 * time)
    signal = low + high
    np.testing.assert_allclose(low_pass_filter(signal, 15, 256), low, atol=1e-10)
    np.testing.assert_allclose(high_pass_filter(signal, 15, 256), high, atol=1e-10)


def test_filters_cover_complex_multiaxis_taper_and_padding():
    signal = np.vstack([np.arange(8), np.arange(8)[::-1]]).astype(complex)
    low = low_pass_filter(
        signal,
        2,
        16,
        axis=1,
        transition_width_hz=2,
        pad_to_length=16,
        pad_mode="reflect",
        return_complex=True,
    )
    high = high_pass_filter(
        signal,
        2,
        16,
        axis=1,
        transition_width_hz=2,
        pad_to_length=16,
        return_complex=False,
    )
    assert low.shape == high.shape == signal.shape
    assert np.iscomplexobj(low)
    assert not np.iscomplexobj(high)


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (low_pass_filter, {"adata": np.array([])}),
        (low_pass_filter, {"adata": np.array(["x"])}),
        (low_pass_filter, {"adata": np.ones(4), "sampling_rate": 0}),
        (low_pass_filter, {"adata": np.ones(4), "bandlimit": -1}),
        (low_pass_filter, {"adata": np.ones(4), "bandlimit": 6, "sampling_rate": 10}),
        (low_pass_filter, {"adata": np.ones(4), "transition_width_hz": -1}),
        (low_pass_filter, {"adata": np.ones(4), "pad_to_length": 2}),
        (low_pass_filter, {"adata": np.ones(4), "response_shape": "bad"}),
        (high_pass_filter, {"adata": np.ones(4), "response_shape": "bad"}),
    ],
)
def test_filters_validate_arguments(function, kwargs):
    with pytest.raises((ValueError, TypeError)):
        function(**kwargs)


@pytest.mark.parametrize(
    "mode",
    [
        "none",
        "raw",
        "l1",
        "l2",
        "minmax",
        "min-max",
        "zscore",
        "standard",
        "robust",
        "maxabs",
    ],
)
def test_normalize_signal_modes_are_finite(mode):
    output = normalize_signal(np.array([[1, 2, 4], [3, 3, 3]]), mode)
    assert output.shape == (2, 3)
    assert np.all(np.isfinite(output))


def test_normalize_signal_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalize_signal(np.ones((2, 3)), "mystery")
