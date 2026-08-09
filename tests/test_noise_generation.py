import numpy as np
import pytest

from DeepPeak import (
    GaussianNoise,
    LaplaceNoise,
    NonstationaryGaussianNoise,
    SignalGenerator,
)
from DeepPeak.generation import Gaussian, UniformCount


def _collapse(signals: np.ndarray) -> np.ndarray:
    return np.sum(signals, axis=1) if signals.ndim == 3 else signals


def test_generator_accepts_gaussian_noise_object():
    np.random.seed(123)
    generator = SignalGenerator(sequence_length=64)
    kernel = Gaussian(amplitude=(3.0, 3.0), position=(20.0, 20.0), width=3.0)

    clean = generator.generate(
        n_samples=6,
        kernel=kernel,
        peak_count=UniformCount(bounds=(1, 1)),
        noise_std=0.0,
    )
    np.random.seed(123)
    noisy = generator.generate(
        n_samples=6,
        kernel=kernel,
        peak_count=UniformCount(bounds=(1, 1)),
        noise=GaussianNoise(std=0.2),
    )

    clean_signals = _collapse(np.asarray(clean.signals))
    noisy_signals = _collapse(np.asarray(noisy.signals))
    assert np.mean(np.var(noisy_signals, axis=1)) > np.mean(
        np.var(clean_signals, axis=1)
    )


def test_generator_accepts_laplace_noise_object():
    np.random.seed(321)
    generator = SignalGenerator(sequence_length=64)
    kernel = Gaussian(amplitude=(0.0, 0.0), position=(20.0, 20.0), width=3.0)

    dataset = generator.generate(
        n_samples=10,
        kernel=kernel,
        peak_count=UniformCount(bounds=(0, 0)),
        noise=LaplaceNoise(scale=0.1),
    )

    signals = _collapse(np.asarray(dataset.signals))
    assert signals.shape == (10, 64)
    assert np.any(np.abs(signals) > 0.0)


def test_generator_rejects_noise_and_noise_std_together():
    generator = SignalGenerator(sequence_length=32)
    kernel = Gaussian(amplitude=(1.0, 1.0), position=(10.0, 10.0), width=2.0)

    with pytest.raises(ValueError, match="mutually exclusive"):
        generator.generate(
            n_samples=2,
            kernel=kernel,
            peak_count=UniformCount(bounds=(1, 1)),
            noise_std=0.1,
            noise=GaussianNoise(std=0.1),
        )


def test_nonstationary_gaussian_noise_increases_variance_across_trace():
    noise = NonstationaryGaussianNoise(std=0.1, end_scale=3.0)
    samples = noise.sample((2000, 128), rng=np.random.default_rng(10))

    assert np.mean(np.var(samples[:, -16:], axis=1)) > 5.0 * np.mean(
        np.var(samples[:, :16], axis=1)
    )


def test_generator_applies_instrument_response_baseline_saturation_and_quantization():
    generator = SignalGenerator(sequence_length=64)
    kernel = Gaussian(amplitude=(10.0, 10.0), position=(32.0, 32.0), width=2.0)
    dataset = generator.generate(
        n_samples=2,
        kernel=kernel,
        peak_count=UniformCount(bounds=(1, 1)),
        baseline_level=0.25,
        instrument_response=np.array([0.25, 0.5, 0.25]),
        saturation=1.0,
        quantization_step=0.25,
    )

    assert np.max(dataset.signals) <= 1.0
    assert np.allclose(dataset.signals / 0.25, np.round(dataset.signals / 0.25))
    assert not np.allclose(dataset.signals, dataset.clean_signals)


def test_generator_rejects_invalid_realism_options():
    generator = SignalGenerator(sequence_length=32)
    kernel = Gaussian(amplitude=(1.0, 1.0), position=(10.0, 10.0), width=2.0)

    with pytest.raises(ValueError, match="instrument_response"):
        generator.generate(
            n_samples=1,
            kernel=kernel,
            peak_count=UniformCount(bounds=(1, 1)),
            instrument_response=np.array([]),
        )
    with pytest.raises(ValueError, match="quantization_step"):
        generator.generate(
            n_samples=1,
            kernel=kernel,
            peak_count=UniformCount(bounds=(1, 1)),
            quantization_step=0.0,
        )
