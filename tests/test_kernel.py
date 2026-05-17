from __future__ import annotations

import numpy as np
import pytest
import DeepPeak.kernels as kernels_module


@pytest.fixture(scope="module")
def kernel_module():
    return kernels_module


def test_kernel_repr_includes_constructor_fields(kernel_module):
    kernel = kernel_module.Gaussian(amplitude=(1.0, 2.0), position=0.5, width=0.1)

    assert repr(kernel) == "Gaussian(amplitude=(1.0, 2.0), position=0.5, width=0.1)"


@pytest.mark.parametrize("Kernel", ["Gaussian", "Lorentzian", "Square"])
def test_width_based_kernels_reject_non_positive_width(kernel_module, Kernel):
    kernel_type = getattr(kernel_module, Kernel)

    with pytest.raises(ValueError, match="width values must be > 0.0"):
        kernel_type(amplitude=1.0, position=0.5, width=0.0)


def test_get_kwargs_requires_evaluate_first(kernel_module):
    kernel = kernel_module.Dirac(amplitude=1.0, position=0.5)

    with pytest.raises(AttributeError, match=r"call evaluate\(\) first"):
        kernel.get_kwargs()


def test_custom_kernel_rejects_empty_shape(kernel_module):
    with pytest.raises(ValueError, match="at least one sample"):
        kernel_module.CustomKernel(kernel=np.array([]), amplitude=1.0, position=0.5)


def test_evaluate_rejects_non_vector_x_values(kernel_module):
    kernel = kernel_module.Gaussian(amplitude=1.0, position=0.5, width=0.1)

    with pytest.raises(ValueError, match="one-dimensional array"):
        kernel.evaluate(np.zeros((2, 2)), n_samples=1, n_peaks=(1, 1))
