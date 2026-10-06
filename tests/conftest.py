"""Shared pytest configuration for non-interactive plotting tests."""

import matplotlib
import pytest


# Tests should render figures in memory and never open GUI windows.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close figures created by each test, including after failed tests."""

    plt.close("all")
    yield
    plt.close("all")
