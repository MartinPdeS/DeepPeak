import matplotlib.pyplot as plt
import numpy as np
import pytest

from DeepPeak.detection import CholeskySolver, ClosedFormSolver, NonMaximumSuppression
from DeepPeak.io import CsvTrace


def test_amplitude_solvers_recover_known_values_and_plot():
    centers = np.array([[0.0, 4.0], [1.0, 6.0]])
    amplitudes = np.array([[1.0, 2.0], [0.5, 1.5]])

    closed = ClosedFormSolver(sigma=1.0)
    response = closed._response_matrix_from_centers(centers, 1.0)
    samples = np.einsum("bij,bj->bi", response, amplitudes)
    np.testing.assert_allclose(closed.run(centers, samples), amplitudes)

    cholesky = CholeskySolver(sigma=1.0)
    gram = cholesky._gram_from_centers(centers, 1.0)
    matched = np.einsum("bij,bj->bi", gram, amplitudes)
    np.testing.assert_allclose(cholesky.run(centers, matched), amplitudes)
    assert len(cholesky.plot(amplitudes[0]).axes) == 1
    assert len(cholesky.plot_gram().axes) == 2


def test_solvers_reject_bad_shapes_and_orders():
    with pytest.raises(ValueError, match="shape"):
        CholeskySolver(1)._coerce_to_2d(np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        CholeskySolver(1).run(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="A must"):
        CholeskySolver(1).run(np.ones(4), np.ones(4))
    with pytest.raises(RuntimeError, match="Run the solver"):
        CholeskySolver(1).plot(np.ones(1))


def test_non_maximum_suppression_result_and_plots():
    x = np.arange(128, dtype=float)
    signal = np.exp(-0.5 * ((x - 32) / 3) ** 2) + 0.8 * np.exp(
        -0.5 * ((x - 91) / 3) ** 2
    )
    detector = NonMaximumSuppression(
        gaussian_sigma=3, threshold=0.5, maximum_number_of_pulses=3
    )
    result = detector.run(x, signal)
    np.testing.assert_allclose(result.peak_indices, [32, 91], atol=1)
    assert result.number_of_peaks == 2
    assert result.summary()["K_detected"] == 2
    assert result.to_dict()["signal"] is signal
    assert len(result.plot(show_kernel=True).axes) == 1
    assert detector.full_width_half_maximum_to_sigma(2.35482) == pytest.approx(
        1.0, rel=1e-4
    )


def test_csv_trace_load_process_detect_and_plot(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(
        "Time,Channel A\nms,V\n0,0\n1,0\n2,1\n3,0\n4,∞\n5,0\n",
        encoding="utf-8",
    )
    trace = CsvTrace(path)
    assert trace.dx == pytest.approx(0.001)
    assert trace.delta_x == pytest.approx(0.005)
    assert trace.sampling_rate == pytest.approx(1000)
    assert np.isfinite(trace.robust_sigma_from_diff())
    assert trace.get_height_based_on_noise(3) >= 0
    trace.remove_dc()
    trace.low_pass_filter(bandlimit=100)
    positions, heights, widths = trace.find_peaks(height=0.1)
    assert positions.shape == heights.shape == widths.shape
    assert len(trace.plot_overview(end_idx=6, nbins=3).axes) == 2
    plt.close("all")


def test_csv_trace_sigma_height_and_row_limit(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("Time,Channel A\nms,V\n0,0\n1,1\n2,0\n3,1\n", encoding="utf-8")
    trace = CsvTrace(path, n_rows=3)
    assert len(trace.x) == 3
    assert trace._process_height("2sigma") == pytest.approx(
        2 * trace.robust_sigma_from_diff()
    )
