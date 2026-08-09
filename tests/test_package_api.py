import DeepPeak
from DeepPeak import models


def test_top_level_public_api_exports_core_types():
    assert DeepPeak.DataSet.__name__ == "DataSet"
    assert DeepPeak.StandardDilutionSeries.__name__ == "StandardDilutionSeries"
    assert DeepPeak.FlashDilutionSeries.__name__ == "FlashDilutionSeries"
    assert DeepPeak.SignalGenerator.__name__ == "SignalGenerator"
    assert DeepPeak.Gaussian.__name__ == "Gaussian"
    assert DeepPeak.TwoLobeGaussian.__name__ == "TwoLobeGaussian"
    assert DeepPeak.WaveNetTraceAnalyzer.__name__ == "WaveNetTraceAnalyzer"


def test_top_level_public_api_drops_compatibility_series_exports():
    assert not hasattr(DeepPeak, "_BaseDilutionSeries")
    assert not hasattr(DeepPeak, "DilutionSeries")
    assert not hasattr(DeepPeak, "PeakCountSeries")
    assert not hasattr(DeepPeak, "SignalDatasetGenerator")


def test_generation_package_exposes_domain_modules():
    from DeepPeak.generation import DataSet
    from DeepPeak.generation import PeakCount
    from DeepPeak.generation import SignalGenerator
    from DeepPeak.generation import Gaussian
    from DeepPeak.generation import GaussianNoise

    assert DataSet.__name__ == "DataSet"
    assert PeakCount.__name__ == "PeakCount"
    assert SignalGenerator.__name__ == "SignalGenerator"
    assert Gaussian.__name__ == "Gaussian"
    assert GaussianNoise.__name__ == "GaussianNoise"


def test_neural_network_package_declares_lazy_public_api():
    assert models.__all__ == [
        "BinaryIoU",
        "DenseNet",
        "ShapeAwarePulseLoss",
        "SmoothBinaryCrossentropy",
        "UNet1D",
        "WaveNet",
        "WeightedBinaryCrossentropy",
        "WeightedHuber",
        "plot_predictions",
        "shape_aware_pulse_loss",
        "smooth_bce",
        "weighted_bce",
        "weighted_huber",
    ]


def test_top_level_api_declares_lazy_ml_exports():
    assert "WaveNet" in DeepPeak.__all__
    assert "UNet1D" in DeepPeak.__all__
    assert "DenseNet" in DeepPeak.__all__
    assert "ShapeAwarePulseLoss" in DeepPeak.__all__
    assert "SmoothBinaryCrossentropy" in DeepPeak.__all__
    assert "WeightedBinaryCrossentropy" in DeepPeak.__all__
    assert "WeightedHuber" in DeepPeak.__all__
    assert "shape_aware_pulse_loss" in DeepPeak.__all__
    assert "smooth_bce" in DeepPeak.__all__
    assert "weighted_bce" in DeepPeak.__all__
    assert "weighted_huber" in DeepPeak.__all__
