import DeepPeak
from DeepPeak.machine_learning import classifier


def test_top_level_public_api_exports_core_types():
    assert DeepPeak.DataSet.__name__ == "DataSet"
    assert DeepPeak.DilutionSeries.__name__ == "DilutionSeries"
    assert DeepPeak.SignalGenerator.__name__ == "SignalGenerator"
    assert DeepPeak.Gaussian.__name__ == "Gaussian"
    assert DeepPeak.TwoLobeGaussian.__name__ == "TwoLobeGaussian"
    assert DeepPeak.PeakCountSeries.__name__ == "PeakCountSeries"
    assert DeepPeak.WaveNetTraceAnalyzer.__name__ == "WaveNetTraceAnalyzer"


def test_classifier_package_declares_lazy_public_api():
    assert classifier.__all__ == [
        "Autoencoder",
        "BinaryIoU",
        "DenseNet",
        "ShapeAwarePulseLoss",
        "WaveNet",
        "WeightedBinaryCrossentropy",
        "WeightedHuber",
        "plot_predictions",
        "shape_aware_pulse_loss",
        "weighted_bce",
        "weighted_huber",
    ]


def test_top_level_api_declares_lazy_ml_exports():
    assert "WaveNet" in DeepPeak.__all__
    assert "DenseNet" in DeepPeak.__all__
    assert "Autoencoder" in DeepPeak.__all__
    assert "ShapeAwarePulseLoss" in DeepPeak.__all__
    assert "WeightedBinaryCrossentropy" in DeepPeak.__all__
    assert "WeightedHuber" in DeepPeak.__all__
    assert "shape_aware_pulse_loss" in DeepPeak.__all__
    assert "weighted_bce" in DeepPeak.__all__
    assert "weighted_huber" in DeepPeak.__all__
