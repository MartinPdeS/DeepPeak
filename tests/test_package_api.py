import DeepPeak
from DeepPeak.machine_learning import classifier


def test_top_level_public_api_exports_core_types():
    assert DeepPeak.DataSet.__name__ == "DataSet"
    assert DeepPeak.DilutionSeries.__name__ == "DilutionSeries"
    assert DeepPeak.SignalGenerator.__name__ == "SignalGenerator"
    assert DeepPeak.Gaussian.__name__ == "Gaussian"
    assert DeepPeak.PeakCountSeries.__name__ == "PeakCountSeries"
    assert DeepPeak.WaveNetTraceAnalyzer.__name__ == "WaveNetTraceAnalyzer"


def test_classifier_package_declares_lazy_public_api():
    assert classifier.__all__ == [
        "Autoencoder",
        "BinaryIoU",
        "DenseNet",
        "WaveNet",
        "plot_predictions",
    ]


def test_top_level_api_declares_lazy_ml_exports():
    assert "WaveNet" in DeepPeak.__all__
    assert "DenseNet" in DeepPeak.__all__
    assert "Autoencoder" in DeepPeak.__all__
