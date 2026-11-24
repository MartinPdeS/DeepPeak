from DeepPeak.core import AnalysisConfig, GenerationConfig, ModelConfig, NoiseConfig


def test_experiment_configs_round_trip_through_json():
    configs = [
        AnalysisConfig(detector="standard", dx=0.5, dilution=0.5, concentration=1.0),
        GenerationConfig(
            sequence_length=64,
            noise_profile="linear",
            noise_end_scale=(1.0, 2.0),
            instrument_response=(0.2, 0.6, 0.2),
            missing_peak_probability=0.1,
        ),
        ModelConfig(architecture="wavenet", sequence_length=64),
        NoiseConfig(kind="correlated_gaussian", scale=0.1, correlation_length=4.0),
    ]

    for config in configs:
        restored = type(config).from_json(config.to_json())
        assert restored.to_dict() == config.to_dict()


def test_noise_config_builds_a_noise_object():
    noise = NoiseConfig(
        kind="nonstationary_gaussian",
        scale=0.1,
        end_scale=2.0,
    ).build()
    assert noise.sample((2, 8)).shape == (2, 8)
