import pytest

from DeepPeak.models import TrainingConfig


def test_training_config_builds_validation_callbacks_and_checkpoint(tmp_path):
    config = TrainingConfig(
        validation_split=0.2,
        monitor="val_binary_iou",
        min_delta=0.01,
        checkpoint_path=tmp_path / "best.weights.h5",
        verbose=0,
    )

    callbacks = config.callbacks()

    assert [type(callback).__name__ for callback in callbacks] == [
        "EarlyStopping",
        "ReduceLROnPlateau",
        "ModelCheckpoint",
    ]
    assert callbacks[0].mode == "max"
    assert callbacks[0].min_delta == pytest.approx(0.01)


def test_training_config_supports_explicit_validation_data_callbacks():
    config = TrainingConfig(validation_split=0.0)

    callbacks = config.callbacks(validation_available=True)

    assert len(callbacks) == 2
    assert all(
        type(callback).__name__ in {"EarlyStopping", "ReduceLROnPlateau"}
        for callback in callbacks
    )


def test_training_config_does_not_create_val_checkpoint_without_validation(tmp_path):
    config = TrainingConfig(checkpoint_path=tmp_path / "best.weights.h5")

    assert config.callbacks() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_delta": -1.0},
        {"monitor_mode": "invalid"},
        {"checkpoint_path": ""},
    ],
)
def test_training_config_rejects_invalid_improvement_options(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)
