from __future__ import annotations

from pathlib import Path

import pytest

from object_detector_trainer.config import load_config
from object_detector_trainer.tests.helpers import minimal_params, write_params


def test_minimal_config_loads_with_defaults(tmp_path: Path) -> None:
    config = load_config(write_params(tmp_path / "params.yaml", minimal_params()))

    assert config.seed == 42
    assert config.prepare.val_split == 0.1
    assert config.train.device == "auto"
    assert config.evaluation.baseline_weights_path is None
    options = config.model_options("yolo-small")
    assert options.asset_id == "yolov8n.pt"
    assert options.freeze_backbone is False


def test_typo_in_model_entry_names_the_model_and_field(tmp_path: Path) -> None:
    params = minimal_params(models={"yolo-small": {"backend": "yolo", "asset_id": "yolov8n.pt", "epoch": 5}})

    with pytest.raises(ValueError, match=r"models\.yolo-small(.|\n)*epoch"):
        load_config(write_params(tmp_path / "params.yaml", params))


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    params = minimal_params(models={"yolo-small": {"backend": "rtmdet", "asset_id": "x.pth"}})

    with pytest.raises(ValueError, match="Unknown backend 'rtmdet'"):
        load_config(write_params(tmp_path / "params.yaml", params))


def test_unknown_section_field_is_rejected(tmp_path: Path) -> None:
    params = minimal_params(prepare={"unknown_option": True})

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(write_params(tmp_path / "params.yaml", params))


def test_classes_must_be_configured(tmp_path: Path) -> None:
    params = minimal_params(data={"dataset_name": "sample"})

    with pytest.raises(ValueError, match="custom_classes"):
        load_config(write_params(tmp_path / "params.yaml", params))


def test_selected_model_must_exist(tmp_path: Path) -> None:
    params = minimal_params(train={"model": "missing"})

    with pytest.raises(ValueError, match="train.model 'missing'"):
        load_config(write_params(tmp_path / "params.yaml", params))


@pytest.mark.parametrize("field", ["image_size", "epochs"])
def test_training_values_must_be_positive(tmp_path: Path, field: str) -> None:
    params = minimal_params(train={"model": "yolo-small", field: 0})

    with pytest.raises(ValueError, match="greater than 0"):
        load_config(write_params(tmp_path / "params.yaml", params))


def test_model_defaults_apply_and_model_values_win(tmp_path: Path) -> None:
    params = minimal_params(
        models_defaults={"yolo": {"cache_dir": "cache/yolo", "epochs": 10}},
        models={"yolo-small": {"backend": "yolo", "asset_id": "yolov8n.pt", "epochs": 3}},
    )
    config = load_config(write_params(tmp_path / "params.yaml", params))

    options = config.model_options("yolo-small")
    assert options.cache_dir == "cache/yolo"
    assert options.epochs == 3


def test_overrides_are_applied_after_defaults(tmp_path: Path) -> None:
    params = minimal_params(models_defaults={"yolo": {"epochs": 10}})
    overrides = ["models.yolo-small.epochs=4", "models.yolo-small.batch_size=2", "seed=7"]
    config = load_config(write_params(tmp_path / "params.yaml", params), overrides)

    assert config.model_options("yolo-small").epochs == 4
    assert config.model_options("yolo-small").batch_size == 2
    assert config.seed == 7


def test_rfdetr_requires_variant(tmp_path: Path) -> None:
    params = minimal_params(
        train={"model": "detr"},
        models={"detr": {"backend": "rfdetr", "asset_id": "rf-detr-nano.pth"}},
    )

    with pytest.raises(ValueError, match="variant"):
        load_config(write_params(tmp_path / "params.yaml", params))


def test_rfdetr_resolution_sets_image_size(tmp_path: Path) -> None:
    params = minimal_params(
        train={"model": "detr"},
        models={"detr": {"backend": "rfdetr", "variant": "nano", "asset_id": "rf-detr-nano.pth", "resolution": 384}},
    )
    config = load_config(write_params(tmp_path / "params.yaml", params))

    assert config.model_options("detr").image_size == 384


def test_ignored_class_cannot_also_be_trained(tmp_path: Path) -> None:
    data = {"dataset_name": "sample", "custom_classes": ["waste"], "ignore_classes": ["Waste"]}
    with pytest.raises(ValueError, match="ignore_classes lists configured classes"):
        load_config(write_params(tmp_path / "params.yaml", minimal_params(data=data)))


@pytest.mark.parametrize("classes", [["waste", "waste"], ["Waste", "waste"]])
def test_repeated_class_names_are_rejected(tmp_path: Path, classes: list[str]) -> None:
    params = minimal_params(data={"dataset_name": "sample", "custom_classes": classes})
    with pytest.raises(ValueError, match="more than once"):
        load_config(write_params(tmp_path / "params.yaml", params))


@pytest.mark.parametrize("backend", ["yolo", "rfdetr"])
@pytest.mark.parametrize("field", ["batch_size", "effective_batch_size"])
def test_model_batch_settings_must_be_positive(tmp_path, backend, field):
    model = {"backend": backend, "asset_id": "unused.pt", field: 0}
    if backend == "rfdetr":
        model["variant"] = "nano"
    params = minimal_params(models={"yolo-small": model})
    with pytest.raises(ValueError, match="greater than 0"):
        load_config(write_params(tmp_path / "params.yaml", params))


@pytest.mark.parametrize("backend,key", [
    ("yolo", "nbs"), ("yolo", "batch"), ("yolo", "epochs"),
    ("rfdetr", "grad_accum_steps"), ("rfdetr", "checkpoint_interval"),
    ("rfdetr", "device"), ("rfdetr", "accelerator"), ("rfdetr", "devices"),
])
def test_extra_train_kwargs_cannot_set_trainer_owned_arguments_when_loading(tmp_path, backend, key):
    model = {"backend": backend, "asset_id": "unused.pt", "extra_train_kwargs": {key: 4, "patience": 5}}
    if backend == "rfdetr":
        model["variant"] = "nano"
    params = minimal_params(models={"yolo-small": model})
    with pytest.raises(ValueError, match=rf"models\.yolo-small\.extra_train_kwargs cannot set \['{key}'\]"):
        load_config(write_params(tmp_path / "params.yaml", params))
