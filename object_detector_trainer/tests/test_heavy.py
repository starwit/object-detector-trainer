"""Real training, saved-model reload and evaluation for every backend.

Each test downloads its pretrained weights into a fresh workspace.
Run with ``pytest --heavy``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from object_detector_trainer.backends import BACKENDS
from object_detector_trainer.evaluation.evaluator import evaluate_model
from object_detector_trainer.pipeline import set_seed
from object_detector_trainer.runs import load_model
from object_detector_trainer.tests.smoke_data import create_sources

pytestmark = pytest.mark.heavy

MODELS = {
    "yolo": {
        "backend": "yolo", "asset_id": "yolo26n.pt", "batch_size": 1, "effective_batch_size": 2,
        "extra_train_kwargs": {"warmup_epochs": 0},
    },
    "rfdetr": {
        "backend": "rfdetr", "variant": "nano", "asset_id": "rf-detr-nano.pth",
        "resolution": 128, "batch_size": 1, "effective_batch_size": 2,
        "extra_train_kwargs": {"num_workers": 0},
    },
}
METRIC_KEYS = ("precision", "recall", "f1_score", "map50", "map", "fitness", "per_class", "scenes", "merged_classes")


def run_cli(workspace: Path, *arguments: str) -> None:
    command = [sys.executable, "-m", "object_detector_trainer.cli", "--workspace-root", str(workspace)]
    subprocess.run(command + list(arguments), check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_train_reload_evaluate(tmp_path: Path, backend: str) -> None:
    create_sources(tmp_path)
    params = {
        "data": {
            "dataset_name": "smoke",
            "custom_classes": ["person", "object"],
        },
        "prepare": {"val_split": 0, "test_split": 0},
        "train": {"model": "smoke-model", "image_size": 128, "epochs": 1},
        "models": {"smoke-model": dict(MODELS[backend])},
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params))

    run_cli(tmp_path, "--stage", "all", "--run-name", "smoke")
    run_dir = tmp_path / "runs/smoke"
    metrics_from_all = read_json(run_dir / "metrics.json")
    run_cli(tmp_path, "--stage", "evaluate", "--run-name", "smoke")
    metrics_from_evaluate = read_json(run_dir / "metrics.json")
    for key in METRIC_KEYS:
        assert metrics_from_all[key] == metrics_from_evaluate[key]
    assert set(metrics_from_all["per_class"]) == {"person", "object"}
    prepared = tmp_path / "datasets/smoke"
    assert [len(list((prepared / split / "images").iterdir())) for split in ("train", "val", "test")] == [5, 2, 3]
    assert (prepared / "train/labels/background__scene_negatives.txt").read_text() == ""
    assert (prepared / "test/labels/background__scene_camera.txt").read_text() == ""

    weights = run_dir / "train/weights/best.pt"
    metadata = yaml.safe_load((run_dir / "train/metadata.yaml").read_text())
    assert metadata["model_backend"] == backend
    assert metadata["class_names"] == {0: "person", 1: "object"}
    assert metadata["epochs"] == 1
    assert metadata["model_options"]["batch_size"] == 1
    assert metadata["model_options"]["effective_batch_size"] == 2
    if backend == "yolo":
        native = yaml.safe_load((run_dir / "train/args.yaml").read_text())
        assert (native["batch"], native["nbs"]) == (1, 2)
    else:
        native = read_json(run_dir / "train/training_config.json")["train_config"]
        assert (native["batch_size"], native["grad_accum_steps"]) == (1, 2)
        assert native["checkpoint_interval"] == 2  # One epoch: no intermediate checkpoint.
        assert not list((run_dir / "train").glob("checkpoint_[0-9]*"))
    assert (run_dir / "train/config.yaml").is_file()
    assert (run_dir / "eval/report.txt").is_file()

    # An independently loaded model scores exactly what the command line reported.
    set_seed(42)
    model = load_model(weights)
    direct = evaluate_model(model, tmp_path / "datasets/smoke", merged_classes={})
    for key in METRIC_KEYS:
        assert direct[key] == metrics_from_all[key]

    # Fine-tune from the trained weights, then compare against them as the baseline.
    finetune = dict(MODELS[backend], init_weights="runs/smoke/train/weights/best.pt", epochs=1)
    if backend == "yolo":
        finetune.update(
            freeze_backbone=True, extra_train_kwargs={"optimizer": "AdamW", "lr0": 0.0002, "warmup_epochs": 0},
        )
    else:
        finetune.update(lr=0.00001, extra_train_kwargs={**finetune["extra_train_kwargs"], "lr_encoder": 0.000015})
    params["models"]["finetuned"] = finetune
    params["train"]["model"] = "finetuned"
    params["evaluation"] = {"baseline_weights_path": "runs/smoke/train/weights/best.pt"}
    params_path.write_text(yaml.safe_dump(params))
    run_cli(tmp_path, "--stage", "train", "--run-name", "tuned")
    # Fine-tuning really uses the lower learning rate.
    if backend == "yolo":
        native = yaml.safe_load((tmp_path / "runs/tuned/train/args.yaml").read_text())
        assert (native["optimizer"], native["lr0"], native["warmup_epochs"]) == ("AdamW", 0.0002, 0)
    else:
        native = read_json(tmp_path / "runs/tuned/train/training_config.json")["train_config"]
        assert (native["lr"], native["lr_encoder"]) == (0.00001, 0.000015)
    run_cli(tmp_path, "--stage", "evaluate", "--run-name", "tuned")

    results = read_json(tmp_path / "runs/tuned/eval/results.json")
    assert list(results) == ["baseline", "tuned"]
    for key in METRIC_KEYS:
        assert results["baseline"][key] == metrics_from_all[key]
    assert (tmp_path / "runs/tuned/train/weights/best.pt").stat().st_size > 0
    assert len(list((tmp_path / "runs/tuned/eval/comparisons").glob("comparison_*.jpg"))) == 3

    # Nothing is written outside the inputs, datasets/, models/ and runs/.
    expected = {"raw_data", "params.yaml", "datasets", "models", "runs"}
    assert {path.name for path in tmp_path.iterdir()} <= expected
    assert {path.name for path in (tmp_path / "runs").iterdir()} == {"smoke", "tuned"}


def test_yolo_musgd_uses_compatible_memory_layout(tmp_path: Path) -> None:
    create_sources(tmp_path)
    params = {
        "data": {"dataset_name": "smoke", "custom_classes": ["person", "object"]},
        "prepare": {"val_split": 0, "test_split": 0},
        "train": {"model": "musgd", "image_size": 128, "epochs": 1},
        "models": {"musgd": {
            **MODELS["yolo"],
            "extra_train_kwargs": {"optimizer": "MuSGD", "warmup_epochs": 0, "workers": 0},
        }},
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    run_cli(tmp_path, "--stage", "all", "--run-name", "musgd")
    native = yaml.safe_load((tmp_path / "runs/musgd/train/args.yaml").read_text())
    assert native["optimizer"] == "MuSGD"
    assert native["channels_last"] is False
    assert (tmp_path / "runs/musgd/train/weights/best.pt").stat().st_size > 0
    assert read_json(tmp_path / "runs/musgd/metrics.json")["fitness"] >= 0
