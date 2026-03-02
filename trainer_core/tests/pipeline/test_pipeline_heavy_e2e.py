"""Heavy end-to-end tests that run real backend training.

These tests are opt-in via ``pytest --heavy`` (see tests-level ``trainer_core/tests/conftest.py``).
They validate full prepare + train/eval execution with real model backends.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trainer_core.pipeline.evaluate_stage import run_evaluate_stage
from trainer_core.pipeline.prepare_stage import run_prepare_stage
from trainer_core.pipeline.train_stage import run_train_stage
from trainer_core.tests.support.pipeline_test_utils import (
    build_args,
    create_minimal_dataset,
    write_params_yaml,
)

pytestmark = pytest.mark.heavy


def run_train_eval_stage(args):
    train_result = run_train_stage(args)
    run_evaluate_stage(args, train_result=train_result)
    return train_result


def _require_repo_weight(filename: str) -> Path:
    """Resolve a required local checkpoint from repo root, or skip with context."""
    repo_root = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists()),
        Path.cwd(),
    )
    candidate = repo_root / filename
    if not candidate.exists() or candidate.stat().st_size == 0:
        pytest.skip(
            f"Heavy test prerequisite missing: {candidate}. "
            f"Provide local '{filename}' before running --heavy tests."
        )
    return candidate


def _assert_common_pipeline_artifacts(workspace: Path, dataset_name: str) -> None:
    dataset_root = workspace / "datasets" / dataset_name
    assert dataset_root.exists()

    results_csv = workspace / "results_comparison" / "results.csv"
    assert results_csv.exists()
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "results.csv should contain at least one row"

    metrics_path = workspace / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("precision", "recall", "map", "map50", "fitness", "f1_score"):
        assert key in metrics


def test_heavy_e2e_yolo_one_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run real YOLO training/evaluation for one epoch on a tiny synthetic dataset."""
    monkeypatch.chdir(tmp_path)

    yolo_ckpt = _require_repo_weight("yolov8n.pt")
    dataset_name = "heavy_e2e_yolo"

    create_minimal_dataset(tmp_path)
    write_params_yaml(
        tmp_path,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "image_size": 128,
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": str(yolo_ckpt),
                }
            },
            "evaluation": {
                "baseline_weights_path": str(yolo_ckpt),
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_common_pipeline_artifacts(tmp_path, dataset_name)
    assert list((tmp_path / "runs").glob("**/weights/best.pt")), "Expected YOLO weights/best.pt"


def test_heavy_e2e_rfdetr_one_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run real RF-DETR training/evaluation for one epoch on a tiny synthetic dataset."""
    monkeypatch.chdir(tmp_path)

    yolo_ckpt = _require_repo_weight("yolov8n.pt")
    rfdetr_ckpt = _require_repo_weight("rf-detr-nano.pth")
    dataset_name = "heavy_e2e_rfdetr"

    create_minimal_dataset(tmp_path)

    write_params_yaml(
        tmp_path,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": "rfdetr-nano",
                "image_size": 128,
                "epochs": 1,
                "batch_size": 1,
            },
            "models": {
                "rfdetr-nano": {
                    "backend": "rfdetr",
                    "variant": "nano",
                    "resolution": 128,
                    "epochs": 1,
                    "batch_size": 1,
                    "grad_accum_steps": 1,
                    "pretrain_weights": str(rfdetr_ckpt),
                    "extra_train_kwargs": {
                        "checkpoint_interval": 1,
                        "run_test": False,
                    },
                }
            },
            "evaluation": {
                "baseline_weights_path": str(yolo_ckpt),
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_common_pipeline_artifacts(tmp_path, dataset_name)

    runs_rfdetr = tmp_path / "runs" / "rfdetr"
    assert runs_rfdetr.exists()
    assert list(runs_rfdetr.glob("**/weights/best.pt")), "Expected RF-DETR weights/best.pt"


def _require_rtmdet_config(filename: str = "rtmdet_tiny_8xb32-300e_coco.py") -> Path:
    """Resolve RTMDet config from installed mmdet package, or skip if unavailable."""
    try:
        import mmdet  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in heavy environments
        pytest.skip(f"Heavy test prerequisite missing: mmdet import failed ({exc}).")

    try:
        import mmcv._ext  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised in heavy environments
        pytest.skip(
            "Heavy test prerequisite missing: full mmcv ops are unavailable "
            f"(install `mmcv`, not `mmcv-lite`; got {exc})."
        )

    root = Path(mmdet.__file__).resolve().parent
    candidate = root / ".mim" / "configs" / "rtmdet" / filename
    if not candidate.exists():
        pytest.skip(
            f"Heavy test prerequisite missing: RTMDet config not found at {candidate}."
        )
    return candidate


def test_heavy_e2e_rtmdet_one_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run real RTMDet/MMDetection training/evaluation for one epoch on a tiny synthetic dataset."""
    monkeypatch.chdir(tmp_path)

    yolo_ckpt = _require_repo_weight("yolov8n.pt")
    rtmdet_cfg = _require_rtmdet_config()
    dataset_name = "heavy_e2e_rtmdet"

    create_minimal_dataset(tmp_path)

    write_params_yaml(
        tmp_path,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": "rtmdet-tiny",
                "image_size": 128,
                "epochs": 1,
                "batch_size": 1,
            },
            "models": {
                "rtmdet-tiny": {
                    "backend": "rtmdet",
                    "config_path": str(rtmdet_cfg),
                    "epochs": 1,
                    "batch_size": 1,
                    "image_size": 128,
                    "allow_download": False,
                }
            },
            "evaluation": {
                "baseline_weights_path": str(yolo_ckpt),
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_common_pipeline_artifacts(tmp_path, dataset_name)
    runs_rtmdet = tmp_path / "runs" / "rtmdet"
    assert runs_rtmdet.exists()
    assert list(runs_rtmdet.glob("**/weights/best.pt")), "Expected RTMDet weights/best.pt"
