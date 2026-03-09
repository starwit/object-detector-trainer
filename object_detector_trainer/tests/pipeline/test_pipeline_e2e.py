"""End-to-end pipeline scenarios for the strict model/baseline contract.

Each test builds a tiny synthetic dataset inside ``tmp_path``, runs the prepare
stage, and executes the training/evaluation stage. We assert that expected
artifacts (datasets, results CSV, metrics) are created and that missing
configured model assets fail instead of triggering fallback behavior.

Baseline comparisons follow a two-state contract:
- No baseline promoted yet: no metadata.yaml next to evaluation.baseline_weights_path; evaluate trained model only.
- Baseline promoted: metadata.yaml exists; missing/empty baseline weights is a hard failure.

Tests rely on a lightweight YOLO stub so we can exercise orchestration logic
without triggering real Ultralytics downloads or GPU-heavy training, keeping
the suite fast, deterministic, and CI-friendly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from object_detector_trainer.pipeline.evaluate_stage import run_evaluate_stage
from object_detector_trainer.pipeline.prepare_stage import run_prepare_stage
from object_detector_trainer.pipeline.train_stage import run_train_stage
from object_detector_trainer.tests.support.pipeline_test_utils import (
    build_args,
    create_baseline_artifact,
    create_local_yolo_checkpoint,
    create_minimal_dataset,
    write_params_yaml,
)
from object_detector_trainer.tests.support.ultralytics_stub import StubYOLO


def run_train_eval_stage(args):
    train_result = run_train_stage(args)
    run_evaluate_stage(args, train_result=train_result)
    return train_result


@pytest.fixture
def stubbed_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Prepare a clean workspace and replace Ultralytics YOLO with a deterministic stub."""

    monkeypatch.chdir(tmp_path)
    StubYOLO.workspace = tmp_path
    StubYOLO.recorded_models = []
    StubYOLO.raise_on_official = False

    # Replace YOLO constructors with the stub
    # Swap in the stub everywhere the pipeline imports YOLO so training/eval stays local.
    monkeypatch.setattr("object_detector_trainer.pipeline.model_state._load_yolo_model", StubYOLO)
    monkeypatch.setattr("object_detector_trainer.backends.yolo.YOLO", StubYOLO)

    # Silence heavy post-processing during tests
    # Skip expensive visualisations/scene scanning; return deterministic metrics instead.
    def _stub_side_by_side_comparisons(*, output_dir: Path, **_kwargs) -> None:
        (Path(output_dir) / "side_by_side_comparisons").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "object_detector_trainer.evaluation.visual_comparison.generate_side_by_side_comparisons",
        _stub_side_by_side_comparisons,
    )
    monkeypatch.setattr(
        "object_detector_trainer.evaluation.scene_metrics.calculate_scene_metrics",
        lambda *args, **kwargs: {"scene_sourceT_fitness": 0.75},
    )

    return StubYOLO


def _assert_results_exist(
    base_dir: Path,
    dataset_name: str,
    scene_suffix: str = "sourceT",
    *,
    expect_scene_metrics: bool = True,
    expect_side_by_side: bool = True,
) -> None:
    dataset_root = base_dir / "datasets" / dataset_name
    assert dataset_root.exists()

    summary_dir = base_dir / "results_comparison"
    results_csv = summary_dir / "results.csv"
    assert results_csv.exists()
    assert (summary_dir / "results.txt").exists()

    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    col_name = f"scene_{scene_suffix}_fitness"
    if expect_scene_metrics:
        assert col_name in header
    assert rows, "results.csv should contain at least one row"

    if expect_scene_metrics:
        for row in rows:
            val_str = (row.get(col_name) or "").strip()
            if val_str and val_str != "-":
                assert float(val_str) >= 0.0

    assert (base_dir / "metrics.json").exists()

    extras = sorted(p.name for p in summary_dir.iterdir() if p.name not in {"results.csv", "results.txt"})
    assert not extras, f"results_comparison/ must be summary-only, found: {extras}"

    marker = base_dir / "runs" / ".last_train_result.json"
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    run_dir = Path(payload["train_output_dir"])
    assert run_dir.exists()
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "metadata.yaml").exists()
    assert (run_dir / "plots").is_dir()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "results.txt").exists()
    assert (run_dir / "test_results.csv").exists()
    if expect_side_by_side:
        assert (run_dir / "side_by_side_comparisons").is_dir()
    else:
        assert not (run_dir / "side_by_side_comparisons").exists()


def test_pipeline_succeeds_without_local_baseline_file(stubbed_pipeline: StubYOLO):
    """Evaluation should succeed when there is no promoted baseline yet."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})
    create_local_yolo_checkpoint(workspace)

    args = build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name, expect_side_by_side=False)

    baseline_path = workspace / "models" / "current_best" / "best.pt"
    assert str(baseline_path) not in StubYOLO.recorded_models


def test_pipeline_fails_when_promoted_baseline_weights_missing(stubbed_pipeline: StubYOLO):
    """If baseline metadata exists (baseline promoted), missing weights must fail loudly."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})
    create_local_yolo_checkpoint(workspace)

    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "metadata.yaml").write_text(
        "experiment_name: promoted-baseline\nmodel_backend: yolo\nimage_size: 320\n",
        encoding="utf-8",
    )
    weights_path = baseline_dir / "best.pt"
    if weights_path.exists():
        weights_path.unlink()

    args = build_args(dataset_name)

    run_prepare_stage(args)
    with pytest.raises(FileNotFoundError, match="Promoted baseline metadata exists"):
        run_train_eval_stage(args)


def test_prepare_stage_fails_when_no_training_data(stubbed_pipeline: StubYOLO):
    """Prepare should fail early with a clear message when no raw training data exists."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = build_args(dataset_name)

    with pytest.raises(ValueError, match="Prepare stage produced 0 training frames"):
        run_prepare_stage(args)


def test_pipeline_requires_local_model_checkpoint(stubbed_pipeline: StubYOLO):
    """Training must fail if the configured model checkpoint is missing."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    create_minimal_dataset(workspace)
    baseline_path = create_baseline_artifact(workspace, experiment_name="baseline")
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": "models/pretrained/yolo/missing.pt",
                }
            },
            "evaluation": {"baseline_weights_path": str(baseline_path)},
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    with pytest.raises(FileNotFoundError, match="models.<key>.checkpoint"):
        run_train_stage(args)


def test_pipeline_uses_local_baseline_when_available(stubbed_pipeline: StubYOLO):
    """When promoted baseline weights exist, they should be loaded as configured."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    baseline_path = create_baseline_artifact(
        workspace,
        experiment_name="promoted-baseline",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "evaluation": {"baseline_weights_path": str(baseline_path)},
        },
    )
    create_local_yolo_checkpoint(workspace)

    args = build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    assert str(baseline_path) in StubYOLO.recorded_models


def test_pipeline_finetune_missing_weights_fails(stubbed_pipeline: StubYOLO):
    """Fine-tune mode with missing weights should fail explicitly."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "finetune": {
                    "enabled": True,
                    "weights": "models/current_best/missing.pt",
                    "epochs": 1,
                },
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    args = build_args(dataset_name)

    run_prepare_stage(args)
    with pytest.raises(FileNotFoundError, match="Fine-tuning mode is enabled"):
        run_train_eval_stage(args)


def test_pipeline_missing_baseline_does_not_fall_back_to_finetune_weights(
    stubbed_pipeline: StubYOLO,
):
    """Missing baseline must not trigger any fallback to finetune weights during evaluation."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    finetune_weights = workspace / "models" / "finetune" / "best.pt"
    finetune_weights.parent.mkdir(parents=True, exist_ok=True)
    finetune_weights.write_bytes(b"stub-finetune-weights")

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "finetune": {
                    "enabled": True,
                    "weights": str(finetune_weights),
                    "epochs": 1,
                },
            },
            "evaluation": {
                "baseline_weights_path": "models/current_best/missing.pt",
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name, expect_side_by_side=False)

    assert StubYOLO.recorded_models.count(str(finetune_weights)) == 1
