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

import cv2
import numpy as np
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


def _write_image(path: Path, fill: int) -> None:
    image = np.full((96, 96, 3), fill, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (72, 72), (255, 255, 255), -1)
    cv2.imwrite(str(path), image)


def _create_class_mapping_dataset(base_dir: Path) -> None:
    train_images = base_dir / "raw_data" / "train" / "source1" / "images"
    train_labels = base_dir / "raw_data" / "train" / "source1" / "labels"
    test_images = base_dir / "raw_data" / "test" / "sourceT" / "images"
    test_labels = base_dir / "raw_data" / "test" / "sourceT" / "labels"

    for path in (train_images, train_labels, test_images, test_labels):
        path.mkdir(parents=True, exist_ok=True)

    train_specs = [
        ("train_waste.jpg", "0 0.5 0.5 0.4 0.4\n", 60),
        ("train_cigarette.jpg", "1 0.5 0.5 0.3 0.3\n", 110),
        ("train_cigarette_2.jpg", "1 0.4 0.4 0.2 0.2\n", 150),
    ]
    for filename, label_text, fill in train_specs:
        _write_image(train_images / filename, fill)
        (train_labels / f"{Path(filename).stem}.txt").write_text(label_text, encoding="utf-8")

    _write_image(test_images / "test_cigarette.jpg", 200)
    (test_labels / "test_cigarette.txt").write_text(
        "1 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )


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


def test_pipeline_fails_when_promoted_baseline_metadata_is_only_dvc_pointer(stubbed_pipeline: StubYOLO):
    """A DVC-tracked promoted baseline must fail loudly until artifacts are pulled."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})
    create_local_yolo_checkpoint(workspace)

    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "metadata.yaml.dvc").write_text("outs:\n- md5: deadbeef\n", encoding="utf-8")
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
                    "asset_id": "missing.pt",
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


def test_pipeline_writes_merged_class_metrics_for_class_mapping(
    stubbed_pipeline: StubYOLO,
) -> None:
    workspace = Path.cwd()
    dataset_name = "mapped-dataset"
    _create_class_mapping_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {
                "dataset_name": dataset_name,
                "custom_classes": ["waste", "cigarette"],
                "use_coco_classes": False,
                "class_mapping": {
                    "waste": ["waste", "cigarette"],
                },
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    args = build_args(dataset_name, {"val_split": 0.5})
    run_prepare_stage(args)

    raw_test_label = workspace / "raw_data" / "test" / "sourceT" / "labels" / "test_cigarette.txt"
    assert raw_test_label.read_text(encoding="utf-8").strip().split()[0] == "1"

    prepared_test_labels = workspace / "datasets" / dataset_name / "test" / "val" / "labels"
    prepared_label = next(prepared_test_labels.glob("*.txt"))
    prepared_tokens = prepared_label.read_text(encoding="utf-8").strip().split()
    assert prepared_tokens[0] == "0"

    run_train_eval_stage(args)
    _assert_results_exist(workspace, dataset_name, expect_side_by_side=False)

    metrics_payload = json.loads((workspace / "metrics.json").read_text(encoding="utf-8"))
    assert "cigarette_as_waste_ap50" in metrics_payload
    assert "cigarette_as_waste_n_objects" in metrics_payload

    marker = json.loads((workspace / "runs" / ".last_train_result.json").read_text(encoding="utf-8"))
    run_dir = Path(marker["train_output_dir"])
    merged_results = run_dir / "merged_class_results.csv"
    assert merged_results.exists()
    merged_rows = merged_results.read_text(encoding="utf-8")
    assert "cigarette" in merged_rows
    assert "waste" in merged_rows


def test_train_stage_builds_replay_set_when_auto_replay_enabled(
    stubbed_pipeline: StubYOLO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Path.cwd()
    dataset_name = "replay-dataset"

    val_images = workspace / "datasets" / dataset_name / "train" / "val" / "images"
    val_labels = workspace / "datasets" / dataset_name / "train" / "val" / "labels"
    for path in (val_images, val_labels):
        path.mkdir(parents=True, exist_ok=True)

    replay_image = val_images / "replay_candidate.jpg"
    _write_image(replay_image, 90)
    (val_labels / "replay_candidate.txt").write_text("", encoding="utf-8")

    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "prepare": {
                "auto_replay": {
                    "enabled": True,
                    "max_new": 1,
                    "max_total": 2,
                    "include_empty": True,
                    "dest": "raw_data/train/replay",
                }
            },
        },
    )

    class _ReplayModel:
        class_names = {0: "waste"}

        def predict(self, *args, **kwargs):
            class _Result:
                boxes = []

            return [_Result()]

    def _fake_train_backend(*_args, **_kwargs):
        run_dir = workspace / "runs" / "replay-contract"
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"replay-trained-weights")
        return _ReplayModel(), run_dir, "replay-contract", 320, 1

    monkeypatch.setattr(
        "object_detector_trainer.pipeline.train_stage.train_backend",
        _fake_train_backend,
    )

    args = build_args(dataset_name)
    result = run_train_stage(args)

    replay_root = workspace / "raw_data" / "train" / "replay"
    assert (replay_root / "images" / replay_image.name).exists()
    assert (replay_root / "labels" / "replay_candidate.txt").exists()
    index_csv = replay_root / "index.csv"
    assert index_csv.exists()
    assert "replay-contract" in index_csv.read_text(encoding="utf-8")
    assert result.train_output_dir == workspace / "runs" / "replay-contract"
