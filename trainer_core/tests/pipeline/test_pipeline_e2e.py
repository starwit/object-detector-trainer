"""End-to-end pipeline scenarios mirroring a freshly cloned repository.

Each test builds a tiny synthetic dataset inside ``tmp_path``, runs the prepare
stage, and executes the training/evaluation stage. We assert that expected
artifacts (datasets, results CSV, metrics) are created and that fallbacks work
correctly.

Tests rely on a lightweight YOLO stub so we can exercise orchestration logic
without triggering real Ultralytics downloads or GPU-heavy training, keeping
the suite fast, deterministic, and CI-friendly.
"""

from __future__ import annotations

import csv
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
from trainer_core.tests.support.ultralytics_stub import StubYOLO

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
    monkeypatch.setattr("trainer_core.pipeline.model_state._load_yolo_model", StubYOLO)
    monkeypatch.setattr("trainer_core.backends.yolo.YOLO", StubYOLO)

    # Silence heavy post-processing during tests
    # Skip expensive visualisations/scene scanning; return deterministic metrics instead.
    monkeypatch.setattr(
        "trainer_core.evaluation.visual_comparison.generate_side_by_side_comparisons",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "trainer_core.evaluation.scene_metrics.calculate_scene_metrics",
        lambda *args, **kwargs: {"scene_sourceT_fitness": 0.75},
    )

    return StubYOLO


def _assert_results_exist(
    base_dir: Path,
    dataset_name: str,
    scene_suffix: str = "sourceT",
    *,
    expect_scene_metrics: bool = True,
) -> None:
    dataset_root = base_dir / "datasets" / dataset_name
    assert dataset_root.exists()

    results_csv = base_dir / "results_comparison" / "results.csv"
    assert results_csv.exists()

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

    # Check per-class results CSV was generated
    per_class_csv = base_dir / "results_comparison" / "per_class_results.csv"
    if per_class_csv.exists():
        with open(per_class_csv, newline="", encoding="utf-8") as f:
            pc_reader = csv.DictReader(f)
            pc_rows = list(pc_reader)
        assert pc_rows, "per_class_results.csv should have at least one row"
        for row in pc_rows:
            assert "CLASS" in row
            assert "precision" in row
            assert "recall" in row
            assert "ap50" in row
            assert "ap" in row
            assert "f1_score" in row


def test_pipeline_fresh_clone_uses_fallback(stubbed_pipeline: StubYOLO):
    """Fresh clone with missing baseline weights should succeed via fallback checkpoints."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Recreate the minimal data tree and defaults a new contributor would see.
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    # Ensure fallback checkpoint was requested at least once
    assert any(model.startswith("yolov8") for model in StubYOLO.recorded_models)


def test_prepare_stage_fails_when_no_training_data(stubbed_pipeline: StubYOLO):
    """Prepare should fail early with a clear message when no raw training data exists."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = build_args(dataset_name)

    with pytest.raises(ValueError, match="Prepare stage produced 0 training frames"):
        run_prepare_stage(args)


def test_pipeline_offline_error_when_no_weights(stubbed_pipeline: StubYOLO):
    """If official checkpoints cannot be loaded, training should fail fast."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Same fresh clone setup, but simulate a network-less environment via the stub.
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = build_args(dataset_name)
    run_prepare_stage(args)

    StubYOLO.raise_on_official = True

    with pytest.raises(RuntimeError):
        run_train_eval_stage(args)


def test_pipeline_uses_local_baseline_when_available(stubbed_pipeline: StubYOLO):
    """When promoted baseline weights exist, they should be loaded instead of COCO."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "best.pt").write_bytes(b"stub-weights")

    # Mirrors a repo where baseline weights have been tracked via DVC/export script.
    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"finetune": {"weights": str(baseline_dir / "best.pt")}},
        },
    )

    args = build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    assert str(baseline_dir / "best.pt") in StubYOLO.recorded_models


def test_pipeline_finetune_missing_weights_fails(stubbed_pipeline: StubYOLO):
    """Fine-tune mode with missing weights should fail explicitly."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Enable finetune mode but omit weights to ensure warning + fallback path is exercised.
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

    args = build_args(dataset_name)

    run_prepare_stage(args)
    with pytest.raises(FileNotFoundError, match="Fine-tuning mode is enabled"):
        run_train_eval_stage(args)


def test_pipeline_finetune_uses_finetune_weights_as_baseline_when_primary_missing(
    stubbed_pipeline: StubYOLO,
):
    """If evaluation baseline is missing, finetune weights should be used for comparison."""

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

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    # Training and baseline comparison both load the finetune weights.
    assert StubYOLO.recorded_models.count(str(finetune_weights)) >= 2
    # No official YOLO fallback should be needed in this path.
    assert not any(model.startswith("yolov8") for model in StubYOLO.recorded_models)


def test_pipeline_loads_rfdetr_baseline_from_metadata(
    stubbed_pipeline: StubYOLO, monkeypatch: pytest.MonkeyPatch
):
    """A promoted RF-DETR baseline should be loaded via adapter (not YOLO fallback)."""

    import numpy as np

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"

    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_weights = baseline_dir / "best.pt"
    baseline_weights.write_bytes(b"stub-rfdetr-checkpoint")
    (baseline_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "experiment_name: promoted-rfdetr",
                "model_backend: rfdetr",
                "model_variant: nano",
                "image_size: 320",
                "",
            ]
        ),
        encoding="utf-8",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "evaluation": {"baseline_weights_path": str(baseline_weights)},
        },
    )

    rfdetr_load_calls: list[dict[str, str]] = []

    class _StubRFDETRBaseline:
        def predict(self, img, threshold=0.5):
            class _EmptyDets:
                xyxy = np.empty((0, 4))
                confidence = np.empty(0)
                class_id = np.empty(0, dtype=int)

                def __len__(self):
                    return 0

            return _EmptyDets()

    def _fake_get_rfdetr_model(
        model_variant,
        pretrain_weights=None,
        device=None,
        resolution=None,
        gradient_checkpointing=None,
    ):
        rfdetr_load_calls.append(
            {
                "variant": str(model_variant),
                "weights": str(pretrain_weights),
                "resolution": str(resolution),
            }
        )
        return _StubRFDETRBaseline()

    monkeypatch.setattr(
        "trainer_core.backends.rfdetr._get_rfdetr_model", _fake_get_rfdetr_model
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)
    assert rfdetr_load_calls
    assert rfdetr_load_calls[0]["variant"] == "nano"
    assert rfdetr_load_calls[0]["weights"] == str(baseline_weights)
    assert rfdetr_load_calls[0]["resolution"] == "320"
    # One official YOLO load for training is expected; no extra fallback baseline load.
    assert sum(1 for model in StubYOLO.recorded_models if model.startswith("yolov8")) == 1


def test_pipeline_can_select_rfdetr_backend_via_params(
    stubbed_pipeline: StubYOLO, monkeypatch: pytest.MonkeyPatch
):
    """RF-DETR backend selection should be driven by params.yaml (train.model).

    The RF-DETR path now runs the full evaluation pipeline (baseline comparison,
    scene metrics, side-by-side images, weights saving) — identical to YOLO.
    """

    import json

    import numpy as np

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": "rfdetr-nano",
            },
            "models": {
                "rfdetr-nano": {
                    "backend": "rfdetr",
                    "variant": "nano",
                    "resolution": 320,
                    "epochs": 1,
                    "batch_size": 1,
                    "grad_accum_steps": 1,
                }
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)

    def _fake_prepare_yolo_layout(*, training_path: Path, test_path: Path, dataset_name: str):
        output_dir = Path(".tmp") / "rfdetr_datasets" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal YOLO structure so RF-DETR path doesn't crash
        (output_dir / "train").mkdir(exist_ok=True)
        (output_dir / "valid").mkdir(exist_ok=True)
        (output_dir / "test").mkdir(exist_ok=True)
        return output_dir

    class _StubRFDETRModel:
        """Minimal stand-in for an RF-DETR model used by the adapter."""

        def predict(self, img, threshold=0.5):
            class _EmptyDets:
                xyxy = np.empty((0, 4))
                confidence = np.empty(0)
                class_id = np.empty(0, dtype=int)

                def __len__(self):
                    return 0

            return _EmptyDets()

    def _fake_train_rfdetr(*, output_root: Path, **_kwargs):
        output_dir = Path(output_root) / "stub-run"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy checkpoint so _save_rfdetr_weights finds something
        (output_dir / "checkpoint_best_total.pth").write_bytes(b"stub-checkpoint")
        payload = {
            "class_map": [
                {
                    "class": "all",
                    "precision": 0.5,
                    "recall": 0.6,
                    "map@50": 0.4,
                    "map@50:95": 0.3,
                }
            ]
        }
        (output_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
        return _StubRFDETRModel(), output_dir

    monkeypatch.setattr(
        "trainer_core.backends.rfdetr._prepare_rfdetr_yolo_layout",
        _fake_prepare_yolo_layout,
    )
    monkeypatch.setattr("trainer_core.backends.rfdetr.train_rfdetr", _fake_train_rfdetr)

    run_train_eval_stage(args)

    # Full evaluation pipeline now runs for RF-DETR, same as YOLO
    _assert_results_exist(workspace, dataset_name)

    metrics = json.loads((workspace / "metrics.json").read_text(encoding="utf-8"))
    # metrics.json now has the same schema as YOLO (written by validate_model)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "map" in metrics
    assert "map50" in metrics
    assert "fitness" in metrics
    assert "f1_score" in metrics

    # RF-DETR path now loads a YOLO baseline for comparison
    assert any(model.startswith("yolov8") for model in StubYOLO.recorded_models)

    # Verify weights were saved in YOLO-compatible layout (weights/best.pt)
    runs_rfdetr = workspace / "runs" / "rfdetr"
    assert runs_rfdetr.exists()
    run_dirs = [d for d in runs_rfdetr.iterdir() if d.is_dir()]
    assert run_dirs, "Expected at least one run directory under runs/rfdetr"
    weights_dir = run_dirs[0] / "weights"
    assert weights_dir.exists(), "weights/ directory should be created for RF-DETR"
    assert (weights_dir / "best.pt").exists(), "best.pt should be copied from RF-DETR checkpoint"


def _build_stub_rtmdet_adapter(model_name: str = "RTMDet-stub"):
    import numpy as np
    stub_name = model_name

    class _StubValMetrics:
        def __init__(self):
            self.speed = {"preprocess": 0.2, "inference": 0.8, "postprocess": 0.2}
            self.results_dict = {
                "metrics/precision(B)": 0.51,
                "metrics/recall(B)": 0.61,
                "metrics/mAP50(B)": 0.41,
                "metrics/mAP50-95(B)": 0.31,
                "metrics/f1(B)": 0.55,
            }
            self.fitness = 0.32
            self.per_class = {
                "waste": {
                    "precision": 0.51,
                    "recall": 0.61,
                    "map50": 0.41,
                    "map": 0.31,
                    "f1_score": 0.55,
                }
            }

    class _StubAdapter:
        model_backend = "rtmdet"
        model_variant = "rtmdet_tiny_8xb32-300e_coco"
        model_config_path = "runs/rtmdet/stub-run/model_config.py"
        rtmdet_config_name = "rtmdet_tiny_8xb32-300e_coco"
        rtmdet_cache_dir = "models/pretrained/rtmdet"
        rtmdet_allow_download = False
        resolution = 320
        model_name = stub_name
        model = type("_M", (), {"yaml": {"model_name": stub_name}})()

        def predict(self, *args, **kwargs):
            class _Result:
                boxes = []

                @staticmethod
                def plot():
                    return np.zeros((8, 8, 3), dtype=np.uint8)

            return [_Result()]

        def val(self, **kwargs):
            return _StubValMetrics()

    return _StubAdapter()


def test_pipeline_loads_rtmdet_baseline_from_metadata(
    stubbed_pipeline: StubYOLO, monkeypatch: pytest.MonkeyPatch
):
    workspace = Path.cwd()
    dataset_name = "e2e_dataset"

    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_weights = baseline_dir / "best.pt"
    baseline_weights.write_bytes(b"stub-rtmdet-checkpoint")
    (baseline_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "experiment_name: promoted-rtmdet",
                "model_backend: rtmdet",
                "model_variant: rtmdet_tiny_8xb32-300e_coco",
                "image_size: 320",
                "",
            ]
        ),
        encoding="utf-8",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "evaluation": {"baseline_weights_path": str(baseline_weights)},
        },
    )

    load_calls: list[dict[str, str]] = []

    def _fake_load_rtmdet_baseline(*, weights_path, metadata, display_name):
        load_calls.append(
            {
                "weights": str(weights_path),
                "display_name": str(display_name),
                "variant": str(metadata.get("model_variant")),
            }
        )
        return _build_stub_rtmdet_adapter(str(display_name))

    monkeypatch.setattr(
        "trainer_core.backends.rtmdet.load_rtmdet_baseline",
        _fake_load_rtmdet_baseline,
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)
    assert load_calls
    assert load_calls[0]["weights"] == str(baseline_weights)
    assert load_calls[0]["variant"] == "rtmdet_tiny_8xb32-300e_coco"
    assert sum(1 for model in StubYOLO.recorded_models if model.startswith("yolov8")) == 1


def test_pipeline_can_select_rtmdet_backend_via_params(
    stubbed_pipeline: StubYOLO, monkeypatch: pytest.MonkeyPatch
):
    workspace = Path.cwd()
    dataset_name = "e2e_dataset"

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"model": "rtmdet-tiny"},
            "models": {
                "rtmdet-tiny": {
                    "backend": "rtmdet",
                    "config_name": "rtmdet_tiny_8xb32-300e_coco",
                    "epochs": 1,
                    "batch_size": 1,
                    "image_size": 320,
                    "allow_download": False,
                }
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)

    def _fake_train_rtmdet_backend(
        *,
        training_path: Path,
        test_path: Path,
        dataset_name: str,
        resolved_cfg: dict,
        experiment_name: str | None,
    ):
        output_dir = Path("runs") / "rtmdet" / "stub-run"
        (output_dir / "weights").mkdir(parents=True, exist_ok=True)
        (output_dir / "weights" / "best.pt").write_bytes(b"stub-rtmdet")
        (output_dir / "model_config.py").write_text("# stub config\n", encoding="utf-8")
        return _build_stub_rtmdet_adapter("rtmdet-stub"), output_dir, "rtmdet-stub", 320, 1

    monkeypatch.setattr(
        "trainer_core.backends.rtmdet.train_rtmdet_backend",
        _fake_train_rtmdet_backend,
    )

    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)
    runs_rtmdet = workspace / "runs" / "rtmdet"
    assert runs_rtmdet.exists()
    assert list(runs_rtmdet.glob("**/weights/best.pt")), "Expected RTMDet weights/best.pt"
    assert any(model.startswith("yolov8") for model in StubYOLO.recorded_models)
