from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from object_detector_trainer.cli import run_all_stages
from object_detector_trainer.pipeline.evaluate_stage import run_evaluate_stage
from object_detector_trainer.pipeline.prepare_stage import run_prepare_stage
from object_detector_trainer.pipeline.train_stage import run_train_stage
from object_detector_trainer.tests.support.pipeline_test_utils import (
    BASE_PARAMS,
    create_baseline_artifact,
    create_local_yolo_checkpoint,
    create_minimal_dataset,
    write_params_yaml,
)
from object_detector_trainer.tests.support.ultralytics_stub import StubYOLO


REQUIRED_METRIC_KEYS = (
    "precision",
    "recall",
    "map",
    "map50",
    "fitness",
    "f1_score",
    "ms_per_frame",
)


def _discover_backend_cases() -> list[tuple[str, str]]:
    cases: dict[str, str] = {}
    models_cfg = BASE_PARAMS.get("models", {})
    if not isinstance(models_cfg, dict):
        return []
    for model_key, model_cfg in sorted(models_cfg.items()):
        if not isinstance(model_cfg, dict):
            continue
        backend = str(model_cfg["backend"]).strip().lower()
        cases.setdefault(backend, str(model_key))
    return [(model_key, backend) for backend, model_key in sorted(cases.items())]


class _ContractBox:
    def __init__(self) -> None:
        self.ap_class_index = np.array([0])
        self.p = np.array([0.5])
        self.r = np.array([0.6])
        self._ap50 = np.array([0.4])
        self._ap = np.array([0.3])

    @property
    def ap50(self):
        return self._ap50

    @property
    def ap(self):
        return self._ap


class _ContractValMetrics:
    def __init__(self) -> None:
        self.speed = {"preprocess": 0.5, "inference": 1.0, "postprocess": 0.5}
        self.results_dict = {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.4,
            "metrics/mAP50-95(B)": 0.3,
            "metrics/f1(B)": 0.5454545,
        }
        self.fitness = 0.35
        self.box = _ContractBox()


class _ContractModel:
    def __init__(self, *, model_name: str, model_backend: str, image_size: int = 320) -> None:
        self.model_name = model_name
        self.model_backend = model_backend
        self.model = SimpleNamespace(yaml={"model_name": model_name})
        self.trainer = None
        self.resolution = int(image_size)
        self.class_names = {0: "waste"}

    def val(self, **kwargs):
        return _ContractValMetrics()

    def predict(self, *args, **kwargs):
        class _Result:
            boxes = []

            @staticmethod
            def plot():
                return np.zeros((8, 8, 3), dtype=np.uint8)

        return [_Result()]


def _contract_args(
    *,
    dataset_name: str,
    model: str = "yolov8n",
    stage: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        stage=stage,
        seed=42,
        dataset_name=dataset_name,
        model=model,
        val_split=0.5,
        test_split=0.0,
        recreate_dataset=True,
        augment_multiplier=1,
        folder_subset=None,
        set=[],
        config="params.yaml",
    )


def _assert_numeric_metric_contract(metrics_path: Path) -> dict:
    assert metrics_path.exists(), f"Expected metrics file: {metrics_path}"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in REQUIRED_METRIC_KEYS:
        assert key in payload, f"Missing metric key: {key}"
        value = payload[key]
        assert isinstance(value, (int, float)), f"Metric '{key}' should be numeric, got {type(value)}"
        assert math.isfinite(float(value)), f"Metric '{key}' must be finite, got {value!r}"

    img_size = payload.get("img_size")
    assert isinstance(img_size, int), "img_size must be an integer"
    assert img_size > 0, "img_size must be > 0"
    return payload


def _assert_results_csv_contract(results_csv: Path, *, expect_two_rows: bool = True) -> list[dict]:
    assert results_csv.exists(), f"Expected results CSV: {results_csv}"
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "results.csv must contain at least one result row"
    if expect_two_rows:
        assert len(rows) >= 2, "results.csv should contain baseline + trained rows"
    return rows


def _latest_metadata_file(runs_dir: Path) -> Path:
    candidates = list(runs_dir.rglob("metadata.yaml"))
    assert candidates, "No metadata.yaml file found under runs/"
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _assert_summary_only_results(summary_dir: Path) -> None:
    _assert_results_csv_contract(summary_dir / "results.csv")
    assert (summary_dir / "results.txt").exists()
    extras = sorted(
        entry.name
        for entry in summary_dir.iterdir()
        if entry.name not in {"results.csv", "results.txt"}
    )
    assert not extras, f"results_comparison must contain summary artifacts only, found: {extras}"


@pytest.fixture
def contract_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    StubYOLO.workspace = tmp_path
    StubYOLO.recorded_models = []
    StubYOLO.raise_on_official = False
    return tmp_path


def _write_contract_params(workspace: Path, *, dataset_name: str, model: str) -> None:
    baseline_path = create_baseline_artifact(workspace)

    common = {
        "data": {"dataset_name": dataset_name},
        "train": {"model": model, "epochs": 1, "batch_size": 1, "image_size": 320},
        "evaluation": {"baseline_weights_path": str(baseline_path)},
    }

    if model == "yolov8n":
        common["models"] = {"yolov8n": {"backend": "yolo", "checkpoint": "yolov8n.pt"}}
    elif model == "rfdetr-nano":
        common["models"] = {
            "rfdetr-nano": {
                "backend": "rfdetr",
                "variant": "nano",
                "pretrain_weights": "models/pretrained/rfdetr/rf-detr-nano.pth",
                "resolution": 320,
                "epochs": 1,
                "batch_size": 1,
                "grad_accum_steps": 1,
            }
        }
    elif model == "rtmdet-tiny":
        common["models"] = {
            "rtmdet-tiny": {
                "backend": "rtmdet",
                "config_name": "rtmdet_tiny_8xb32-300e_coco",
                "allow_download": False,
                "epochs": 1,
                "batch_size": 1,
                "image_size": 320,
            }
        }
    else:
        raise AssertionError(f"Unsupported model for test setup: {model}")

    write_params_yaml(workspace, common)
    if model == "yolov8n":
        create_local_yolo_checkpoint(workspace)


def _patch_lightweight_trainers(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    def _mk_runner(backend: str):
        def _runner(*args, **kwargs):
            run_dir = workspace / "runs" / f"{backend}-contract"
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"trained-stub")
            model = _ContractModel(model_name=f"{backend}-contract", model_backend=backend, image_size=320)
            return model, run_dir, f"{backend}-contract", 320, 1

        return _runner

    monkeypatch.setattr("object_detector_trainer.backends.yolo.train_backend", _mk_runner("yolo"))
    monkeypatch.setattr("object_detector_trainer.backends.rfdetr.train_backend", _mk_runner("rfdetr"))
    monkeypatch.setattr("object_detector_trainer.backends.rtmdet.train_backend", _mk_runner("rtmdet"))
    monkeypatch.setattr("object_detector_trainer.pipeline.model_state._load_yolo_model", StubYOLO)
    monkeypatch.setattr(
        "object_detector_trainer.evaluation.visual_comparison.generate_side_by_side_comparisons",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("object_detector_trainer.evaluation.scene_metrics.calculate_scene_metrics", lambda *a, **k: {})
    monkeypatch.setattr("object_detector_trainer.plugins.replay.build_or_update_replay_set", lambda *a, **k: None)


def test_stage_contract_prepare_train_evaluate_all(
    contract_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_minimal_dataset(contract_workspace)
    _write_contract_params(contract_workspace, dataset_name="contract-stages", model="yolov8n")
    _patch_lightweight_trainers(monkeypatch, contract_workspace)

    args = _contract_args(dataset_name="contract-stages", model="yolov8n")
    run_prepare_stage(args)
    assert (contract_workspace / "datasets" / "contract-stages").exists()
    assert not (contract_workspace / "results_comparison" / "results.csv").exists()

    train_out = run_train_stage(args)
    assert train_out is not None
    assert list((contract_workspace / "runs").glob("**/weights/best.pt"))
    assert not (contract_workspace / "results_comparison" / "results.csv").exists()

    run_evaluate_stage(args, train_result=train_out)
    _assert_numeric_metric_contract(contract_workspace / "metrics.json")
    _assert_summary_only_results(contract_workspace / "results_comparison")
    assert (train_out.train_output_dir / "metadata.yaml").exists()
    assert (train_out.train_output_dir / "results.csv").exists()
    assert (train_out.train_output_dir / "results.txt").exists()

    # all-stage flow should also satisfy the same contracts end-to-end
    create_minimal_dataset(contract_workspace)
    _write_contract_params(contract_workspace, dataset_name="contract-all", model="yolov8n")
    args_all = _contract_args(dataset_name="contract-all", model="yolov8n")
    run_all_stages(args_all)
    _assert_numeric_metric_contract(contract_workspace / "metrics.json")
    _assert_summary_only_results(contract_workspace / "results_comparison")


@pytest.mark.parametrize(
    ("model_key", "expected_backend"),
    _discover_backend_cases(),
)
def test_backend_metric_contracts(
    contract_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
    expected_backend: str,
) -> None:
    create_minimal_dataset(contract_workspace)
    _write_contract_params(contract_workspace, dataset_name=f"contract-{model_key}", model=model_key)
    _patch_lightweight_trainers(monkeypatch, contract_workspace)

    args = _contract_args(dataset_name=f"contract-{model_key}", model=model_key)
    run_all_stages(args)

    _assert_numeric_metric_contract(contract_workspace / "metrics.json")
    _assert_summary_only_results(contract_workspace / "results_comparison")
    metadata_path = _latest_metadata_file(contract_workspace / "runs")
    metadata = json.loads("{}")
    if metadata_path.exists():
        import yaml

        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    assert metadata.get("model_backend") == expected_backend
