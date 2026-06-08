"""Split evaluate-stage coverage for non-YOLO backends.

These tests intentionally run:
1) prepare
2) train
3) evaluate with ``train_result=None``

That forces the evaluate stage to reload the trained model from persisted run
artifacts (the same code path used by split DVC stages).

If a new non-YOLO backend is introduced, this test will fail until it gets an
explicit train/reload patch pair here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from object_detector_trainer.backends.registry import SUPPORTED_BACKEND_NAMES
from object_detector_trainer.pipeline.evaluate_stage import run_evaluate_stage
from object_detector_trainer.pipeline.prepare_stage import run_prepare_stage
from object_detector_trainer.pipeline.train_stage import run_train_stage
from object_detector_trainer.tests.support.pipeline_test_utils import (
    build_args,
    create_baseline_artifact,
    create_local_yolo_checkpoint,
    create_minimal_dataset,
    representative_model_key_for_backend,
    write_params_yaml,
)
from object_detector_trainer.tests.support.ultralytics_stub import StubYOLO


class _StubValMetrics:
    def __init__(self) -> None:
        self.speed = {"preprocess": 0.2, "inference": 0.8, "postprocess": 0.2}
        self.results_dict = {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.4,
            "metrics/mAP50-95(B)": 0.3,
            "metrics/f1(B)": 0.5454545,
        }
        self.fitness = 0.35
        self.per_class = {
            "waste": {
                "precision": 0.5,
                "recall": 0.6,
                "map50": 0.4,
                "map": 0.3,
                "f1_score": 0.5454545,
            }
        }


class _StubEvalModel:
    def __init__(
        self,
        *,
        model_name: str,
        model_backend: str,
        model_variant: str | None = None,
        resolution: int = 320,
        model_config_path: str | None = None,
        rtmdet_config_name: str | None = None,
        rtmdet_cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_backend = model_backend
        self.model_variant = model_variant
        self.resolution = int(resolution)
        self.model = type("_M", (), {"yaml": {"model_name": model_name}})()
        self.trainer = None
        self.class_names = {0: "waste"}
        self.model_config_path = model_config_path
        self.rtmdet_config_name = rtmdet_config_name
        self.rtmdet_cache_dir = rtmdet_cache_dir

    def val(self, **kwargs):
        return _StubValMetrics()

    def predict(self, *args, **kwargs):
        class _Result:
            boxes = []

            @staticmethod
            def plot():
                return np.zeros((8, 8, 3), dtype=np.uint8)

        return [_Result()]


class _StubRFDETRPredictor:
    def predict(self, img, threshold=0.5):
        class _EmptyDets:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty(0, dtype=np.float32)
            class_id = np.empty(0, dtype=np.int64)

            def __len__(self):
                return 0

        return _EmptyDets()


@pytest.fixture
def split_eval_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    StubYOLO.workspace = tmp_path
    StubYOLO.recorded_models = []
    StubYOLO.raise_on_official = False

    monkeypatch.setattr("object_detector_trainer.pipeline.model_state._load_yolo_model", StubYOLO)
    monkeypatch.setattr("object_detector_trainer.backends.yolo.YOLO", StubYOLO)
    monkeypatch.setattr(
        "object_detector_trainer.dataprep.find_duplicates.DuplicateDetector.find_duplicates",
        lambda self, _image_paths: {},
    )
    monkeypatch.setattr(
        "object_detector_trainer.dataprep.find_duplicates.DuplicateDetector.get_unique_images",
        lambda self, image_paths: set(image_paths),
    )
    monkeypatch.setattr(
        "object_detector_trainer.dataprep.find_duplicates.DuplicateDetector.compare_folders",
        lambda self, _training_path, _test_path: {},
    )
    monkeypatch.setattr(
        "object_detector_trainer.evaluation.visual_comparison.generate_side_by_side_comparisons",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "object_detector_trainer.evaluation.scene_metrics.calculate_scene_metrics",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr("object_detector_trainer.plugins.replay.build_or_update_replay_set", lambda *a, **k: None)
    return tmp_path


def _patch_rfdetr_train(monkeypatch: pytest.MonkeyPatch) -> Path:
    run_dir = Path(".dvc_artifacts") / "train_runs" / "split-reload-rfdetr"

    def _fake_train_backend(
        *,
        training_path: Path,
        test_path: Path,
        dataset_name: str,
        resolved_cfg: dict,
        experiment_name: str | None,
    ):
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"rfdetr-split-weights")
        model = _StubEvalModel(
            model_name="rfdetr-split",
            model_backend="rfdetr",
            model_variant="nano",
            resolution=320,
        )
        return model, run_dir, "rfdetr-split", 320, 1

    monkeypatch.setattr("object_detector_trainer.backends.rfdetr.train_backend", _fake_train_backend)
    return run_dir / "weights" / "best.pt"


def _patch_rfdetr_reload(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, str]]) -> None:
    def _fake_get_rfdetr_model(
        model_variant,
        checkpoint_path=None,
        device=None,
        resolution=None,
        gradient_checkpointing=None,
    ):
        calls.append(
            {
                "variant": str(model_variant),
                "weights": str(checkpoint_path),
                "resolution": str(resolution),
            }
        )
        return _StubRFDETRPredictor()

    monkeypatch.setattr("object_detector_trainer.backends.rfdetr._get_rfdetr_model", _fake_get_rfdetr_model)


def _patch_rtmdet_train(monkeypatch: pytest.MonkeyPatch) -> Path:
    run_dir = Path(".dvc_artifacts") / "train_runs" / "split-reload-rtmdet"

    def _fake_train_backend(
        *,
        training_path: Path,
        test_path: Path,
        dataset_name: str,
        resolved_cfg: dict,
        experiment_name: str | None,
    ):
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"rtmdet-split-weights")
        model_config = run_dir / "model_config.py"
        model_config.write_text("# split-eval-stub\n", encoding="utf-8")
        model = _StubEvalModel(
            model_name="rtmdet-split",
            model_backend="rtmdet",
            model_variant="rtmdet_tiny_8xb32-300e_coco",
            resolution=320,
            model_config_path=str(model_config),
            rtmdet_config_name="rtmdet_tiny_8xb32-300e_coco",
            rtmdet_cache_dir="models/pretrained/rtmdet",
        )
        return model, run_dir, "rtmdet-split", 320, 1

    monkeypatch.setattr("object_detector_trainer.backends.rtmdet.train_backend", _fake_train_backend)
    return run_dir / "weights" / "best.pt"


def _patch_rtmdet_reload(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, str]]) -> None:
    def _fake_load_rtmdet_baseline(*, weights_path: Path, metadata: dict, display_name: str):
        calls.append(
            {
                "weights": str(weights_path),
                "backend": str(metadata.get("model_backend")),
                "variant": str(metadata.get("model_variant")),
                "display_name": str(display_name),
            }
        )
        return _StubEvalModel(
            model_name=str(display_name),
            model_backend="rtmdet",
            model_variant=str(metadata.get("model_variant") or "rtmdet_tiny_8xb32-300e_coco"),
            resolution=int(metadata.get("image_size", 320) or 320),
            model_config_path=str(metadata.get("model_config_path", "")),
            rtmdet_config_name=str(metadata.get("rtmdet_config_name", "")),
            rtmdet_cache_dir=str(metadata.get("rtmdet_cache_dir", "")),
        )

    monkeypatch.setattr("object_detector_trainer.backends.rtmdet.load_rtmdet_baseline", _fake_load_rtmdet_baseline)


_PATCHERS_BY_BACKEND = {
    "rfdetr": {
        "patch_train": _patch_rfdetr_train,
        "patch_reload": _patch_rfdetr_reload,
    },
    "rtmdet": {
        "patch_train": _patch_rtmdet_train,
        "patch_reload": _patch_rtmdet_reload,
    },
}


def _discover_non_yolo_backend_cases() -> dict[str, dict[str, object]]:
    expected_backends = {backend for backend in SUPPORTED_BACKEND_NAMES if backend != "yolo"}
    if expected_backends != set(_PATCHERS_BY_BACKEND):
        missing = sorted(expected_backends - set(_PATCHERS_BY_BACKEND))
        extra = sorted(set(_PATCHERS_BY_BACKEND) - expected_backends)
        raise RuntimeError(
            "Split evaluate patch coverage drifted. "
            f"Missing backends: {missing or 'none'}. Extra backends: {extra or 'none'}."
        )

    model_by_backend: dict[str, str] = {}
    for backend in sorted(expected_backends):
        model_by_backend[backend] = representative_model_key_for_backend(backend)

    return {
        backend: {
            "model_key": model_by_backend[backend],
            "patch_train": patchers["patch_train"],
            "patch_reload": patchers["patch_reload"],
        }
        for backend, patchers in sorted(_PATCHERS_BY_BACKEND.items())
    }


BACKEND_CASES = _discover_non_yolo_backend_cases()


def _assert_summary_only(summary_dir: Path) -> None:
    assert (summary_dir / "results.csv").exists()
    assert (summary_dir / "results.txt").exists()
    extras = sorted(
        entry.name
        for entry in summary_dir.iterdir()
        if entry.name not in {"results.csv", "results.txt"}
    )
    assert not extras, f"results_comparison must stay summary-only, found: {extras}"


@pytest.mark.parametrize("backend_key", sorted(BACKEND_CASES.keys()))
def test_split_evaluate_reloads_trained_backend_model(
    split_eval_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_key: str,
) -> None:
    case = BACKEND_CASES[backend_key]
    workspace = Path.cwd()
    dataset_name = f"split-eval-{backend_key}"

    create_minimal_dataset(workspace)
    baseline_path = create_baseline_artifact(
        workspace,
        experiment_name=f"{backend_key}-baseline",
    )
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"model": case["model_key"]},
            "evaluation": {"baseline_weights_path": str(baseline_path)},
        },
    )
    args = build_args(dataset_name, {"model": case["model_key"]})

    run_prepare_stage(args)
    best_weights_path = case["patch_train"](monkeypatch)
    reload_calls: list[dict[str, str]] = []
    case["patch_reload"](monkeypatch, reload_calls)

    run_train_stage(args)
    assert best_weights_path.exists()
    marker_payload = json.loads(
        (workspace / ".dvc_artifacts" / "last_train_result.json").read_text(encoding="utf-8")
    )
    assert marker_payload.get("reload_metadata", {}).get("model_backend") == backend_key

    run_evaluate_stage(args, train_result=None)

    assert reload_calls, f"{backend_key} reload path was not used during split evaluate stage."
    assert str(best_weights_path) not in StubYOLO.recorded_models
    _assert_summary_only(workspace / "results_comparison")
    assert (workspace / "metrics.json").exists()
    published_run_dir = workspace / "runs" / Path(marker_payload["train_output_dir"]).name
    assert (published_run_dir / "metadata.yaml").exists()
    assert (published_run_dir / "results.csv").exists()
    assert (published_run_dir / "results.txt").exists()
    assert (published_run_dir / "plots").is_dir()


def test_split_evaluate_uses_current_baseline_path_and_keeps_runs_immutable(
    split_eval_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Path.cwd()
    dataset_name = "split-eval-yolo-baseline"

    baseline_a = create_baseline_artifact(
        workspace,
        weights_path="models/current_best/baseline_a.pt",
        experiment_name="baseline-a",
    )
    baseline_b = create_baseline_artifact(
        workspace,
        weights_path="models/current_best/baseline_b.pt",
        experiment_name="baseline-b",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"model": representative_model_key_for_backend("yolo")},
            "evaluation": {"baseline_weights_path": str(baseline_a)},
        },
    )
    create_local_yolo_checkpoint(workspace)
    args = build_args(dataset_name, {"model": representative_model_key_for_backend("yolo")})

    run_prepare_stage(args)

    run_dir = Path(".dvc_artifacts") / "train_runs" / "split-baseline-check"

    def _fake_train_backend(
        *,
        training_path: Path,
        test_path: Path,
        dataset_name: str,
        resolved_cfg: dict,
        experiment_name: str | None,
    ):
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"trained-yolo")
        model = _StubEvalModel(
            model_name="yolo-split",
            model_backend="yolo",
            resolution=320,
        )
        return model, run_dir, "yolo-split", 320, 1

    monkeypatch.setattr("object_detector_trainer.backends.yolo.train_backend", _fake_train_backend)
    train_result = run_train_stage(args)
    assert train_result.train_output_dir == run_dir
    assert not (run_dir / "metadata.yaml").exists()

    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"model": representative_model_key_for_backend("yolo")},
            "evaluation": {"baseline_weights_path": str(baseline_b)},
        },
    )

    captured: dict[str, str] = {}

    def _fake_load_model_from_weights(path_candidate, metadata_override=None):
        if metadata_override is None:
            captured["baseline_weights_path"] = str(path_candidate)
            model = _StubEvalModel(
                model_name="baseline",
                model_backend="yolo",
                resolution=320,
            )
            return model, "baseline"

        model = _StubEvalModel(
            model_name="yolo-trained-reloaded",
            model_backend="yolo",
            resolution=320,
        )
        return model, "yolo-trained-reloaded"

    monkeypatch.setattr(
        "object_detector_trainer.pipeline.evaluate_stage.load_model_from_weights",
        _fake_load_model_from_weights,
    )

    run_evaluate_stage(args, train_result=None)

    assert Path(captured["baseline_weights_path"]).resolve() == baseline_b.resolve()
    published_run_dir = workspace / "runs" / run_dir.name
    assert (published_run_dir / "metadata.yaml").exists()
    assert (published_run_dir / "results.csv").exists()
    assert (published_run_dir / "results.txt").exists()
    _assert_summary_only(workspace / "results_comparison")
