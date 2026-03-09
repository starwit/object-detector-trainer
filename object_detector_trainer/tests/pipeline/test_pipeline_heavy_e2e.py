"""Shared heavy backend contract tests (real one-epoch training)."""

from __future__ import annotations

import copy
import csv
import json
import shutil
from pathlib import Path

import pytest

from object_detector_trainer.backends.training_config import normalize_backend_name
from object_detector_trainer.pipeline import bootstrap_stage
from object_detector_trainer.pipeline.bootstrap_stage import run_bootstrap_stage
from object_detector_trainer.pipeline.evaluate_stage import run_evaluate_stage
from object_detector_trainer.pipeline.prepare_stage import run_prepare_stage
from object_detector_trainer.pipeline.train_stage import run_train_stage
from object_detector_trainer.tests.support.pipeline_test_utils import (
    BASE_PARAMS,
    build_args,
    create_baseline_artifact,
    create_minimal_dataset,
    write_params_yaml,
)

pytestmark = pytest.mark.heavy


def _require_rtmdet_runtime() -> None:
    try:
        import mim  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"openmim not installed; install openmim to run RTMDet heavy contract tests. ({exc})")
    try:
        import mmcv._ext  # type: ignore  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        pytest.fail(
            "RTMDet heavy contract prerequisite missing: full mmcv ops are unavailable "
            f"(install `mmcv`, not `mmcv-lite`; got {exc})."
        )


def _discover_backend_cases() -> list[tuple[str, str]]:
    models_cfg = BASE_PARAMS.get("models", {})
    if not isinstance(models_cfg, dict):
        return []

    by_backend: dict[str, str] = {}
    for model_key, model_cfg in sorted(models_cfg.items()):
        if not isinstance(model_cfg, dict):
            continue
        backend = normalize_backend_name(model_cfg["backend"])
        by_backend.setdefault(backend, str(model_key))

    return sorted((backend, model_key) for backend, model_key in by_backend.items())


BACKEND_CASES = _discover_backend_cases()
if not BACKEND_CASES:
    raise RuntimeError("Heavy backend contract setup failed: no backend cases discovered from BASE_PARAMS.")

def _ensure_yolo_checkpoint(workspace: Path) -> Path:
    """Ensure we have a real YOLO checkpoint for baseline/model loading in heavy tests."""

    checkpoint = workspace / "models" / "pretrained" / "yolo" / "yolov8n.pt"
    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        return checkpoint

    # Use the same download logic as the bootstrap stage so paths stay canonical.
    bootstrap_stage._download_yolo_asset(checkpoint)  # noqa: SLF001
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        pytest.fail(f"Failed to download required YOLO checkpoint: {checkpoint}")
    return checkpoint


def _assert_metrics_contract(workspace: Path) -> None:
    metrics_path = workspace / "metrics.json"
    assert metrics_path.exists(), "metrics.json must be written"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("precision", "recall", "map", "map50", "fitness", "f1_score", "ms_per_frame"):
        assert key in payload, f"Missing metric key: {key}"


def _assert_summary_contract(workspace: Path) -> None:
    summary_dir = workspace / "results_comparison"
    assert summary_dir.exists(), "results_comparison/ must exist"
    csv_path = summary_dir / "results.csv"
    txt_path = summary_dir / "results.txt"
    assert csv_path.exists(), "results_comparison/results.csv must exist"
    assert txt_path.exists(), "results_comparison/results.txt must exist"

    extras = sorted(p.name for p in summary_dir.iterdir() if p.name not in {"results.csv", "results.txt"})
    assert not extras, f"results_comparison/ must stay summary-only; found extras: {extras}"

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "results.csv must contain at least one row"


def _assert_run_contract(run_dir: Path) -> None:
    assert run_dir.exists(), f"Run directory missing: {run_dir}"
    assert (run_dir / "weights" / "best.pt").exists(), "Run must contain weights/best.pt"
    assert (run_dir / "metadata.yaml").exists(), "Run must contain metadata.yaml"
    assert (run_dir / "train_dataset.yaml").exists(), "Run must contain train_dataset.yaml"
    assert (run_dir / "test_dataset.yaml").exists(), "Run must contain test_dataset.yaml"
    assert (run_dir / "plots").is_dir(), "Run must contain plots/"
    assert (run_dir / "test_results.csv").exists(), "Run must contain test_results.csv"
    assert (run_dir / "results.csv").exists(), "Run must contain results.csv"
    assert (run_dir / "results.txt").exists(), "Run must contain results.txt"
    assert (run_dir / "side_by_side_comparisons").is_dir(), "Run must contain side_by_side_comparisons/"


def _write_backend_contract_params(
    workspace: Path,
    *,
    dataset_name: str,
    backend: str,
    model_key: str,
) -> Path | None:
    models_cfg = BASE_PARAMS.get("models", {})
    if not isinstance(models_cfg, dict):
        raise AssertionError("BASE_PARAMS.models must be a mapping.")
    source_cfg = models_cfg.get(model_key)
    if not isinstance(source_cfg, dict):
        raise AssertionError(f"BASE_PARAMS.models.{model_key} must be a mapping.")

    model_cfg: dict[str, object] = copy.deepcopy(source_cfg)
    model_cfg["backend"] = backend
    for key, value in (
        ("epochs", 1),
        ("batch_size", 1),
        ("grad_accum_steps", 1),
        ("image_size", 128),
        ("resolution", 128),
    ):
        if key in model_cfg:
            model_cfg[key] = value

    cache_dir: Path | None = None
    if "cache_dir" in model_cfg:
        cache_dir = workspace / "models" / "pretrained" / backend
        model_cfg["cache_dir"] = str(cache_dir)
    if "allow_download" in model_cfg:
        model_cfg["allow_download"] = False

    yolo_checkpoint = _ensure_yolo_checkpoint(workspace)
    if backend == "yolo":
        model_cfg["checkpoint"] = str(yolo_checkpoint)

    baseline_path = create_baseline_artifact(
        workspace,
        experiment_name=f"{backend}-baseline",
        model_backend="yolo",
        image_size=128,
    )
    shutil.copy2(yolo_checkpoint, baseline_path)

    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": model_key,
                "image_size": 128,
                "epochs": 1,
                "batch_size": 1,
            },
            "models": {model_key: model_cfg},
            "evaluation": {"baseline_weights_path": str(baseline_path)},
        },
    )
    return cache_dir


def _assert_wrapper_contract(model: object, backend: str) -> None:
    assert callable(getattr(model, "predict", None)), f"{backend} adapter must implement predict()."
    assert callable(getattr(model, "val", None)), f"{backend} adapter must implement val()."
    assert hasattr(model, "model_backend"), f"{backend} adapter must set model_backend."
    model_backend = str(getattr(model, "model_backend")).strip().lower()
    assert model_backend == backend, f"{backend} adapter must set model_backend={backend!r}."


@pytest.mark.parametrize(("backend", "model_key"), BACKEND_CASES)
def test_heavy_backend_contract_one_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    model_key: str,
) -> None:
    if backend == "rtmdet":
        _require_rtmdet_runtime()

    monkeypatch.chdir(tmp_path)
    dataset_name = f"heavy-contract-{backend}"

    create_minimal_dataset(tmp_path)
    backend_cache_dir = _write_backend_contract_params(
        tmp_path,
        dataset_name=dataset_name,
        backend=backend,
        model_key=model_key,
    )
    args = build_args(dataset_name, {"model": model_key})
    run_bootstrap_stage(args)
    run_prepare_stage(args)

    train_result = run_train_stage(args)
    run_evaluate_stage(args, train_result=train_result)

    _assert_wrapper_contract(train_result.model, backend)
    assert train_result.train_output_dir.parent.resolve() == (tmp_path / "runs").resolve()
    _assert_run_contract(train_result.train_output_dir)
    _assert_summary_contract(tmp_path)
    _assert_metrics_contract(tmp_path)

    marker = tmp_path / "runs" / ".last_train_result.json"
    assert marker.exists(), "runs/.last_train_result.json must exist"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["reload_metadata"]["model_backend"] == backend
    assert Path(payload["best_weights_path"]).exists()

    if backend_cache_dir is not None:
        cache_files = [path for path in backend_cache_dir.rglob("*") if path.is_file()]
        assert cache_files, f"{backend} contract expected local assets under cache_dir."
