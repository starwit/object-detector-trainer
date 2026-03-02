from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trainer_core.pipeline.check_optional_weight_deps import (
    ensure_optional_weight_placeholders,
    main as check_optional_weight_deps_main,
)


def _write_params(
    workspace: Path,
    *,
    baseline_weights_path: str | None = None,
    finetune_weights_path: str | None = None,
) -> None:
    payload: dict = {
        "data": {"dataset_name": "sample-ds"},
        "train": {"finetune": {}},
        "evaluation": {},
    }
    if baseline_weights_path is not None:
        payload["evaluation"]["baseline_weights_path"] = baseline_weights_path
    if finetune_weights_path is not None:
        payload["train"]["finetune"]["weights"] = finetune_weights_path
    (workspace / "params.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_new_clone_requires_setup_then_passes_after_placeholder_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_params(
        tmp_path,
        baseline_weights_path="models/current_best/best.pt",
        finetune_weights_path="models/finetune/start.pt",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="Optional DVC weight dependencies are missing"):
        check_optional_weight_deps_main()

    ensure_optional_weight_placeholders(tmp_path)
    check_optional_weight_deps_main()

    status_file = tmp_path / ".tmp" / "dvc_bootstrap.txt"
    assert status_file.exists()
    assert status_file.read_text(encoding="utf-8") == "ok\n"


def test_check_optional_weight_deps_accepts_existing_baseline_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_params(tmp_path, baseline_weights_path="models/current_best/best.pt")
    baseline = tmp_path / "models" / "current_best" / "best.pt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_bytes(b"real-baseline")
    monkeypatch.chdir(tmp_path)

    check_optional_weight_deps_main()
    assert (tmp_path / ".tmp" / "dvc_bootstrap.txt").exists()


def test_check_optional_weight_deps_dvc_pointer_without_pt_is_not_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_params(tmp_path, baseline_weights_path="models/current_best/best.pt")
    pointer = tmp_path / "models" / "current_best" / "best.pt.dvc"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("outs:\n- md5: deadbeef\n  path: best.pt\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        check_optional_weight_deps_main()

    message = str(exc_info.value)
    assert "evaluation.baseline_weights_path" in message
    assert "models/current_best/best.pt" in message


def test_check_optional_weight_deps_allows_unconfigured_optional_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_params(tmp_path)
    monkeypatch.chdir(tmp_path)

    check_optional_weight_deps_main()
    assert (tmp_path / ".tmp" / "dvc_bootstrap.txt").exists()


def test_check_optional_weight_deps_fails_fast_without_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="Missing params.yaml"):
        check_optional_weight_deps_main()
