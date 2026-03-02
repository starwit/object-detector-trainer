from __future__ import annotations

import re
from pathlib import Path

from trainer_core.utils.path_ops import resolve_unique_run_dir, safe_dataset_dirname


def test_safe_dataset_dirname_is_stable_and_safe() -> None:
    assert safe_dataset_dirname("") == "dataset"
    assert safe_dataset_dirname("   ") == "dataset"
    assert safe_dataset_dirname("../my dataset") == "my_dataset"
    assert safe_dataset_dirname("___---") == "dataset"

    sanitized = safe_dataset_dirname("My:Dataset v1")
    assert sanitized
    assert re.fullmatch(r"[A-Za-z0-9_-]+", sanitized)


def test_resolve_unique_run_dir_appends_suffixes(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()

    first = resolve_unique_run_dir(root, "exp")
    assert first == root / "exp"

    first.mkdir()
    second = resolve_unique_run_dir(root, "exp")
    assert second == root / "exp_1"

    second.mkdir()
    third = resolve_unique_run_dir(root, "exp")
    assert third == root / "exp_2"
