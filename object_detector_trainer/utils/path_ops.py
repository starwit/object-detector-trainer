"""Shared path and file-transfer primitives used in multiple core modules.

Why this file exists:
1) run-dir uniqueness logic is reused by multiple backends,
2) dataset-name sanitization rules must stay consistent across exporters,
3) copy/link behavior should be implemented once (same fallback semantics).
"""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path


def resolve_unique_run_dir(root: Path, run_name: str) -> Path:
    """Return a unique directory path under ``root``.

    If ``root/run_name`` exists, suffix the name with ``_N`` (N starting at 1)
    until a free directory name is found.
    """
    candidate = root / run_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while (root / f"{run_name}_{suffix}").exists():
        suffix += 1
    return root / f"{run_name}_{suffix}"


def safe_dataset_dirname(dataset_name: str) -> str:
    """Return a filesystem-safe single path component for a dataset name."""
    raw = str(dataset_name or "").strip()
    raw = Path(raw).name  # drop any path components (prevents traversal/absolute paths)
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in raw)
    safe = safe.strip("_-")
    return safe or "dataset"


def link_or_copy(src: Path, dst: Path, *, prefer_hardlink: bool = True) -> None:
    """Create ``dst`` from ``src`` via hardlink when possible, else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                dst.unlink()
                os.link(src, dst)
                return
    shutil.copy2(src, dst)
