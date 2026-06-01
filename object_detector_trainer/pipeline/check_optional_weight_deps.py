from __future__ import annotations

"""DVC preflight helpers for optional model-weight paths.

This module is intentionally lightweight and side-effect free unless invoked
directly (``python -m object_detector_trainer.pipeline.check_optional_weight_deps``)
or called by DVC as a stage command.

It exists to support fresh clones where DVC stage dependencies reference weight
paths (e.g. ``models/current_best/best.pt``) that may not exist locally yet.
We create empty placeholders so DVC dependency checks do not fail before the
pipeline code can emit a clear, domain-specific error message.

Runtime code treats empty files as missing; this does not introduce fallback
behavior.
"""

from pathlib import Path
import yaml

from object_detector_trainer.utils.path_ops import resolve_workspace_path


def ensure_optional_weight_placeholders(workspace_root: Path) -> list[Path]:
    """Touch missing configured weight paths so DVC deps exist.

    This only creates files when they are missing. It never truncates existing
    files (even if empty).
    """

    params_file = workspace_root / "params.yaml"
    if not params_file.exists():
        return []

    params = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    train_cfg = params.get("train", {})
    finetune_cfg = train_cfg.get("finetune", {})
    eval_cfg = params.get("evaluation", {})

    raw_paths = (
        eval_cfg.get("baseline_weights_path"),
        finetune_cfg.get("weights"),
    )

    created: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        path = resolve_workspace_path(raw_path, root=workspace_root)
        if path is None:
            continue
        if path in seen:
            continue
        seen.add(path)

        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        created.append(path)

    return created


def main() -> int:
    workspace_root = Path.cwd()
    ensure_optional_weight_placeholders(workspace_root)

    marker = workspace_root / ".tmp" / "dvc_bootstrap.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
