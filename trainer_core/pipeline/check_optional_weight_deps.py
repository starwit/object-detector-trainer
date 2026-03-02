from pathlib import Path

from ruamel.yaml import YAML


def ensure_optional_weight_placeholders(root: Path) -> None:
    params_file = root / "params.yaml"
    if not params_file.exists():
        return

    with params_file.open("r", encoding="utf-8") as handle:
        params = YAML(typ="safe").load(handle) or {}
    if not isinstance(params, dict):
        return

    finetune_cfg = (params.get("train") or {}).get("finetune") or {}
    raw_paths = (
        (params.get("evaluation") or {}).get("baseline_weights_path"),
        finetune_cfg.get("weights"),
    )

    seen: set[Path] = set()
    for raw_path in raw_paths:
        path = _resolve_path(root, raw_path)
        if path is None or path in seen:
            continue
        seen.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        path.touch()


def _resolve_path(repo_root: Path, raw: object) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def main() -> None:
    repo_root = Path.cwd()
    params_file = repo_root / "params.yaml"
    if not params_file.exists():
        raise SystemExit(
            "Missing params.yaml in workspace root. "
            "Create it (or run your project bootstrap) before starting the pipeline."
        )

    with params_file.open("r", encoding="utf-8") as handle:
        params = YAML(typ="safe").load(handle) or {}
    if not isinstance(params, dict):
        raise SystemExit("Invalid params.yaml (expected a YAML mapping).")

    finetune_cfg = (params.get("train") or {}).get("finetune") or {}
    baseline_path = _resolve_path(
        repo_root, (params.get("evaluation") or {}).get("baseline_weights_path")
    )
    finetune_path = _resolve_path(repo_root, finetune_cfg.get("weights"))

    missing: list[tuple[str, Path]] = []
    if baseline_path is not None and not baseline_path.exists():
        missing.append(("evaluation.baseline_weights_path", baseline_path))
    if finetune_path is not None and not finetune_path.exists():
        missing.append(("train.finetune.weights", finetune_path))

    if missing:
        lines = ["Optional DVC weight dependencies are missing before the pipeline can start:"]
        for key, path in missing:
            try:
                display = path.relative_to(repo_root)
            except ValueError:
                display = path
            lines.append(f"- {key}: {display}")
        lines.append(
            "Run your project bootstrap to create the 0-byte placeholders, "
            "or create the configured files manually."
        )
        raise SystemExit("\n".join(lines))

    status_file = repo_root / ".tmp" / "dvc_bootstrap.txt"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
