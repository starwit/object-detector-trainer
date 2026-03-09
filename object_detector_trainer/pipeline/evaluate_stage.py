from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from object_detector_trainer.config.loader import load_config
from object_detector_trainer.evaluation import reports, validate
from object_detector_trainer.evaluation import merged_subset_metrics, visual_comparison
from object_detector_trainer.pipeline.model_state import (
    load_model_from_weights,
    load_persisted_train_result,
    resolve_baseline_model,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    model: object
    experiment_name: str
    train_output_dir: Path
    test_path: Path
    training_path: Path
    image_size: int
    train_epochs: int
    baseline_weights_path: str
    params: dict


def _organize_training_outputs(
    train_output_dir: Path,
    training_path: Path,
    test_path: Path,
    metadata: dict,
) -> None:
    with (train_output_dir / "metadata.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    train_dataset_yaml_path = training_path / "dataset.yaml"
    test_dataset_yaml_path = test_path / "dataset.yaml"
    if train_dataset_yaml_path.exists():
        shutil.copy2(train_dataset_yaml_path, train_output_dir / "train_dataset.yaml")
    if test_dataset_yaml_path.exists():
        shutil.copy2(test_dataset_yaml_path, train_output_dir / "test_dataset.yaml")

    plots_dir = train_output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_files = sorted(train_output_dir.glob("*.png")) + sorted(train_output_dir.glob("*.jpg"))
    for plot_file in plot_files:
        shutil.move(str(plot_file), str(plots_dir / plot_file.name))

    logger.info("Experiment data organized in %s", train_output_dir)
    logger.info("Plots saved to %s", plots_dir)


def _delete_unused_folders() -> None:
    current_runs_dir = Path("runs")
    if not current_runs_dir.exists():
        return
    for folder in current_runs_dir.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def _log_export_guidance(train_output_dir: Path, experiment_name: str) -> None:
    guidance_lines = [
        "",
        "=" * 70,
        "Training complete! Next steps:",
        "=" * 70,
        "",
        "To make this run the baseline for future comparisons:",
        f"  python tools/export_baseline.py --run-dir {train_output_dir}",
        "",
        "Then track it with DVC:",
        "  dvc add models/current_best/best.pt models/current_best/metadata.yaml",
        "  dvc push",
        "  git add models/current_best/best.pt.dvc models/current_best/metadata.yaml.dvc",
        f'  git commit -m "Update baseline to {experiment_name}"',
        "=" * 70,
    ]
    for line in guidance_lines:
        logger.info(line)


def _build_evaluation_context(args, cfg, train_result) -> EvaluationContext:
    baseline_weights_path = str(cfg.evaluation.baseline_weights_path or "").strip()
    if not baseline_weights_path:
        raise ValueError(
            "evaluation.baseline_weights_path must be configured for evaluation. "
            "Baseline comparisons are optional only when no promoted baseline exists yet."
        )

    if train_result is None:
        persisted = load_persisted_train_result()
        model, _display_name = load_model_from_weights(
            persisted.best_weights_path,
            metadata_override=persisted.reload_metadata,
        )
        if model is None:
            raise FileNotFoundError(
                f"Could not load trained model from persisted path: {persisted.best_weights_path}"
            )

        return EvaluationContext(
            model=model,
            experiment_name=persisted.experiment_name,
            train_output_dir=persisted.train_output_dir,
            test_path=persisted.test_path,
            training_path=persisted.training_path,
            image_size=int(persisted.image_size),
            train_epochs=int(persisted.train_epochs),
            baseline_weights_path=baseline_weights_path,
            params=cfg.model_dump(),
        )

    return EvaluationContext(
        model=train_result.model,
        experiment_name=train_result.experiment_name,
        train_output_dir=train_result.train_output_dir,
        test_path=train_result.test_path,
        training_path=train_result.training_path,
        image_size=int(train_result.image_size),
        train_epochs=int(train_result.train_epochs),
        baseline_weights_path=baseline_weights_path,
        params=cfg.model_dump(),
    )


def _append_merged_subset_metrics_to_json(
    *,
    merged_metrics: dict[str, dict],
    metrics_path: Path,
) -> None:
    if not metrics_path.exists() or not merged_metrics:
        return

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    for src_class, metric_values in merged_metrics.items():
        target = metric_values["target_class"]
        for key in ("ap50", "ap", "precision", "recall", "f1_score", "n_objects"):
            metrics_data[f"{src_class}_as_{target}_{key}"] = metric_values[key]

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)


def _run_merged_class_evaluation(
    *,
    context: EvaluationContext,
    output_dir: Path,
    metrics_path: Path,
    baseline_model: object | None,
    baseline_display_name: str | None,
) -> None:
    params = context.params if isinstance(context.params, dict) else {}
    data_cfg = params.get("data", {}) if isinstance(params, dict) else {}
    custom_classes = list(data_cfg.get("custom_classes") or [])
    class_mapping_config = dict(data_cfg.get("class_mapping") or {})
    raw_test_path = Path("raw_data") / "test"

    if not class_mapping_config or not raw_test_path.exists():
        return

    has_merged = any(
        src != target
        for target, sources in class_mapping_config.items()
        for src in (sources if isinstance(sources, list) else [sources])
    )
    if not has_merged:
        return

    merged_class_results: list[tuple[str, dict]] = []

    if baseline_model is not None:
        try:
            baseline_merged = merged_subset_metrics.evaluate_merged_class_subsets(
                baseline_model,
                baseline_display_name or "baseline",
                context.test_path,
                raw_test_path,
                class_mapping_config,
                custom_classes,
                imgsz=context.image_size,
            )
        except merged_subset_metrics.MergedSubsetEvaluationError as exc:
            logger.warning("Skipping merged-class subset evaluation for baseline: %s", exc)
            baseline_merged = {}
        if baseline_merged:
            merged_class_results.append((baseline_display_name or "baseline", baseline_merged))

    try:
        trained_merged = merged_subset_metrics.evaluate_merged_class_subsets(
            context.model,
            context.experiment_name,
            context.test_path,
            raw_test_path,
            class_mapping_config,
            custom_classes,
            imgsz=context.image_size,
        )
    except merged_subset_metrics.MergedSubsetEvaluationError as exc:
        logger.warning("Skipping merged-class subset evaluation for trained model: %s", exc)
        trained_merged = {}
    if trained_merged:
        merged_class_results.append((context.experiment_name, trained_merged))

    if not merged_class_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    reports.write_merged_class_results(output_dir, merged_class_results)
    _append_merged_subset_metrics_to_json(merged_metrics=trained_merged, metrics_path=metrics_path)

    results_csv = output_dir / "results.csv"
    if results_csv.exists():
        reports.create_formatted_table(results_csv, output_dir=output_dir, include_details=True)


def _cleanup_summary_dir(summary_dir: Path) -> None:
    allowed = {"results.csv", "results.txt"}
    for entry in summary_dir.iterdir():
        if entry.name in allowed:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _resolve_optional_baseline_model(
    baseline_weights_path: str,
) -> tuple[object | None, str | None]:
    """Resolve baseline model only when a real baseline artifact is present.

    Baseline comparisons are optional only when *no promoted baseline exists yet*.

    Contract:
    - If the baseline weights file is missing/empty AND there is no nearby
      metadata.yaml, treat this as "no baseline yet" and evaluate the trained
      model only.
    - If metadata.yaml exists next to the baseline path, we consider a promoted
      baseline to exist, and missing/empty weights is an error (likely a missing
      `dvc pull`).
    - If weights exist and are non-empty, we load strictly via resolve_baseline_model()
      (which requires valid metadata including model_backend).
    """

    candidate = Path(baseline_weights_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    metadata_candidates = (
        candidate.parent / "metadata.yaml",
        candidate.parent.parent / "metadata.yaml",
    )
    metadata_dvc_candidates = [path.with_name(f"{path.name}.dvc") for path in metadata_candidates]
    baseline_promoted = any(path.exists() for path in metadata_candidates) or any(
        path.exists() for path in metadata_dvc_candidates
    )

    if candidate.exists() and not candidate.is_file():
        raise FileNotFoundError(
            "evaluation.baseline_weights_path must point to a file, "
            f"got: {candidate}"
        )

    baseline_ready = candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0
    if not baseline_ready:
        if baseline_promoted:
            raise FileNotFoundError(
                "Promoted baseline metadata exists, but the baseline weights file is missing/empty at "
                f"{candidate}. Fetch the baseline (e.g. `dvc pull {candidate}` or run the bootstrap stage)."
            )
        logger.warning(
            "No promoted baseline present yet at %s; skipping baseline comparison.",
            candidate,
        )
        return None, None

    baseline_model, baseline_display_name = resolve_baseline_model(str(candidate))
    return baseline_model, baseline_display_name


def run_evaluate_stage(args, train_result=None, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    raw_val_split = getattr(args, "val_split", None)
    val_split = float(cfg.prepare.val_split if raw_val_split is None else raw_val_split)

    context = _build_evaluation_context(args, cfg, train_result)

    evaluation_output_dir = context.train_output_dir
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_dir = Path("results_comparison")
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_summary_dir(summary_output_dir)

    metrics_path = Path("metrics.json")

    baseline_model, baseline_display_name = _resolve_optional_baseline_model(
        context.baseline_weights_path,
    )

    baseline_results = None
    if baseline_model is not None:
        _, baseline_results = validate.evaluate_and_log_model_results(
            model=baseline_model,
            model_name=baseline_display_name or "baseline",
            test_path=context.test_path,
            image_size=context.image_size,
            output_dir=evaluation_output_dir,
            val_split=val_split,
            train_epochs=0,
            is_original=True,
            metrics_json_path=None,
        )

    retrained_metadata, retrained_results = validate.evaluate_and_log_model_results(
        model=context.model,
        model_name=context.experiment_name,
        test_path=context.test_path,
        image_size=context.image_size,
        output_dir=evaluation_output_dir,
        val_split=val_split,
        train_epochs=context.train_epochs,
        metrics_json_path=metrics_path,
    )

    _organize_training_outputs(
        evaluation_output_dir,
        context.training_path,
        context.test_path,
        retrained_metadata,
    )

    if baseline_model is not None:
        visual_comparison.generate_side_by_side_comparisons(
            original_model=baseline_model,
            retrained_model=context.model,
            test_img_dir=context.test_path / "val" / "images",
            output_dir=evaluation_output_dir,
        )

    _run_merged_class_evaluation(
        context=context,
        output_dir=evaluation_output_dir,
        metrics_path=metrics_path,
        baseline_model=baseline_model,
        baseline_display_name=baseline_display_name,
    )

    reports.mean_table(
        baseline_results,
        retrained_results,
        context.experiment_name,
        baseline_results is not None,
        baseline_display_name,
        output_dir=evaluation_output_dir,
        include_per_class=True,
        include_details=True,
    )
    reports.mean_table(
        baseline_results,
        retrained_results,
        context.experiment_name,
        baseline_results is not None,
        baseline_display_name,
        output_dir=summary_output_dir,
        include_per_class=False,
        include_details=False,
    )

    _delete_unused_folders()
    _log_export_guidance(context.train_output_dir, context.experiment_name)
