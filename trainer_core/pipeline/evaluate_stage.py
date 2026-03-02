from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from trainer_core.config.loader import load_config
from trainer_core.evaluation import reports, validate
from trainer_core.evaluation import merged_subset_metrics, visual_comparison
from trainer_core.pipeline.model_state import (
    load_model_from_weights,
    load_persisted_train_result,
    resolve_baseline_model,
)
from trainer_core.plugins.replay import build_or_update_replay_set

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
    baseline_weights_path: str | None
    fallback_checkpoint: str
    finetune_weights_path: str | None
    params: dict
    split_evaluate_mode: bool


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

        baseline_weights_path = cfg.evaluation.baseline_weights_path
        if baseline_weights_path is None:
            baseline_weights_path = persisted.baseline_weights_path

        return EvaluationContext(
            model=model,
            experiment_name=persisted.experiment_name,
            train_output_dir=persisted.train_output_dir,
            test_path=persisted.test_path,
            training_path=persisted.training_path,
            image_size=int(persisted.image_size),
            train_epochs=int(persisted.train_epochs),
            baseline_weights_path=baseline_weights_path,
            fallback_checkpoint=persisted.fallback_checkpoint,
            finetune_weights_path=persisted.finetune_weights_path,
            params=cfg.model_dump(),
            split_evaluate_mode=True,
        )

    return EvaluationContext(
        model=train_result.model,
        experiment_name=train_result.experiment_name,
        train_output_dir=train_result.train_output_dir,
        test_path=train_result.test_path,
        training_path=train_result.training_path,
        image_size=int(train_result.image_size),
        train_epochs=int(train_result.train_epochs),
        baseline_weights_path=train_result.baseline_weights_path,
        fallback_checkpoint=train_result.fallback_checkpoint,
        finetune_weights_path=train_result.finetune_weights_path,
        params=train_result.params,
        split_evaluate_mode=False,
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
        reports.create_formatted_table(results_csv, output_dir=output_dir)


def run_evaluate_stage(args, train_result=None, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    val_split = float(getattr(args, "val_split", cfg.prepare.val_split))

    context = _build_evaluation_context(args, cfg, train_result)

    evaluation_output_dir = (
        Path("results_comparison") if context.split_evaluate_mode else context.train_output_dir
    )
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path("metrics.json")

    baseline_model, baseline_display_name = resolve_baseline_model(
        context.baseline_weights_path,
        context.fallback_checkpoint,
        finetune_weights_path=context.finetune_weights_path,
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
            baseline_model=None,
            metrics_json_path=None,
        )

    retrained_metadata, _ = validate.evaluate_and_log_model_results(
        model=context.model,
        model_name=context.experiment_name,
        test_path=context.test_path,
        image_size=context.image_size,
        output_dir=evaluation_output_dir,
        val_split=val_split,
        train_epochs=context.train_epochs,
        baseline_model=baseline_model,
        baseline_display_name=baseline_display_name,
        baseline_results=baseline_results,
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

    auto_replay_cfg = cfg.prepare.auto_replay
    if context.split_evaluate_mode and auto_replay_cfg and auto_replay_cfg.get("enabled", False):
        build_or_update_replay_set(
            model=context.model,
            training_path=context.training_path,
            train_output_dir=context.train_output_dir,
            config=auto_replay_cfg,
        )

    if not context.split_evaluate_mode:
        _delete_unused_folders()
    _log_export_guidance(context.train_output_dir, context.experiment_name)
