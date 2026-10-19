"""The training pipeline: prepare the dataset, train a model, evaluate it.

Every function takes the validated config and the workspace folder. Paths in
the config are relative to the workspace. Outputs::

    datasets/<data.dataset_name>/   prepared dataset (see dataset.py)
    runs/<run-name>/                trained model, reports and metrics (see runs.py)
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from object_detector_trainer.backends import get_backend
from object_detector_trainer.backends.common import ModelOptions, TrainJob, resolve_device
from object_detector_trainer.classes import merged_classes
from object_detector_trainer.config import Config
from object_detector_trainer.dataset import build_dataset, log_dataset_summary, read_dataset_class_names
from object_detector_trainer.evaluation.comparison import write_comparison_images
from object_detector_trainer.evaluation.evaluator import evaluate_model
from object_detector_trainer.evaluation.reports import write_reports
from object_detector_trainer.runs import best_weights_path, load_model, read_metadata, run_dir, write_metadata

log = logging.getLogger(__name__)


def run_all(config: Config, workspace: Path, run_name: str) -> None:
    prepare(config, workspace)
    train(config, workspace, run_name)
    evaluate(config, workspace, run_name)


def prepare(config: Config, workspace: Path) -> Path:
    """Build ``datasets/<name>`` from the sources. Always rebuilds from scratch."""
    dataset_dir = workspace / "datasets" / config.data.dataset_name
    log.info("Preparing dataset %s", dataset_dir)
    build_dataset(config, workspace)
    log_dataset_summary(dataset_dir)
    return dataset_dir


def train(config: Config, workspace: Path, run_name: str) -> Path:
    """Train ``train.model`` into ``runs/<run_name>/train`` and return the best weights."""
    model_key = config.train.model
    options = config.model_options(model_key)
    backend = get_backend(options.backend)

    dataset_dir = workspace / "datasets" / config.data.dataset_name
    if not (dataset_dir / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_dir}. Run --stage prepare first.")
    train_dir = run_dir(workspace, run_name) / "train"
    if train_dir.exists():
        raise FileExistsError(f"Run folder already exists: {train_dir}. Choose another --run-name or delete it.")
    train_dir.mkdir(parents=True)

    job = TrainJob(
        dataset_dir=dataset_dir,
        class_names=list(read_dataset_class_names(dataset_dir).values()),
        output_dir=train_dir,
        start_weights=find_start_weights(options, workspace),
        epochs=options.epochs or config.train.epochs,
        image_size=options.image_size or config.train.image_size,
        batch_size=options.batch_size,
        seed=config.seed,
        device=resolve_device(config.train.device),
        options=options,
    )
    log.info(
        "Training %s (%s) from %s: %d epochs, image size %d, batch size %d, device %s, output %s",
        model_key, options.backend, job.start_weights, job.epochs, job.image_size,
        job.batch_size, job.device, train_dir,
    )

    set_seed(config.seed)
    weights_path = backend.train(job)
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise RuntimeError(f"Training did not produce weights: {weights_path}")

    write_metadata(train_dir, {
        "experiment_name": run_name,
        "model_key": model_key,
        "model_backend": options.backend,
        "image_size": job.image_size,
        "epochs": job.epochs,
        "class_names": dict(enumerate(job.class_names)),
        "model_options": options.model_dump(mode="json"),
    })
    with (train_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
    log.info("Best weights: %s", weights_path)
    return weights_path


def evaluate(config: Config, workspace: Path, run_name: str) -> dict:
    """Evaluate the run's best weights on the test split, and the baseline when it can be compared."""
    dataset_dir = workspace / "datasets" / config.data.dataset_name
    weights_path = best_weights_path(workspace, run_name)
    if not weights_path.is_file():
        raise FileNotFoundError(f"No trained weights for run {run_name!r}: {weights_path}")
    eval_dir = run_dir(workspace, run_name) / "eval"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)

    set_seed(config.seed)
    merged = merged_classes(config.data)
    results = {}
    baseline_model, baseline_note = load_baseline(config, workspace, dataset_dir)
    if baseline_note:
        log.warning(baseline_note)
    if baseline_model is not None:
        baseline_path = workspace / config.evaluation.baseline_weights_path
        results["baseline"] = evaluate_weights(baseline_model, baseline_path, dataset_dir, merged)
    trained_model = load_model(weights_path, device=config.train.device)
    results[run_name] = evaluate_weights(
        trained_model, weights_path, dataset_dir, merged, predictions_dir=eval_dir / "predictions"
    )

    write_reports(eval_dir, results, baseline_note)
    with (run_dir(workspace, run_name) / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results[run_name], handle, indent=2)
    if baseline_model is not None:
        test_images = dataset_dir / "test" / "images"
        write_comparison_images(baseline_model, trained_model, test_images, eval_dir / "comparisons")

    log.info("\n%s", (eval_dir / "report.txt").read_text(encoding="utf-8"))
    return results[run_name]


def evaluate_weights(
    model, weights_path: Path, dataset_dir: Path, merged: dict[str, str], predictions_dir: Path | None = None,
) -> dict:
    """Evaluate a loaded model and record which weights it came from."""
    log.info("Evaluating %s", weights_path)
    result = evaluate_model(model, dataset_dir, merged, predictions_dir)
    result["weights"] = str(weights_path)
    result["image_size"] = int(read_metadata(weights_path)["image_size"])
    return result


def load_baseline(config: Config, workspace: Path, dataset_dir: Path):
    """The baseline model, or None and the reason why there is no comparison.

    A missing baseline or one with other classes never stops the evaluation:
    the trained model is still evaluated and the report says why the
    baseline is not compared.
    """
    setting = config.evaluation.baseline_weights_path
    if not setting:
        return None, None
    path = workspace / setting
    if not path.is_file() or path.stat().st_size == 0:
        if path.with_name(path.name + ".dvc").is_file():
            return None, (
                f"No baseline comparison: {setting} is tracked by DVC but not downloaded. "
                f"Run `dvc pull {setting}.dvc`, then run the experiment again."
            )
        return None, f"No baseline comparison: there is no baseline at {setting} yet."

    dataset_classes = read_dataset_class_names(dataset_dir)
    try:
        model = load_model(path, device=config.train.device)
    except Exception as error:
        # Baselines are optional; checkpoint/metadata loaders can raise backend-specific errors.
        return None, (
            f"No baseline comparison: could not load {setting}: {error}. "
            "Check or restore the baseline weights and metadata.yaml."
        )
    if model.class_names != dataset_classes:
        return None, (
            "No baseline comparison: the baseline was trained with other classes "
            f"({list(model.class_names.values())}) than this dataset ({list(dataset_classes.values())})."
        )
    return model, None


def find_start_weights(options: ModelOptions, workspace: Path) -> Path:
    """``init_weights`` when fine-tuning, otherwise the pretrained ``asset_id`` (downloaded if needed)."""
    if options.init_weights:
        path = workspace / options.init_weights
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"init_weights must point to an existing, non-empty file: {path}")
        return path

    cache_dir = workspace / (options.cache_dir or f"models/pretrained/{options.backend}")
    path = cache_dir / options.asset_id
    if not path.is_file() or path.stat().st_size == 0:
        if not options.allow_download:
            raise FileNotFoundError(f"Pretrained weights are missing and allow_download is false: {path}")
        log.info("Downloading pretrained weights %s to %s", options.asset_id, path)
        get_backend(options.backend).download_weights(options.asset_id, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Pretrained weights could not be downloaded: {path}")
    return path


def set_seed(seed: int) -> None:
    """Seed every random number generator and ask PyTorch for deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Some backend operations have no deterministic version; warn about them instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
