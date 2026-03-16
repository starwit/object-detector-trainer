# Object Detector Trainer

`object-detector-trainer` is a reusable multi-backend training pipeline for object detection.
It prepares datasets from raw labeled data, bootstraps pretrained assets, trains a selected backend, and writes comparable evaluation outputs across model families.

Supported backends:

- Ultralytics YOLO (`backend: yolo`)
- RF-DETR (`backend: rfdetr`)
- RTMDet via MMDetection (`backend: rtmdet`)

This repository documents the trainer itself.
Project-specific concerns such as DVC wiring, repo-local experiment workflows, baseline promotion policy, and large parameter catalogs belong in consumer project repositories.
That also includes any helper that exports or promotes a trained run into a project baseline location.

## What This Repo Covers

- Dataset preparation from `raw_data/`
- Multi-stage training and evaluation CLI
- Backend adapters and pretrained asset bootstrap
- Shared reporting and metrics outputs
- Trainer-level tests, including real backend integration coverage

## Install

Python `3.11+` is required.

Base install:

```bash
poetry install
```

Run the CLI:

```bash
poetry run object-detector-trainer --help
```

You can also invoke it as a module:

```bash
poetry run python -m object_detector_trainer.cli --help
```

### RTMDet Setup

RTMDet needs the optional dependencies plus compiled `mmcv` ops.

Install the extra:

```bash
poetry install -E rtmdet
```

Method 1: OpenMMLab prebuilt wheels

```bash
poetry run mim install mmcv==2.1.0
poetry run python -c "import mmcv._ext"
```

Method 2: build from source

```bash
MMCV_WITH_OPS=1 poetry run pip install "mmcv==2.1.0" --no-binary=mmcv --no-build-isolation --no-cache-dir
poetry run python -c "import mmcv._ext"
```

## Quick Start

The trainer expects a workspace root containing:

```text
workspace/
  params.yaml
  raw_data/
    train/
    test/
```

Minimal `params.yaml`:

```yaml
data:
  dataset_name: demo
  custom_classes: [waste, cigarette]
  use_coco_classes: false

train:
  model: yolo11n
  image_size: 1280
  epochs: 100
  batch_size: 8

models:
  yolo11n:
    backend: yolo
    asset_id: yolo11n.pt

evaluation:
  baseline_weights_path: models/current_best/best.pt
```

Run the pipeline:

```bash
poetry run object-detector-trainer --stage bootstrap --workspace-root /path/to/workspace
poetry run object-detector-trainer --stage prepare --workspace-root /path/to/workspace
poetry run object-detector-trainer --stage train --workspace-root /path/to/workspace
poetry run object-detector-trainer --stage evaluate --workspace-root /path/to/workspace
```

Or run everything in one command:

```bash
poetry run object-detector-trainer --stage all --workspace-root /path/to/workspace
```

`--config` defaults to `params.yaml` inside the workspace root.
Relative config paths are resolved from `--workspace-root`.
When using the trainer CLI directly, `prepare` and `all` skip rebuilding an existing
`datasets/<dataset_name>/` unless you pass `--recreate-dataset`.

## CLI Stages

- `bootstrap`: resolve or download pretrained assets for the selected model
- `prepare`: build a YOLO-style dataset under `datasets/<dataset_name>/`
- `train`: train the selected backend and write run artifacts under `runs/`
- `evaluate`: evaluate the trained model and compare it against the configured baseline when available
- `all`: run `bootstrap`, `prepare`, `train`, and `evaluate` in order

Common overrides:

- `--model <key>` to select a key from `models.*`
- `--set key=value` for ad hoc config overrides
- `--dataset-name <name>` to override `data.dataset_name`
- `--val-split` and `--test-split` to adjust data splits
- `--recreate-dataset` to rebuild `datasets/<dataset_name>/`
- `--augment-multiplier` to increase dataset augmentation during preparation
- `--folder-subset <folder> <ratio>` to override `prepare.folder_subsets`
- `--all-models` to bootstrap every configured model instead of only the active one

## Input Data

Put source data under:

- `raw_data/train/` for train and validation data
- `raw_data/test/` for a holdout test set

The importer accepts these source layouts:

- Standard YOLO folders with `images/` and `labels/`
- CVAT YOLO exports with `train.txt`
- Scene-based folders inside `raw_data/test/`
- Folders containing `data.yaml` so classes can be remapped by name

Typical layout:

```text
raw_data/
  train/
    source_a/
      images/
      labels/
  test/
    source_b/
      images/
      labels/
```

## Configuration

The trainer keeps its top-level configuration intentionally small:

- `data` describes dataset naming and class strategy
- `prepare` controls splits, augmentation, folder subsets, and optional auto-replay
- `train` selects the model key and shared training defaults
- `models` defines backend-specific model entries
- `evaluation.baseline_weights_path` tells evaluation where a comparison baseline would live

For larger model catalogs, use `models_defaults` to share backend-specific defaults such as `cache_dir` and `allow_download`.

Keep project-specific experiment matrices and parameter-heavy workflows in consumer project repositories rather than here.

For the exact validated config shape, see `object_detector_trainer/config/schema.py`.
For backend-specific resolution logic, see `object_detector_trainer/backends/training_config.py`.

### Baseline Behavior

`evaluation.baseline_weights_path` must always be configured.

Evaluation behaves like this:

- If the baseline weights file is missing or empty and there is no nearby `metadata.yaml`, evaluation runs on the trained model only
- If `metadata.yaml` exists next to the baseline path, the baseline is treated as promoted and the weights file must exist and be non-empty
- Fine-tune weights and baseline weights are separate concerns

This keeps the trainer usable in a fresh standalone workspace without forcing a baseline to exist on day one.

## Outputs

The trainer writes:

- `datasets/<dataset_name>/` for prepared YOLO-style datasets
- `runs/<run_name>/` for backend outputs, weights, plots, metadata, and evaluation artifacts
- `runs/.last_train_result.json` so `evaluate` can reload the latest trained model
- `metrics.json` for machine-readable metrics
- `results_comparison/results.csv` and `results_comparison/results.txt` for summary comparison outputs

## Backend Notes

### YOLO

- `models.<key>.asset_id` is the checkpoint filename, for example `yolo11n.pt`
- Bootstrap stores checkpoints under `models/pretrained/yolo/` by default

### RF-DETR

- `models.<key>.variant` is required
- `models.<key>.asset_id` is the checkpoint filename
- Bootstrap stores checkpoints under `models/pretrained/rfdetr/` by default
- The trainer adapts prepared datasets into the layout expected by RF-DETR during training

### RTMDet

- `models.<key>.asset_id` is the MMDetection config name, for example `rtmdet_m_8xb32-300e_coco`
- Bootstrap stores configs and checkpoints under `models/pretrained/rtmdet/` by default
- Training and evaluation use a temporary COCO export internally

For reproducible offline RTMDet runs, you can predownload model assets:

```bash
poetry run mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest models/pretrained/rtmdet
```

## Testing

Fast test suite:

```bash
poetry run pytest
```

Heavy integration suite:

```bash
poetry run pytest object_detector_trainer/tests --heavy
```

Notes:

- Heavy tests run real backend training
- The first heavy run may download pretrained assets into `models/pretrained/<backend>/`
- Later heavy runs reuse that shared cache
- If you add a new backend, also add a representative heavy-test model in `object_detector_trainer/tests/support/pipeline_test_utils.py`

## Code Map

- `object_detector_trainer/cli.py`: CLI entrypoint
- `object_detector_trainer/pipeline/`: stage orchestration
- `object_detector_trainer/backends/`: backend integrations
- `object_detector_trainer/dataprep/`: raw-data ingestion and dataset building
- `object_detector_trainer/evaluation/`: reports and metrics
- `object_detector_trainer/config/`: schema, loading, and overrides
- `object_detector_trainer/tests/`: trainer-level test suite
