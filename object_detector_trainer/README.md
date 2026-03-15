# Trainer Core — Multi-backend Object Detection Pipeline

Trainer Core is a reusable training/evaluation engine for object detection.
It prepares YOLO-style datasets, trains a selected backend, and produces
comparable metrics/reports across model families.

It is designed to be consumed from a project repository (for example a
`waste-detection` repo) where that project owns `params.yaml`, DVC wiring, and
its own thin entrypoint wrapper.

## Supported backends

- Ultralytics YOLO (`backend: yolo`)
- RF-DETR (`backend: rfdetr`)
- RTMDet via MMDetection (`backend: rtmdet`)

## Repository layout

- Backends (`yolo`, `rfdetr`, `rtmdet`): `object_detector_trainer/backends/`
- Dataset preparation/import: `object_detector_trainer/dataprep/`
- Pipeline stages: `object_detector_trainer/pipeline/`
- Evaluation + reporting: `object_detector_trainer/evaluation/`
- Model adapters (so non-YOLO models look like Ultralytics for eval): `object_detector_trainer/wrappers/`
- Config schema + overrides: `object_detector_trainer/config/`

## Pipeline stages

The pipeline is split into three explicit lifecycle stages:

1. **Prepare** (`--stage prepare`)
   - Reads raw images/labels under `raw_data/`
   - Builds a YOLO-style dataset under `datasets/<dataset_name>/`
   - Applies class mapping / class merging (if configured)
2. **Train** (`--stage train`)
   - Trains the selected backend (`yolo`, `rfdetr`, or `rtmdet`)
   - Writes run artifacts under `runs/` (including `weights/best.pt`)
   - Persists `runs/.last_train_result.json` for the evaluate stage
3. **Evaluate** (`--stage evaluate`)
   - Loads the trained model (from the persisted pointer if needed)
   - Resolves a baseline model for comparison
   - Writes `metrics.json`
   - Writes run-owned artifacts into `runs/<run_name>/`
   - Writes summary-only comparison outputs to `results_comparison/` (`results.csv`, `results.txt`)

## Running

Run the stages via the core CLI:

```bash
python -m object_detector_trainer.cli --stage bootstrap --model <model-key>
python -m object_detector_trainer.cli --stage prepare --dataset-name <name>
python -m object_detector_trainer.cli --stage train --dataset-name <name> --model <model-key>
python -m object_detector_trainer.cli --stage evaluate --dataset-name <name> --model <model-key>
python -m object_detector_trainer.cli --stage all --dataset-name <name> --model <model-key>
```

`--stage all` runs bootstrap, prepare, train, and evaluate in order.

### Typical project wrapper (`train.py`)

In a project repo that depends on `object-detector-trainer` (import namespace `object_detector_trainer`), use a tiny wrapper that
injects project-local defaults for workspace + config:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from object_detector_trainer.cli import main as core_main

PROJECT_ROOT = Path(__file__).resolve().parent


def _inject_defaults(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage")
    parser.add_argument("--workspace-root")
    parser.add_argument("--config")
    known, _ = parser.parse_known_args(argv)

    out = list(argv)
    if known.stage is None:
        out.extend(["--stage", "train"])
    if known.workspace_root is None:
        out.extend(["--workspace-root", str(PROJECT_ROOT)])
    if known.config is None:
        out.extend(["--config", str(PROJECT_ROOT / "params.yaml")])
    return out


if __name__ == "__main__":
    raise SystemExit(core_main(_inject_defaults(sys.argv[1:])))
```

### Common CLI options

- `--workspace-root .`: root directory for pipeline I/O (`raw_data/`, `datasets/`, `runs/`, `results_comparison/`, `metrics.json`)
- `--config params.yaml`: config path (defaults to `params.yaml`)
- `--model <key>`: selects a key under `models.*` (overrides `train.model`)
- `--set key=value`: override config keys (supports dot paths)
- `--val-split`, `--test-split`: adjust dataset splits
- `--recreate-dataset`: rebuild `datasets/<dataset_name>/` from `raw_data/`
- `--augment-multiplier`: increase augmentation rate during preparation
- `--folder-subset <folder> <ratio>`: override `prepare.folder_subsets`

## Tests

Core test suites live under `object_detector_trainer/tests/`:

- `data/`
- `pipeline/`
- `wrappers/`
- `utils/`

When split into a dedicated repository, keep this test layout unchanged.

## Split Checklist

When extracting Trainer Core into its own repository/module (for example
`object-detector-trainer`):

1. Keep the Python import package name as `object_detector_trainer` initially (avoid churn).
2. Move `object_detector_trainer/` and `object_detector_trainer/tests/` as-is into the new repo.
3. Add package metadata (`pyproject.toml`) in the new repo root.
4. In consumer project repos:
   - add dependency on the published core package,
   - keep a project-local `train.py` wrapper,
   - keep project-local `params.yaml`, `dvc.yaml`, and setup/bootstrap scripts.

## Configuration (`params.yaml`)

Backends are selected via `train.model` (a key under `models.*`) and `models.<key>.backend`.

See `object_detector_trainer/config/schema.py` for the validated shape and defaults.

Minimal example:

```yaml
data:
  dataset_name: waste-detection
  # Either set a class list...
  custom_classes: [waste, cigarette]
  use_coco_classes: false
  # Optional: merge multiple source classes into one during training.
  class_mapping: {}

prepare:
  val_split: 0.1
  test_split: 0.1
  augment_multiplier: 1
  folder_subsets: {}

train:
  model: yolo11m                # key under models.*
  image_size: 1280
  epochs: 100
  batch_size: 4
  finetune:
    enabled: false
    weights: models/current_best/best.pt

models:
  yolo11m:
    backend: yolo
    asset_id: yolo11m.pt
  rfdetr-medium:
    backend: rfdetr
    variant: medium
    asset_id: rf-detr-medium.pth
    resolution: 1280
  rtmdet-m:
    backend: rtmdet
    asset_id: rtmdet_m_8xb32-300e_coco

evaluation:
  baseline_weights_path: models/current_best/best.pt
```

### `data.custom_classes` vs `data.use_coco_classes`

- If `data.custom_classes` is non-empty, those names become class 0…n-1.
- If it’s empty and `data.use_coco_classes: true`, the pipeline falls back to the configured COCO subset.

### Baseline & fine-tune weights

- `evaluation.baseline_weights_path` must always be configured. On fresh clones that path may point to a missing/empty file until a baseline is promoted; evaluation then runs on the trained model only.
- If `metadata.yaml` exists next to `evaluation.baseline_weights_path`, the baseline is considered promoted and the weights file must also exist and be non-empty (otherwise evaluation fails loudly and you need to fetch/export the baseline).
- Fine-tuning weights (`train.finetune.weights`) are required when `train.finetune.enabled: true` and must be a non-empty file.
- Bootstrap provisions model assets only. Promoted baselines remain an explicit fetch/export step.
- In a consumer project, use the project wrapper or `python -m object_detector_trainer.cli --stage bootstrap --config <params.yaml>` to prefetch model assets explicitly.
- In this repo, use `python scripts/provision_heavy_test_assets.py` to prefetch the local asset cache required by backend-heavy tests.

## Backends

### Ultralytics YOLO (`backend: yolo`)

- Uses `ultralytics.YOLO`.
- `models.<key>.asset_id` is the pretrained checkpoint filename (e.g. `yolo11m.pt`).
- The checkpoint is expected at `models_defaults.yolo.cache_dir / asset_id` after bootstrap.
- Fine-tuning is supported via `train.finetune.*`.

### RF-DETR (`backend: rfdetr`)

- Uses the `rfdetr` Python package.
- `rfdetr` is a standard project dependency.
- `models.<key>.asset_id` is the pretrained checkpoint filename (e.g. `rf-detr-medium.pth`).
- The checkpoint is expected at `models_defaults.rfdetr.cache_dir / asset_id` after bootstrap.
- The backend trains via RF-DETR’s Roboflow dataset loader; the pipeline creates a tiny bridge layout under `.tmp/` and cleans it up after training.
- RF-DETR resolution has divisibility constraints; the pipeline will auto-adjust and print a warning if needed.

### RTMDet / MMDetection (`backend: rtmdet`)

- Requires `mmdet`, `mmengine`, and **full `mmcv` ops** (`mmcv`, not `mmcv-lite`).
- `models.<key>.asset_id` is the MMDetection config name (e.g. `rtmdet_m_8xb32-300e_coco`).
- Config/checkpoint files are expected under `models_defaults.rtmdet.cache_dir` after bootstrap.
- Training/evaluation runs do not auto-download RTMDet assets; bootstrap is the explicit provisioning step.
- Uses a temporary COCO export under `.tmp/` for training and evaluation.
- You can predownload configs/checkpoints for reproducible offline runs:

```bash
mim download rtmdet --config rtmdet_m_8xb32-300e_coco --dest models/pretrained/rtmdet
```

## Raw data layouts (`raw_data/`)

Put raw data in:

- `raw_data/train/` for train+val
- `raw_data/test/` for the final hold-out set (optional)

The importer accepts any of the following layouts:

| # | Layout | What to do | Notes |
|---|--------|------------|-------|
| 1 | **CVAT YOLO export** | Drop the whole export folder (`data.yaml`, `images/`, `labels/`, `train.txt`). | `train.txt` is automatically parsed. |
| 2 | **Standard YOLO** | Inside a subfolder create `images/` & `labels/`. | Class IDs will be remapped if needed. |
| 3 | **Scene-based test sets** | One subfolder per scene, each with its own `images/` & `labels/`. | Scene name is appended to filenames so metrics stay separate. |
| 4 | **Any folder containing `data.yaml` / `dataset.yaml`** | Copy it in. | Class IDs will be remapped by name matching if needed. |

## Custom classes

Configure custom classes in `params.yaml`:

```yaml
data:
  custom_classes: [waste, cigarette]
  use_coco_classes: false
```

When importing datasets that include a `data.yaml`, the pipeline can remap class IDs based on name matching.

## Class mapping (merging classes)

The class mapping feature allows you to **merge multiple classes into one** during training/evaluation without modifying your raw data.

Example:

```yaml
data:
  custom_classes: [waste, cigarette]
  use_coco_classes: false
  class_mapping:
    waste: [waste, cigarette]
```

With this configuration:

- Raw data remains unchanged in `raw_data/`
- During dataset preparation, labels are remapped
- The trained model sees only the merged target classes

## Folder subsets (`prepare.folder_subsets`)

You can limit or oversample specific source folders during dataset preparation.

Example:

```yaml
prepare:
  folder_subsets:
    uavvaste: 0.5        # use 50% of images from this folder
    taco: 0.2            # use 20%
    cw32-08-07-train: 2  # 200% = oversample (training split only)
```

Behavior:

- `0 < ratio < 1.0`: subsample a folder proportionally
- `ratio > 1.0` (float): oversample a folder (applied to training split only)
- `ratio >= 2` (int): treat as an **absolute count** (“use exactly N images”)

CLI override (multiple allowed):

```bash
python -m object_detector_trainer.cli \
  --stage prepare -d waste-detection \
  --folder-subset uavvaste 0.5 \
  --folder-subset cw32-08-07-train 2.0
```

## Fine-tuning (YOLO)

Fine-tuning is supported for YOLO backends:

```yaml
train:
  finetune:
    enabled: true
    weights: models/current_best/best.pt
    lr: 0.0001
    epochs: 60
    freeze_backbone: false
```

Notes:

- Fine-tuning rejects missing or empty weight files.
- Evaluation compares against the configured baseline only; it does not substitute fine-tune weights or alternate checkpoints.

## Outputs

- Prepared datasets: `datasets/<dataset_name>/train` and `datasets/<dataset_name>/test`
- Training runs: `runs/<run_name>/` for all backends, with `weights/best.pt`, `metadata.yaml`, `plots/`, run-level `results.csv`/`results.txt`, and evaluation artifacts
- Metrics JSON: `metrics.json`
- Results summary: `results_comparison/results.csv` and `results_comparison/results.txt` only
- Persisted pointer for evaluate: `runs/.last_train_result.json`

## Testing

This repo includes:

- Unit tests + E2E pipeline smoke tests using stubs (fast, default)
- Opt-in heavy integration tests (real backend training, requires preprovisioned local assets)

Run:

```bash
python -m pip install -e ".[dev,rtmdet]"
python -m pytest -q
python scripts/provision_heavy_test_assets.py
python -m pytest object_detector_trainer/tests -q --heavy
```

`pytest -m heavy` is not a valid substitute. The test suite requires the explicit `--heavy` flag and fails fast if the local heavy-test asset cache is missing.
