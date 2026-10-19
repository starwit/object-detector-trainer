# Object Detector Trainer

Prepare a dataset, train an object detector and evaluate it, all from one
`params.yaml`. Supported backends:

- Ultralytics YOLO (`backend: yolo`)
- RF-DETR (`backend: rfdetr`)

Every model is evaluated by the same code with the same metric definitions, so
results are comparable across backends.

Project concerns (DVC wiring, baseline promotion, special data selection)
belong in the project that uses the trainer. See the
[training template](https://github.com/starwit/object-detection-training-template)
for a ready project layout.

## Quick start

Python 3.11+ is required.

```bash
poetry install
```

1. Put labeled sources in a project folder (YOLO format), for example:

   ```text
   my-project/raw_data/train/<source>/images/*.jpg
   my-project/raw_data/train/<source>/labels/*.txt
   ```

   Other layouts (a whole unpacked CVAT project export, `train/val/test` folders, or both
   combined) are listed under [Input data](#input-data). The
   [training template](https://github.com/starwit/object-detection-training-template)
   has a step-by-step guide.

2. Write `my-project/params.yaml`:

   ```yaml
   data:
     dataset_name: my-dataset
     custom_classes: [waste, cigarette]
   train:
     model: yolo26n
     epochs: 50
   models:
     yolo26n: {backend: yolo, asset_id: yolo26n.pt}
   ```

3. Run everything:

   ```bash
   poetry run object-detector-trainer --workspace-root my-project --stage all --run-name first-try
   ```

4. Read `my-project/runs/first-try/eval/report.txt`.

## Command line

```text
object-detector-trainer --stage {prepare,train,evaluate,all}
    [--workspace-root .] [--config params.yaml]
    [--model KEY] [--seed N] [--run-name NAME] [--set KEY=VALUE ...]
```

| Stage | What it does |
|---|---|
| `prepare` | Builds `datasets/<dataset_name>` from the sources. Always rebuilds from scratch. |
| `train` | Trains `train.model` into `runs/<run-name>/train`. |
| `evaluate` | Evaluates `runs/<run-name>` (and the baseline, if configured) on the test split. Needs `--run-name`. |
| `all` | `prepare`, `train` and `evaluate`. |

- `--run-name` defaults to `<model>-<UTC time>`. `train` refuses to overwrite an existing run.
- `--set` overrides any config value, for example `--set train.epochs=5` or
  `--set models.yolo26n.batch_size=2`. `--model` and `--seed` are shortcuts for
  `--set train.model=...` and `--set seed=...`.
- Paths in the config are relative to the workspace.

The same functions are available in Python:

```python
from pathlib import Path
from object_detector_trainer import load_config, prepare, train, evaluate, load_model

workspace = Path("my-project")
config = load_config(workspace / "params.yaml")
prepare(config, workspace)
train(config, workspace, run_name="first-try")
metrics = evaluate(config, workspace, run_name="first-try")
model = load_model(workspace / "runs/first-try/train/weights/best.pt")
```

## Configuration

```yaml
seed: 42                       # Seeds splitting and training.

data:
  dataset_name: my-dataset     # Folder name under datasets/.
  custom_classes: [waste, cigarette, bottle]   # Source classes, in label-ID order.
  use_coco_classes: false      # Use the 80 COCO classes instead of custom_classes.
  class_mapping:               # Optional: train classes under another name.
    waste: [waste, bottle]     # final_class: [source classes]
  ignore_classes: [leaf_pile]  # Optional: classes in a source's data.yaml whose labels are dropped.

prepare:
  val_split: 0.1               # Fraction of train used for validation, unless validation data is supplied.
  test_split: 0.1              # Fraction of train used for test, unless test data is supplied.

train:
  model: yolo26n               # Key under models.
  image_size: 640              # Shared defaults, overridable per model:
  epochs: 100
  device: auto                 # "auto" = first GPU if available, else CPU. Or "cpu", "cuda:1", ...

models_defaults:               # Optional per-backend defaults; model entries win.
  rfdetr: {cache_dir: models/pretrained/rfdetr}

models:
  yolo26n:
    backend: yolo
    asset_id: yolo26n.pt       # Pretrained weights, downloaded on first use.
    batch_size: 8             # Images processed at once.
    effective_batch_size: 64  # Optional YOLO accumulation target.
  rfdetr-medium:
    backend: rfdetr
    variant: medium
    asset_id: rf-detr-medium.pth
    resolution: 1280           # RF-DETR's name for image_size.
    batch_size: 2
    effective_batch_size: 16  # Optional RF-DETR accumulation target.

evaluation:
  baseline_weights_path: models/current_best/best.pt   # Compared when it exists; null = never.
```

Every model entry accepts:

| Key | Meaning |
|---|---|
| `backend` | `yolo` or `rfdetr`. |
| `asset_id` | Pretrained weights file name, downloaded to `cache_dir` when missing. |
| `cache_dir` | Where pretrained weights are stored. Default `models/pretrained/<backend>`. |
| `allow_download` | Set `false` to fail instead of downloading. Default `true`. |
| `init_weights` | Start from these weights instead of `asset_id`, for fine-tuning. |
| `epochs`, `image_size` | Override the `train.*` defaults for this model. |
| `batch_size` | Images processed at once. Default 8; set it on the model. |
| `extra_train_kwargs` | Passed unchanged to the backend's own training function. |

YOLO also accepts `freeze_backbone` (freeze the backbone layers). RF-DETR also
accepts `variant` (required: `nano`, `small`, `medium`, `base`, `large`),
`resolution`, `lr` and `gradient_checkpointing`.

YOLO and RF-DETR also accept `effective_batch_size` (default 64 for YOLO, 16
for RF-DETR). Training then updates the model only after about this many
images, so a small `batch_size` trains like a larger one.

Unknown keys are errors, so a typo never goes unnoticed.

`extra_train_kwargs` go to Ultralytics `model.train()` or RF-DETR
`model.train()`. They cannot change values the trainer sets from its config
(data, epochs, image size, batch size, device, seed, output folder).
YOLO's `workers: 4` and `amp: true` are defaults that `extra_train_kwargs` can change.

### Fine-tuning

Fine-tuning is a model entry with `init_weights`:

```yaml
models:
  finetune:
    backend: yolo
    asset_id: yolo26m.pt
    init_weights: models/finetune/best.pt
    epochs: 30
    freeze_backbone: true
    extra_train_kwargs: {optimizer: AdamW, lr0: 0.0002, warmup_epochs: 0, patience: 20}
```

YOLO only uses `lr0` when `optimizer` is set; with the default `optimizer: auto`
it picks its own learning rate. For RF-DETR, set a lower `lr` instead.

Use `init_weights` with either backend. The example's `freeze_backbone`
setting is YOLO-specific; an RF-DETR entry also needs its `variant`.
Copy the chosen starting checkpoint to the configured path before training.

## Input data

Unpack whole CVAT **Ultralytics YOLO Detection 1.0** exports, including
images, into a folder under `raw_data/`:

```bash
unzip my-export.zip -d raw_data/my-export
```

The export's `data.yaml` defines source classes and split references, for example:

```yaml
path: .
Train: Train.txt
Validation: Validation.txt
Test: Test.txt
names: [waste, cigarette]
```

Split keys are case-insensitive; `val`, `valid` and `validation` all
mean validation. References can be image lists or image directories.
Files retain their original path spelling; CVAT's `data/images/...`
list prefix is accepted. Images and matching labels can use either
`images/<subset>/`, `labels/<subset>/` or
`<subset>/images/`, `<subset>/labels/`.

Alternatively, assign individual task exports or plain sources by placement:

```text
raw_data/train/<source>/data.yaml     individual training task export
raw_data/train/<source>/images/...     training
raw_data/train/<source>/labels/...     YOLO labels
raw_data/val/<source>/...              optional fixed validation
raw_data/test/<source>/...             optional fixed test
```

Outer folder names are also case-insensitive and accept validation aliases.
For one source per set, images and labels can also sit directly under
`raw_data/train/images/` and `raw_data/train/labels/`, likewise for val/test.
An outer folder overrides a source's single exported subset; a source with
several subsets must go directly under `raw_data/`.
Older exports can provide subset image lists without `data.yaml`.
Independent exports and folder sources can be combined: their supplied sets are
merged before generating any missing validation/test set from the combined
training pool. A supplied set in one source applies to the whole dataset.

- Accepted images: jpg, jpeg, png, bmp; subdirectories and uppercase extensions work.
- A missing or empty label file marks a background image.
- Polygon rows are converted to their bounding box.
- Source class IDs are mapped by name using `data.yaml`; without it, IDs
  follow `custom_classes`. Class mapping and ignored classes apply normally.
- Source files are never modified.

Duplicate and near-duplicate images are removed first, within and across all
sets, in one pass. Images are compared in two stages: a 64-bit perceptual hash
(pHash) of each image finds candidates that differ in at most 2 bits; each
candidate pair is then compared pixel by pixel on 256x256 grayscale copies and
counts as a duplicate at SSIM >= 0.95. Of each group one image is kept: in test
if the group has a test image, otherwise in validation, otherwise in train;
within a set, the image with the first path. A training copy of an evaluation
image is therefore dropped, never the evaluation image. The log shows how many
images were removed per set; `manifest.csv` lists the images that remain.

Missing validation/test sets are then generated from the remaining training
images using `prepare.val_split` / `prepare.test_split` and the seed. A
declared empty split or a broken reference is an error.

To update data, replace the export's folder with the new export and run
preparation again. Do not combine a project export with exports of its own tasks.

## Outputs

```text
datasets/<dataset_name>/
  dataset.yaml                      train / val / test / class names
  train|val|test/images/            hard links to the source images
  train|val|test/labels/            labels with final class IDs
  manifest.csv                      per image: split, source, source image, source classes

runs/<run-name>/
  train/weights/best.pt             best checkpoint (chosen by the backend's own validation)
  train/metadata.yaml               what is needed to load the model
  train/config.yaml                 the full config of the run
  train/...                         the backend's own logs and plots
  eval/report.txt                   readable results
  eval/results.json                 all results, per model
  eval/results.csv, per_class.csv, scenes.csv, merged_classes.csv
  eval/predictions/                 test annotations (left) vs. trained predictions (right), every run
  eval/comparisons/                 baseline vs. trained images (when a baseline is available)
  metrics.json                      final test metrics of the trained model
```

`metadata.yaml` makes a saved model loadable on its own. It must contain
`model_backend` and `image_size`; RF-DETR models also need `class_names` and the
variant. Projects copy it next to exported baseline weights.

## Evaluation

Final evaluation always reloads the saved best checkpoint and predicts every
test image once. Overall, per-class, per-scene and merged-class results all
come from those predictions and `evaluation/coco.py`:

- AP uses predictions with confidence >= 0.001, at most 500 per image.
- `map50` uses IoU 0.5; `map` averages IoU 0.5 to 0.95.
- Precision, recall and F1 use confidence >= 0.25 and IoU 0.5.
- Overall values average the classes present in the test split. Present classes
  without detections score zero; detections on background images are false positives.
- `fitness` is `0.1 * map50 + 0.9 * map`.
- A scene is the source an image came from.
- Merged-class results score the test images that contained a merged source
  class (from `data.class_mapping`).

A baseline is evaluated on the same test split when its weights exist and it
has the same classes as the dataset. Otherwise only the trained model is
evaluated, and the first line of `report.txt` says why the baseline was not
compared (not downloaded yet, unreadable weights/metadata, or different classes).
Problems loading the optional baseline do not stop the trained model's evaluation.

Every evaluation saves test images under `eval/predictions/`, with annotations
on the left and predictions on the right. Predictions show class names and
confidence scores at the same 0.25 threshold used for precision/recall.
An image with no predictions above that threshold is still saved.
When a compatible baseline exists, `eval/comparisons/` additionally shows
baseline predictions on the left and trained-model predictions on the right.

The backends' own validation during training (checkpoint selection, early
stopping) uses their upstream metrics, so training logs can differ from the
final report.

## Adding a backend

The trainer is designed to keep backend integration small: implement one
adapter module and register it. Dataset preparation, run metadata, final
metrics and reports are shared by all backends.

Copy `backends/yolo.py`, the shortest complete backend, and adapt it. The
parts below are what every backend must provide:

```python
from pathlib import Path

import numpy as np

from object_detector_trainer.backends.common import Detections, ModelOptions, TrainJob


class Options(ModelOptions):
    ...  # Declare any additional settings accepted under models.<key>.


def download_weights(asset_id: str, path: Path) -> None:
    ...  # Write the requested pretrained checkpoint to path.


def train(job: TrainJob) -> Path:
    ...  # Train, save and return job.output_dir / "weights/best.pt".


def load(weights_path: Path, metadata: dict, *, device: str):
    ...  # Return a model on this device with class_names and predict(), as below.


class Model:
    class_names: dict[int, str]

    def predict(self, image: np.ndarray, conf: float, max_det: int) -> Detections:
        ...
```

| Part of the contract | What the backend supplies |
|---|---|
| `Options` | Extends the common settings with its own validated parameters |
| `download_weights` | Creates a non-empty weights file at the requested path |
| `train` | Uses `job.start_weights`, settings and prepared dataset; writes all outputs inside `job.output_dir` |
| `load` | Reloads the saved checkpoint on the given device and returns a prediction model |
| `class_names` | Final dataset class IDs mapped to their names, read from the checkpoint. Loading fails if they differ from `metadata.yaml` |
| `predict` | Detections in the original image's coordinates, filtered by confidence and detection limit |

`TrainJob` provides the dataset directory, ordered final class names,
starting weights, output directory, epochs, image size, batch size, seed,
device and validated options. Options specific to a backend stay in its
`Options` class.

The prepared dataset contains
`dataset.yaml` and `train/val/test` image and label folders. Convert this
layout inside the backend when an upstream library needs another format.

Every registered backend is imported whenever the trainer runs. Import a
large or optional library inside the backend's functions, as
`backends/rfdetr.py` does: importing `rfdetr` changes PyTorch settings for the
whole process, which would otherwise affect the other backends.

Prediction inputs are BGR NumPy images from OpenCV. Convert to RGB if the
model requires it. Return `Detections` with three aligned arrays:
`xyxy` of shape `(N, 4)` in original-image pixels, confidence `scores`
of shape `(N,)`, and integer `class_ids` of shape `(N,)`. Empty images
still return correctly shaped empty arrays. IDs must follow `class_names`.

To connect the adapter:

1. Add its module and an entry to `BACKENDS` in `backends/__init__.py`.
2. Add any required package dependency and update the Poetry lock file.
3. Define a model under `models:` with `backend:` set to your registered name,
   then select that entry with `train.model`.
4. Add a small runnable model configuration to `MODELS` in
   `tests/test_heavy.py` and run `poetry run pytest --heavy`.

The pipeline writes the completed run's metadata, including `model_options`
and `class_names`; the backend's loader can use these to recreate its model.
Checkpoint selection and early stopping use the backend's own validation.
The shared evaluator computes final test metrics from `predict()`, so a
new backend does not need a separate evaluator or report implementation.

## Testing

```bash
poetry run pytest              # fast tests
poetry run pytest --heavy      # real training for every backend (GPU recommended)
poetry run ruff check object_detector_trainer
```

The heavy tests train each backend for one epoch, reload the saved model in a
separate process, fine-tune from it, and compare it as a baseline. Every test
uses a fresh workspace, so pretrained weights are downloaded each time.

## Code map

| File | Responsibility |
|---|---|
| `cli.py` | Command line |
| `pipeline.py` | The stages: prepare, train, evaluate |
| `config.py` | params.yaml schema and loading |
| `classes.py` | Class lists and class_mapping |
| `labels.py` | YOLO label reading and normalization |
| `dataset.py` | Building the prepared dataset |
| `find_duplicates.py` | Duplicate image detection (unchanged from 1.x) |
| `runs.py` | Run folders, metadata.yaml, loading saved models |
| `backends/` | YOLO and RF-DETR |
| `evaluation/` | Metrics, reports and comparison images |
