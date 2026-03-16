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

CLI entrypoints:

```bash
poetry run object-detector-trainer --help
```

You can also invoke it as a module:

```bash
poetry run python -m object_detector_trainer.cli --help
```

Note: when invoked as a module, `argparse` will show `usage: cli.py ...` in the help output.

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

## Using This In A Project

This repo is the trainer engine. In our workflow it is typically consumed from a project repository or a template repository that provides:

- the expected workspace layout (`raw_data/`, `datasets/`, `runs/`, `params.yaml`, etc.)
- DVC wiring and dataset versioning policy
- baseline promotion/export conventions
- parameter catalogs and experiment orchestration

If you have an existing project repo (for example [`starwit/waste-detection`](https://github.com/starwit/waste-detection)), prefer following that repo's template/docs for the canonical end-to-end workflow.

At the trainer level, the only thing you normally need is running stages against a workspace root:

```bash
poetry run object-detector-trainer --stage all --workspace-root /path/to/workspace
```

For the authoritative CLI surface, use `--help`. For the validated config shape and backend resolution logic, see:

- `object_detector_trainer/config/schema.py`
- `object_detector_trainer/backends/training_config.py`

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
