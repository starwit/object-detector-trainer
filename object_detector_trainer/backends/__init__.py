"""Backends: the model families the trainer can train and load.

Each backend is one module that provides:

``Options``
    A pydantic class (subclass of ``ModelOptions``) for its keys under
    ``models.<key>`` in params.yaml.

``download_weights(asset_id, path)``
    Download the pretrained weights named ``asset_id`` to ``path``.

``train(job: TrainJob) -> Path``
    Train on ``job.dataset_dir`` and return ``job.output_dir / "weights/best.pt"``.
    Native validation for checkpoint selection and early stopping stays inside
    the backend.

``load(weights_path, metadata, *, device)``
    Return a model object with ``class_names`` (dict of class ID to name) and
    ``predict(image, conf, max_det) -> Detections``. ``image`` is a BGR NumPy
    array as returned by OpenCV.

Final evaluation does not belong in a backend: every model is scored by
``evaluation/evaluator.py`` from its predictions.

To add a backend, add its module here and to ``BACKENDS``.
"""

from __future__ import annotations

from types import ModuleType

from object_detector_trainer.backends import rfdetr, yolo

BACKENDS: dict[str, ModuleType] = {
    "yolo": yolo,
    "rfdetr": rfdetr,
}


def get_backend(name: str) -> ModuleType:
    if name not in BACKENDS:
        available = ", ".join(BACKENDS)
        raise ValueError(f"Unknown backend {name!r}. Available backends: {available}")
    return BACKENDS[name]
