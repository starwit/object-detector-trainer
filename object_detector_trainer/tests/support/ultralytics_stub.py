from __future__ import annotations

"""Minimal Ultralytics stand-in so tests never hit real downloads or GPU work."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np


class StubBox:
    """Mimics ``ultralytics.utils.metrics.Metric`` for per-class metrics.

    Required because ``_extract_per_class_metrics()`` in evaluate.py now reads
    ``metrics.box.ap_class_index``, ``.p``, ``.r``, ``.ap50``, and ``.ap`` to
    produce per-class results.  Without this stub the E2E smoke tests would
    crash when that code path runs against ``StubValMetrics``.
    """

    def __init__(self):
        self.ap_class_index = np.array([0])
        self.p = np.array([0.5])
        self.r = np.array([0.6])
        self._ap50 = np.array([0.4])
        self._ap = np.array([0.3])

    @property
    def ap50(self):
        return self._ap50

    @property
    def ap(self):
        return self._ap


class StubValMetrics:
    def __init__(self):
        # Full speed dict required: validate_model() now computes ms_per_frame as
        # preprocess + inference + postprocess (previously only inference was used).
        self.speed = {"preprocess": 0.5, "inference": 1.0, "postprocess": 0.3}
        self.results_dict = {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.4,
            "metrics/mAP50-95(B)": 0.3,
        }
        self.fitness = 0.42
        # Per-class metrics extraction (see _extract_per_class_metrics) reads
        # metrics.box.ap_class_index / .p / .r / .ap50 / .ap.
        self.box = StubBox()


class StubYOLO:
    """Drop-in replacement for `ultralytics.YOLO` used in tests and DVC checks."""

    recorded_models: list[str] = []
    raise_on_official: bool = False
    workspace: Path | None = None

    def __init__(self, model_path: str | Path):
        model_str = str(model_path)
        workspace = StubYOLO.workspace or Path.cwd()
        if StubYOLO.raise_on_official and model_str.startswith("yolov8"):
            raise RuntimeError(f"Simulated offline failure loading {model_str}")

        StubYOLO.recorded_models.append(model_str)
        self.model_source = model_str
        self.model_name = Path(model_str).stem or "stub"
        self._workspace = workspace
        self.trainer = None
        self.model = SimpleNamespace(yaml={"model_name": "yolov8n"})

    def train(self, **kwargs):
        project = kwargs.get("project", "runs")
        name = kwargs.get("name") or "train"
        save_dir = self._workspace / project / name
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"stub-trained-weights")
        self.trainer = SimpleNamespace(save_dir=str(save_dir))
        return SimpleNamespace(save_dir=str(save_dir))

    def val(self, **kwargs):
        return StubValMetrics()

    def predict(self, *args, **kwargs):
        class _Result:
            boxes = []

            @staticmethod
            def plot():
                return np.zeros((8, 8, 3), dtype=np.uint8)

        return [_Result()]
