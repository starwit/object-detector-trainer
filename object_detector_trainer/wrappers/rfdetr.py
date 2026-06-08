from __future__ import annotations

import io
import sys
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_detector_trainer.wrappers.prediction_types import (
    CocoEvalResults,
    PredictionResult,
    ValMetrics,
)
from object_detector_trainer.wrappers.prediction_utils import build_prediction_result, load_image
from object_detector_trainer.wrappers.yolo_eval import (
    compute_coco_metrics as _compute_coco_metrics_base,
    evaluate_yolo_dataset,
)


class RFDETRModelAdapter:
    """Expose RF-DETR through the small YOLO-like API used by the pipeline."""

    def __init__(
        self,
        rfdetr_model: Any,
        model_name: str = "RF-DETR",
        resolution: int = 560,
        class_names: dict[int, str] | None = None,
        model_variant: str | None = None,
    ) -> None:
        self._model = rfdetr_model
        self.model_name = model_name
        self.model_backend = "rfdetr"
        self.model_variant = str(model_variant).lower() if model_variant else None
        self.resolution = int(resolution)
        self._class_names = class_names or {}
        self.class_names = dict(self._class_names)
        self.model = type("_Stub", (), {"yaml": {"model_name": model_name}})()
        self.trainer = None

    def _predict_arrays(
        self,
        img: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # RF-DETR expects RGB, while OpenCV and the plotting path use BGR.
        img_rgb = (
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.ndim == 3 and img.shape[2] == 3
            else img
        )
        detections = self._model.predict(img_rgb, threshold=threshold)
        if detections is None or len(detections) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return (
            np.asarray(detections.xyxy, dtype=np.float32),
            np.asarray(detections.confidence, dtype=np.float32),
            np.asarray(detections.class_id, dtype=np.int64),
        )

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        save: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[PredictionResult]:
        img = load_image(source)
        bboxes, scores, labels = self._predict_arrays(img, float(conf))
        return [
            build_prediction_result(
                xyxy=bboxes,
                scores=scores,
                labels=labels,
                img=img,
                class_names=self._class_names,
            )
        ]

    def val(
        self,
        data: str | None = None,
        verbose: bool = False,
        save: bool = False,
        plots: bool = False,
        **kwargs: Any,
    ) -> ValMetrics:
        if data is None:
            raise ValueError("data (path to dataset.yaml) is required for val()")

        imgsz = kwargs.get("imgsz")
        if imgsz is not None and int(imgsz) != int(self.resolution):
            raise ValueError(
                f"RF-DETR adapter resolution is {self.resolution}, "
                f"but validation requested imgsz={int(imgsz)}."
            )

        classes_filter: set[int] | None = None
        if kwargs.get("classes") is not None:
            classes_filter = {int(c) for c in kwargs["classes"]}

        gt_images, gt_annotations, dt_results, class_names, avg_inference = evaluate_yolo_dataset(
            data=data,
            conf_threshold=float(kwargs.get("conf", 0.001)),
            classes_filter=classes_filter,
            infer_fn=lambda img, threshold: self._predict_arrays(img, threshold),
        )

        metrics = _compute_coco_metrics(
            gt_images,
            gt_annotations,
            dt_results,
            [{"id": cls_id, "name": name} for cls_id, name in class_names.items()],
        )
        fitness = 0.1 * float(metrics.map50) + 0.9 * float(metrics.map50_95)
        return ValMetrics(
            results_dict={
                "metrics/precision(B)": float(metrics.precision),
                "metrics/recall(B)": float(metrics.recall),
                "metrics/mAP50(B)": float(metrics.map50),
                "metrics/mAP50-95(B)": float(metrics.map50_95),
                "metrics/f1(B)": float(metrics.macro_f1),
            },
            fitness=fitness,
            speed={"preprocess": 0.0, "inference": avg_inference, "postprocess": 0.0},
            per_class=metrics.per_class,
        )


def _coco_eval_pycocotools(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
) -> CocoEvalResults:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    if len(detections) == 0 or len(annotations) == 0:
        return CocoEvalResults(0.0, 0.0, 0.0, 0.0, {}, macro_f1=0.0)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        # Match native RF-DETR evaluation, which summarizes AP at maxDets=500.
        coco_eval.params.maxDets = [1, 10, 500]
        from rfdetr.datasets.coco_eval import patched_pycocotools_summarize

        coco_eval.evaluate()
        coco_eval.accumulate()
        patched_pycocotools_summarize(coco_eval)
    finally:
        sys.stdout = old_stdout

    from rfdetr.engine import coco_extended_metrics

    extended = coco_extended_metrics(coco_eval)
    per_class: dict[str, dict[str, float]] = {}
    for row in extended["class_map"]:
        cls_name = str(row["class"]).strip()
        if cls_name.lower() == "all":
            continue
        per_class[cls_name] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "map50": float(row["map@50"]),
            "map": float(row["map@50:95"]),
            "f1_score": float(row["f1_score"]),
        }

    return CocoEvalResults(
        precision=float(extended["precision"]),
        recall=float(extended["recall"]),
        map50=float(coco_eval.stats[1]),
        map50_95=float(coco_eval.stats[0]),
        per_class=per_class,
        macro_f1=float(extended["f1_score"]),
    )


_compute_coco_metrics = partial(_compute_coco_metrics_base, eval_fn=_coco_eval_pycocotools)

__all__ = ["RFDETRModelAdapter"]
