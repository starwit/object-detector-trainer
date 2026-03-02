"""Adapter that wraps an RF-DETR model to expose *.val()* and *.predict()*
compatible with ``ultralytics.YOLO``, so the trainer pipeline can evaluate
all model families through one interface.

Usage::

    from trainer_core.wrappers.rfdetr import RFDETRModelAdapter

    adapter = RFDETRModelAdapter(trained_rfdetr_model, model_name="RF-DETR-medium",
                                  resolution=1280, class_names={0: "waste", 1: "leaves"})
    # Now usable everywhere a YOLO model is expected:
    results = adapter.predict(img_path, conf=0.25)
    metrics = adapter.val(data="path/to/dataset.yaml", imgsz=1280)
"""

from __future__ import annotations

import io
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from trainer_core.wrappers.prediction_types import (
    CocoEvalResults as _CocoEvalResults,
    PredictionResult as _PredictionResult,
    ValMetrics as _ValMetrics,
)
from trainer_core.wrappers.prediction_utils import (
    build_prediction_result,
    load_image,
)
from trainer_core.wrappers.yolo_eval import (
    compute_coco_metrics as _compute_coco_metrics_base,
    evaluate_yolo_dataset as _evaluate_yolo_dataset,
    parse_class_names as _parse_class_names,  # retained for compatibility in tests
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class RFDETRModelAdapter:
    """Makes an RF-DETR model look like ``ultralytics.YOLO`` for evaluation.

    After training, wrap the model:

    >>> adapter = RFDETRModelAdapter(model, ...)
    >>> evaluate_and_log_model_results(adapter, ...)       # works
    >>> generate_side_by_side_comparisons(adapter, ...)    # works
    """

    def __init__(
        self,
        rfdetr_model: Any,
        model_name: str = "RF-DETR",
        resolution: int = 560,
        class_names: Dict[int, str] | None = None,
        model_variant: str | None = None,
    ):
        self._model = rfdetr_model
        self.model_name = model_name
        self.model_backend = "rfdetr"
        self.model_variant = str(model_variant).lower() if model_variant else None
        self._resolution = int(resolution)
        # Public attr so shared tooling can read the effective eval resolution.
        self.resolution = int(resolution)
        self._class_names = class_names or {}
        # The adapter also needs to look like it has a `model` attr (see train_model code).
        self.model = type("_Stub", (), {"yaml": {"model_name": model_name}})()
        self.trainer = None

    # ------------------------------------------------------------------
    # predict()  — compatible with ultralytics.YOLO.predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        save: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[_PredictionResult]:
        """Run inference on a single image.

        Returns a list with one ``_PredictionResult`` per image (always length 1).
        """
        img = load_image(source)

        # RF-DETR expects RGB channel order. OpenCV loads images in BGR.
        img_rgb = (
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.ndim == 3 and img.shape[2] == 3
            else img
        )
        detections = self._model.predict(img_rgb, threshold=conf)
        return [self._wrap_detections(detections, img)]

    def _wrap_detections(
        self, detections: Any, img: np.ndarray
    ) -> _PredictionResult:
        """Convert ``supervision.Detections`` to our Ultralytics-compatible wrapper."""
        if detections is None or len(detections) == 0:
            return build_prediction_result(
                xyxy=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
                labels=np.empty(0, dtype=np.float32),
                img=img,
                class_names=self._class_names,
            )

        return build_prediction_result(
            xyxy=detections.xyxy,
            scores=detections.confidence,
            labels=detections.class_id,
            img=img,
            class_names=self._class_names,
        )

    # ------------------------------------------------------------------
    # val()  — compatible with ultralytics.YOLO.val()
    # ------------------------------------------------------------------

    def val(
        self,
        data: str | None = None,
        verbose: bool = False,
        save: bool = False,
        plots: bool = False,
        **kwargs: Any,
    ) -> _ValMetrics:
        """Evaluate the model on a YOLO-format dataset.

        Parameters match the keyword-style call used by ``validate_model()``:

            model.val(data=..., classes=[...], imgsz=1280, workers=0, ...)

        Metrics are computed via ``pycocotools`` to stay comparable with standard
        COCO evaluation.
        """
        if data is None:
            raise ValueError("data (path to dataset.yaml) is required for val()")

        imgsz = kwargs.get("imgsz")
        if imgsz is not None:
            try:
                imgsz_int = int(imgsz)
            except (TypeError, ValueError):
                imgsz_int = None
            if imgsz_int is not None and imgsz_int != int(self._resolution):
                logger.warning(
                    "RF-DETR adapter ignores imgsz=%s; using resolution=%s.",
                    imgsz_int,
                    self._resolution,
                )

        conf_threshold = kwargs.get("conf", 0.001)
        classes_filter: set[int] | None = None
        if kwargs.get("classes") is not None:
            classes_filter = set(int(c) for c in kwargs["classes"])

        def _infer_fn(img: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        gt_images, gt_annotations, dt_results, class_names, avg_inference = _evaluate_yolo_dataset(
            data=data,
            conf_threshold=float(conf_threshold),
            classes_filter=classes_filter,
            infer_fn=_infer_fn,
        )

        # --- Compute metrics via pycocotools ---
        categories = [{"id": i, "name": n} for i, n in class_names.items()]
        coco_metrics = _compute_coco_metrics(gt_images, gt_annotations, dt_results, categories)
        precision, recall, map50, map50_95, per_class = coco_metrics
        macro_f1 = getattr(coco_metrics, "macro_f1", None)

        # Ultralytics fitness formula: 0.1 * mAP50 + 0.9 * mAP50-95
        fitness = 0.1 * map50 + 0.9 * map50_95

        results_dict = {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50_95,
        }
        # Native RF-DETR reports F1 from a confidence-threshold sweep (macro-F1),
        # not derived from a single (precision, recall) pair. Surface it so the
        # shared validate_model() can log the same value for RF-DETR runs.
        if macro_f1 is not None:
            try:
                results_dict["metrics/f1(B)"] = float(macro_f1)
            except (TypeError, ValueError):
                pass

        return _ValMetrics(
            results_dict=results_dict,
            fitness=fitness,
            # RF-DETR timing includes the full predict() call (preprocess+inference+postprocess)
            # so we report it all under "inference" with the others zeroed out.
            speed={"preprocess": 0.0, "inference": avg_inference, "postprocess": 0.0},
            per_class=per_class,
        )


def _coco_eval_pycocotools(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
) -> _CocoEvalResults:
    """COCO evaluation using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build ground-truth COCO object
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    if len(detections) == 0 or len(annotations) == 0:
        # No detections or no ground-truth → everything is zero
        return _CocoEvalResults(0.0, 0.0, 0.0, 0.0, {}, macro_f1=0.0)

    # Suppress pycocotools stdout spam
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        # Align COCOeval settings with native RF-DETR evaluation defaults.
        # Native RF-DETR uses eval_max_dets=500 and a patched summarize()
        # to compute AP@[.5:.95] at maxDets=maxDets[2] (not hardcoded 100).
        coco_eval.params.maxDets = [1, 10, 500]
        from rfdetr.datasets.coco_eval import patched_pycocotools_summarize
        coco_eval.evaluate()
        coco_eval.accumulate()
        patched_pycocotools_summarize(coco_eval)
    finally:
        sys.stdout = old_stdout

    # stats: [AP@.5:.95, AP@.5, AP@.75, APsmall, APmed, APlarge,
    #         AR@1, AR@10, AR@100, ARsmall, ARmed, ARlarge]
    map50_95 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])

    # Native RF-DETR reports macro precision/recall/F1 by sweeping confidence thresholds
    # using COCOeval's raw matching data (evalImgs). Use its implementation directly
    # so adapter numbers match RF-DETR evaluation output.
    from rfdetr.engine import coco_extended_metrics
    extended = coco_extended_metrics(coco_eval)
    precision = float(extended.get("precision", 0.0))
    recall = float(extended.get("recall", 0.0))
    macro_f1 = float(extended.get("f1_score", 0.0))

    per_class: dict[str, dict[str, float]] = {}
    for row in extended.get("class_map", []) or []:
        cls_name = str(row.get("class", "")).strip()
        if not cls_name or cls_name.lower() == "all":
            continue
        per_class[cls_name] = {
            "precision": float(row.get("precision", 0.0) or 0.0),
            "recall": float(row.get("recall", 0.0) or 0.0),
            "map50": float(row.get("map@50", 0.0) or 0.0),
            "map": float(row.get("map@50:95", 0.0) or 0.0),
            "f1_score": float(row.get("f1_score", 0.0) or 0.0),
        }

    return _CocoEvalResults(precision, recall, map50, map50_95, per_class, macro_f1=macro_f1)


_compute_coco_metrics = partial(_compute_coco_metrics_base, eval_fn=_coco_eval_pycocotools)
