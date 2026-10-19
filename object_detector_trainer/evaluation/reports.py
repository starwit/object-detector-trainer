"""Write evaluation results as JSON, CSV tables and a readable report.txt.

Every file is rewritten on each evaluation; nothing is appended.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tabulate import tabulate

from object_detector_trainer.evaluation.coco import DETECTION_CONFIDENCE, MAX_DETECTIONS, METRIC_CONFIDENCE

OVERALL_COLUMNS = ["precision", "recall", "f1_score", "map50", "map", "fitness", "ms_per_frame"]
CLASS_COLUMNS = ["precision", "recall", "f1_score", "map50", "map"]
MERGED_COLUMNS = ["merged_into", "images", "objects", "precision", "recall", "f1_score", "map50", "map"]

METRIC_DESCRIPTIONS = f"""\
Metric definitions (identical for every backend):
  map50         Mean average precision at IoU 0.5.
  map           Mean average precision averaged over IoU 0.5 to 0.95.
  precision     Correct detections / all detections, at confidence >= {METRIC_CONFIDENCE} and IoU 0.5.
  recall        Found objects / all objects, at confidence >= {METRIC_CONFIDENCE} and IoU 0.5.
  f1_score      Harmonic mean of precision and recall per class.
  fitness       0.1 * map50 + 0.9 * map.
  ms_per_frame  Prediction time per image in milliseconds.
AP uses detections with confidence >= {DETECTION_CONFIDENCE}, at most {MAX_DETECTIONS} per image.
Overall values average the classes that occur in the test split.
Scene fitness is computed on the test images of one source.
Merged-class rows score all classes on the test images that contain the original class.
"""


def write_reports(eval_dir: Path, results: dict[str, dict], note: str | None = None) -> None:
    """Write the reports for one or more models. ``results`` maps model name to its metrics.

    ``note`` is shown at the top of report.txt, for example why a baseline is not compared.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    overall_rows = overall_table(results)
    class_rows = per_class_table(results)
    scene_rows = scene_table(results)
    merged_rows = merged_class_table(results)

    write_csv(eval_dir / "results.csv", ["model"] + OVERALL_COLUMNS, overall_rows)
    write_csv(eval_dir / "per_class.csv", ["model", "class"] + CLASS_COLUMNS, class_rows)
    write_csv(eval_dir / "scenes.csv", ["model", "scene", "fitness"], scene_rows)
    write_csv(eval_dir / "merged_classes.csv", ["model", "class"] + MERGED_COLUMNS, merged_rows)

    sections = [note] if note else []
    sections += [
        "Overall\n" + tabulate(overall_rows, headers=["model"] + OVERALL_COLUMNS, floatfmt=".4f"),
        "Per class\n" + tabulate(class_rows, headers=["model", "class"] + CLASS_COLUMNS, floatfmt=".4f"),
        "Scenes\n" + tabulate(scene_rows, headers=["model", "scene", "fitness"], floatfmt=".4f"),
    ]
    if merged_rows:
        sections.append(
            "Merged classes\n" + tabulate(merged_rows, headers=["model", "class"] + MERGED_COLUMNS, floatfmt=".4f")
        )
    sections.append(METRIC_DESCRIPTIONS)
    (eval_dir / "report.txt").write_text("\n\n".join(sections), encoding="utf-8")


def overall_table(results: dict[str, dict]) -> list[list]:
    rows = []
    for model_name, result in results.items():
        rows.append([model_name] + [result[column] for column in OVERALL_COLUMNS])
    return rows


def per_class_table(results: dict[str, dict]) -> list[list]:
    rows = []
    for model_name, result in results.items():
        for class_name, metrics in result["per_class"].items():
            rows.append([model_name, class_name] + [metrics[column] for column in CLASS_COLUMNS])
    return rows


def scene_table(results: dict[str, dict]) -> list[list]:
    rows = []
    for model_name, result in results.items():
        for scene, fitness in result["scenes"].items():
            rows.append([model_name, scene, fitness])
    return rows


def merged_class_table(results: dict[str, dict]) -> list[list]:
    rows = []
    for model_name, result in results.items():
        for class_name, metrics in result["merged_classes"].items():
            rows.append([model_name, class_name] + [metrics[column] for column in MERGED_COLUMNS])
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
