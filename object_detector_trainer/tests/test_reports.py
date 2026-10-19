from __future__ import annotations

import csv
from pathlib import Path

from object_detector_trainer.evaluation.reports import write_reports


def result(precision: float) -> dict:
    return {
        "precision": precision, "recall": 0.2, "f1_score": 0.13, "map50": 0.4, "map": 0.3,
        "fitness": 0.31, "ms_per_frame": 2.0,
        "per_class": {"waste": {"precision": precision, "recall": 0.2, "f1_score": 0.13, "map50": 0.4, "map": 0.3}},
        "scenes": {"camera": 0.31},
        "merged_classes": {},
    }


def test_reports_are_rewritten_not_appended(tmp_path: Path) -> None:
    results = {"baseline": result(0.1), "trained": result(0.5)}

    write_reports(tmp_path, results)
    first = {path.name: path.read_text() for path in tmp_path.iterdir()}
    write_reports(tmp_path, results)

    assert {path.name: path.read_text() for path in tmp_path.iterdir()} == first
    with (tmp_path / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["baseline", "trained"]
    assert "confidence >= 0.25" in first["report.txt"]
    assert "Merged classes" not in first["report.txt"]
