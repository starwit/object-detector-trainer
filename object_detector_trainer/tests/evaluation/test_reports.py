from __future__ import annotations

import csv
from pathlib import Path

from object_detector_trainer.evaluation.reports import mean_table


def _metrics_payload() -> dict[str, float]:
    return {
        "img_size": 128,
        "precision": 0.1,
        "recall": 0.2,
        "map": 0.3,
        "map50": 0.4,
        "fitness": 0.31,
        "f1_score": 0.13,
        "ms_per_frame": 2.0,
    }


def test_mean_table_overwrites_ultralytics_training_results_csv(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "epoch,time,train/box_loss\n1,0.1,1.2\n",
        encoding="utf-8",
    )

    mean_table(
        _metrics_payload(),
        _metrics_payload(),
        "trained-model",
        "baseline-model",
        output_dir=tmp_path,
    )

    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["MODEL"] == "baseline-model"
    assert rows[1]["MODEL"] == "trained-model"
