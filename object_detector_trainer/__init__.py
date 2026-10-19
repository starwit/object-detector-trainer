"""Train and evaluate object detectors (YOLO, RF-DETR) from a params.yaml config.

Python use mirrors the command line::

    from pathlib import Path
    from object_detector_trainer import load_config, prepare, train, evaluate

    workspace = Path("my-project")
    config = load_config(workspace / "params.yaml")
    prepare(config, workspace)
    train(config, workspace, run_name="first-try")
    evaluate(config, workspace, run_name="first-try")
"""

from object_detector_trainer.config import load_config
from object_detector_trainer.pipeline import evaluate, prepare, run_all, train
from object_detector_trainer.runs import load_model

__all__ = ["evaluate", "load_config", "load_model", "prepare", "run_all", "train"]
