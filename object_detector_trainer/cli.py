"""Command line: ``object-detector-trainer --stage {prepare,train,evaluate,all}``.

This is the only module that reads command-line arguments. It turns them into
one validated config and calls the matching function in ``pipeline.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

from object_detector_trainer import pipeline
from object_detector_trainer.config import load_config
from object_detector_trainer.runs import default_run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="object-detector-trainer",
        description="Prepare a dataset, train an object detector and evaluate it.",
    )
    parser.add_argument("--stage", required=True, choices=["prepare", "train", "evaluate", "all"])
    parser.add_argument(
        "--workspace-root", default=".",
        help="Project folder with params.yaml; datasets/ and runs/ are written here. Default: current folder.",
    )
    parser.add_argument("--config", default="params.yaml", help="Config file, relative to the workspace.")
    parser.add_argument("--model", help="Model key from models.* (overrides train.model).")
    parser.add_argument("--seed", type=int, help="Random seed (overrides seed).")
    parser.add_argument(
        "--run-name",
        help="Run folder name under runs/. Required for --stage evaluate. Default: <model>-<UTC time>.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config value, for example --set train.epochs=5. Can be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    workspace = Path(args.workspace_root).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Workspace folder does not exist: {workspace}")
    if args.stage == "evaluate" and not args.run_name:
        parser.error("--stage evaluate needs --run-name to know which run to evaluate.")

    overrides = list(args.set)
    if args.model:
        overrides.append(f"train.model={args.model}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    config = load_config(workspace / args.config, overrides)
    run_name = args.run_name or default_run_name(config.train.model)

    # Ultralytics writes helper files (for example AMP check weights) into the
    # current folder; keep them inside the workspace.
    os.chdir(workspace)

    if args.stage == "prepare":
        pipeline.prepare(config, workspace)
    elif args.stage == "train":
        pipeline.train(config, workspace, run_name)
    elif args.stage == "evaluate":
        pipeline.evaluate(config, workspace, run_name)
    else:
        pipeline.run_all(config, workspace, run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
