from __future__ import annotations

import multiprocessing as mp

import pytest


# Avoid fork-related teardown issues from threaded backend workers in heavy tests.
mp.set_start_method("spawn", force=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="Run heavy integration tests (real backend training).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--heavy"):
        return

    skip_heavy = pytest.mark.skip(reason="Need --heavy option to run heavy tests.")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
