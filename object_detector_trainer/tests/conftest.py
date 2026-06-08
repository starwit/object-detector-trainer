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
        help="Run heavy integration tests (real backend training; first run may download model assets).",
    )


def _explicit_heavy_selection(config: pytest.Config, items: list[pytest.Item]) -> bool:
    markexpr = str(getattr(config.option, "markexpr", "") or "").strip()
    if markexpr == "heavy":
        return True
    return bool(items) and all("heavy" in item.keywords for item in items)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--heavy"):
        return
    if _explicit_heavy_selection(config, items):
        raise pytest.UsageError(
            "Heavy tests are enabled via the `--heavy` flag. "
            "Use `pytest --heavy` (optionally with a nodeid)."
        )

    skip_heavy = pytest.mark.skip(reason="Need --heavy option to run heavy tests.")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
