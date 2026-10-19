import pytest


def pytest_addoption(parser):
    parser.addoption("--heavy", action="store_true", help="Run real training integration tests.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--heavy"):
        return
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Use --heavy to run real training."))
