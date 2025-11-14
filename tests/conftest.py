import pytest


def pytest_addoption(parser):
    """Add command line options to pytest"""
    parser.addoption(
        "--runvisualization",
        action="store_true",
        default=False,
        help="Run tests that generate visualizations",
    )


def pytest_configure(config):
    """Configure pytest based on command line options"""
    config.addinivalue_line(
        "markers",
        "visualization: mark test to run only when --runvisualization is specified",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options"""
    if not config.getoption("--runvisualization"):
        skip_visualization = pytest.mark.skip(
            reason="Need --runvisualization option to run"
        )
        for item in items:
            if "visualization" in item.keywords:
                item.add_marker(skip_visualization)
