"""Shared pytest configuration for benchmark tests.

Configures pytest-benchmark to save JSON results automatically so that the
companion ``summarise_benchmarks.py`` script can produce heatmap tables.

Usage
-----
Run benchmarks with JSON export::

    pytest -m "grid and small" --benchmark-json=benchmark_results.json

Or rely on the autosave directory (``./benchmark_results/``)::

    pytest -m "grid and small" --benchmark-autosave

Then generate the summary::

    python src/photon_mosaic/core/tests/summarise_benchmarks.py benchmark_results.json
"""


def pytest_configure(config):
    """Set sensible defaults for pytest-benchmark when the user hasn't
    explicitly configured them."""
    # Only touch settings when benchmark plugin is available
    if config.pluginmanager.has_plugin("benchmark"):
        # Use a fixed storage directory so results accumulate across runs
        if not config.getoption("benchmark_storage", default=None):
            config.option.benchmark_storage = "./benchmark_results"
