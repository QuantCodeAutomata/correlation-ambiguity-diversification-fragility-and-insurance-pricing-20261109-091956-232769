"""Shared pytest fixtures and marker registration.

The ``market_data`` marker is used for tests that exercise the QuantX
Dataloader REST API (data/data_loader.py). They are skipped automatically
when DATALOADER_API_KEY / MASSIVE_API_KEY are not set in the environment, so
the default test run (``python -m pytest -m "not market_data"``) never
depends on a live network call.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test loads real data through data/data_loader.py"
    )


def _credentials_available() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def market_return_slice(request: pytest.FixtureRequest) -> pd.Series:
    """A short, cached quarterly market-return series (2020-2021) loaded
    through data/data_loader.py, used by market_data-marked tests only."""
    if not _credentials_available():
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set: skipping market_data test")

    cache_dir = request.config.cache.mkdir("market_return_slice")
    cache_file = cache_dir / "spy_2020_2021.json"
    if cache_file.exists():
        import json

        data = json.loads(cache_file.read_text())
        series = pd.Series(data["values"], index=pd.PeriodIndex(data["index"], freq="Q"))
        return series

    from exp.exp_1 import load_market_return

    series, _meta = load_market_return("2020-01-01", "2021-12-31", "SPY")

    import json

    cache_file.write_text(
        json.dumps({"index": [str(p) for p in series.index], "values": series.tolist()})
    )
    return series
