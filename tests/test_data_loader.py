"""Market-data smoke test for the QuantX Dataloader path used by exp_1
(the market-return proxy for the CRSP value-weighted total market return,
paper Sec. 1). Skipped when DATALOADER_API_KEY / MASSIVE_API_KEY are absent."""

from __future__ import annotations

import pytest


@pytest.mark.market_data
def test_market_return_slice_has_expected_shape(market_return_slice) -> None:
    series = market_return_slice
    assert len(series) > 0
    assert series.notna().all()
    assert series.index.freqstr == "Q-DEC"
    # Quarterly total returns should be moderate in magnitude (not raw prices).
    assert (series.abs() < 1.0).all()
