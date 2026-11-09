"""Unit tests for src/claims_factor.py (Deng and Pang, Sec. 1, footnote 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.claims_factor import (
    aggregate_claims_performance_factor,
    build_claims_performance_factor,
    claims_performance,
    expanding_insurer_effects_forecast,
    scaled_forecast_errors,
)


def _toy_panel() -> pd.DataFrame:
    """Two insurers, six quarters, deterministic values so forecasts and
    scaling can be checked by hand."""
    quarters = pd.period_range("2020Q1", periods=6, freq="Q")
    records = []
    # Insurer 0: constant performance of -losses/premiums = 0.5 every quarter.
    for q in quarters:
        records.append({"insurer_id": 0, "quarter": q, "losses_lae": -50.0,
                         "premiums_earned": 100.0, "admitted_assets": 1000.0})
    # Insurer 1: performance alternates, premiums sometimes non-positive.
    perf_losses = [-60.0, -40.0, 30.0, -55.0, -45.0, -50.0]
    premiums = [100.0, 100.0, 0.0, 100.0, 100.0, 100.0]
    for q, loss, prem in zip(quarters, perf_losses, premiums):
        records.append({"insurer_id": 1, "quarter": q, "losses_lae": loss,
                         "premiums_earned": prem, "admitted_assets": 500.0})
    return pd.DataFrame.from_records(records)


def test_claims_performance_sign_and_definition() -> None:
    """Favorable claims performance is minus incurred losses/LAE over
    positive net premiums earned (footnote 1)."""
    panel = _toy_panel()
    performance = claims_performance(panel)
    insurer0 = performance[panel["insurer_id"] == 0]
    assert np.allclose(insurer0, 0.5)

    insurer1 = performance[panel["insurer_id"] == 1].to_numpy()
    assert insurer1[0] == pytest.approx(0.6)  # -(-60)/100
    assert np.isnan(insurer1[2])  # premiums_earned == 0 is excluded, not negative-premium


def test_claims_performance_requires_columns() -> None:
    with pytest.raises(AssertionError):
        claims_performance(pd.DataFrame({"a": [1]}))


def test_expanding_forecast_uses_only_prior_quarters() -> None:
    """The forecast for insurer i at quarter q must be a function of
    quarters strictly before q (no look-ahead, footnote 1)."""
    panel = _toy_panel()
    performance = claims_performance(panel)
    forecast = expanding_insurer_effects_forecast(panel, performance)

    insurer0_forecast = forecast[panel["insurer_id"] == 0].to_numpy()
    assert np.isnan(insurer0_forecast[0])  # no prior quarter at the first observation
    assert np.allclose(insurer0_forecast[1:], 0.5)  # expanding mean of a constant series

    # Changing a later observation of insurer 0 must not alter the forecast
    # for an earlier quarter (the forecast at quarter q is a function of
    # quarters < q only).
    panel_altered = panel.copy()
    mask = (panel_altered["insurer_id"] == 0) & (panel_altered["quarter"] == panel["quarter"].iloc[-1])
    panel_altered.loc[mask, "losses_lae"] = -999.0
    performance_altered = claims_performance(panel_altered)
    forecast_altered = expanding_insurer_effects_forecast(panel_altered, performance_altered)
    first_three = (panel["insurer_id"] == 0) & panel["quarter"].isin(panel["quarter"].iloc[:3])
    assert np.allclose(
        forecast[first_three].dropna().to_numpy(),
        forecast_altered[first_three].dropna().to_numpy(),
    )


def test_scaled_forecast_errors_requires_full_scaling_window() -> None:
    """With a 20-quarter scaling window and only 6 quarters of data, no
    scaled error can be computed (insufficient trailing history)."""
    panel = _toy_panel()
    performance = claims_performance(panel)
    forecast = expanding_insurer_effects_forecast(panel, performance)
    scaled = scaled_forecast_errors(panel, performance, forecast, scaling_window_q=20)
    assert scaled.isna().all()

    scaled_short = scaled_forecast_errors(panel, performance, forecast, scaling_window_q=2)
    assert scaled_short.notna().any()


def test_aggregate_uses_positive_lagged_assets_only() -> None:
    """Aggregation at quarter q uses only insurers with strictly positive
    admitted assets lagged one quarter (footnote 1); an insurer with a
    non-positive lagged weight must not enter the weighted average."""
    panel = _toy_panel()
    scaled_errors = pd.Series(1.0, index=panel.index)  # every insurer contributes 1.0
    panel_zero_asset = panel.copy()
    mask_q1 = (panel_zero_asset["insurer_id"] == 1) & (panel_zero_asset["quarter"] == panel["quarter"].iloc[0])
    panel_zero_asset.loc[mask_q1, "admitted_assets"] = 0.0

    factor = aggregate_claims_performance_factor(panel_zero_asset, scaled_errors)
    # At quarter index 1 insurer 1's lagged assets (from quarter 0) are zero,
    # so only insurer 0 should contribute; the weighted mean of all-1.0
    # values is still 1.0 regardless, so instead check insurer count via a
    # differentiated scaled_errors series.
    differentiated = pd.Series(np.where(panel_zero_asset["insurer_id"] == 0, 1.0, 3.0), index=panel.index)
    factor_diff = aggregate_claims_performance_factor(panel_zero_asset, differentiated)
    second_quarter = panel["quarter"].iloc[1]
    assert factor_diff.loc[second_quarter] == pytest.approx(1.0)  # insurer 1 excluded at q1


def test_build_claims_performance_factor_end_to_end() -> None:
    panel = _toy_panel()
    factor = build_claims_performance_factor(panel, scaling_window_q=2)
    assert isinstance(factor, pd.Series)
    assert factor.notna().any()
    assert factor.index.is_monotonic_increasing
