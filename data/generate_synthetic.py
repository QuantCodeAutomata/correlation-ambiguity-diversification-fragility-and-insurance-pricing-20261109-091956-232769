"""Synthetic U.S. property-casualty statutory insurer panel.

U.S. property-casualty statutory data (incurred losses and loss-adjustment
expenses, net premiums earned, admitted assets, at insurer-quarter
granularity) is published by SNL Financial / S&P Global Market Intelligence,
a commercial subscription database. It is not served by the QuantX
Dataloader, by Massive, or by Yahoo Finance, all of which cover traded
market instruments rather than insurance regulatory filings.

This module generates a synthetic insurer-quarter panel with the same
columns and the same panel structure (insurer fixed effects, a common
market-correlated shock, idiosyncratic noise, and heteroskedastic scaling)
that the claims-performance construction in src/claims_factor.py expects.
It is used only to exercise and validate that construction end to end; the
resulting rolling-correlation figure is reported as an illustration of the
method, not as a reproduction of the paper's Figure 1.1 (see
results/RESULTS.md, "Not reproduced").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_insurer_panel(
    n_insurers: int,
    start_quarter: str,
    end_quarter: str,
    market_return: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Synthetic insurer-quarter panel correlated with `market_return`.

    Each insurer i has a fixed effect mu_i, and its favorable claims
    performance is mu_i + beta_i * market_return_q + idiosyncratic noise,
    with beta_i heterogeneous across insurers (so the admitted-asset
    weighted aggregate factor is only partially, not perfectly, correlated
    with the market return, matching the qualitative pattern described in
    the paper: wide confidence intervals that often include zero).

    Returns a panel with columns: insurer_id, quarter, losses_lae,
    premiums_earned, admitted_assets.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.period_range(start_quarter, end_quarter, freq="Q")
    market_return = market_return.reindex(quarters)
    assert market_return.notna().any(), "market_return must cover the requested quarters"
    market_return = market_return.fillna(market_return.mean())

    insurer_fe = rng.normal(loc=-0.60, scale=0.05, size=n_insurers)  # average loss ratio ~60%
    betas = rng.normal(loc=0.05, scale=0.08, size=n_insurers)  # heterogeneous market loading
    log_assets0 = rng.normal(loc=12.0, scale=1.5, size=n_insurers)  # starting log admitted assets

    records = []
    for i in range(n_insurers):
        premiums = rng.uniform(50.0, 500.0, size=len(quarters))  # premiums earned, $mm
        idio = rng.normal(scale=0.08, size=len(quarters))
        performance = insurer_fe[i] + betas[i] * market_return.to_numpy() + idio
        losses_lae = -performance * premiums
        asset_growth = rng.normal(loc=0.01, scale=0.03, size=len(quarters)).cumsum()
        admitted_assets = np.exp(log_assets0[i] + asset_growth)
        for q_idx, quarter in enumerate(quarters):
            records.append(
                {
                    "insurer_id": i,
                    "quarter": quarter,
                    "losses_lae": losses_lae[q_idx],
                    "premiums_earned": premiums[q_idx],
                    "admitted_assets": admitted_assets[q_idx],
                }
            )
    return pd.DataFrame.from_records(records)
