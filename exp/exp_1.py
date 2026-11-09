"""Experiment 1: rolling underwriting-market dependence with block-bootstrap
confidence intervals (Deng and Pang, Sec. 1, Figure 1.1).

Methodology (paper, Sec. 1 and footnote 1):
  1. Construct a common claims-performance factor from U.S. property-
     casualty statutory data: favorable claims performance is minus incurred
     losses and loss-adjustment expenses divided by positive net premiums
     earned; expanding insurer-effects forecasts use only prior quarters;
     forecast errors are scaled by their preceding 20-quarter standard
     deviation and aggregated using positive lagged admitted-asset weights.
  2. Convert monthly CRSP value-weighted total market returns to calendar
     quarters by compounding.
  3. Compute rolling 32-quarter correlations between the two series.
  4. Build nominal 90% confidence intervals from a paired circular four-
     quarter block-bootstrap with 2,000 draws (5th/95th percentiles).
  5. The empirical reference half-width is the median of (upper-lower)/2
     across rolling windows.

Data: U.S. property-casualty statutory insurer data (SNL Financial / S&P
Global Market Intelligence) is a commercial subscription database that is
not served by the QuantX Dataloader, by Massive, or by Yahoo Finance (see
results/RESULTS.md, "Deviations from the paper" and "Not reproduced"). This
script therefore uses a synthetic insurer panel (data/generate_synthetic.py)
in its place, and reports the resulting figure only as an illustration of
the full construction, not as a reproduction of the paper's Figure 1.1
numbers. The market-return side uses real data: SPY (dividend-adjusted)
total return from the QuantX Dataloader, as a documented substitute for the
proprietary CRSP value-weighted index.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config
from src.bootstrap import rolling_correlation_with_ci
from src.claims_factor import build_claims_performance_factor

# Market data: QuantX Dataloader — https://quantx.limex.pro/
#
# Price history in this project is loaded through the QuantX Dataloader, a
# research-ready market-data service available through a REST API and as an
# MCP server for AI agents. It is built on market data from Massive
# (https://massive.com/) and prepares every stock history for backtesting:
#   - prices adjusted for dividends, ready for total-return research;
#   - one continuous history when a company changes its ticker;
#   - clean separation when a ticker has belonged to different companies;
#   - careful corporate-action handling and fine-grained data polishing.
#
# Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
# MASSIVE_API_KEY at https://massive.com/.
from data.data_loader import load_ohlcv
from data.generate_synthetic import generate_synthetic_insurer_panel


def load_market_return(start_date: str, end_date: str, ticker: str) -> pd.Series:
    """Quarterly total-return series compounded from monthly closes of a
    dividend-adjusted broad-market ETF (documented substitute for the CRSP
    value-weighted total market return, see results/RESULTS.md)."""
    bars, meta = load_ohlcv([ticker], start_date, end_date, timeframe="1d")
    assert not bars.empty, f"no bars returned for {ticker}"
    close = bars.set_index("date")["close"].sort_index()
    monthly = close.resample("ME").last()
    monthly_return = monthly.pct_change().dropna()
    quarterly_return = (1.0 + monthly_return).resample("QE").prod() - 1.0
    quarterly_return.index = quarterly_return.index.to_period("Q")
    quarterly_return.name = "market_return"
    return quarterly_return, meta


def run() -> dict:
    market_return, meta = load_market_return(
        config.EXP1_PANEL_START, config.EXP1_PANEL_END, config.EXP1_MARKET_PROXY_TICKER
    )
    print(f"Loaded {len(market_return)} quarterly market-return observations "
          f"from {market_return.index.min()} to {market_return.index.max()}")
    print("ticker_details:", meta.get("ticker_details"))

    panel = generate_synthetic_insurer_panel(
        n_insurers=config.EXP1_SYN_N_INSURERS,
        start_quarter=str(market_return.index.min()),
        end_quarter=str(market_return.index.max()),
        market_return=market_return,
        seed=config.EXP1_SYN_SEED,
    )
    claims_factor = build_claims_performance_factor(panel, config.EXP1_SCALING_WINDOW_Q)
    print(f"Claims-performance factor: {claims_factor.notna().sum()} quarterly values")

    aligned = pd.concat([claims_factor, market_return], axis=1).dropna()
    aligned.columns = ["claims_factor", "market_return"]
    assert len(aligned) > config.EXP1_ROLLING_WINDOW_Q, (
        "too few overlapping quarters for a 32-quarter rolling window"
    )

    table = rolling_correlation_with_ci(
        aligned["claims_factor"],
        aligned["market_return"],
        window=config.EXP1_ROLLING_WINDOW_Q,
        block_size=config.EXP1_BOOTSTRAP_BLOCK_Q,
        n_draws=config.EXP1_BOOTSTRAP_DRAWS,
        lower_pctl=config.EXP1_CI_LOWER_PCTL,
        upper_pctl=config.EXP1_CI_UPPER_PCTL,
        seed=config.EXP1_BOOTSTRAP_SEED,
    )
    median_half_width = float(table["ci_half_width"].median())
    return {
        "table": table,
        "median_half_width": median_half_width,
        "correlation_range": (float(table["correlation"].min()), float(table["correlation"].max())),
        "ticker_details": meta.get("ticker_details"),
        "non_massive_source": meta.get("non_massive_source"),
    }


if __name__ == "__main__":
    out = run()
    os.makedirs("results", exist_ok=True)
    out["table"].to_csv("results/exp1_rolling_correlation.csv")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    table = out["table"]
    ax.plot(table.index.to_timestamp(), table["correlation"], color="C0", label="Rolling 32Q correlation")
    ax.fill_between(table.index.to_timestamp(), table["ci_lower"], table["ci_upper"],
                     color="C0", alpha=0.2, label="90% block-bootstrap CI")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Rolling-window end quarter")
    ax.set_ylabel("Correlation")
    ax.set_title("Illustrative rolling underwriting-market dependence\n"
                 "(synthetic insurer panel; SNL statutory data unavailable, see RESULTS.md)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp1_rolling_correlation.png", dpi=150)

    summary = {
        "median_half_width": out["median_half_width"],
        "correlation_range": out["correlation_range"],
        "non_massive_source": out["non_massive_source"],
    }
    print(summary)
    pd.Series(summary).to_json("results/exp1_summary.json")
