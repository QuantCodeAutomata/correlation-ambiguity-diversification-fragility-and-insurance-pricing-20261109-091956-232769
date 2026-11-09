"""Construction of the common claims-performance factor used in Figure 1.1
of Deng and Pang, "Correlation Ambiguity, Diversification Fragility, and
Insurance Pricing" (paper, Sec. 1 and footnote 1).

Footnote 1 (verbatim): "Favorable claims performance is measured as minus
incurred losses and loss-adjustment expenses divided by positive net
premiums earned. Expanding insurer-effects forecasts use only prior
quarters; forecast errors are scaled by their preceding 20-quarter standard
deviation and aggregated using positive lagged admitted-asset weights."

The functions below implement this construction for any insurer-quarter
panel with columns: insurer_id, quarter (pandas Period, quarterly), incurred
losses and loss-adjustment expenses (losses_lae), net premiums earned
(premiums_earned), and admitted assets (admitted_assets). The panel may be
real SNL statutory data or the synthetic substitute generated in
data/generate_synthetic.py when SNL data is unavailable (see
results/RESULTS.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "insurer_id",
    "quarter",
    "losses_lae",
    "premiums_earned",
    "admitted_assets",
)


def claims_performance(panel: pd.DataFrame) -> pd.Series:
    """Favorable claims performance y_it = -losses_lae / premiums_earned,
    defined only where premiums_earned is strictly positive (footnote 1).
    """
    for col in REQUIRED_COLUMNS:
        assert col in panel.columns, f"panel is missing required column '{col}'"
    positive_premiums = panel["premiums_earned"] > 0
    performance = pd.Series(np.nan, index=panel.index, dtype=float)
    performance[positive_premiums] = (
        -panel.loc[positive_premiums, "losses_lae"] / panel.loc[positive_premiums, "premiums_earned"]
    )
    return performance


def expanding_insurer_effects_forecast(panel: pd.DataFrame, performance: pd.Series) -> pd.Series:
    """Expanding insurer-effects forecast of claims performance, using only
    prior quarters (footnote 1): the forecast for insurer i at quarter q is
    the insurer's own expanding mean of y_i over all quarters strictly
    before q. This is the fitted value of a panel regression of
    performance on insurer fixed effects re-estimated each quarter on the
    expanding window of quarters already observed, which, for an
    intercept-only (fixed-effects) design, coincides with the insurer's
    expanding average.
    """
    df = panel[["insurer_id", "quarter"]].copy()
    df["performance"] = performance.values
    df = df.sort_values(["insurer_id", "quarter"])
    # Shift by one period before the expanding mean so that quarter q's
    # forecast uses only quarters < q (prevents look-ahead).
    forecast = (
        df.groupby("insurer_id")["performance"]
        .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    forecast.index = df.index
    return forecast.reindex(panel.index)


def scaled_forecast_errors(
    panel: pd.DataFrame, performance: pd.Series, forecast: pd.Series, scaling_window_q: int
) -> pd.Series:
    """Forecast errors e_it = y_it - forecast_it, scaled by each insurer's
    preceding `scaling_window_q`-quarter rolling standard deviation of its
    own forecast errors (footnote 1). The scaling window is computed on
    past errors only (shifted by one quarter) to avoid look-ahead.
    """
    df = panel[["insurer_id", "quarter"]].copy()
    df["error"] = (performance - forecast).values
    df = df.sort_values(["insurer_id", "quarter"])

    def _scale(group: pd.Series) -> pd.Series:
        trailing_std = group.shift(1).rolling(scaling_window_q, min_periods=scaling_window_q).std()
        return group / trailing_std

    scaled = df.groupby("insurer_id")["error"].apply(_scale).reset_index(level=0, drop=True)
    scaled.index = df.index
    return scaled.reindex(panel.index)


def aggregate_claims_performance_factor(
    panel: pd.DataFrame, scaled_errors: pd.Series
) -> pd.Series:
    """Aggregate scaled forecast errors across insurers at each quarter
    using positive lagged admitted-asset weights (footnote 1): the weighted
    average of scaled errors, weighted by each insurer's admitted assets
    one quarter prior, restricted to insurers with strictly positive lagged
    assets.
    """
    df = panel[["insurer_id", "quarter", "admitted_assets"]].copy()
    df["scaled_error"] = scaled_errors.values
    df = df.sort_values(["insurer_id", "quarter"])
    df["lagged_assets"] = df.groupby("insurer_id")["admitted_assets"].shift(1)

    valid = df["lagged_assets"] > 0
    df = df[valid & df["scaled_error"].notna()]

    def _weighted_mean(group: pd.DataFrame) -> float:
        weights = group["lagged_assets"]
        return float(np.average(group["scaled_error"], weights=weights))

    factor = df.groupby("quarter").apply(_weighted_mean)
    factor.name = "claims_performance_factor"
    return factor.sort_index()


def build_claims_performance_factor(panel: pd.DataFrame, scaling_window_q: int) -> pd.Series:
    """Full pipeline: favorable claims performance -> expanding insurer-
    effects forecast -> scaled forecast errors -> admitted-asset-weighted
    cross-sectional aggregation, producing one claims-performance factor
    value per quarter (paper, Sec. 1, footnote 1)."""
    performance = claims_performance(panel)
    forecast = expanding_insurer_effects_forecast(panel, performance)
    scaled = scaled_forecast_errors(panel, performance, forecast, scaling_window_q)
    return aggregate_claims_performance_factor(panel, scaled)
