"""Unit tests for src/bootstrap.py (Deng and Pang, Sec. 1, footnote 1;
Figure 1.1: rolling correlation with circular block-bootstrap CIs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import block_bootstrap_ci, rolling_correlation, rolling_correlation_with_ci


def test_rolling_correlation_recovers_known_correlation() -> None:
    """A rolling window over two perfectly correlated series must yield a
    correlation of 1.0 everywhere the window is full."""
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=60))
    x = base
    y = 2.0 * base + 3.0  # perfectly linearly related to x
    corr = rolling_correlation(x, y, window=10)
    assert np.allclose(corr.dropna(), 1.0, atol=1e-8)


def test_rolling_correlation_window_length() -> None:
    """The first `window - 1` rolling correlations must be NaN."""
    x = pd.Series(np.arange(20, dtype=float))
    y = pd.Series(np.arange(20, dtype=float) ** 2)
    corr = rolling_correlation(x, y, window=8)
    assert corr.iloc[: 8 - 1].isna().all()
    assert corr.iloc[8 - 1 :].notna().all()


def test_block_bootstrap_ci_bounds_are_ordered() -> None:
    """The bootstrap confidence interval must have a lower bound no larger
    than the upper bound, and both must lie in [-1, 1]."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = 0.5 * x + rng.normal(size=40) * 0.5
    lower, upper = block_bootstrap_ci(
        x, y, block_size=4, n_draws=200, lower_pctl=5, upper_pctl=95, seed=42
    )
    assert lower <= upper
    assert -1.0 <= lower <= 1.0
    assert -1.0 <= upper <= 1.0


def test_block_bootstrap_ci_contains_point_estimate_for_strong_correlation() -> None:
    """For strongly correlated series, the 90% bootstrap interval should be
    tight and bracket the true (near-1) correlation most of the time."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=200)
    x = base
    y = base + rng.normal(size=200) * 0.05
    point_estimate = float(np.corrcoef(x, y)[0, 1])
    lower, upper = block_bootstrap_ci(
        x, y, block_size=4, n_draws=500, lower_pctl=5, upper_pctl=95, seed=7
    )
    assert lower <= point_estimate <= upper


def test_block_bootstrap_requires_matching_lengths() -> None:
    with pytest.raises(AssertionError):
        block_bootstrap_ci(
            np.arange(10.0), np.arange(5.0), block_size=4, n_draws=10,
            lower_pctl=5, upper_pctl=95, seed=0,
        )


def test_rolling_correlation_with_ci_output_shape_and_half_width_positive() -> None:
    """End-to-end rolling correlation + CI: one row per window end, with a
    non-negative half-width at every row (Figure 1.1 methodology)."""
    rng = np.random.default_rng(3)
    n = 50
    base = rng.normal(size=n)
    x = pd.Series(base, index=pd.period_range("2010Q1", periods=n, freq="Q"))
    y = pd.Series(base * 0.3 + rng.normal(size=n) * 0.9, index=x.index)

    result = rolling_correlation_with_ci(
        x, y, window=20, block_size=4, n_draws=100, lower_pctl=5, upper_pctl=95, seed=11
    )
    assert len(result) == n - 20 + 1
    assert (result["ci_half_width"] >= 0.0).all()
    assert (result["ci_lower"] <= result["ci_upper"]).all()
    assert (result["correlation"] >= -1.0 - 1e-9).all() and (result["correlation"] <= 1.0 + 1e-9).all()
