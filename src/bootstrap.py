"""Rolling correlation and paired circular block-bootstrap confidence
intervals (Deng and Pang, Sec. 1 and footnote 1; Figure 1.1).

"Rolling 32-quarter correlations are evaluated using paired circular
four-quarter block-bootstrap samples, with 2,000 draws and fifth/ninety-
fifth percentile endpoints. First-stage estimates are held fixed during
resampling."

The circular block bootstrap is implemented with
arch.bootstrap.CircularBlockBootstrap (McElroy/Politis-Romano circular
block scheme), applied jointly ("paired") to the two series so that
observations of the claims-performance factor and the market-return series
are resampled together, preserving their contemporaneous pairing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import CircularBlockBootstrap


def rolling_correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation of two aligned series over `window`
    trailing observations (paper: 32 quarters, Sec. 1)."""
    aligned = pd.concat([x, y], axis=1).dropna()
    aligned.columns = ["x", "y"]
    return aligned["x"].rolling(window).corr(aligned["y"])


def block_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    n_draws: int,
    lower_pctl: float,
    upper_pctl: float,
    seed: int,
) -> tuple[float, float]:
    """Nominal confidence interval for corr(x, y) via a paired circular
    block bootstrap: resample (x, y) pairs jointly in circular blocks of
    `block_size`, recompute the correlation on each draw, and take the
    `lower_pctl`/`upper_pctl` percentiles of the resulting distribution
    (paper, footnote 1). The point estimate itself ("first-stage
    estimate") is not re-centered: the percentiles of the bootstrap
    distribution are reported directly as the interval endpoints.
    """
    assert len(x) == len(y), "x and y must have the same length"
    assert len(x) > block_size, "series must be longer than the block size"

    paired = np.column_stack([x, y])
    bs = CircularBlockBootstrap(block_size, paired, seed=seed)
    correlations = np.empty(n_draws)
    for i, data in enumerate(bs.bootstrap(n_draws)):
        sample = data[0][0]
        correlations[i] = np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]
    lower, upper = np.percentile(correlations, [lower_pctl, upper_pctl])
    return float(lower), float(upper)


def rolling_correlation_with_ci(
    x: pd.Series,
    y: pd.Series,
    window: int,
    block_size: int,
    n_draws: int,
    lower_pctl: float,
    upper_pctl: float,
    seed: int,
) -> pd.DataFrame:
    """Rolling correlation together with paired circular block-bootstrap
    confidence intervals at every window end date (paper, Figure 1.1)."""
    aligned = pd.concat([x, y], axis=1).dropna()
    aligned.columns = ["x", "y"]

    records = []
    for end in range(window - 1, len(aligned)):
        window_slice = aligned.iloc[end - window + 1 : end + 1]
        corr = float(np.corrcoef(window_slice["x"], window_slice["y"])[0, 1])
        lower, upper = block_bootstrap_ci(
            window_slice["x"].to_numpy(),
            window_slice["y"].to_numpy(),
            block_size=block_size,
            n_draws=n_draws,
            lower_pctl=lower_pctl,
            upper_pctl=upper_pctl,
            seed=seed + end,  # distinct but reproducible seed per window
        )
        records.append(
            {
                "window_end": aligned.index[end],
                "correlation": corr,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_half_width": (upper - lower) / 2.0,
            }
        )
    return pd.DataFrame.from_records(records).set_index("window_end")
