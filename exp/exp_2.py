"""Experiment 2: conditional financial exposure under correlation ambiguity
at fixed underwriting shares (Deng and Pang, Figure 5.1).

For each fixed underwriting share x in {0.25, 0.5, 0.75} and each ambiguity
half-width phi on a grid from 0 to 1 (step 0.1), with correlation midpoint
rho_0 = 0, this script computes the insurer's optimal financial position
y_t(x) from Proposition 2 (Eq. 3.4) and the normalized exposure sigma*y/eta,
and verifies the two qualitative properties stated for Figure 5.1: (a) the
absolute financial investment is weakly decreasing in phi, and (b) the slope
of the normalized exposure on the positive-investment branch equals -x
(Corollary 1, Eq. 3.8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.equilibrium import Primitives, conditional_financial_position


def run(
    underwriting_shares: tuple[float, ...] = config.EXP2_UNDERWRITING_SHARES,
    phi_min: float = config.PHI_GRID_MIN,
    phi_max: float = config.PHI_GRID_MAX,
    phi_step: float = config.PHI_GRID_STEP,
) -> pd.DataFrame:
    """Compute y_t(x) and sigma*y/eta over the (x, phi) grid.

    Returns a DataFrame with columns: x, phi, y, normalized_exposure,
    regime ('positive', 'zero', 'negative').
    """
    p = Primitives(
        l=config.BENCH_L, eta=config.BENCH_ETA, r=config.BENCH_R, mu=config.BENCH_MU,
        sigma=config.BENCH_SIGMA, alpha=config.BENCH_ALPHA, gamma=config.BENCH_GAMMA,
        T=config.BENCH_T, t=config.BENCH_T0,
    )
    phi_grid = np.round(np.arange(phi_min, phi_max + 1e-9, phi_step), 10)
    rows = []
    for x in underwriting_shares:
        for phi in phi_grid:
            rho_lower, rho_upper = config.BENCH_RHO0 - phi, config.BENCH_RHO0 + phi
            y = float(conditional_financial_position(p, x, rho_lower, rho_upper))
            k = (p.mu - p.r) * p.disc / (p.gamma * p.eta * p.sigma * x)
            if k > rho_upper:
                regime = "positive"
            elif k < rho_lower:
                regime = "negative"
            else:
                regime = "zero"
            rows.append(
                {
                    "x": x,
                    "phi": float(phi),
                    "y": y,
                    "normalized_exposure": p.sigma * y / p.eta,
                    "regime": regime,
                }
            )
    return pd.DataFrame(rows)


def validate(results: pd.DataFrame) -> dict:
    """Check the two qualitative properties stated in the paper for
    Figure 5.1: weak monotonicity of |y| in phi, and slope -x on the
    positive branch (Corollary 1, Eq. 3.8)."""
    checks = {}
    for x, group in results.groupby("x"):
        group = group.sort_values("phi")
        abs_y = group["y"].abs().to_numpy()
        checks[f"monotone_x={x}"] = bool(np.all(np.diff(abs_y) <= 1e-9))

        positive = group[group["regime"] == "positive"]
        if len(positive) >= 2:
            slope = np.polyfit(positive["phi"], positive["normalized_exposure"], 1)[0]
            checks[f"slope_x={x}"] = float(slope)
            assert abs(slope - (-x)) < 1e-6, f"slope {slope} != -{x} for x={x} (Eq. 3.8)"
    return checks


if __name__ == "__main__":
    import os

    results = run()
    checks = validate(results)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/exp2_financial_exposure.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for x, group in results.groupby("x"):
        group = group.sort_values("phi")
        ax.plot(group["phi"], group["normalized_exposure"], marker="o", label=f"x = {x}")
    ax.axvline(config.EMPIRICAL_PHI_REFERENCE, color="grey", linestyle="--",
               label=f"empirical reference phi = {config.EMPIRICAL_PHI_REFERENCE}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Ambiguity half-width phi")
    ax.set_ylabel("Normalized financial exposure sigma*y/eta")
    ax.set_title("Conditional financial exposure under correlation ambiguity (Fig. 5.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp2_financial_exposure.png", dpi=150)

    print("Validation checks:", checks)
    print(results.to_string(index=False))
