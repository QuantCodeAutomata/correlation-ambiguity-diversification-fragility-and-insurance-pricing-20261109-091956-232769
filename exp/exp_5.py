"""Experiment 5: equilibrium regime structure across correlation midpoint and
ambiguity half-width, benchmark parameters (Deng and Pang, Sec. 4, Theorem 2,
Figure 5.4).

Unlike Experiment 3 (which fixes the correlation midpoint rho0 = 0 and widens
phi), this experiment varies both the correlation midpoint rho0 and the
ambiguity half-width phi over a grid, building the correlation interval
[rho0 - phi, rho0 + phi] at each point and classifying the resulting
competitive-equilibrium regime via Theorem 2 (Eq. 4.3-4.12). The admissible
domain is the open triangle -1 < rho0 - phi, rho0 + phi < 1, phi >= 0.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config
from src.equilibrium import (
    Primitives,
    equilibrium_allocation,
    full_underwriting_threshold,
    pure_underwriting_threshold,
    zero_underwriting_threshold,
)
from exp.exp_3 import benchmark_primitives


def run(p: Primitives | None = None) -> pd.DataFrame:
    """Regime classification over the (rho0, phi) grid; returns a long-format
    DataFrame with one row per admissible grid point, columns rho0, phi,
    regime."""
    p = p or benchmark_primitives()

    rho0_vals = np.linspace(-0.98, 0.98, config.EXP5_GRID_POINTS)
    phi_vals = np.linspace(0.0, 0.98, config.EXP5_GRID_POINTS)

    rows: list[dict] = []
    for rho0 in rho0_vals:
        for phi in phi_vals:
            rho_lower, rho_upper = rho0 - phi, rho0 + phi
            if rho_lower > -1.0 and rho_upper < 1.0:
                allocation = equilibrium_allocation(p, rho_lower, rho_upper)
                rows.append({"rho0": float(rho0), "phi": float(phi), "regime": allocation["regime"]})

    return pd.DataFrame(rows, columns=["rho0", "phi", "regime"])


def thresholds(p: Primitives) -> dict:
    """Theorem 2 correlation thresholds at the given primitives."""
    return {
        "full_underwriting": full_underwriting_threshold(p),
        "pure_underwriting": pure_underwriting_threshold(p),
        "zero_underwriting": zero_underwriting_threshold(p),
    }


def validate(results: pd.DataFrame, p: Primitives) -> dict:
    """Checks: the empirical reference point (rho0=0, phi=0.23) falls in the
    interior-positive-investment regime; the zero-underwriting regime is
    unattainable at the benchmark (its threshold exceeds 1); and the set of
    distinct regimes observed on the grid."""
    dist2 = (results["rho0"] - 0.0) ** 2 + (results["phi"] - config.EMPIRICAL_PHI_REFERENCE) ** 2
    nearest_idx = dist2.idxmin()
    nearest_regime = results.loc[nearest_idx, "regime"]

    regimes_observed = sorted(results["regime"].unique().tolist())

    return {
        "empirical_reference_is_interior_positive_investment": bool(
            nearest_regime == "interior_positive_investment"
        ),
        "zero_underwriting_attainable": bool("zero_underwriting" in regimes_observed),
        "regimes_observed": regimes_observed,
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    p = benchmark_primitives()

    results = run(p)
    thr = thresholds(p)
    print("Thresholds:", thr)

    checks = validate(results, p)
    print("Validation checks:", checks)

    results.to_csv("results/exp5_regime_map.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    for regime_name in sorted(results["regime"].unique()):
        mask = results["regime"] == regime_name
        ax.scatter(results.loc[mask, "rho0"], results.loc[mask, "phi"], s=4, label=regime_name)

    ax.scatter(
        [0.0], [config.EMPIRICAL_PHI_REFERENCE],
        marker="*", s=200, color="black",
        label="empirical reference (rho0=0, phi=0.23)",
    )

    ax.set_xlabel("Correlation midpoint rho0")
    ax.set_ylabel("Ambiguity half-width phi")
    ax.set_title("Equilibrium regime map (Fig. 5.4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp5_regime_map.png", dpi=150)
