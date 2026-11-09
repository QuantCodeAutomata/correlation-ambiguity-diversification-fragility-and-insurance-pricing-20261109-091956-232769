"""Experiment 3: competitive equilibrium adjustment under fixed-midpoint
correlation widening, benchmark parameters (Deng and Pang, Sec. 4-5,
Figure 5.2).

For each ambiguity half-width phi on a grid from 0 to 1 (step 0.1), with
correlation midpoint rho_0 = 0 and the benchmark primitives of Sec. 5.1,
this script determines the competitive-equilibrium regime from Theorem 2
(Eq. 4.3-4.12) and reports coverage x*, absolute financial exposure
sigma|y*|/eta, joint exposure x*sigma|y*|/eta, and loading theta*.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config
from src.equilibrium import (
    Primitives,
    equilibrium_path,
    full_underwriting_threshold,
    pure_underwriting_threshold,
    zero_underwriting_threshold,
)


def benchmark_primitives(alpha: float = config.BENCH_ALPHA) -> Primitives:
    return Primitives(
        l=config.BENCH_L, eta=config.BENCH_ETA, r=config.BENCH_R, mu=config.BENCH_MU,
        sigma=config.BENCH_SIGMA, alpha=alpha, gamma=config.BENCH_GAMMA,
        T=config.BENCH_T, t=config.BENCH_T0,
    )


def run(
    rho0: float = config.BENCH_RHO0,
    phi_min: float = config.PHI_GRID_MIN,
    phi_max: float = config.PHI_GRID_MAX,
    phi_step: float = config.PHI_GRID_STEP,
    p: Primitives | None = None,
) -> pd.DataFrame:
    """Equilibrium path over the phi grid; returns a DataFrame with columns
    phi, x, theta, y, regime, coverage_pct, abs_financial_exposure,
    joint_exposure, loading_pct."""
    p = p or benchmark_primitives()
    phi_grid = np.round(np.arange(phi_min, phi_max + 1e-9, phi_step), 10)
    path = equilibrium_path(p, rho0, phi_grid)
    df = pd.DataFrame(path)
    df["coverage_pct"] = 100.0 * df["x"]
    df["abs_financial_exposure"] = p.sigma * df["y"].abs() / p.eta
    df["joint_exposure"] = df["x"] * df["abs_financial_exposure"]
    df["loading_pct"] = 100.0 * df["theta"]
    return df


def validate(results: pd.DataFrame) -> dict:
    """Checks: joint exposure is non-increasing while positive, and once it
    reaches zero it stays zero for all wider intervals (paper, Sec. 5.2)."""
    joint = results.sort_values("phi")["joint_exposure"].to_numpy()
    diffs = np.diff(joint)
    non_increasing = bool(np.all(diffs <= 1e-9))
    zero_idx = np.argmax(joint <= 1e-9) if np.any(joint <= 1e-9) else None
    stays_zero = bool(zero_idx is None or np.all(joint[zero_idx:] <= 1e-9))
    return {"joint_exposure_non_increasing": non_increasing, "joint_exposure_stays_zero": stays_zero}


if __name__ == "__main__":
    p = benchmark_primitives()
    print(
        "Thresholds: full-underwriting", full_underwriting_threshold(p),
        "pure-underwriting", pure_underwriting_threshold(p),
        "zero-underwriting", zero_underwriting_threshold(p),
    )
    results = run(p=p)
    checks = validate(results)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/exp3_equilibrium_adjustment.csv", index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panels = [
        ("coverage_pct", "Coverage x* (%)"),
        ("abs_financial_exposure", "Absolute financial exposure sigma|y*|/eta"),
        ("joint_exposure", "Joint exposure x*sigma|y*|/eta"),
        ("loading_pct", "Loading theta* (%)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.plot(results["phi"], results[col], marker="o")
        ax.axvline(config.EMPIRICAL_PHI_REFERENCE, color="grey", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Ambiguity half-width phi")
    fig.suptitle("Competitive equilibrium adjustment, benchmark parameters (Fig. 5.2)")
    fig.tight_layout()
    fig.savefig("results/exp3_equilibrium_adjustment.png", dpi=150)

    print("Validation checks:", checks)
    print(results.to_string(index=False))
