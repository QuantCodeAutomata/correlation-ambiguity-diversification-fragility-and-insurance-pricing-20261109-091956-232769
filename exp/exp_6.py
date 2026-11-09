"""Experiment 6: marginal ambiguity cost and incidence analysis (Deng and
Pang, Sec. 4.3, Figure 5.5).

Along the competitive-equilibrium path of correlation intervals
[rho0 - phi, rho0 + phi] with benchmark parameters and rho0 = 0, this script
computes the direct ambiguity cost gamma*e^{r(T-t)}*eta*sigma*x*|y*|, the
loading effect l*x*dtheta*/dphi, and the envelope derivatives
dCE^I/dphi, dCE^H/dphi, dCE^T/dphi of Proposition 5 (Eq. 4.19), with
dtheta*/dphi obtained by central differences on the equilibrium path
(one-sided at phi = 0 and at regime boundaries, as in equilibrium_path_derivative).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config
from src.equilibrium import (
    Primitives,
    ce_derivatives,
    direct_ambiguity_cost,
    equilibrium_allocation,
    equilibrium_path_derivative,
)
from exp.exp_3 import benchmark_primitives


def run(
    p: Primitives | None = None,
    rho0: float = config.BENCH_RHO0,
    phi_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    p = p or benchmark_primitives()
    if phi_grid is None:
        phi_grid = np.round(np.arange(config.PHI_GRID_MIN, config.PHI_GRID_MAX + 1e-9,
                                       config.EXP6_PHI_GRID_STEP), 10)
    rows = []
    for phi in phi_grid:
        allocation = equilibrium_allocation(p, rho0 - phi, rho0 + phi)
        x, y = allocation["x"], allocation["y"]
        dtheta_dphi = equilibrium_path_derivative(p, rho0, phi)
        derivatives = ce_derivatives(p, x, y, dtheta_dphi)
        rows.append(
            {
                "phi": float(phi),
                "x": x,
                "y": y,
                "theta": allocation["theta"],
                "regime": allocation["regime"],
                "direct_ambiguity_cost": direct_ambiguity_cost(p, x, y),
                "dtheta_dphi": dtheta_dphi,
                "loading_effect": derivatives["loading_effect"],
                "d_ce_insurer": derivatives["d_ce_insurer"],
                "d_ce_household": derivatives["d_ce_household"],
                "d_ce_total": derivatives["d_ce_total"],
            }
        )
    return pd.DataFrame(rows)


def validate(results: pd.DataFrame) -> dict:
    """Checks from Proposition 5: the direct ambiguity cost is non-negative,
    and the total local certainty-equivalent derivative is non-positive
    whenever joint exposure is positive (and zero once it vanishes)."""
    joint_positive = (results["x"] * results["y"].abs()) > 1e-9
    return {
        "direct_cost_nonnegative": bool((results["direct_ambiguity_cost"] >= -1e-9).all()),
        "total_ce_derivative_nonpositive_while_joint_exposure_positive": bool(
            (results.loc[joint_positive, "d_ce_total"] <= 1e-9).all()
        ),
        "total_ce_derivative_zero_after_joint_exposure_vanishes": bool(
            np.allclose(results.loc[~joint_positive, "d_ce_total"], 0.0, atol=1e-9)
        ),
    }


if __name__ == "__main__":
    results = run()
    checks = validate(results)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/exp6_marginal_ambiguity_cost.csv", index=False)

    panel_a_points = results[results["phi"].isin(config.EXP6_TABLE_PHI_POINTS)]
    panel_a_points.to_csv("results/exp6_panel_a_selected_points.csv", index=False)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(results["phi"], results["direct_ambiguity_cost"], marker="o", label="Direct ambiguity cost")
    ax1.plot(results["phi"], results["loading_effect"], marker="s", label="Loading effect")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.axvline(config.EMPIRICAL_PHI_REFERENCE, color="grey", linestyle="--")
    ax1.set_xlabel("Ambiguity half-width phi")
    ax1.set_title("Panel A: direct cost and loading effect")
    ax1.legend()

    ax2.plot(results["phi"], results["d_ce_insurer"], marker="o", label="dCE^I/dphi")
    ax2.plot(results["phi"], results["d_ce_household"], marker="s", label="dCE^H/dphi")
    ax2.plot(results["phi"], results["d_ce_total"], marker="^", label="dCE^T/dphi")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.axvline(config.EMPIRICAL_PHI_REFERENCE, color="grey", linestyle="--")
    ax2.set_xlabel("Ambiguity half-width phi")
    ax2.set_title("Panel B: certainty-equivalent derivatives")
    ax2.legend()

    fig.suptitle("Marginal ambiguity cost and incidence (Fig. 5.5)")
    fig.tight_layout()
    fig.savefig("results/exp6_marginal_ambiguity_cost.png", dpi=150)

    print("Validation checks:", checks)
    print(results.to_string(index=False))
