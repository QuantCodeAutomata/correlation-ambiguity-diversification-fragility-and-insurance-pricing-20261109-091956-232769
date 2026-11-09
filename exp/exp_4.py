"""Experiment 4: equilibrium investment adjustment with low household risk
aversion (Deng and Pang, Sec. 5.3, Theorem 2, Remark 3, Figure 5.3).

With alpha reduced from the benchmark value (Sec. 5.1) to alpha = 0.2, all
other primitives held at their Sec. 5.1 benchmark values and rho_0 = 0, this
script traces the competitive-equilibrium underwriting share x* and the
(signed) normalized financial exposure sigma*y*/eta over the ambiguity
half-width grid of Figure 5.3. The qualitative point (Remark 3) is that,
unlike the benchmark-alpha case of exp_3, the signed exposure need not be
monotone in phi: as x* shrinks under low risk aversion the household's
desired financial position can first become less negative and later rise
further, producing a local "turning point" before x* reaches zero.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config
from src.equilibrium import (
    Primitives,
    equilibrium_path,
    pure_underwriting_threshold,
    zero_underwriting_threshold,
)
from exp.exp_3 import benchmark_primitives


def run(
    p: Primitives | None = None,
    rho0: float = config.BENCH_RHO0,
    phi_grid: "tuple[float, ...]" = config.EXP4_PHI_GRID,
) -> pd.DataFrame:
    """Equilibrium path over phi_grid at low household risk aversion; returns
    a DataFrame with columns phi, x, signed_normalized_exposure,
    joint_exposure, regime."""
    p = p if p is not None else benchmark_primitives(alpha=config.LOW_ALPHA)
    path = equilibrium_path(p, rho0, np.asarray(phi_grid, dtype=float))
    df = pd.DataFrame(path)
    df["signed_normalized_exposure"] = p.sigma * df["y"] / p.eta
    df["joint_exposure"] = df["x"] * df["signed_normalized_exposure"].abs()
    df = df[["phi", "x", "signed_normalized_exposure", "joint_exposure", "regime"]]
    return df


def validate(results: pd.DataFrame) -> dict:
    """Reports (without raising): joint-exposure monotonicity, the phi of the
    pre-x*=0 turning point in |signed_normalized_exposure|, the regime
    thresholds, and the Remark 3 sufficient condition on the primitives."""
    sorted_results = results.sort_values("phi")
    joint = sorted_results["joint_exposure"].to_numpy()
    diffs = np.diff(joint)
    non_increasing = bool(np.all(diffs <= 1e-9))
    zero_idx = np.argmax(joint <= 1e-9) if np.any(joint <= 1e-9) else None
    stays_zero = bool(zero_idx is None or np.all(joint[zero_idx:] <= 1e-9))

    positive_x = sorted_results[sorted_results["x"] > 1e-9]
    turning_point_phi = None
    has_turning_point = False
    if len(positive_x) > 0:
        abs_exposure = positive_x["signed_normalized_exposure"].abs().to_numpy()
        phis = positive_x["phi"].to_numpy()
        min_idx = int(np.argmin(abs_exposure))
        turning_point_phi = float(phis[min_idx])
        # a genuine local minimum strictly before x* = 0 requires an interior point
        has_turning_point = bool(0 < min_idx < len(abs_exposure) - 1)

    p = benchmark_primitives(alpha=config.LOW_ALPHA)
    sufficient_condition_satisfied = bool((p.mu - p.r) / p.sigma > p.alpha * p.eta)

    return {
        "joint_exposure_non_increasing": non_increasing,
        "joint_exposure_stays_zero": stays_zero,
        "turning_point_phi": turning_point_phi,
        "turning_point_is_interior": has_turning_point,
        "zero_underwriting_threshold": float(zero_underwriting_threshold(p)),
        "pure_underwriting_threshold": float(pure_underwriting_threshold(p)),
        "sufficient_condition_satisfied": sufficient_condition_satisfied,
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    results = run()
    checks = validate(results)
    results.to_csv("results/exp4_low_risk_aversion.csv", index=False)

    print(checks)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(results["phi"], results["x"], marker="o", label="x*")
    ax.axvline(
        config.EMPIRICAL_PHI_REFERENCE, color="grey", linestyle="--",
        label=f"empirical phi = {config.EMPIRICAL_PHI_REFERENCE}",
    )
    ax.set_xlabel("Ambiguity half-width phi")
    ax.set_ylabel("Underwriting share x*")
    ax.set_title("Equilibrium underwriting x*")
    ax.legend()

    ax = axes[1]
    ax.plot(
        results["phi"], results["signed_normalized_exposure"], marker="o",
        label="sigma*y*/eta",
    )
    ax.axvline(
        config.EMPIRICAL_PHI_REFERENCE, color="grey", linestyle="--",
        label=f"empirical phi = {config.EMPIRICAL_PHI_REFERENCE}",
    )
    ax.set_xlabel("Ambiguity half-width phi")
    ax.set_ylabel("Signed normalized exposure sigma*y*/eta")
    ax.set_title("Signed normalized financial exposure")
    ax.legend()

    fig.suptitle(
        "Equilibrium investment adjustment with low household risk aversion "
        "(alpha=0.2), Fig. 5.3"
    )
    fig.tight_layout()
    fig.savefig("results/exp4_low_risk_aversion.png", dpi=150)

    print(results.to_string(index=False))
