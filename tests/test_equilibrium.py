"""Unit tests for src/equilibrium.py against properties stated in Deng and
Pang, "Correlation Ambiguity, Diversification Fragility, and Insurance
Pricing" (Proposition 2, Corollary 1, Theorem 2, Proposition 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.equilibrium import (
    Primitives,
    ce_derivatives,
    conditional_financial_position,
    direct_ambiguity_cost,
    equilibrium_allocation,
    equilibrium_path,
    full_underwriting_threshold,
    pure_underwriting_threshold,
    zero_underwriting_threshold,
)


def benchmark() -> Primitives:
    """Sec. 5.1 benchmark primitives."""
    return Primitives(l=2.0, eta=0.28, r=0.015, mu=0.035, sigma=0.18, alpha=2.0,
                       gamma=2.0, T=5.0, t=0.0)


def test_thresholds_match_benchmark_worked_example() -> None:
    """At the Sec. 5.1 benchmark parameters, the regime thresholds of
    Theorem 2 work out to approximately -0.912 (full-underwriting), 0.3825
    (pure-underwriting) and 5.04 (zero-underwriting, Sec. 5.2)."""
    p = benchmark()
    assert full_underwriting_threshold(p) == pytest.approx(-0.912, abs=5e-3)
    assert pure_underwriting_threshold(p) == pytest.approx(0.3825, abs=5e-3)
    assert zero_underwriting_threshold(p) == pytest.approx(5.04, abs=5e-2)


def test_conditional_position_reduces_to_stand_alone_at_zero_ambiguity() -> None:
    """At phi = 0 (rho_lower = rho_upper = rho0 = 0), y_t(x) must equal the
    stand-alone optimum y_t(0) = (mu - r) e^{-r(T-t)} / (gamma sigma^2),
    since the ambiguity term vanishes (Eq. 3.4 with rho = 0)."""
    p = benchmark()
    y = conditional_financial_position(p, x=0.5, rho_lower=0.0, rho_upper=0.0)
    assert float(y) == pytest.approx(float(p.y0), rel=1e-9)


def test_conditional_position_zero_investment_region() -> None:
    """When the threshold ratio k lies within [rho_lower, rho_upper], the
    optimal financial position is exactly zero (Proposition 2, Eq. 3.4)."""
    p = benchmark()
    x = 0.5
    k = (p.mu - p.r) * p.disc / (p.gamma * p.eta * p.sigma * x)
    y_zero = conditional_financial_position(p, x=x, rho_lower=k - 0.05, rho_upper=k + 0.05)
    assert float(y_zero) == pytest.approx(0.0, abs=1e-9)


def test_conditional_position_weakly_decreasing_in_phi() -> None:
    """Figure 5.1: absolute financial investment is weakly decreasing in the
    ambiguity half-width phi, at each fixed underwriting share."""
    p = benchmark()
    for x in (0.25, 0.5, 0.75):
        phi_grid = np.linspace(0.0, 1.0, 21)
        y = np.array([
            conditional_financial_position(p, x, -phi, phi) for phi in phi_grid
        ])
        assert np.all(np.diff(np.abs(y)) <= 1e-9), f"|y| not weakly decreasing for x={x}"


def test_conditional_position_slope_minus_x_on_positive_branch() -> None:
    """Corollary 1 (Eq. 3.8): on the positive-investment branch, the slope of
    the normalized exposure sigma*y/eta with respect to phi equals -x."""
    p = benchmark()
    for x in (0.25, 0.5, 0.75):
        phi_grid = np.linspace(0.0, 0.05, 6)  # small phi: stays on positive branch
        y = np.array([conditional_financial_position(p, x, -phi, phi) for phi in phi_grid])
        normalized = p.sigma * y / p.eta
        slope = np.polyfit(phi_grid, normalized, 1)[0]
        assert slope == pytest.approx(-x, abs=1e-6)


def test_equilibrium_allocation_matches_pure_underwriting_worked_example() -> None:
    """At the benchmark, phi >= 0.3825 (rho0 = 0) implies pure underwriting
    (Eq. 4.7-4.8), with x* = (disc/gamma) / (1/alpha + disc/gamma) and y* = 0."""
    p = benchmark()
    allocation = equilibrium_allocation(p, rho_lower=-0.5, rho_upper=0.5)
    expected_x = (p.disc / p.gamma) / (1.0 / p.alpha + p.disc / p.gamma)
    assert allocation["regime"] == "pure_underwriting"
    assert allocation["x"] == pytest.approx(expected_x, rel=1e-9)
    assert allocation["y"] == pytest.approx(0.0, abs=1e-12)


def test_equilibrium_allocation_interior_positive_at_small_phi() -> None:
    """Below the pure-underwriting threshold (phi < 0.3825 at rho0 = 0), the
    benchmark equilibrium is interior with positive investment (Eq. 4.5-4.6)."""
    p = benchmark()
    allocation = equilibrium_allocation(p, rho_lower=-0.1, rho_upper=0.1)
    assert allocation["regime"] == "interior_positive_investment"
    assert 0.0 < allocation["x"] < 1.0
    assert allocation["y"] > 0.0


def test_equilibrium_path_joint_exposure_non_increasing() -> None:
    """Sec. 5.2: joint exposure x*|y*| strictly decreases while positive and
    stays at zero once it vanishes, as phi widens at fixed rho0 = 0."""
    p = benchmark()
    phi_grid = np.linspace(0.0, 1.0, 41)
    path = equilibrium_path(p, rho0=0.0, phi_grid=phi_grid)
    joint = np.array([abs(row["x"] * row["y"]) for row in path])
    assert np.all(np.diff(joint) <= 1e-9)
    zero_from = np.argmax(joint <= 1e-9)
    assert np.all(joint[zero_from:] <= 1e-9)


def test_low_alpha_zero_underwriting_threshold() -> None:
    """Sec. 5.3: with alpha = 0.2, the zero-underwriting threshold
    alpha*eta*sigma/(mu-r) falls to approximately 0.504, inside [0, 1]."""
    p = Primitives(l=2.0, eta=0.28, r=0.015, mu=0.035, sigma=0.18, alpha=0.2,
                   gamma=2.0, T=5.0, t=0.0)
    assert zero_underwriting_threshold(p) == pytest.approx(0.504, abs=5e-3)
    allocation = equilibrium_allocation(p, rho_lower=-0.9, rho_upper=0.9)
    assert allocation["regime"] == "zero_underwriting"
    assert allocation["x"] == pytest.approx(0.0, abs=1e-12)


def test_direct_ambiguity_cost_nonnegative() -> None:
    """The direct ambiguity cost gamma e^{r(T-t)} eta sigma x |y| is a
    product of non-negative terms given x >= 0 (Sec. 4.3)."""
    p = benchmark()
    assert direct_ambiguity_cost(p, x=0.3, y=-0.2) >= 0.0
    assert direct_ambiguity_cost(p, x=0.0, y=0.5) == pytest.approx(0.0, abs=1e-12)


def test_ce_derivatives_total_equals_minus_direct_cost() -> None:
    """Proposition 5 (Eq. 4.19): dCE^T/dphi = -gamma e^{r(T-t)} eta sigma
    x |y|, independent of the loading effect (the loading effect only
    redistributes surplus between insurer and household)."""
    p = benchmark()
    x, y, dtheta_dphi = 0.4, 0.1, -0.02
    out = ce_derivatives(p, x, y, dtheta_dphi)
    assert out["d_ce_total"] == pytest.approx(-direct_ambiguity_cost(p, x, y), rel=1e-9)
    assert out["d_ce_insurer"] + out["d_ce_household"] == pytest.approx(out["d_ce_total"], rel=1e-9)


def test_conditional_financial_position_rejects_nonpositive_x() -> None:
    p = benchmark()
    with pytest.raises(AssertionError):
        conditional_financial_position(p, x=0.0, rho_lower=-0.1, rho_upper=0.1)


def test_equilibrium_allocation_rejects_invalid_interval() -> None:
    p = benchmark()
    with pytest.raises(AssertionError):
        equilibrium_allocation(p, rho_lower=0.2, rho_upper=-0.2)
