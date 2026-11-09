"""Analytic implementation of the insurer's conditional investment problem
(Sec. 3) and the competitive insurance equilibrium (Sec. 4) of Deng and Pang,
"Correlation Ambiguity, Diversification Fragility, and Insurance Pricing".

All quantities are closed-form and deterministic given the model primitives
l, eta, r, mu, sigma, alpha, gamma, T, t, and a correlation interval
[rho_lower, rho_upper]. No numerical optimizer is used: the formulas below
reproduce Proposition 2 (Eq. 3.4), Corollary 1 (Eq. 3.8), Theorem 2
(Eq. 4.3-4.12), Proposition 4 (Eq. 4.18) and Proposition 5 (Eq. 4.19-4.20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Primitives:
    """Model primitives shared by the conditional-investment and equilibrium
    problems (paper, Sec. 2-4)."""

    l: float
    eta: float
    r: float
    mu: float
    sigma: float
    alpha: float
    gamma: float
    T: float
    t: float = 0.0

    @property
    def tau(self) -> float:
        """Time to maturity T - t."""
        return self.T - self.t

    @property
    def disc(self) -> float:
        """Discount factor e^{-r(T-t)}."""
        return np.exp(-self.r * self.tau)

    @property
    def y0(self) -> np.ndarray:
        """Stand-alone optimal financial position y_t(0), Eq. 3.5."""
        return (self.mu - self.r) / (self.gamma * self.sigma**2) * self.disc


def conditional_financial_position(
    p: Primitives, x: np.ndarray, rho_lower: np.ndarray, rho_upper: np.ndarray
) -> np.ndarray:
    """Optimal financial position y_t(x) at a fixed underwriting share x > 0,
    for correlation interval [rho_lower, rho_upper] (Proposition 2, Eq. 3.4).

    Parameters are broadcast with numpy; x, rho_lower, rho_upper may be
    scalars or arrays of the same broadcastable shape.
    """
    x = np.asarray(x, dtype=float)
    rho_lower = np.asarray(rho_lower, dtype=float)
    rho_upper = np.asarray(rho_upper, dtype=float)
    assert np.all(x > 0), "conditional_financial_position requires x > 0 (paper Prop. 2)"
    assert np.all(rho_lower <= rho_upper), "rho_lower must not exceed rho_upper"

    k = (p.mu - p.r) * p.disc / (p.gamma * p.eta * p.sigma * x)  # threshold ratio
    y_pos = (p.mu - p.r) / (p.gamma * p.sigma**2) * p.disc - rho_upper * p.eta * x / p.sigma
    y_neg = (p.mu - p.r) / (p.gamma * p.sigma**2) * p.disc - rho_lower * p.eta * x / p.sigma

    y = np.where(k > rho_upper, y_pos, np.where(k < rho_lower, y_neg, 0.0))
    return y


# ---------------------------------------------------------------------------
# Theorem 2: competitive equilibrium regimes
# ---------------------------------------------------------------------------

REGIME_FULL_UNDERWRITING = "full_underwriting"
REGIME_INTERIOR_POSITIVE = "interior_positive_investment"
REGIME_PURE_UNDERWRITING = "pure_underwriting"
REGIME_INTERIOR_NEGATIVE = "interior_negative_investment"
REGIME_ZERO_UNDERWRITING = "zero_underwriting"


def full_underwriting_threshold(p: Primitives) -> float:
    """Upper-bound threshold in condition (4.3)."""
    a = (p.mu - p.r) * p.disc / (2 * p.gamma * p.eta * p.sigma)
    return a - np.sqrt(1.0 + a**2)


def pure_underwriting_threshold(p: Primitives) -> float:
    """Threshold correlation in conditions (4.7)/(A.8)."""
    return (p.mu - p.r) * (1.0 / p.alpha + p.disc / p.gamma) / (p.eta * p.sigma)


def zero_underwriting_threshold(p: Primitives) -> float:
    """Threshold in condition (4.11)."""
    return p.alpha * p.eta * p.sigma / (p.mu - p.r)


def equilibrium_allocation(p: Primitives, rho_lower: float, rho_upper: float) -> dict:
    """Competitive equilibrium (x*, theta*, y*, regime) for one correlation
    interval [rho_lower, rho_upper], following Theorem 2 (Eq. 4.3-4.12).

    Returns a dict with keys 'x', 'theta', 'y', 'regime'.
    """
    assert rho_lower <= rho_upper, "rho_lower must not exceed rho_upper"
    # Assumption 1 requires an open admissible domain (-1, 1); the closed
    # boundary rho = +-1 is allowed here only because the full- and
    # zero-underwriting formulas (4.4)/(4.12) do not involve 1 - rho^2 and
    # remain well defined there, matching the grids used in Figures 5.1-5.4.
    assert -1.0 <= rho_lower and rho_upper <= 1.0, "correlation interval must lie in [-1, 1]"

    full_thr = full_underwriting_threshold(p)
    pure_thr = pure_underwriting_threshold(p)
    zero_thr = zero_underwriting_threshold(p)

    if rho_upper <= full_thr:
        # (i) Full underwriting, Eq. 4.4.
        theta = 0.0
        x = 1.0
        y = (p.mu - p.r) / (p.gamma * p.sigma**2) * p.disc - rho_upper * p.eta / p.sigma
        regime = REGIME_FULL_UNDERWRITING
    elif rho_upper >= zero_thr:
        # (v) Zero underwriting, Eq. 4.12.
        theta = p.alpha * p.eta**2 / p.l
        x = 0.0
        y = (p.mu - p.r) / (p.gamma * p.sigma**2) * p.disc
        regime = REGIME_ZERO_UNDERWRITING
    elif rho_lower <= pure_thr <= rho_upper:
        # (iii) Pure underwriting, Eq. 4.8.
        x = (p.disc / p.gamma) / (1.0 / p.alpha + p.disc / p.gamma)
        theta = p.eta**2 / (p.l * (1.0 / p.alpha + p.disc / p.gamma))
        y = 0.0
        regime = REGIME_PURE_UNDERWRITING
    elif pure_thr < rho_lower:
        # (iv) Interior underwriting with negative investment, Eq. 4.10.
        rho = rho_lower
        x = (p.disc / p.gamma) * (1.0 - rho * (p.mu - p.r) / (p.alpha * p.eta * p.sigma)) / (
            (1.0 - rho**2) / p.alpha + p.disc / p.gamma
        )
        theta = p.alpha * p.eta**2 / p.l * (1.0 - x)
        y = (p.mu - p.r) * p.disc / (p.gamma * p.sigma**2) - rho * p.eta * x / p.sigma
        regime = REGIME_INTERIOR_NEGATIVE
    else:
        # (ii) Interior underwriting with positive investment, Eq. 4.6.
        rho = rho_upper
        x = (p.disc / p.gamma) * (1.0 - rho * (p.mu - p.r) / (p.alpha * p.eta * p.sigma)) / (
            (1.0 - rho**2) / p.alpha + p.disc / p.gamma
        )
        theta = p.alpha * p.eta**2 / p.l * (1.0 - x)
        y = (p.mu - p.r) * p.disc / (p.gamma * p.sigma**2) - rho * p.eta * x / p.sigma
        regime = REGIME_INTERIOR_POSITIVE

    return {"x": float(x), "theta": float(theta), "y": float(y), "regime": regime}


def equilibrium_path(p: Primitives, rho0: float, phi_grid: np.ndarray) -> "list[dict]":
    """Equilibrium allocation at each ambiguity half-width in phi_grid, along
    the fixed-midpoint widening [rho0 - phi, rho0 + phi] (paper, Sec. 4.2,
    Eq. 4.16)."""
    out = []
    for phi in phi_grid:
        rho_lower, rho_upper = rho0 - phi, rho0 + phi
        allocation = equilibrium_allocation(p, rho_lower, rho_upper)
        allocation["phi"] = float(phi)
        out.append(allocation)
    return out


# ---------------------------------------------------------------------------
# Proposition 4 / 5: envelope derivatives, ambiguity cost and incidence
# ---------------------------------------------------------------------------


def direct_ambiguity_cost(p: Primitives, x: float, y: float) -> float:
    """Direct marginal ambiguity cost gamma * e^{r(T-t)} * eta * sigma * x * |y|
    at the fixed-loading optimum (paper, Sec. 4.3, discussion following
    Eq. 4.18; discount factor e^{r(T-t)} = 1/disc)."""
    accrual = np.exp(p.r * p.tau)
    return p.gamma * accrual * p.eta * p.sigma * x * abs(y)


def ce_derivatives(
    p: Primitives, x: float, y: float, dtheta_dphi: float
) -> dict:
    """Envelope derivatives of insurer, household and total local certainty
    equivalents with respect to phi (Proposition 5, Eq. 4.19), given the
    equilibrium allocation (x, y) and dtheta*/dphi at that point."""
    accrual = np.exp(p.r * p.tau)
    direct_cost = p.gamma * accrual * p.eta * p.sigma * x * abs(y)
    loading_effect = p.l * x * dtheta_dphi
    d_ce_insurer = -direct_cost + loading_effect
    d_ce_household = -loading_effect
    d_ce_total = -direct_cost
    return {
        "direct_cost": direct_cost,
        "loading_effect": loading_effect,
        "d_ce_insurer": d_ce_insurer,
        "d_ce_household": d_ce_household,
        "d_ce_total": d_ce_total,
    }


def dtheta_dx(p: Primitives) -> float:
    """d(theta*)/d(x*) in the interior regimes (ii)/(iv), from theta* =
    alpha*eta^2*(1-x*)/l (Eq. 4.6/4.10)."""
    return -p.alpha * p.eta**2 / p.l


def equilibrium_path_derivative(p: Primitives, rho0: float, phi: float, eps: float = 1e-5) -> float:
    """Numerical derivative d(theta*)/d(phi) along the fixed-midpoint path
    rho0 +- phi (paper, Sec. 4.3, Proposition 5), by central differences on
    equilibrium_allocation. A one-sided difference is used at phi = 0 (the
    left endpoint of the admissible grid) and wherever the central step
    would cross outside [0, 1 - |rho0|); this reproduces the one-sided
    derivatives used at regime boundaries and at phi = 0 (paper, Figure 5.5
    construction notes)."""
    phi_max = 1.0 - abs(rho0)
    phi_lo = max(phi - eps, 0.0)
    phi_hi = min(phi + eps, phi_max)
    if phi_hi == phi_lo:
        return 0.0
    theta_lo = equilibrium_allocation(p, rho0 - phi_lo, rho0 + phi_lo)["theta"]
    theta_hi = equilibrium_allocation(p, rho0 - phi_hi, rho0 + phi_hi)["theta"]
    return (theta_hi - theta_lo) / (phi_hi - phi_lo)
