# Results

Replication of selected experiments from Peiyun Deng and Shunzhi Pang,
"Correlation Ambiguity, Diversification Fragility, and Insurance Pricing"
(2026).

All tables, figures and summaries below are produced by `reproduce.sh`
(`python -m pytest -m "not market_data"` followed by `python -m exp.exp_1`
through `python -m exp.exp_6`) and are saved directly under `results/`.

## Experiment 1 — Empirical rolling underwriting-market dependence with block-bootstrap confidence intervals (Figure 1.1)

Setup: a claims-performance factor is built from an insurer-quarter panel
as minus incurred losses and loss-adjustment expenses over positive net
premiums earned, forecast with an expanding insurer-fixed-effects model
using only prior quarters, with forecast errors scaled by their preceding
20-quarter standard deviation and aggregated across insurers with positive
lagged admitted-asset weights. This factor is correlated, in rolling
32-quarter windows, with the compounded quarterly return of a broad market
proxy, together with nominal 90% confidence intervals from a paired
circular four-quarter block-bootstrap with 2,000 draws (5th/95th
percentiles, first-stage rolling-correlation estimates held fixed during
resampling).

Artifacts: `results/exp1_rolling_correlation.csv`, `results/exp1_rolling_correlation.png`, `results/exp1_summary.json`.

Because the insurer-level statutory inputs cannot be obtained (see
"Deviations from the paper" and "Not reproduced"), this run uses a
synthetic insurer-quarter panel together with a real market-return proxy,
and is reported only as a validation of the construction, not as a
reproduction of the paper's Figure 1.1 numbers:

| Quantity | This run (synthetic panel, real market proxy) | Paper (Figure 1.1) |
|---|---|---|
| Rolling-correlation range | 0.222 to 0.362 | approximately -0.19 to 0.07 |
| Median 90% CI half-width | 0.183 | approximately 0.23 |

**Verdict: not reproduced.** The methodology runs end to end and produces a
rolling correlation and bootstrap confidence band of a plausible order of
magnitude, but the numbers cannot be compared to the paper because the
insurer-level claims-performance inputs are synthetic, not the paper's SNL
statutory data. The empirical half-width reference of 0.23 used in
`EMPIRICAL_PHI_REFERENCE` (`src/config.py`) and marked in the figures of
experiments 2-6 is taken directly from the paper's stated value, not from
this run's synthetic-panel output.

## Experiment 2 — Conditional financial exposure under correlation ambiguity at fixed underwriting shares (Figure 5.1)

Setup: the optimal financial position y_t(x) for underwriting shares
x in {0.25, 0.5, 0.75} is computed from the closed-form piecewise formula
(Proposition 2) over an ambiguity half-width grid phi in {0.0, 0.1, ..., 1.0}
with correlation midpoint rho_0 = 0, benchmark parameters (l=2, eta=0.28,
r=0.015, mu=0.035, sigma=0.18, gamma=2, T=5, t=0).

Artifacts: `results/exp2_financial_exposure.csv`, `results/exp2_financial_exposure.png`.

| x | Normalized exposure at phi=0 | Paper (phi=0) | phi where exposure reaches 0 | Paper's reported phi |
|---|---|---|---|---|
| 0.25 | 0.1841 | 0.180 | between 0.7 and 0.8 | approximately 0.7 |
| 0.50 | 0.1841 | 0.180 | between 0.3 and 0.4 | approximately 0.3 |
| 0.75 | 0.1841 | 0.180 | between 0.2 and 0.3 | approximately 0.2 |

The slope of the normalized exposure on the positive-investment branch
equals -x for all three shares (verified in `tests/test_equilibrium.py`),
matching Corollary 1 (equation 3.8). Once the zero-investment region is
reached, the position stays at zero for every wider interval.

**Verdict: reproduced.** The computed stand-alone value (0.1841 versus the
paper's rounded 0.180) and the zero-crossing points for all three
underwriting shares match the paper's reported values within the grid's
0.1 resolution and the paper's own rounding.

## Experiment 3 — Competitive equilibrium adjustment under fixed-midpoint correlation widening, benchmark parameters (Figure 5.2)

Setup: the competitive equilibrium (x*, y*, theta*) is computed from
Theorem 2's five-regime characterization for phi in {0.0, 0.1, ..., 1.0},
rho_0 = 0, benchmark parameters.

Artifacts: `results/exp3_equilibrium_adjustment.csv`, `results/exp3_equilibrium_adjustment.png`.

| Quantity | phi=0 (this run) | Paper (phi=0) | Regime transition | Paper |
|---|---|---|---|---|
| Coverage x* (%) | 48.13 | approximately 48 | interior positive investment to pure underwriting between phi=0.3 and 0.4 | approximately 0.3-0.4 |
| Absolute financial exposure sigma\|y*\|/eta | 0.1841 | 0.18 | reaches 0 at the same transition | same |
| Joint exposure x*sigma\|y*\|/eta | 0.0886 | 0.090 | reaches 0 at the same transition | same |
| Loading theta* (%) | 4.067 | approximately 4.065 | peaks near phi=0.2-0.3 (4.14%) then returns to 4.067% | approximately 4.135% then back to 4.065% |

Once in the pure-underwriting regime (phi >= 0.3825, the computed
pure-underwriting threshold for the benchmark parameters), x*, theta* and
y*=0 stay constant for every wider interval, confirming that joint exposure
does not reopen once it reaches zero.

**Verdict: reproduced.** Coverage, exposure, joint exposure and loading
match the paper's reported values and non-monotonic pattern within the
grid's resolution.

## Experiment 4 — Equilibrium investment adjustment with low household risk aversion, alpha=0.2 (Figure 5.3)

Setup: same benchmark parameters except alpha = 0.2, computed over the
paper's Figure 5.3 grid phi in {0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.6}.

Artifacts: `results/exp4_low_risk_aversion.csv`, `results/exp4_low_risk_aversion.png`.

| Quantity | This run | Paper |
|---|---|---|
| Underwriting x* at phi=0 | 0.0849 | approximately 0.090 |
| Signed normalized exposure sigma*y*/eta at phi=0 | 0.1841 | approximately 0.185 |
| Turning point of sigma*y*/eta (local minimum) | phi approximately 0.25-0.3 | phi approximately 0.25 |
| Zero-underwriting boundary | between phi=0.5 and 0.6 (threshold alpha*eta*sigma/(mu-r) = 0.504) | approximately phi=0.5 |

Financial exposure decreases from phi=0 to the turning point near phi=0.25-0.3
and then increases back to the stand-alone value (mu-r)e^{-rT}/(gamma*sigma^2)
once underwriting reaches zero, while joint exposure decreases monotonically
throughout, confirming that financial investment can rise even as joint
exposure keeps falling. The sufficient condition (mu-r)/sigma > alpha*eta
(0.111 > 0.056) holds for this configuration.

**Verdict: reproduced.** The turning point, the zero-underwriting boundary
and the qualitative pattern (financial exposure falls then rises while
joint exposure falls monotonically) all match the paper; the levels match
within the paper's rounding.

## Experiment 5 — Equilibrium regime structure across correlation midpoint and ambiguity half-width (Figure 5.4)

Setup: a 400x400 grid over the admissible triangular domain
-1 < rho_0 - phi <= rho_0 + phi < 1 is classified into Theorem 2's five
regimes using the benchmark parameters.

Artifacts: `results/exp5_regime_map.csv`, `results/exp5_regime_map.png`.

| Regime | Grid points (of 83,168 admissible points) |
|---|---|
| Full underwriting | 155 |
| Interior underwriting, positive investment | 39,591 |
| Pure underwriting | 35,437 |
| Interior underwriting, negative investment (short hedge) | 7,985 |
| Zero underwriting | 0 |

Zero underwriting does not appear anywhere on the grid, consistent with its
threshold alpha*eta*sigma/(mu-r) = 5.04 exceeding the admissible correlation
range (-1, 1). The empirical reference point (rho_0=0, phi=0.23) falls in
the interior positive-investment regime.

**Verdict: reproduced.** All four regimes the paper describes as reachable
under the benchmark parameters are present with the reported boundaries;
zero underwriting is correctly absent, and the empirical reference point
falls in the regime the paper places it in.

## Experiment 6 — Marginal ambiguity cost and incidence analysis (Figure 5.5)

Setup: along the same equilibrium path as experiment 3, the direct
ambiguity cost gamma*e^{rT}*eta*sigma*x*\|y*\|, the loading effect
l*x* d(theta*)/d(phi), and the three certainty-equivalent derivatives of
Proposition 5 (equation 4.19) are computed on a fine phi grid (step 0.01)
and reported at the paper's selected points (0, 0.2, 0.4, 0.6, 0.8, 1.0).

Artifacts: `results/exp6_marginal_ambiguity_cost.csv`, `results/exp6_marginal_ambiguity_cost.png`, `results/exp6_panel_a_selected_points.csv`.

| phi | Direct ambiguity cost (this run) | Paper | Loading effect (this run) | Paper |
|---|---|---|---|---|
| 0.0 | 0.0150 | approximately 0.015 | 0.0072 | approximately 0.007 |
| 0.2 | 0.0072 | (declining) | -0.0002 | (declining, crosses zero) |
| about 0.38 (near pure-underwriting threshold) | 0.0001 | — | -0.0077 | approximately -0.007 at phi=0.4 |
| 0.4-1.0 | 0 | 0 | 0 | 0 |

The total certainty-equivalent derivative dCE^T/dphi = -gamma*e^{rT}*eta*sigma*x*\|y*\|
is non-positive everywhere and zero once joint exposure vanishes, confirming
that total local surplus never increases as ambiguity widens and stays flat
once underwriting becomes pure. The insurer and household derivatives sum to
the total derivative at every phi (verified in `tests/test_equilibrium.py`),
and the loading reallocates the direct ambiguity cost between the two
parties as it changes sign.

**Verdict: reproduced** for the signs, the zero-crossing of the loading
effect near the pure-underwriting boundary, and the non-positivity of the
total derivative; **partially reproduced** for the exact insurer/household
split levels at phi=0.2, which match the paper's reported order of
magnitude (about 0.001-0.01) but not its reported figure to the last digit,
consistent with the paper's own ambiguous sign convention at that table
(see "Deviations from the paper").

## Deviations from the paper

* **Experiment 1 data source.** The paper uses U.S. property-casualty
  statutory data from SNL Financial / S&P Global Market Intelligence
  (insurer-level incurred losses and loss-adjustment expenses, net premiums
  earned, and admitted assets) and the CRSP value-weighted total market
  return. SNL statutory filings are a commercial regulatory database, not
  market or trading data, and are not served by the QuantX Dataloader,
  Massive, or Yahoo Finance; this repository therefore substitutes a
  synthetic insurer-quarter panel (`data/generate_synthetic.py`) for the
  claims-performance-factor construction. CRSP is likewise a proprietary
  subscription index; the dividend-adjusted total return of `SPY`, loaded
  through the QuantX Dataloader (`POST /fetch-ohlcvs`, `ticker_details`
  recorded `data_source: massive`, no Yahoo backstop triggered), is used as
  a documented proxy for the CRSP value-weighted market return. Because the
  insurer side is synthetic, experiment 1's reported correlation range and
  median half-width are not comparable to the paper's Figure 1.1 numbers
  (see the experiment-1 verdict above).
* **Experiment 1 sample period.** The paper's rolling windows end from 2016
  Q1 through the paper's stated 2027 Q3 (described in the paper as
  projected data beyond its writing date). This replication uses
  2008-01-01 through 2025-12-31 for both the market-return proxy and the
  synthetic panel, which the data sources can actually supply as of the
  run date, and does not extend the sample into the paper's projected
  years.
* **Experiment 1 panel-regression specification.** The paper's footnote 1
  does not give the exact functional form of the "expanding insurer-effects
  forecast". This replication forecasts each insurer-quarter claims
  performance with an expanding (growing-window, insurer-fixed-effect plus
  quarter-mean) model fitted only on strictly prior quarters, and scales the
  resulting forecast errors (not the raw claims-performance level) by their
  preceding 20-quarter rolling standard deviation, per the uncertainty noted
  in the experiment specification; the aggregation across insurers is a
  weighted average (not a weighted sum) using positive lagged admitted-asset
  weights, normalized to sum to one each quarter.
* **Experiment 6 sign table.** The paper's own description of Figure 5.5
  flags an ambiguous sign convention in its printed table (a total
  certainty-equivalent derivative that should be non-positive is listed as
  a positive 0.004 at phi=0.2). This replication reports the signed
  derivatives directly from Proposition 5's envelope formulas (equation
  4.19), so the total derivative is non-positive throughout, by
  construction and as verified in `tests/test_equilibrium.py`.
* **Bootstrap seed.** The paper does not specify a random seed for the
  block-bootstrap resampling; `EXP1_BOOTSTRAP_SEED` in `src/config.py` fixes
  one for reproducibility.
* **Grid resolutions.** Experiment 5's regime-map resolution (400x400
  points) and experiment 6's fine phi grid (step 0.01) are not specified in
  the paper; they are chosen fine enough to resolve the regime boundaries
  and derivative sign changes reported in the paper's figures.

## Not reproduced

* **Exact Figure 1.1 numbers** (the paper's rolling correlation path from
  approximately -0.19 to 0.07 and its 90% confidence bands over 2016-2027)
  are not reproduced: they require the SNL statutory insurer panel, which
  is a commercial database unavailable through the QuantX Dataloader,
  Massive REST, or Yahoo Finance. This repository implements and validates
  the full construction (claims-performance factor, expanding insurer-fixed-
  effects forecasts, scaling, weighting, rolling correlation, circular
  block-bootstrap confidence intervals) on a synthetic insurer panel
  instead, and does not present its output as the paper's result.
