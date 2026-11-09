"""Model and experiment parameters for Deng and Pang,
"Correlation Ambiguity, Diversification Fragility, and Insurance Pricing"
(2026). Every value below is a module-level constant, grouped by experiment,
with a comment on its meaning, unit, and source (the paper's section,
equation or table, or "not specified in the paper").

No classes or parameter dicts are used here; downstream code imports these
names directly.
"""

# ---------------------------------------------------------------------------
# Experiment 1: rolling underwriting-market dependence (Figure 1.1, footnote 1)
# ---------------------------------------------------------------------------

# Rolling correlation window, in quarters (paper, Sec. 1 and footnote 1).
EXP1_ROLLING_WINDOW_Q = 32

# Trailing window used to scale insurer forecast errors, in quarters
# (footnote 1: "forecast errors are scaled by their preceding 20-quarter
# standard deviation").
EXP1_SCALING_WINDOW_Q = 20

# Circular block-bootstrap block size, in quarters (footnote 1: "paired
# circular four-quarter block-bootstrap samples").
EXP1_BOOTSTRAP_BLOCK_Q = 4

# Number of bootstrap draws (footnote 1: "2,000 draws").
EXP1_BOOTSTRAP_DRAWS = 2000

# Nominal confidence-interval endpoints (footnote 1: "fifth/ninety-fifth
# percentile endpoints"), i.e. a nominal 90% interval.
EXP1_CI_LOWER_PCTL = 5.0
EXP1_CI_UPPER_PCTL = 95.0

# Bootstrap random seed (not specified in the paper; fixed here for
# reproducibility of the resampling distribution).
EXP1_BOOTSTRAP_SEED = 20261109

# Sample period for the market-return proxy and the synthetic insurer panel
# used in place of the unavailable SNL statutory data (paper, Sec. 1: data
# "span at least from approximately 2008 through 2025" to produce rolling
# 32-quarter windows ending from 2016 Q1 onward; the paper's own Figure 1.1
# table runs through 2026 using data described in the paper as projected).
EXP1_PANEL_START = "2008-01-01"
EXP1_PANEL_END = "2025-12-31"

# Market-return proxy ticker: the paper uses the CRSP value-weighted total
# market return, a proprietary CRSP series. Not specified in the paper beyond
# its definition; the ticker below is a documented substitution (broad
# U.S.-market total-return ETF) recorded in results/RESULTS.md.
EXP1_MARKET_PROXY_TICKER = "SPY"

# Synthetic insurer panel (used only because SNL statutory data cannot be
# obtained from the market-data sources used in this repository; not a
# paper value).
EXP1_SYN_N_INSURERS = 60  # cross-sectional size, not specified in the paper
EXP1_SYN_SEED = 20261109


# ---------------------------------------------------------------------------
# Experiments 2-6: numerical benchmark (Sec. 5.1) and household-aversion
# variant (Sec. 5.3), shared across the equilibrium experiments.
# ---------------------------------------------------------------------------

# Benchmark primitives (paper, Sec. 5.1).
BENCH_L = 2.0        # household risk-aversion scale in the loading domain [0, alpha*eta^2/l]
BENCH_ETA = 0.28      # underwriting-risk volatility, per unit underwriting share
BENCH_R = 0.015       # risk-free rate, annual
BENCH_MU = 0.035      # risky-asset expected return, annual
BENCH_SIGMA = 0.18    # risky-asset return volatility, annual
BENCH_ALPHA = 2.0     # household risk-aversion coefficient
BENCH_GAMMA = 2.0     # insurer CARA risk-aversion coefficient
BENCH_T = 5.0         # horizon, years
BENCH_T0 = 0.0        # evaluation date t
BENCH_RHO0 = 0.0      # correlation midpoint rho_0 (exp_2, exp_3, exp_4, exp_6)

# Reduced household risk aversion used in exp_4 (paper, Sec. 5.3).
LOW_ALPHA = 0.2

# Empirical reference ambiguity half-width, from the median 90% confidence
# interval half-width computed in exp_1 (paper, Sec. 5.1: "approximately
# 0.23"; Figure 5.1's table marks the reference near phi = 0.25).
EMPIRICAL_PHI_REFERENCE = 0.23

# exp_2: fixed underwriting shares (paper, Figure 5.1 description).
EXP2_UNDERWRITING_SHARES = (0.25, 0.5, 0.75)
# exp_2/exp_3: ambiguity half-width grid, step 0.1 from 0 to 1 (paper, Figure 5.1/5.2).
PHI_GRID_STEP = 0.1
PHI_GRID_MIN = 0.0
PHI_GRID_MAX = 1.0

# exp_4: half-width grid matching the paper's Figure 5.3 table points.
EXP4_PHI_GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.6)

# exp_5: resolution of the (rho_0, phi) regime map over the admissible
# triangular domain -1 < rho_0 - phi <= rho_0 + phi < 1 (not specified in the
# paper; chosen fine enough to resolve the regime boundaries visually).
EXP5_GRID_POINTS = 400

# exp_6: fine phi grid for the marginal-cost and incidence curves (paper,
# Figure 5.5 panel points are at 0, 0.2, 0.4, 0.6, 0.8, 1.0; a finer grid is
# used for the plotted curves, step not specified in the paper).
EXP6_PHI_GRID_STEP = 0.01
EXP6_TABLE_PHI_POINTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
