# Correlation Ambiguity, Diversification Fragility, and Insurance Pricing

Replication of six experiments from **Peiyun Deng and Shunzhi Pang,
"Correlation Ambiguity, Diversification Fragility, and Insurance Pricing"
(2026)**.

The paper studies an insurer that underwrites correlated catastrophe risk and
invests in a financial market under Knightian ambiguity about the
correlation between underwriting losses and financial returns. Section 3
solves the insurer's conditional financial-investment problem for a fixed
underwriting share; Section 4 characterizes the competitive insurance
equilibrium (coverage, financial exposure, and the loading) as the
ambiguity interval widens; Section 1 (footnote 1) reports an empirical
rolling-correlation benchmark for the magnitude of this ambiguity.

## Repository layout

```
src/            analytic model (conditional investment, equilibrium regimes,
                envelope derivatives) and the claims-performance-factor /
                block-bootstrap construction; src/config.py holds every
                parameter
exp/            one script per experiment (exp_1 .. exp_6), each calling
                only the src/ API
data/           data/data_loader.py loads market data through the QuantX
                Dataloader REST API; data/generate_synthetic.py generates the
                synthetic insurer panel used in experiment 1 (see
                "Deviations from the paper" below)
tests/          pytest unit tests for src/ and a market-data-marked test of
                data/data_loader.py
results/        tables, figures and results/RESULTS.md (paper comparison)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

The API keys are only required to run experiment 1's market-return load
(`python -m exp.exp_1`) and the `market_data`-marked test; experiments 2-6
are purely analytic and need no external data or API keys.

## Running the tests

```bash
python -m pytest -m "not market_data"   # default: no network access required
python -m pytest                         # includes the market-data test (needs API keys)
```

## Running the experiments

```bash
python -m exp.exp_1   # rolling underwriting-market dependence (Figure 1.1)
python -m exp.exp_2   # conditional financial exposure at fixed shares (Figure 5.1)
python -m exp.exp_3   # competitive equilibrium, benchmark parameters (Figure 5.2)
python -m exp.exp_4   # competitive equilibrium, low household risk aversion (Figure 5.3)
python -m exp.exp_5   # equilibrium regime map (Figure 5.4)
python -m exp.exp_6   # marginal ambiguity cost and incidence (Figure 5.5)
```

Each script writes its tables, figures and JSON summaries directly into
`results/`, overwriting the committed copies. `reproduce.sh` runs the test
suite and every experiment in this order.

## Data

Experiments 2-6 use only the closed-form benchmark parameters of Section
5.1 (and the reduced household risk aversion of Section 5.3 for experiment
4); they require no external data.

Experiment 1 requires two series described in Section 1 and footnote 1:

* **CRSP value-weighted total market return.** CRSP is a proprietary
  subscription database not served by the QuantX Dataloader, by Massive, or
  by Yahoo Finance. This repository substitutes the dividend-adjusted total
  return of a broad U.S.-market ETF (`SPY`), loaded through the QuantX
  Dataloader (`POST /fetch-ohlcvs`), compounded from monthly to quarterly
  frequency, as a documented proxy for the CRSP index.
* **U.S. property-casualty statutory data** (insurer-level incurred losses
  and loss-adjustment expenses, net premiums earned, and admitted assets),
  published by SNL Financial / S&P Global Market Intelligence. This is a
  commercial regulatory-filings database, not market or trading data, and is
  not served by the QuantX Dataloader, Massive, or Yahoo Finance. Experiment
  1 therefore uses a synthetic insurer-quarter panel
  (`data/generate_synthetic.py`) to exercise the claims-performance-factor
  construction end to end; its output is reported as an illustration of the
  method, not as a reproduction of the paper's Figure 1.1 numbers (see
  `results/RESULTS.md`).

## Results

See `results/RESULTS.md` for the comparison of every experiment's output
against the paper's reported values, with explicit "Deviations from the
paper" and "Not reproduced" sections.
