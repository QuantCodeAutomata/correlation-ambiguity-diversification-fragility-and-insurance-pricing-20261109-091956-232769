# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of six experiments from Peiyun Deng and Shunzhi Pang,
"Correlation Ambiguity, Diversification Fragility, and Insurance Pricing"
(2026): the insurer's conditional financial-investment problem (Sec. 3), the
competitive insurance equilibrium under correlation ambiguity (Sec. 4), and
an empirical rolling-correlation benchmark for the ambiguity magnitude
(Sec. 1, footnote 1).

## Layout

```
src/        flat modules: config.py (all parameters), equilibrium.py
            (Sec. 3/4 closed-form model), claims_factor.py (claims-
            performance factor construction), bootstrap.py (rolling
            correlation + circular block bootstrap)
exp/        exp_1.py .. exp_6.py, one script per experiment, each calling
            only src/ and data/
data/       data_loader.py (QuantX Dataloader REST client) and
            generate_synthetic.py (synthetic insurer panel for exp_1);
            downloaded data is never committed (data/cache/ is git-ignored)
tests/      pytest tests for src/ and data/; tests/conftest.py defines the
            session-scoped market-data fixture and the market_data marker
results/    tables (.csv), figures (.png), summaries (.json) and RESULTS.md
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # DATALOADER_API_KEY / MASSIVE_API_KEY, only for exp_1

python -m pytest -m "not market_data"   # default test run, no network
python -m pytest                         # also runs the market-data test

python -m exp.exp_1   # ... python -m exp.exp_6   # run each experiment
bash reproduce.sh      # tests + every experiment, in one command
```

## Conventions

* All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE constants with a comment on meaning, unit and source. No
  classes or parameter dicts; no parameter is hard-coded elsewhere.
* Data access goes only through `data/data_loader.py` (market data via the
  QuantX Dataloader REST API) and `data/generate_synthetic.py` (synthetic
  insurer panel). No module under `src/` or `exp/` fetches or generates data
  on its own.
* `exp/*.py` scripts call the `src/` functions by their documented
  signatures and return keys; they add no private data paths or shortcuts.
* Tests: unit tests of `src/` run on synthetic fixtures with no network
  access. The one test that calls the QuantX Dataloader is marked
  `@pytest.mark.market_data` and is skipped automatically when
  `DATALOADER_API_KEY` / `MASSIVE_API_KEY` are not set. Run
  `python -m pytest -m "not market_data"` for the default, offline suite and
  the full `python -m pytest` (with keys set) before any change to the data
  path.
* API keys are read from the environment via `.env` (never committed); only
  empty placeholders live in `.env.example`.
* Running any `exp/*.py` script overwrites its output files in `results/`.
  After a rerun, the numbers and verdicts in `results/RESULTS.md` must be
  checked against the newly written files and updated by hand; they are not
  regenerated automatically.
