#!/usr/bin/env bash
# Runs the offline test suite and every experiment of Deng and Pang,
# "Correlation Ambiguity, Diversification Fragility, and Insurance Pricing".
set -e

python -m pytest -m "not market_data"

python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
python -m exp.exp_6
