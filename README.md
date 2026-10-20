#Educational use only. Not investment advice. I am not a registered adviser. Past performance does not predict future results.”

# Quant Toolkit

A command-line toolkit for price generation, strategy backtests
(moving-average crossover, mean reversion, pairs trading), parameter
screening and bootstrap scenario simulation.

Educational use only. Not investment advice. Synthetic data says nothing
about real markets, and past performance does not predict future results.

## Setup

pip install -r requirements.txt

## Usage

python3 quant_toolkit.py pipeline --model gbm --points 5000
python3 quant_toolkit.py pipeline --csv prices.csv --column Close
python3 quant_toolkit.py backtest --strategy ma --model ou
python3 quant_toolkit.py sample --distribution student-t
python3 quant_toolkit.py --help
