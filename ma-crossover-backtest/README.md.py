# Moving Average Crossover Backtest

A simple dual moving average crossover strategy backtested on SPY (2018–2024).

## Strategy Logic
- Buy when the 50-day MA crosses above the 200-day MA
- Sell when the 50-day MA crosses below the 200-day MA
- 1-day lag applied to avoid lookahead bias

## Results
| Parameters | Strategy Return | Buy & Hold |
|---|---|---|
| 50/200 | 48.24% | 95.54% |
| 20/50 | 36.93% | 95.54% |

## Why It Underperforms
The MA crossover is a lagging indicator — it signals after the move happens.
On a long-term uptrending asset like SPY, being out of the market during
recoveries hurts returns significantly. Shorter windows trade more frequently
and suffer worse whipsaw effects.

## Libraries
- yfinance, pandas, matplotlib