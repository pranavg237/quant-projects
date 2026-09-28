# Options Pricing & Greeks Toolkit

> **Earlier, smaller project.** [options-pricing-engine](../options-pricing-engine) does
> everything here and more (lattices, Monte Carlo, Heston, a real volatility surface),
> with far more testing. This folder is kept as a compact, dependency-light
> Black-Scholes calculator.

A from-scratch Black-Scholes implementation: pricer, full Greeks, an implied
vol solver, and a theta/gamma decay visualizer aimed at the 0-2 DTE zone.
No external pricing library (no `py_vollib`, no `mibian`). Every formula is
written out.

```bash
pip install -r requirements.txt
python cli.py price --S 230 --K 235 --dte 2 --r 0.045 --sigma 0.55 --type call
python -m pytest     # 14 tests
```

This is a calculator, not a trading strategy, so there is no Sharpe ratio,
drawdown or turnover to report.

## Why this exists

Directly supports the short-dated (0-2 DTE) TSLA/NVDA/AAPL premium-capture
approach: knowing the price is not enough at 0-2 DTE, since theta and gamma
both blow up in the final days and a "favorable" fill can be caused by IV
crush rather than the underlying moving your way. `theta_decay.py` makes that
visible instead of theoretical.

## Files

- `black_scholes.py` - `price()`, `greeks()` (delta/gamma/vega/theta/rho), `implied_vol()`
- `cli.py` - command-line pricer / IV solver
- `theta_decay.py` - plots price, theta, and gamma vs. days-to-expiry, with the
  0-2 DTE zone shaded, plus a printed table for the 2/1/0.25 DTE marks
- `tests/test_black_scholes.py` - 14 unit tests: textbook benchmark values,
  put-call parity, finite-difference checks against the analytic Greeks,
  implied-vol round-trips, and the no-arbitrage bounds

## A bug worth knowing about (fixed)

The implied-vol solver used to reject prices below *undiscounted* intrinsic
value (K - S for a put). That is wrong for European options: a deep
in-the-money European put is worth less than K - S, because the strike is
received at expiry, not today. With S=50, K=100, r=10%, T=1y and 20% vol the
put is worth 40.49, and the solver refused its own model price. It now checks
the correct bounds, `max(0, K e^-rT - S e^-qT) <= P <= K e^-rT` (and the call
equivalent), exposed as `no_arbitrage_bounds()`.

## Usage

```bash
# price + Greeks
python3 cli.py price --S 230 --K 235 --dte 2 --r 0.045 --sigma 0.55 --type call

# solve implied vol from an observed market price
python3 cli.py iv --S 230 --K 235 --dte 2 --r 0.045 --type call --price 3.10

# plot decay into 0-2 DTE (saves theta_decay.png)
python3 theta_decay.py --S 230 --K 235 --sigma 0.55 --type call

# run the tests
python -m pytest
```

## Conventions

- `dte` (days to expiry) can be fractional - e.g. `0.5` for "expires later
  today"
- `sigma` and `r` are annualized (e.g. `0.55` = 55% IV, `0.045` = 4.5% risk-free)
- `theta` is reported **per calendar day** (not per year) and `vega`/`rho`
  are reported **per 1 vol point / 1% rate move**, matching how brokers
  display them - the raw Black-Scholes partial derivatives are annualized
  and get rescaled internally

## Known simplifications (roadmap)

- European-style only (no early exercise / American pricing via binomial or
  finite-difference trees - most single-stock options are American, so this
  is an approximation, tightest for calls on non-dividend-paying names)
- No dividend discreteness (only a continuous yield `q`)
- Flat volatility input - no vol surface / skew modeling yet. A natural next
  step: pull an options chain and back out the smile via `implied_vol()`
  applied strike-by-strike
