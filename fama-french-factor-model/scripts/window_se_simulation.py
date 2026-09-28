"""Which standard errors can be trusted inside a 36-month window?

Monte Carlo on the real FF5 factor returns in the committed snapshot (Jan 2005 to
Jul 2026). The true betas are fixed, so any 95% interval should miss them 5% of the
time and a test of "the beta never changed" should reject 5% of the time. Run:

    python scripts/window_se_simulation.py

It needs no network and takes under a minute. The numbers it prints are the ones
quoted in the README and in the docstrings of ``rolling_regression`` and ``stability_test``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ffmodel import data
from ffmodel.regression import newey_west_lags, stability_test

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "snapshot-2026-09-28"
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
BETA = np.array([0.0, 1.0, 0.8, 0.4, 0.0, 0.0])  # alpha, then FF5 (a small-value-like fund)
RESID_VOL = 0.02  # monthly, between IWN's and BRK-B's
WINDOW = 36


def residuals(rng, market, n, hetero):
    scale = RESID_VOL * (0.3 + 15 * np.abs(market)) if hetero else RESID_VOL
    return rng.normal(0, 1, n) * scale


def interval_miss_rates(factors, reps, hetero, rng, window=WINDOW):
    X_all = sm.add_constant(factors[FF5].to_numpy())
    lags = newey_west_lags(window)
    covs = {"OLS": {}, "White HC1": {"cov_type": "HC1"}, "HC3": {"cov_type": "HC3"},
            f"Newey-West, {lags} lags": {"cov_type": "HAC", "cov_kwds": {"maxlags": lags}}}
    misses = {k: [] for k in covs}
    for _ in range(reps):
        start = rng.integers(0, len(X_all) - window + 1)
        X = X_all[start:start + window]
        y = X @ BETA + residuals(rng, X[:, 1], window, hetero)
        for label, kw in covs.items():
            ci = sm.OLS(y, X).fit(**kw).conf_int(0.05)
            misses[label].append(np.mean((ci[:, 0] > BETA) | (ci[:, 1] < BETA)))
    return {k: float(np.mean(v)) for k, v in misses.items()}


def stability_rejection_rates(factors, reps, rng):
    X = factors[FF5].to_numpy()
    rates = {}
    for se in ("hac", "hc3", "ols"):
        rejections = []
        for _ in range(reps):
            y = pd.Series(X @ BETA[1:] + residuals(rng, X[:, 0], len(X), False), index=factors.index)
            table = stability_test(y, factors, "ff5", WINDOW, excess=True, se=se)
            rejections.append((table["p-value"] < 0.05).to_numpy())
        rates[se] = np.mean(rejections, axis=0)
    return pd.DataFrame(rates, index=["alpha"] + FF5).T


def main() -> None:
    warnings.simplefilter("ignore")
    factors = data.load_factors("ff5", start="2005-01", end="2026-07", data_dir=SNAPSHOT)
    rng = np.random.default_rng(20260928)
    print(f"FF5 factors {factors.index[0]:%b %Y} to {factors.index[-1]:%b %Y}, {WINDOW}-month windows\n")
    print("Share of nominal 95% intervals that miss the true coefficient (should be 0.05), 2,000 windows:")
    for hetero in (False, True):
        label = "residual vol scales with |market|" if hetero else "i.i.d. normal residuals"
        rates = interval_miss_rates(factors, 2000, hetero, rng)
        print(f"  {label:36s}", "  ".join(f"{k} {v:.3f}" for k, v in rates.items()))
    print(f"\nSame, full sample ({len(factors)} months), 2,000 samples:")
    for hetero in (False, True):
        label = "residual vol scales with |market|" if hetero else "i.i.d. normal residuals"
        rates = interval_miss_rates(factors, 2000, hetero, rng, window=len(factors))
        print(f"  {label:36s}", "  ".join(f"{k} {v:.3f}" for k, v in rates.items()))
    print("\nStability test, share of 5%-level rejections when betas never change (should be 0.05), 300 samples:")
    print(stability_rejection_rates(factors, 300, rng).round(3).to_string())


if __name__ == "__main__":
    main()
