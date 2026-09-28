"""Decompose an asset's excess return into alpha, factor contributions and residual."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .regression import RegressionResult


@dataclass
class Attribution:
    # Per-period components: alpha, beta_k * f_k for each factor, residual.
    # Each row sums to that period's excess return.
    contributions: pd.DataFrame
    # Per component: annualized mean return, share of mean excess return, and
    # share of excess-return variance (Euler decomposition, sums to 1).
    summary: pd.DataFrame
    periods_per_year: int

    def cumulative(self) -> pd.DataFrame:
        """Running sum of each component (arithmetic, so the components add up)."""
        return self.contributions.cumsum()


def attribute_returns(result: RegressionResult, factors: pd.DataFrame) -> Attribution:
    index = result.resid.index
    parts = factors.loc[index, result.factors].mul(result.betas, axis=1)
    contributions = pd.concat(
        [pd.Series(result.alpha, index=index, name="alpha"), parts, result.resid.rename("residual")], axis=1
    )
    excess = contributions.sum(axis=1)
    systematic = parts.sum(axis=1)
    var_total = excess.var()

    ppy = result.periods_per_year
    annual = contributions.mean() * ppy
    total = excess.mean() * ppy
    var_share = pd.Series(
        {
            "alpha": 0.0,
            **{k: parts[k].cov(systematic) / var_total for k in result.factors},
            "residual": result.resid.var() / var_total,
        }
    )
    summary = pd.DataFrame(
        {"return (ann.)": annual, "share of return": annual / total, "share of variance": var_share}
    )
    summary.loc["total"] = [total, 1.0, 1.0]
    return Attribution(contributions=contributions, summary=summary, periods_per_year=ppy)
