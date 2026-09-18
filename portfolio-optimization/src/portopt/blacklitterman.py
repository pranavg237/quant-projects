r"""The Black-Litterman model.

Mean-variance optimisation with historical sample means produces portfolios nobody would
hold. The reason is not the optimiser, it is the input: a 20-year sample mean of monthly
equity returns has a standard error of roughly 4% a year, so the *differences* between
assets' estimated returns are mostly noise, and the optimiser responds to those differences
with enormous offsetting positions.

Black and Litterman (1992) fix the input rather than patching the output. Two ideas.

**Start from equilibrium, not from history.** If the market portfolio is what everyone
holds, then reverse-optimise to find the expected returns that would make it optimal:

.. math:: \Pi = \delta \Sigma w_{\text{mkt}}, \qquad
    \delta = \frac{\mathbb{E}[r_m] - r_f}{\sigma_m^2}.

:math:`\Pi` is not estimated from returns at all. It is a statement about what the market
must believe, and it is enormously more stable than a sample mean -- it inherits only the
covariance matrix's error, not the mean's.

**Blend in views as a Bayesian update.** A view is a linear statement :math:`P\mu = Q` with
uncertainty :math:`\Omega`. Treating the equilibrium as a prior
:math:`\mu \sim N(\Pi, \tau\Sigma)` and the views as a noisy observation gives the posterior

.. math::
    \mathbb{E}[\mu] = \big[(\tau\Sigma)^{-1} + P'\Omega^{-1}P\big]^{-1}
        \big[(\tau\Sigma)^{-1}\Pi + P'\Omega^{-1}Q\big],

with posterior covariance of the mean :math:`M = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}`
and total return covariance :math:`\Sigma + M`.

Two properties make it well behaved and are both checked in the tests: with **no views**
the posterior is exactly the prior, and with **infinitely confident views** the posterior
satisfies them exactly. In between, an asset with no view expressed about it moves too --
through its correlation with the assets that do -- which is the behaviour that makes the
resulting portfolios sensible instead of concentrated.

On :math:`\tau`: it scales the uncertainty of the *prior mean* relative to the uncertainty
of returns. Values from 0.01 to 1 appear in the literature. Only the ratio
:math:`\tau\Sigma / \Omega` matters, so if :math:`\Omega` is set proportionally to
:math:`\tau` (the He-Litterman convention used here by default) the answer is invariant to
:math:`\tau` -- which is the sane way to sidestep an otherwise arbitrary parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import FloatArray

__all__ = [
    "BlackLittermanResult",
    "View",
    "black_litterman",
    "implied_equilibrium_returns",
    "market_cap_weights",
    "views_to_matrices",
]


@dataclass(frozen=True)
class View:
    """One linear view on expected returns.

    Attributes:
        assets: Asset weights in the view. ``{"XLK": 1.0}`` is an absolute view on
            technology; ``{"XLK": 1.0, "XLP": -1.0}`` is a relative view that technology
            outperforms staples.
        value: The expected return the view asserts, annualised. For a relative view this
            is the expected *spread*.
        confidence: Relative confidence in ``(0, 1]``. 1.0 means the view is held as firmly
            as the equilibrium prior; smaller values pull the posterior back towards
            equilibrium. Mapped to ``Omega`` by the He-Litterman convention.
    """

    assets: dict[str, float]
    value: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("a view must reference at least one asset")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")

    @property
    def is_relative(self) -> bool:
        """Whether the view's weights sum to (approximately) zero."""
        return abs(sum(self.assets.values())) < 1e-12

    def __str__(self) -> str:
        parts = " ".join(f"{v:+g}*{k}" for k, v in self.assets.items())
        kind = "relative" if self.is_relative else "absolute"
        return f"{parts} = {self.value:.2%} ({kind}, confidence {self.confidence:.0%})"


def market_cap_weights(caps: pd.Series) -> pd.Series:
    """Normalise market capitalisations to weights."""
    if (caps <= 0).any():
        raise ValueError("market capitalisations must be positive")
    return caps / caps.sum()


def implied_equilibrium_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float | None = None,
    market_excess_return: float | None = None,
) -> pd.Series:
    r"""Reverse-optimise the market portfolio: :math:`\Pi = \delta\Sigma w_{\text{mkt}}`.

    Args:
        covariance: Asset covariance matrix.
        market_weights: Market-capitalisation weights, aligned to ``covariance``.
        risk_aversion: :math:`\delta`. If ``None`` it is derived from
            ``market_excess_return`` as :math:`\delta = (\mathbb{E}[r_m]-r_f)/\sigma_m^2`,
            which is the only defensible way to set it: it makes the implied market Sharpe
            ratio equal to whatever you actually believe it is.
        market_excess_return: The market's expected excess return, used when
            ``risk_aversion`` is ``None``. Defaults to 5% a year.

    Returns:
        Equilibrium expected **excess** returns.
    """
    weights = market_weights.reindex(covariance.index)
    if weights.isna().any():
        raise ValueError("market weights must cover every asset in the covariance matrix")
    w = weights.to_numpy(dtype=np.float64)
    sigma = covariance.to_numpy(dtype=np.float64)

    if risk_aversion is None:
        market_variance = float(w @ sigma @ w)
        if market_variance <= 0:
            raise ValueError("the market portfolio has zero variance")
        excess = 0.05 if market_excess_return is None else market_excess_return
        risk_aversion = excess / market_variance
    return pd.Series(risk_aversion * (sigma @ w), index=covariance.index, name="equilibrium")


def views_to_matrices(
    views: list[View], assets: pd.Index
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Turn a list of :class:`View` into the ``(P, Q, confidence)`` matrices.

    Raises:
        ValueError: if a view references an asset not in the universe.
    """
    lookup = {name: i for i, name in enumerate(assets)}
    pick = np.zeros((len(views), len(assets)), dtype=np.float64)
    q = np.zeros(len(views), dtype=np.float64)
    confidence = np.zeros(len(views), dtype=np.float64)
    for row, view in enumerate(views):
        for name, weight in view.assets.items():
            if name not in lookup:
                raise ValueError(f"view references unknown asset {name!r}")
            pick[row, lookup[name]] = weight
        q[row] = view.value
        confidence[row] = view.confidence
    return pick, q, confidence


@dataclass
class BlackLittermanResult:
    """Posterior of a Black-Litterman update.

    Attributes:
        posterior_returns: Blended expected excess returns.
        posterior_covariance: :math:`\\Sigma + M`, the covariance of returns including
            uncertainty about the mean. Using plain :math:`\\Sigma` instead is a common
            simplification and slightly understates risk.
        prior_returns: The equilibrium returns :math:`\\Pi` the update started from.
        tilt: ``posterior - prior``. This is the useful diagnostic: it shows which assets
            the views actually moved, including ones no view mentioned.
        risk_aversion: The :math:`\\delta` used.
    """

    posterior_returns: pd.Series
    posterior_covariance: pd.DataFrame
    prior_returns: pd.Series
    tilt: pd.Series
    risk_aversion: float

    def __str__(self) -> str:
        biggest = str(self.tilt.abs().idxmax())
        return (
            f"Black-Litterman: delta={self.risk_aversion:.2f}, "
            f"largest tilt {biggest} {float(self.tilt.loc[biggest]):+.2%}"
        )


def black_litterman(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    views: list[View] | None = None,
    tau: float = 0.05,
    risk_aversion: float | None = None,
    market_excess_return: float = 0.05,
) -> BlackLittermanResult:
    r"""Blend equilibrium returns with views.

    Args:
        covariance: Asset covariance matrix (annualised).
        market_weights: Market-capitalisation weights.
        views: Views to impose. ``None`` or empty returns the prior exactly.
        tau: Scale of prior-mean uncertainty. With the default He-Litterman
            :math:`\Omega`, the result is invariant to ``tau``.
        risk_aversion: :math:`\delta`, or ``None`` to derive it from
            ``market_excess_return``.
        market_excess_return: Used when ``risk_aversion`` is ``None``.

    Returns:
        A :class:`BlackLittermanResult`.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    prior = implied_equilibrium_returns(
        covariance, market_weights, risk_aversion, market_excess_return
    )
    sigma = covariance.to_numpy(dtype=np.float64)
    w = market_weights.reindex(covariance.index).to_numpy(dtype=np.float64)
    delta = (
        risk_aversion if risk_aversion is not None else market_excess_return / float(w @ sigma @ w)
    )

    if not views:
        return BlackLittermanResult(
            posterior_returns=prior.copy(),
            posterior_covariance=covariance.copy(),
            prior_returns=prior,
            tilt=pd.Series(0.0, index=covariance.index, name="tilt"),
            risk_aversion=delta,
        )

    pick, q, confidence = views_to_matrices(views, covariance.index)
    tau_sigma = tau * sigma
    # He-Litterman: Omega = diag(P tau Sigma P') / confidence. Proportional to tau, so the
    # posterior does not depend on tau -- which removes an otherwise arbitrary parameter.
    base = np.diag(pick @ tau_sigma @ pick.T)
    omega = np.diag(np.maximum(base / confidence, 1e-16))

    prior_precision = np.linalg.inv(tau_sigma)
    view_precision = pick.T @ np.linalg.inv(omega) @ pick
    posterior_mean_covariance = np.linalg.inv(prior_precision + view_precision)
    posterior = posterior_mean_covariance @ (
        prior_precision @ prior.to_numpy(dtype=np.float64) + pick.T @ np.linalg.inv(omega) @ q
    )

    posterior_series = pd.Series(posterior, index=covariance.index, name="posterior")
    return BlackLittermanResult(
        posterior_returns=posterior_series,
        posterior_covariance=pd.DataFrame(
            sigma + posterior_mean_covariance, index=covariance.index, columns=covariance.columns
        ),
        prior_returns=prior,
        tilt=(posterior_series - prior).rename("tilt"),
        risk_aversion=delta,
    )
