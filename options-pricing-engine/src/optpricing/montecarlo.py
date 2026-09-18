r"""Monte Carlo pricing under geometric Brownian motion, with variance reduction.

Under the risk-neutral measure :math:`S_T = S_0\exp\!\big((r-q-\tfrac12\sigma^2)\tau +
\sigma\sqrt{\tau}Z\big)` with :math:`Z\sim N(0,1)`, so a European payoff can be priced by
sampling :math:`Z` directly -- no time stepping and therefore no discretisation bias. The
only error is statistical, and the standard error falls as :math:`O(n^{-1/2})`.

Two variance-reduction techniques are implemented.

**Antithetic variates.** Price each draw twice, at :math:`Z` and :math:`-Z`, and average
the pair. Because a vanilla payoff is monotone in :math:`Z`, the two legs are negatively
correlated and the variance of the pair mean is
:math:`\tfrac12(\operatorname{Var}(f(Z)) + \operatorname{Cov}(f(Z), f(-Z)))`, strictly
below the :math:`\tfrac12\operatorname{Var}` of two independent draws.

**Control variates.** Given a correlated variable :math:`X` with known mean
:math:`\mathbb{E}[X]`, price with :math:`\hat\theta = \bar{Y} - \beta(\bar{X} -
\mathbb{E}[X])`. The variance-minimising coefficient is
:math:`\beta^\* = \operatorname{Cov}(Y, X)/\operatorname{Var}(X)`, giving a variance
reduction of :math:`1 - \rho^2_{XY}`. Two controls are offered:

* ``"underlying"`` -- use :math:`S_T` itself, whose mean is :math:`S_0e^{(r-q)\tau}`.
  Cheap, and for deep-ITM options :math:`\rho\to1`.
* the **geometric Asian** price, used as the control for arithmetic Asian options, where
  :math:`\rho > 0.99` is typical and the variance drops by two orders of magnitude.

Estimating :math:`\beta` from the same sample introduces an :math:`O(1/n)` bias. It is
negligible at the path counts used here and is the standard practice; ``beta`` can be
supplied explicitly to remove it entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .types import FloatArray, OptionType, to_option_type

__all__ = [
    "MCResult",
    "asian_price",
    "european_price",
    "geometric_asian_price",
    "simulate_gbm_paths",
    "simulate_terminal_gbm",
]


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo price together with its sampling error.

    Attributes:
        price: Discounted sample mean of the payoff.
        std_error: Standard error of ``price``.
        n_paths: Number of *effective* samples (antithetic pairs count as one sample,
            because the two legs are not independent).
        beta: Control-variate coefficient actually used, or ``None``.
        control_corr: Sample correlation between payoff and control, or ``None``.
    """

    price: float
    std_error: float
    n_paths: int
    beta: float | None = None
    control_corr: float | None = None

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Two-sided normal confidence interval at the given level."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        z = float(norm.ppf(0.5 * (1.0 + level)))
        return self.price - z * self.std_error, self.price + z * self.std_error

    def __repr__(self) -> str:
        lo, hi = self.confidence_interval()
        return (
            f"MCResult(price={self.price:.6f}, se={self.std_error:.2e}, "
            f"95% CI=[{lo:.6f}, {hi:.6f}])"
        )


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def simulate_terminal_gbm(
    spot: float,
    tau: float,
    rate: float,
    sigma: float,
    n_paths: int,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    seed: int | np.random.Generator | None = None,
) -> FloatArray:
    r"""Exact one-shot draws of :math:`S_\tau` under GBM.

    Args:
        n_paths: Number of draws. With ``antithetic=True`` this is rounded **up** to the
            next even number and returned as ``(n_paths//2, 2)`` pairs flattened to 1-D,
            so ``result[:m]`` are the ``Z`` legs and ``result[m:]`` the ``-Z`` legs.

    Returns:
        1-D array of terminal spot values.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    gen = _rng(seed)
    drift = (rate - dividend_yield - 0.5 * sigma * sigma) * tau
    diffusion = sigma * np.sqrt(max(tau, 0.0))

    if antithetic:
        half = (n_paths + 1) // 2
        z_half = gen.standard_normal(half)
        z = np.concatenate([z_half, -z_half])
    else:
        z = gen.standard_normal(n_paths)
    return np.asarray(spot * np.exp(drift + diffusion * z), dtype=np.float64)


def simulate_gbm_paths(
    spot: float,
    tau: float,
    rate: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    seed: int | np.random.Generator | None = None,
) -> FloatArray:
    r"""Simulate whole GBM paths on a uniform grid.

    Uses the exact lognormal transition over each step, so there is no discretisation
    bias even for coarse grids -- the grid only controls which fixing dates exist.

    Returns:
        Array of shape ``(n_paths, n_steps + 1)`` including the initial spot in column 0.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    gen = _rng(seed)
    dt = tau / n_steps
    drift = (rate - dividend_yield - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        half = (n_paths + 1) // 2
        z_half = gen.standard_normal((half, n_steps))
        z = np.concatenate([z_half, -z_half], axis=0)
    else:
        z = gen.standard_normal((n_paths, n_steps))

    log_increments = drift + diffusion * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((z.shape[0], n_steps + 1), dtype=np.float64)
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_paths)
    return paths


def _summarise(
    payoff: FloatArray,
    discount: float,
    antithetic: bool,
    control: FloatArray | None = None,
    control_mean: float | None = None,
    beta: float | None = None,
) -> MCResult:
    """Fold antithetic pairs, apply the control variate, and compute the standard error.

    Order matters: the antithetic legs are folded into one sample per pair **first**, and
    the control coefficient is then fitted on those folded samples. Fitting on the raw
    legs would minimise the wrong variance, because the estimator we actually report is
    the mean over pairs, not the mean over legs.
    """
    if antithetic:
        half = payoff.size // 2
        # The (Z, -Z) legs are dependent; treating them as 2n independent draws would
        # understate the standard error by up to sqrt(2).
        payoff_s = 0.5 * (payoff[:half] + payoff[half : 2 * half])
        control_s = (
            0.5 * (control[:half] + control[half : 2 * half]) if control is not None else None
        )
    else:
        payoff_s = payoff
        control_s = control

    corr: float | None = None
    used_beta: float | None = None
    adjusted = payoff_s

    if control_s is not None:
        if control_mean is None:  # pragma: no cover - programming error
            raise ValueError("control_mean is required when a control is supplied")
        control_var = float(np.var(control_s, ddof=1))
        if beta is None:
            cov = float(np.cov(payoff_s, control_s, ddof=1)[0, 1])
            used_beta = cov / control_var if control_var > 0.0 else 0.0
        else:
            used_beta = float(beta)
        if control_var > 0.0 and float(np.var(payoff_s, ddof=1)) > 0.0:
            corr = float(np.corrcoef(payoff_s, control_s)[0, 1])
        adjusted = payoff_s - used_beta * (control_s - control_mean)

    n = int(adjusted.size)
    mean = float(np.mean(adjusted))
    std = float(np.std(adjusted, ddof=1)) if n > 1 else 0.0
    return MCResult(
        price=discount * mean,
        std_error=discount * std / np.sqrt(n),
        n_paths=n,
        beta=used_beta,
        control_corr=corr,
    )


def european_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    control_variate: bool = True,
    beta: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> MCResult:
    """Monte Carlo price of a European vanilla.

    Args:
        n_paths: Number of normal draws (halved into antithetic pairs if enabled).
        antithetic: Pair each draw with its negation.
        control_variate: Use :math:`S_\\tau` as a control with known mean
            :math:`S_0e^{(r-q)\\tau}`.
        beta: Fix the control coefficient instead of estimating it from the sample.
        seed: Seed or ``Generator`` for reproducibility.

    Returns:
        An :class:`MCResult`.
    """
    opt = to_option_type(option_type)
    phi = opt.sign
    discount = float(np.exp(-rate * tau))
    terminal = simulate_terminal_gbm(
        spot, tau, rate, sigma, n_paths, dividend_yield, antithetic, seed
    )
    payoff = np.maximum(phi * (terminal - strike), 0.0)

    control: FloatArray | None = None
    control_mean: float | None = None
    if control_variate:
        control = terminal
        control_mean = float(spot * np.exp((rate - dividend_yield) * tau))
    return _summarise(payoff, discount, antithetic, control, control_mean, beta)


def geometric_asian_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    n_fixings: int,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> float:
    r"""Closed-form price of a **discretely monitored geometric-average** Asian option.

    The geometric average :math:`G = \big(\prod_{i=1}^{m} S_{t_i}\big)^{1/m}` over equally
    spaced fixings :math:`t_i = i\tau/m` is lognormal with

    .. math::
        \mu = \ln S_0 + (r - q - \tfrac12\sigma^2)\bar{t},\quad
        \bar{t} = \frac{1}{m}\sum_i t_i = \frac{\tau(m+1)}{2m},

    .. math::
        \sigma_G^2 = \frac{\sigma^2}{m^2}\sum_{i}\sum_{j}\min(t_i, t_j)
                   = \frac{\sigma^2\tau(m+1)(2m+1)}{6m^2},

    so a Black-Scholes-style formula applies. This has no practical use on its own -- it
    exists to be the control variate for the arithmetic Asian Monte Carlo, where it
    removes almost all of the variance.
    """
    if n_fixings < 1:
        raise ValueError("n_fixings must be >= 1")
    opt = to_option_type(option_type)
    phi = opt.sign
    m = float(n_fixings)
    if tau <= 0.0 or sigma <= 0.0:
        return float(max(phi * (spot - strike), 0.0) * np.exp(-rate * max(tau, 0.0)))

    t_bar = tau * (m + 1.0) / (2.0 * m)
    var_g = sigma * sigma * tau * (m + 1.0) * (2.0 * m + 1.0) / (6.0 * m * m)
    sd_g = float(np.sqrt(var_g))
    mu = np.log(spot) + (rate - dividend_yield - 0.5 * sigma * sigma) * t_bar

    d1 = (mu - np.log(strike) + var_g) / sd_g
    d2 = d1 - sd_g
    forward_g = float(np.exp(mu + 0.5 * var_g))
    value = phi * (forward_g * norm.cdf(phi * d1) - strike * norm.cdf(phi * d2))
    return float(np.exp(-rate * tau) * max(value, 0.0))


def asian_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    n_fixings: int = 12,
    n_paths: int = 100_000,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | np.random.Generator | None = None,
) -> MCResult:
    """Monte Carlo price of a discretely monitored **arithmetic**-average Asian option.

    With ``control_variate=True`` the geometric-average payoff -- which has a closed form
    and correlates with the arithmetic payoff at better than 0.99 -- is used as the
    control. This is the textbook demonstration of how much a well-chosen control buys
    you: typically a 30-100x reduction in standard error at the same path count.
    """
    opt = to_option_type(option_type)
    phi = opt.sign
    discount = float(np.exp(-rate * tau))
    paths = simulate_gbm_paths(
        spot, tau, rate, sigma, n_paths, n_fixings, dividend_yield, antithetic, seed
    )
    fixings = paths[:, 1:]  # exclude t=0 from the average
    arithmetic_avg = fixings.mean(axis=1)
    payoff = np.maximum(phi * (arithmetic_avg - strike), 0.0)

    control: FloatArray | None = None
    control_mean: float | None = None
    if control_variate:
        geometric_avg = np.exp(np.log(fixings).mean(axis=1))
        control = np.maximum(phi * (geometric_avg - strike), 0.0)
        # The control's known mean is the *undiscounted* expected geometric payoff.
        control_mean = (
            geometric_asian_price(
                spot, strike, tau, rate, sigma, n_fixings, opt, dividend_yield
            )
            / discount
        )
    return _summarise(payoff, discount, antithetic, control, control_mean)
