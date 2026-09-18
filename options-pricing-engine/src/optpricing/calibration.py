r"""Calibrating Heston to a quoted implied-volatility surface.

Calibration means choosing :math:`(v_0, \kappa, \theta, \xi, \rho)` to minimise the
distance between model and market. Three choices define the problem, and all three are
consequential.

**What is the residual?** Matching *prices* weights the fit towards at-the-money options,
whose prices are the largest, and effectively ignores the wings -- which is where the
model's whole value lies. Matching *implied vols* treats every quote equally, but requires
inverting Black-Scholes inside every objective evaluation. This module uses the standard
compromise, a **vega-normalised price residual**:

.. math::
    \varepsilon_i = \frac{V^{\text{model}}_i - V^{\text{mkt}}_i}{\mathcal{V}_i}
    \;\approx\; \sigma^{\text{model}}_i - \sigma^{\text{mkt}}_i,

where :math:`\mathcal{V}_i` is the market vega. This is a first-order approximation to the
vol error, costs one quadrature per expiry instead of one root-solve per quote, and agrees
with the exact vol residual to well inside a vol point at typical surface accuracy.
:func:`calibrate` can be asked for exact vol residuals when that matters.

**Which quotes?** Every quote is additionally weighted by :math:`\sqrt{\text{vega}}` so
that near-worthless wing options -- whose implied vols are the least reliable numbers on
the surface -- cannot dominate a least-squares fit.

**Where do you start?** The Heston objective is genuinely multimodal, and a single
Levenberg-Marquardt run from a bad seed lands in a local minimum with a plausible-looking
RMSE and nonsense parameters (:math:`\kappa = 19.9`, :math:`\rho = -0.999`). :func:`calibrate`
therefore runs a small multi-start: several economically sensible seeds, each polished
with ``scipy.optimize.least_squares`` (Trust Region Reflective, which respects bounds),
and keeps the best. The spread of the local optima is reported, because a calibration that
is unstable across seeds is telling you something.

The benchmark to beat is not "no model". It is
:func:`fit_flat_vol_per_expiry` -- one Black-Scholes volatility per expiry. That model
already captures the entire term structure and gets every at-the-money option exactly
right; the only thing Heston can add is the smile. Comparing against a single global
Black-Scholes vol instead would be flattering and meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import heston
from .heston import HestonParams
from .types import FloatArray, OptionType, to_float

__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_SEEDS",
    "CalibrationResult",
    "calibrate",
    "fit_flat_vol_per_expiry",
    "fit_global_flat_vol",
    "surface_errors",
]

#: ``(lower, upper)`` bounds on ``[v0, kappa, theta, xi, rho]``.
DEFAULT_BOUNDS: tuple[tuple[float, ...], tuple[float, ...]] = (
    (1e-4, 1e-2, 1e-4, 1e-3, -0.999),
    (1.0, 20.0, 1.0, 5.0, 0.999),
)

#: Multi-start seeds, chosen to span the economically plausible region rather than to be
#: close to any particular answer: slow/fast mean reversion, low/high vol-of-vol, and
#: correlations from mildly to strongly negative (equity indices are always negative).
DEFAULT_SEEDS: tuple[HestonParams, ...] = (
    HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7),
    HestonParams(v0=0.02, kappa=3.0, theta=0.06, xi=0.8, rho=-0.5),
    HestonParams(v0=0.06, kappa=0.5, theta=0.03, xi=0.3, rho=-0.9),
    HestonParams(v0=0.03, kappa=5.0, theta=0.05, xi=1.2, rho=-0.6),
)


@dataclass
class CalibrationResult:
    """Outcome of a Heston calibration.

    Attributes:
        params: Best-fitting parameters.
        rmse_vol: Root-mean-square implied-volatility error, in vol points (0.01 = 1 pt).
        mae_vol: Mean absolute implied-volatility error, in vol points.
        max_abs_vol_error: Worst single-quote vol error.
        n_quotes: Number of quotes used.
        n_function_evals: Objective evaluations in the winning local solve.
        seed_rmses: RMSE reached from each multi-start seed, sorted ascending. A wide
            spread means the objective is multimodal and the answer is seed-dependent.
        errors: Per-quote frame with model vol, market vol and the residual.
        success: Whether the winning local solve reported convergence.
    """

    params: HestonParams
    rmse_vol: float
    mae_vol: float
    max_abs_vol_error: float
    n_quotes: int
    n_function_evals: int
    seed_rmses: list[float] = field(default_factory=list)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    success: bool = True

    def __str__(self) -> str:
        p = self.params
        spread = (
            f"{min(self.seed_rmses):.4f}-{max(self.seed_rmses):.4f}"
            if len(self.seed_rmses) > 1
            else "n/a"
        )
        return (
            f"Heston calibration on {self.n_quotes} quotes\n"
            f"  v0={p.v0:.4f}  kappa={p.kappa:.3f}  theta={p.theta:.4f}  "
            f"xi={p.xi:.3f}  rho={p.rho:+.3f}\n"
            f"  Feller 2*kappa*theta/xi^2 = {p.feller_ratio:.2f} "
            f"({'satisfied' if p.satisfies_feller else 'VIOLATED'})\n"
            f"  RMSE {self.rmse_vol * 100:.2f} vol pts | MAE {self.mae_vol * 100:.2f} | "
            f"max {self.max_abs_vol_error * 100:.2f}\n"
            f"  seed RMSE spread: {spread}"
        )


def _model_prices(
    params: HestonParams,
    spot: float,
    groups: list[tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]],
) -> FloatArray:
    """Price every quote under ``params``, one Fourier quadrature per expiry."""
    out: list[FloatArray] = []
    for tau, strikes, rates, divs, is_call in groups:
        rate = float(rates[0])
        div = float(divs[0])
        call = heston.price(spot, strikes, tau, rate, params, OptionType.CALL, div)
        put = heston.price(spot, strikes, tau, rate, params, OptionType.PUT, div)
        out.append(np.where(is_call.astype(bool), call, put))
    return np.concatenate(out) if out else np.empty(0)


def _prepare(
    surface: pd.DataFrame,
) -> tuple[list[tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]], pd.DataFrame]:
    """Group the surface by expiry so each Fourier integral is reused across strikes.

    A per-expiry ``rate``/``dividend_yield`` is carried through unchanged: those came from
    put-call parity, so calls and puts at the same strike are already consistent.
    """
    ordered = surface.sort_values(["tau", "strike"]).reset_index(drop=True)
    groups: list[tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]] = []
    for tau, g in ordered.groupby("tau", sort=True):
        groups.append(
            (
                to_float(tau),
                np.asarray(g["strike"].to_numpy(), dtype=np.float64),
                np.asarray(g["rate"].to_numpy(), dtype=np.float64),
                np.asarray(g["dividend_yield"].to_numpy(), dtype=np.float64),
                np.asarray((g["option_type"] == "call").to_numpy(), dtype=np.float64),
            )
        )
    return groups, ordered


def surface_errors(
    params: HestonParams, surface: pd.DataFrame, spot: float, exact_vols: bool = True
) -> pd.DataFrame:
    """Per-quote model-vs-market comparison for a given parameter set.

    Args:
        params: Heston parameters.
        surface: Output of :func:`optpricing.surface.build_surface`.
        spot: Underlying price.
        exact_vols: Invert Black-Scholes on the model price to get an exact model implied
            vol. With ``False`` the vega approximation is used, which is what the
            objective sees.

    Returns:
        Frame with ``tau, strike, log_moneyness, option_type, market_vol, model_vol,
        vol_error, market_price, model_price, price_error``.
    """
    from . import implied_vol as iv_mod  # noqa: PLC0415  (avoids an import cycle)

    groups, ordered = _prepare(surface)
    model_px = _model_prices(params, spot, groups)

    out = ordered[
        [
            "tau",
            "strike",
            "option_type",
            "log_moneyness",
            "implied_vol",
            "mid",
            "vega",
            "rate",
            "dividend_yield",
        ]
    ].copy()
    out = out.rename(columns={"implied_vol": "market_vol", "mid": "market_price"})
    out["model_price"] = model_px
    out["price_error"] = out["model_price"] - out["market_price"]

    if exact_vols:
        model_vol = np.empty(len(out), dtype=np.float64)
        for side in ("call", "put"):
            mask = (out["option_type"] == side).to_numpy()
            if not mask.any():
                continue
            model_vol[mask] = np.asarray(
                iv_mod.implied_vol(
                    out.loc[mask, "model_price"].to_numpy(dtype=np.float64),
                    spot,
                    out.loc[mask, "strike"].to_numpy(dtype=np.float64),
                    out.loc[mask, "tau"].to_numpy(dtype=np.float64),
                    out.loc[mask, "rate"].to_numpy(dtype=np.float64),
                    OptionType(side),
                    out.loc[mask, "dividend_yield"].to_numpy(dtype=np.float64),
                )
            )
        out["model_vol"] = model_vol
    else:
        out["model_vol"] = out["market_vol"] + out["price_error"] / out["vega"]

    out["vol_error"] = out["model_vol"] - out["market_vol"]
    return out


def calibrate(
    surface: pd.DataFrame,
    spot: float,
    seeds: tuple[HestonParams, ...] = DEFAULT_SEEDS,
    bounds: tuple[tuple[float, ...], tuple[float, ...]] = DEFAULT_BOUNDS,
    max_nfev: int = 400,
    weight_by_vega: bool = True,
    verbose: bool = False,
) -> CalibrationResult:
    """Calibrate Heston to a quoted surface by multi-start least squares.

    Args:
        surface: Output of :func:`optpricing.surface.build_surface`.
        spot: Underlying price.
        seeds: Starting points for the multi-start.
        bounds: ``(lower, upper)`` on ``[v0, kappa, theta, xi, rho]``.
        max_nfev: Objective-evaluation cap per local solve.
        weight_by_vega: Weight residuals by ``sqrt(vega)`` so illiquid wing quotes, whose
            implied vols are the least trustworthy, cannot dominate the fit.
        verbose: Print each seed's result.

    Returns:
        A :class:`CalibrationResult`.

    Raises:
        ValueError: if the surface is empty or every local solve fails.
    """
    if surface.empty:
        raise ValueError("cannot calibrate to an empty surface")

    groups, ordered = _prepare(surface)
    market_px = ordered["mid"].to_numpy(dtype=np.float64)
    vega = np.maximum(ordered["vega"].to_numpy(dtype=np.float64), 1e-8)
    weights = np.sqrt(vega) if weight_by_vega else np.ones_like(vega)
    weights = weights / np.sqrt(np.mean(weights**2))  # keep the residual scale interpretable

    def residuals(x: FloatArray) -> FloatArray:
        try:
            params = HestonParams.from_array(x)
        except ValueError:  # pragma: no cover - bounds keep us in the valid region
            return np.full(market_px.size, 1e3)
        model = _model_prices(params, spot, groups)
        return np.asarray(weights * (model - market_px) / vega, dtype=np.float64)

    best: CalibrationResult | None = None
    seed_rmses: list[float] = []

    for seed in seeds:
        x0 = np.clip(seed.to_array(), bounds[0], bounds[1])
        solution = least_squares(
            residuals,
            x0,
            bounds=bounds,
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
            max_nfev=max_nfev,
        )
        params = HestonParams.from_array(solution.x)
        errors = surface_errors(params, surface, spot, exact_vols=True)
        valid = errors["vol_error"].notna()
        vol_err = errors.loc[valid, "vol_error"].to_numpy(dtype=np.float64)
        rmse = float(np.sqrt(np.mean(vol_err**2))) if vol_err.size else np.inf
        seed_rmses.append(rmse)
        if verbose:  # pragma: no cover - diagnostic output
            print(f"  seed {seed.to_array().round(3)} -> RMSE {rmse * 100:.3f} vol pts")
        if best is None or rmse < best.rmse_vol:
            best = CalibrationResult(
                params=params,
                rmse_vol=rmse,
                mae_vol=float(np.mean(np.abs(vol_err))) if vol_err.size else np.inf,
                max_abs_vol_error=float(np.max(np.abs(vol_err))) if vol_err.size else np.inf,
                n_quotes=int(valid.sum()),
                n_function_evals=int(solution.nfev),
                errors=errors,
                success=bool(solution.success),
            )

    if best is None:  # pragma: no cover - seeds is non-empty by construction
        raise ValueError("every local solve failed")
    best.seed_rmses = sorted(seed_rmses)
    return best


def fit_global_flat_vol(surface: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Fit a single Black-Scholes volatility to the whole surface (the weakest benchmark).

    Returns:
        ``(vol, errors)`` where ``errors`` has ``model_vol``/``vol_error`` columns shaped
        like :func:`surface_errors`.
    """
    vega = np.maximum(surface["vega"].to_numpy(dtype=np.float64), 1e-8)
    weights = np.sqrt(vega)
    market_vol = surface["implied_vol"].to_numpy(dtype=np.float64)
    vol = float(np.sum(weights * market_vol) / np.sum(weights))
    errors = surface.copy()
    errors["model_vol"] = vol
    errors["vol_error"] = vol - market_vol
    return vol, errors


def fit_flat_vol_per_expiry(
    surface: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one Black-Scholes volatility per expiry -- the benchmark Heston must beat.

    This model reproduces the entire at-the-money term structure exactly and has as many
    free parameters as there are expiries (12 here, against Heston's 5). Whatever error it
    leaves is, by construction, *the smile*: it is the only thing a stochastic volatility
    model can buy you.

    Returns:
        ``(per_expiry_vols, errors)``.
    """
    rows: list[dict[str, float]] = []
    errors = surface.copy()
    model_vol = np.empty(len(surface), dtype=np.float64)
    for tau, group in surface.groupby("tau", sort=True):
        weights = np.sqrt(np.maximum(group["vega"].to_numpy(dtype=np.float64), 1e-8))
        vols = group["implied_vol"].to_numpy(dtype=np.float64)
        fitted = float(np.sum(weights * vols) / np.sum(weights))
        model_vol[surface.index.get_indexer(group.index)] = fitted
        rows.append({"tau": to_float(tau), "flat_vol": fitted, "n_quotes": len(group)})
    errors["model_vol"] = model_vol
    errors["vol_error"] = model_vol - surface["implied_vol"].to_numpy(dtype=np.float64)
    return pd.DataFrame(rows), errors
