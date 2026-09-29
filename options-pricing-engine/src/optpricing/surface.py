r"""Volatility surface construction from a cleaned option chain.

A "volatility surface" is not raw data: it is the result of a chain of modelling choices.
The ones made here, and the reasons, are:

**Out-of-the-money options only.** At a given strike, the call and the put contain the
same information (put-call parity), but the in-the-money leg is mostly intrinsic value.
Its vega-to-price ratio is tiny, so its implied vol is far more sensitive to quote noise,
and it carries early-exercise value on American names. Desks quote from OTM options; so
does this module. The switch happens at the **forward**, not at spot.

**Per-expiry forward and discount factor from put-call parity**, not a single T-bill rate
(see :mod:`optpricing.data`). Using a wrong rate shifts the effective forward, which tilts
the whole smile and creates a fake skew.

**American quotes are de-Americanised** (``exercise="american"``, the default). SPY
options are American, and plain European parity on them gives a forward biased low -- the
in-the-money puts in the parity window carry an early-exercise premium of order
:math:`rK\tau` -- while the out-of-the-money puts carry a smaller premium of their own.
Every quote has its lattice premium removed before it is inverted
(:func:`optpricing.american.deamericanise`) and the forward is re-solved from
de-Americanised parity, the two iterated to a fixed point. ``exercise="european"`` is the
uncorrected pipeline, kept so the before/after comparison reproduces.

**Log-moneyness and total variance as the surface coordinates.**

.. math:: k = \ln(K/F_\tau), \qquad w(k, \tau) = \sigma_{\text{imp}}^2(k,\tau)\,\tau.

In these coordinates the two static no-arbitrage conditions are simple to state:

* *Calendar spread*: :math:`w(k, \tau)` must be non-decreasing in :math:`\tau` at fixed
  :math:`k`. A violation is a free calendar spread.
* *Butterfly*: the undiscounted call price must be convex in :math:`K`. A violation means
  the implied risk-neutral density is negative somewhere.

:func:`arbitrage_report` checks both, and the butterfly test is **spread-aware**, which
matters more than it sounds. SPY lists $1 strikes near the money, and at $1 spacing the
theoretical value of a one-lot butterfly is of the same order as the $0.01 quote tick. A
zero-tolerance convexity test on mid prices therefore flags ~25% of strike triples, almost
all of which are rounding rather than arbitrage. The test implemented here flags a triple
only when the butterfly could be bought for a negative amount *after* paying the bid-ask
spread on all three legs -- i.e. only when it is actually tradeable. Both counts are
reported, because the gap between them is the interesting number.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import american
from . import blackscholes as bs
from . import implied_vol as iv
from .data import ChainSnapshot, RateCurve, implied_forward_curve, parity_pairs
from .dividends import DividendSchedule
from .types import ExerciseStyle, FloatArray, OptionType, to_exercise_style, to_float

__all__ = [
    "ArbitrageReport",
    "SurfaceBuild",
    "arbitrage_report",
    "atm_term_structure",
    "build_surface",
    "build_surface_detailed",
    "call_put_gap",
    "smile_slice",
    "surface_grid",
    "thin_surface",
]


@dataclass(frozen=True)
class SurfaceBuild:
    """A surface plus the record of how its forwards and quotes were corrected.

    Attributes:
        surface: The tidy surface (what :func:`build_surface` returns).
        forwards: One row per expiry: ``expiry, tau, forward_european`` (plain parity),
            ``dividend_yield_european, discount, rate, forward`` (the one the surface
            uses), ``dividend_yield, forward_shift_bp, forward_iterations``.
        exercise: How the quotes were treated.
        outer_iterations: Forward -> surface -> forward passes (1 for European).
        forward_history: Largest relative forward change per pass.
        converged: Whether the outer loop met its tolerance.
        max_fixed_point_iterations: Most per-quote de-Americanisation iterations needed.
        max_contraction: Largest observed slope of the per-quote fixed-point map.
        unconverged_quotes: Quotes whose per-quote fixed point did not meet tolerance.
        dropped_by_correction: Quotes invertible as quoted but not once the premium is
            removed (the corrected price fell to the European no-arbitrage floor).
        dividend_model: ``"continuous"`` (the lattice carries a continuous yield) or
            ``"discrete"`` (cash dividends, escrowed-dividend model). Irrelevant, and
            ``"continuous"``, for European exercise.
    """

    surface: pd.DataFrame
    forwards: pd.DataFrame
    exercise: ExerciseStyle
    outer_iterations: int
    forward_history: list[float]
    converged: bool
    max_fixed_point_iterations: int
    max_contraction: float
    unconverged_quotes: int
    dropped_by_correction: int
    dividend_model: str = "continuous"


_SURFACE_COLUMNS = [
    "expiry",
    "tau",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "mid_market",
    "ee_premium",
    "relative_spread",
    "volume",
    "open_interest",
    "forward",
    "discount",
    "rate",
    "dividend_yield",
    "log_moneyness",
    "implied_vol",
    "iv_bid",
    "iv_ask",
    "total_variance",
    "vega",
    "moneyness_bucket",
]


def build_surface(
    snapshot: ChainSnapshot,
    quotes: pd.DataFrame,
    rate_curve: RateCurve,
    otm_only: bool = True,
    max_abs_log_moneyness: float = 1.0,
    min_vol: float = 0.01,
    max_vol: float = 3.0,
    exercise: ExerciseStyle | str = ExerciseStyle.AMERICAN,
    tree_steps: int = 201,
    dividends: DividendSchedule | None = None,
) -> pd.DataFrame:
    """Turn cleaned quotes into a tidy implied-volatility surface.

    Args:
        snapshot: The chain snapshot (supplies spot and as-of time).
        quotes: Cleaned quotes from :func:`optpricing.data.clean_chain`.
        rate_curve: Discount curve used to solve put-call parity for each expiry's
            forward.
        otm_only: Keep only the out-of-the-money leg at each strike (recommended).
        max_abs_log_moneyness: Drop wings beyond this :math:`|k|`. Deep wings on free data
            are almost all stale.
        min_vol: Drop implied vols below this (numerically degenerate).
        max_vol: Drop implied vols above this (almost always a stale quote, not a real
            300% vol).
        exercise: ``"american"`` (default; SPY options are American) removes each
            quote's early-exercise premium and fits the forward to de-Americanised parity;
            ``"european"`` inverts the quotes as they are, with the plain parity forward
            -- the pre-correction pipeline, kept so the comparison is reproducible.
        tree_steps: Lattice steps for the early-exercise premium.
        dividends: Cash dividends (times from ``snapshot.asof``). With them the premia
            come from the escrowed-dividend lattice
            (:func:`optpricing.american.lattice_prices`) and the parity forward's carry
            is split into the dividends plus a continuous residual; without them
            (default) the lattice carries the parity-implied continuous yield. Ignored
            for European exercise.

    Returns:
        Tidy frame with one row per surviving contract and columns
        ``expiry, tau, strike, option_type, bid, ask, mid, mid_market, ee_premium, ...,
        forward, discount, rate, dividend_yield, log_moneyness, implied_vol, iv_bid,
        iv_ask, total_variance, vega, moneyness_bucket``. ``bid``, ``ask`` and ``mid``
        are *European-equivalent* prices (market minus ``ee_premium``), which is what
        every European model downstream must be compared with; ``mid_market`` is the
        quote as observed.

    Raises:
        ValueError: if no expiry admits a put-call-parity forward fit.
    """
    return build_surface_detailed(
        snapshot,
        quotes,
        rate_curve,
        otm_only=otm_only,
        max_abs_log_moneyness=max_abs_log_moneyness,
        min_vol=min_vol,
        max_vol=max_vol,
        exercise=exercise,
        tree_steps=tree_steps,
        dividends=dividends,
    ).surface


def build_surface_detailed(
    snapshot: ChainSnapshot,
    quotes: pd.DataFrame,
    rate_curve: RateCurve,
    otm_only: bool = True,
    max_abs_log_moneyness: float = 1.0,
    min_vol: float = 0.01,
    max_vol: float = 3.0,
    exercise: ExerciseStyle | str = ExerciseStyle.AMERICAN,
    tree_steps: int = 201,
    forward_tol: float = 1e-7,
    max_forward_iter: int = 10,
    forwards: pd.DataFrame | None = None,
    dividends: DividendSchedule | None = None,
) -> SurfaceBuild:
    r""":func:`build_surface`, plus the forward curve and convergence diagnostics.

    For American exercise the forward and the surface depend on each other: the forward
    needs the early-exercise premia of the in-the-money legs in the parity window, which
    need vols, which come from the surface, which is built at the forward. Starting from
    the European-parity forward, each pass builds the surface (de-Americanising every
    quote by the per-quote fixed point of :func:`optpricing.american.deamericanise`) and
    re-solves parity with the premia removed; it stops when no forward moves by more than
    ``forward_tol`` (relative). The surface returned is built at forwards that reproduce
    themselves to that tolerance.

    Args:
        forward_tol: Relative tolerance on the forward between passes.
        max_forward_iter: Pass cap; ``converged`` is ``False`` if it is hit.
        forwards: The ``forwards`` of an earlier :class:`SurfaceBuild` to reuse instead of
            solving again. Needed when building a restricted view of the same chain (e.g.
            both legs near the money, for :func:`call_put_gap`): a narrow surface cannot
            supply the vols of the in-the-money parity legs 10% away, so re-solving the
            forward on it would bias it.
        Others: as :func:`build_surface`.
    """
    style = to_exercise_style(exercise)
    spot = snapshot.spot
    if style is ExerciseStyle.EUROPEAN:
        dividends = None
    curve = implied_forward_curve(quotes, spot, rate_curve)
    if curve.empty:
        raise ValueError(
            "could not fit a forward for any expiry: the chain has too few matched "
            "call/put strikes near the money after cleaning"
        )
    european = curve[["expiry", "tau", "forward", "dividend_yield"]].rename(
        columns={"forward": "forward_european", "dividend_yield": "dividend_yield_european"}
    )
    forward_iterations = dict.fromkeys(curve["expiry"], 0)
    history: list[float] = []
    converged = style is ExerciseStyle.EUROPEAN
    if forwards is not None:
        fixed = forwards.set_index("expiry")
        curve = curve.loc[curve["expiry"].isin(fixed.index)].reset_index(drop=True)
        curve["forward"] = curve["expiry"].map(fixed["forward"])
        curve["dividend_yield"] = curve["rate"] - np.log(curve["forward"] / spot) / curve["tau"]
        converged, max_forward_iter = True, 0
    outer = 0
    while True:
        outer += 1
        surf, stats = _invert_quotes(
            spot, quotes, curve, style, otm_only, max_abs_log_moneyness, tree_steps, dividends
        )
        if style is ExerciseStyle.EUROPEAN or outer > max_forward_iter:
            break  # European, reused forwards, or the pass cap
        new_curve, iters = _american_forward_curve(quotes, spot, curve, surf, tree_steps, dividends)
        change = float(np.max(np.abs(new_curve["forward"] / curve["forward"] - 1.0)))
        history.append(change)
        for expiry, n in iters.items():
            forward_iterations[expiry] += n
        if change < forward_tol:
            converged = True
            break
        curve = new_curve

    surf = surf.loc[surf["implied_vol"].between(min_vol, max_vol)].copy()
    surf["total_variance"] = surf["implied_vol"] ** 2 * surf["tau"]
    surf["vega"] = bs.vega(
        spot,
        surf["strike"].to_numpy(dtype=np.float64),
        surf["tau"].to_numpy(dtype=np.float64),
        surf["rate"].to_numpy(dtype=np.float64),
        surf["implied_vol"].to_numpy(dtype=np.float64),
        surf["dividend_yield"].to_numpy(dtype=np.float64),
    )
    surf["moneyness_bucket"] = pd.cut(
        surf["log_moneyness"],
        bins=[-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf],
        labels=["deep put wing", "put wing", "ATM", "call wing", "deep call wing"],
    )
    surf = (
        surf[[c for c in _SURFACE_COLUMNS if c in surf.columns]]
        .sort_values(["tau", "strike"])
        .reset_index(drop=True)
    )

    fwd_table = european.merge(
        curve[["expiry", "discount", "rate", "forward", "dividend_yield"]], on="expiry"
    )
    fwd_table["forward_shift_bp"] = 1e4 * (
        fwd_table["forward"] / fwd_table["forward_european"] - 1.0
    )
    fwd_table["forward_iterations"] = fwd_table["expiry"].map(forward_iterations)
    # How the carry splits: PV of the cash dividends before expiry, and the continuous
    # residual r - ln(F / (S - PV)) / tau the lattice carries on top of them. Minus the
    # residual is the funding rate over the discount rate (0 PV for the continuous model).
    pv = [
        dividends.present_value(float(t), float(r)) if dividends is not None else 0.0
        for t, r in zip(fwd_table["tau"], fwd_table["rate"], strict=True)
    ]
    fwd_table["pv_dividends"] = pv
    fwd_table["residual_yield"] = (
        fwd_table["rate"]
        - np.log(fwd_table["forward"] / (spot - fwd_table["pv_dividends"])) / fwd_table["tau"]
    )
    return SurfaceBuild(
        surface=surf,
        forwards=fwd_table,
        exercise=style,
        outer_iterations=outer,
        forward_history=history,
        converged=converged,
        max_fixed_point_iterations=int(stats["max_iterations"]),
        max_contraction=float(stats["max_contraction"]),
        unconverged_quotes=int(stats["unconverged"]),
        dropped_by_correction=int(stats["dropped"]),
        dividend_model="discrete" if dividends is not None else "continuous",
    )


def _invert_quotes(
    spot: float,
    quotes: pd.DataFrame,
    curve: pd.DataFrame,
    style: ExerciseStyle,
    otm_only: bool,
    max_abs_log_moneyness: float,
    tree_steps: int,
    dividends: DividendSchedule | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach the forward, pick the legs, remove premia (if American) and invert."""
    df = quotes.merge(
        curve[["expiry", "forward", "discount", "rate", "dividend_yield"]],
        on="expiry",
        how="inner",
    )
    df["log_moneyness"] = np.log(df["strike"] / df["forward"])
    if otm_only:
        is_otm = ((df["option_type"] == "call") & (df["strike"] >= df["forward"])) | (
            (df["option_type"] == "put") & (df["strike"] < df["forward"])
        )
        df = df.loc[is_otm]
    df = df.loc[df["log_moneyness"].abs() <= max_abs_log_moneyness].reset_index(drop=True)
    if df.empty:
        raise ValueError("no quotes survive the OTM / log-moneyness filters")

    # Price with the parity-implied rate and *carry-adjusted spot* so that calls and puts
    # on the same strike must imply the same vol by construction (Black-76 in effect).
    df["dividend_yield"] = df["rate"] - np.log(df["forward"] / spot) / df["tau"]
    df["mid_market"] = df["mid"]
    n = len(df)
    premium = np.zeros(n)
    vol = np.full(n, np.nan)
    stats = {"max_iterations": 0.0, "max_contraction": 0.0, "unconverged": 0.0, "dropped": 0.0}
    for _, g in df.groupby(["expiry", "option_type"], sort=False):
        pos = df.index.get_indexer(g.index)
        side = str(g["option_type"].iloc[0])
        tau = float(g["tau"].iloc[0])
        rate = float(g["rate"].iloc[0])
        q = float(g["dividend_yield"].iloc[0])
        mids = g["mid"].to_numpy(dtype=np.float64)
        strikes = g["strike"].to_numpy(dtype=np.float64)
        if style is ExerciseStyle.EUROPEAN:
            vol[pos] = np.asarray(iv.implied_vol(mids, spot, strikes, tau, rate, side, q))
            continue
        if dividends is not None:
            # Same forward, carry split into cash dividends plus a residual yield.
            s_star = american.escrowed_spot(spot, tau, rate, dividends.within(tau))
            q = rate - float(np.log(float(g["forward"].iloc[0]) / s_star)) / tau
        res = american.deamericanise(
            mids, spot, strikes, tau, rate, q, side, tree_steps, dividends=dividends
        )
        raw = np.asarray(
            iv.implied_vol(mids, spot, strikes, tau, rate, side, float(g["dividend_yield"].iloc[0]))
        )
        vol[pos] = res.vol
        premium[pos] = np.where(np.isfinite(res.premium), res.premium, 0.0)
        finite = np.isfinite(res.vol)
        stats["dropped"] += float((np.isfinite(raw) & ~finite).sum())
        stats["unconverged"] += float((finite & ~res.converged).sum())
        if finite.any():
            stats["max_iterations"] = max(
                stats["max_iterations"], float(res.iterations[finite].max())
            )
        stats["max_contraction"] = max(stats["max_contraction"], res.contraction)

    df["ee_premium"] = premium
    df["implied_vol"] = vol
    # European-equivalent prices. The bid and ask carry the mid's premium: the premium
    # moves with vol by less than a cent across a quote's bid-ask, so this keeps the
    # tradeable spread exactly as quoted.
    for col in ("mid", "bid", "ask"):
        if col in df.columns:
            df[col] = df[col] - premium
    for out_col, price_col in (("iv_bid", "bid"), ("iv_ask", "ask")):
        if price_col not in df.columns:
            continue
        vols = np.full(n, np.nan)
        for side in ("call", "put"):
            mask = (df["option_type"] == side).to_numpy()
            if not mask.any():
                continue
            vols[mask] = np.asarray(
                iv.implied_vol(
                    df.loc[mask, price_col].to_numpy(dtype=np.float64),
                    spot,
                    df.loc[mask, "strike"].to_numpy(dtype=np.float64),
                    df.loc[mask, "tau"].to_numpy(dtype=np.float64),
                    df.loc[mask, "rate"].to_numpy(dtype=np.float64),
                    OptionType(side),
                    df.loc[mask, "dividend_yield"].to_numpy(dtype=np.float64),
                )
            )
        df[out_col] = vols
    return df, stats


def _smile_interpolator(k_obs: FloatArray, v_obs: FloatArray) -> Callable[[FloatArray], FloatArray]:
    """Vol as a function of log-moneyness: linear inside the quotes, flat beyond."""

    def vol_at(k: FloatArray) -> FloatArray:
        return np.asarray(np.interp(k, k_obs, v_obs), dtype=np.float64)

    return vol_at


def _american_forward_curve(
    quotes: pd.DataFrame,
    spot: float,
    curve: pd.DataFrame,
    surf: pd.DataFrame,
    tree_steps: int,
    dividends: DividendSchedule | None = None,
) -> tuple[pd.DataFrame, dict[object, int]]:
    """Re-solve every expiry's parity forward with the premia priced at ``surf``'s vols."""
    out = curve.copy()
    iters: dict[object, int] = {}
    by_expiry = dict(tuple(quotes.groupby("expiry", sort=False)))
    for i, row in curve.iterrows():
        expiry = row["expiry"]
        sl = surf.loc[(surf["expiry"] == expiry) & np.isfinite(surf["implied_vol"])].sort_values(
            "log_moneyness"
        )
        pairs = parity_pairs(by_expiry[expiry], spot)
        if sl.empty or pairs.empty:  # pragma: no cover - a fitted expiry has both
            iters[expiry] = 0
            continue
        vol_at = _smile_interpolator(
            sl["log_moneyness"].to_numpy(dtype=np.float64),
            sl["implied_vol"].to_numpy(dtype=np.float64),
        )
        tau = float(row["tau"])
        rate = float(row["rate"])
        sol = american.american_forward(
            pairs.index.to_numpy(dtype=np.float64),
            pairs["call"].to_numpy(dtype=np.float64),
            pairs["put"].to_numpy(dtype=np.float64),
            spot,
            tau,
            rate,
            float(row["discount"]),
            vol_at,
            float(row["forward"]),
            steps=tree_steps,
            dividends=dividends,
        )
        out.loc[i, "forward"] = sol.forward
        out.loc[i, "dividend_yield"] = rate - np.log(sol.forward / spot) / tau
        iters[expiry] = sol.iterations
    return out, iters


def thin_surface(
    surface: pd.DataFrame,
    max_per_expiry: int = 40,
    moneyness_range: tuple[float, float] = (-0.45, 0.25),
) -> pd.DataFrame:
    """Subsample a surface to roughly ``max_per_expiry`` quotes per expiry.

    Calibration cost is linear in the number of quotes and a dense SPY surface has ~2,000
    of them, most carrying nearly identical information (adjacent $1 strikes). Thinning to
    an evenly spaced subset in **log-moneyness** -- not in strike, which would
    over-represent the dense at-the-money region -- cuts calibration time by ~50x while
    changing the fitted parameters in the third decimal place.

    Args:
        surface: Output of :func:`build_surface`.
        max_per_expiry: Target number of quotes kept per expiry.
        moneyness_range: Log-moneyness window to keep. The default trims the far put wing,
            where free quotes are least reliable, while keeping the skew that matters.

    Returns:
        A thinned copy, index reset.
    """
    lo, hi = moneyness_range
    windowed = surface.loc[surface["log_moneyness"].between(lo, hi)]
    pieces: list[pd.DataFrame] = []
    for _, group in windowed.groupby("tau", sort=True):
        ordered = group.sort_values("log_moneyness")
        if len(ordered) <= max_per_expiry:
            pieces.append(ordered)
            continue
        targets = np.linspace(
            ordered["log_moneyness"].iloc[0], ordered["log_moneyness"].iloc[-1], max_per_expiry
        )
        obs = ordered["log_moneyness"].to_numpy(dtype=np.float64)
        idx = sorted({int(np.abs(obs - t).argmin()) for t in targets})
        pieces.append(ordered.iloc[idx])
    if not pieces:
        return surface.iloc[:0].copy()
    return pd.concat(pieces).sort_values(["tau", "strike"]).reset_index(drop=True)


def smile_slice(surface: pd.DataFrame, expiry: dt.date) -> pd.DataFrame:
    """Return one expiry's smile, sorted by log-moneyness."""
    sl = surface.loc[surface["expiry"] == expiry]
    if sl.empty:
        raise KeyError(f"expiry {expiry} not present in the surface")
    return sl.sort_values("log_moneyness").reset_index(drop=True)


def atm_term_structure(surface: pd.DataFrame, window: float = 0.05) -> pd.DataFrame:
    r"""ATM implied volatility per expiry, by vega-weighted average near :math:`k = 0`.

    A single "ATM strike" rarely exists on the listed grid, and picking the nearest one
    makes the term structure jump around as the forward drifts between strikes. Averaging
    the strikes within ``window`` of the forward, weighted by vega, is stable and is close
    to what the variance-swap-style ATM quote means.

    Args:
        surface: Output of :func:`build_surface`.
        window: Half-width in log-moneyness.

    Returns:
        Frame with ``expiry, tau, atm_vol, atm_total_variance, n_quotes``.
    """
    near = surface.loc[surface["log_moneyness"].abs() <= window]
    rows: list[dict[str, object]] = []
    for (expiry, tau), group in near.groupby(["expiry", "tau"], sort=True):
        weights = group["vega"].to_numpy(dtype=np.float64)
        vols = group["implied_vol"].to_numpy(dtype=np.float64)
        if weights.sum() <= 0:  # pragma: no cover - vega is positive for live quotes
            continue
        atm = float(np.average(vols, weights=weights))
        rows.append(
            {
                "expiry": expiry,
                "tau": to_float(tau),
                "atm_vol": atm,
                "atm_total_variance": atm * atm * to_float(tau),
                "n_quotes": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)


def call_put_gap(both_legs: pd.DataFrame, band: float = 0.01) -> pd.DataFrame:
    """Call-minus-put implied vol at strikes within ``band`` of the forward, per expiry.

    Under European parity with the right forward a call and a put on one strike imply
    the same vol, so the gap measures how well the surface's forward and quote treatment
    fit together. Needs a surface built with ``otm_only=False``.

    Returns:
        ``expiry, days, gap_vol_points`` (median over strikes) ``, strikes``.
    """
    near = both_legs.loc[both_legs["log_moneyness"].abs() <= band]
    rows: list[dict[str, object]] = []
    for (expiry, tau), g in near.groupby(["expiry", "tau"], sort=True):
        wide = g.pivot_table(index="strike", columns="option_type", values="implied_vol")
        if not {"call", "put"}.issubset(wide.columns):
            continue
        diff = (wide["call"] - wide["put"]).dropna()
        if diff.empty:
            continue
        rows.append(
            {
                "expiry": expiry,
                "days": to_float(tau) * 365.0,
                "gap_vol_points": float(diff.median() * 100.0),
                "strikes": len(diff),
            }
        )
    return pd.DataFrame(rows, columns=["expiry", "days", "gap_vol_points", "strikes"])


def surface_grid(
    surface: pd.DataFrame,
    n_moneyness: int = 41,
    moneyness_range: tuple[float, float] = (-0.4, 0.3),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Interpolate the scattered surface onto a regular :math:`(\tau, k)` grid.

    Interpolation is done in **total variance**, linearly in :math:`k` within each expiry,
    then the grid is returned as volatility. Interpolating total variance rather than
    volatility is what keeps the result closer to calendar-arbitrage-free: total variance
    is the quantity that must be monotone in :math:`\tau`.

    No extrapolation is performed: grid points outside an expiry's quoted strike range
    come back as ``nan``, which is the honest answer and shows up as a hole in the 3-D
    plot rather than as a flat invented wing.

    Returns:
        ``(taus, moneyness, vol_grid)`` with ``vol_grid`` of shape
        ``(len(taus), n_moneyness)``.
    """
    taus = np.array(sorted(surface["tau"].unique()), dtype=np.float64)
    ks = np.linspace(moneyness_range[0], moneyness_range[1], n_moneyness)
    grid = np.full((taus.size, ks.size), np.nan, dtype=np.float64)

    for i, tau in enumerate(taus):
        sl = surface.loc[surface["tau"] == tau].sort_values("log_moneyness")
        if len(sl) < 2:
            continue
        k_obs = sl["log_moneyness"].to_numpy(dtype=np.float64)
        w_obs = sl["total_variance"].to_numpy(dtype=np.float64)
        inside = (ks >= k_obs.min()) & (ks <= k_obs.max())
        w_interp = np.interp(ks[inside], k_obs, w_obs)
        grid[i, inside] = np.sqrt(np.maximum(w_interp, 0.0) / tau)
    return taus, ks, grid


@dataclass(frozen=True)
class ArbitrageReport:
    """Counts of static no-arbitrage violations found in a surface.

    Attributes:
        calendar_violations: Adjacent-expiry pairs where total variance falls with tau.
        calendar_checks: Pairs checked.
        butterfly_violations: Strike triples where the butterfly is negative by more than
            the cost of crossing the spread on all three legs -- i.e. actually tradeable.
        butterfly_checks: Triples checked.
        butterfly_violations_zero_tol: Triples that are negative at all, ignoring spreads.
            The gap against ``butterfly_violations`` is the part explained by quote noise.
        worst_calendar_drop: Largest fall in total variance observed (<= 0).
        worst_butterfly: Most negative spread-adjusted butterfly value, in price units.
    """

    calendar_violations: int
    calendar_checks: int
    butterfly_violations: int
    butterfly_checks: int
    butterfly_violations_zero_tol: int
    worst_calendar_drop: float
    worst_butterfly: float

    @property
    def calendar_rate(self) -> float:
        """Fraction of calendar checks that failed."""
        return self.calendar_violations / self.calendar_checks if self.calendar_checks else 0.0

    @property
    def butterfly_rate(self) -> float:
        """Fraction of butterfly checks that were tradeable violations."""
        return self.butterfly_violations / self.butterfly_checks if self.butterfly_checks else 0.0

    @property
    def butterfly_rate_zero_tol(self) -> float:
        """Fraction of butterfly checks that were negative at all."""
        if not self.butterfly_checks:
            return 0.0
        return self.butterfly_violations_zero_tol / self.butterfly_checks

    def __str__(self) -> str:
        return (
            f"Calendar-spread violations : {self.calendar_violations}/{self.calendar_checks} "
            f"({self.calendar_rate:.1%}); worst total-variance drop "
            f"{self.worst_calendar_drop:.2e}\n"
            f"Butterfly, zero tolerance  : {self.butterfly_violations_zero_tol}/"
            f"{self.butterfly_checks} ({self.butterfly_rate_zero_tol:.1%}) "
            f"<- dominated by the $0.01 quote tick\n"
            f"Butterfly, net of spread   : {self.butterfly_violations}/{self.butterfly_checks} "
            f"({self.butterfly_rate:.1%}); worst {self.worst_butterfly:+.4f} per butterfly"
        )


def arbitrage_report(surface: pd.DataFrame, n_moneyness: int = 31) -> ArbitrageReport:
    r"""Check the two static no-arbitrage conditions on a fitted surface.

    *Calendar*: on the interpolated grid, :math:`w(k, \tau)` must not fall as :math:`\tau`
    increases at fixed :math:`k`.

    *Butterfly*: for each consecutive strike triple :math:`K_{i-1} < K_i < K_{i+1}` within
    an expiry, convexity of the call price is equivalent to the tradeable spread

    .. math::
        B_i = \lambda C_{i-1} + (1-\lambda)C_{i+1} - C_i \ge 0,\qquad
        \lambda = \frac{K_{i+1}-K_i}{K_{i+1}-K_{i-1}},

    having non-negative value -- a long butterfly with a non-negative payoff everywhere.
    The weights handle non-uniform strike grids correctly, which a plain second difference
    does not. The execution cost of the three legs is
    :math:`\tfrac12(\lambda s_{i-1} + s_i + (1-\lambda)s_{i+1})` with :math:`s` the
    bid-ask spreads, and a violation is only counted when :math:`B_i` is below minus that.

    Args:
        surface: Output of :func:`build_surface`.
        n_moneyness: Grid resolution used for the calendar test.
    """
    taus, ks, vol_grid = surface_grid(surface, n_moneyness=n_moneyness)
    total_var = vol_grid**2 * taus[:, None]

    cal_checks = cal_bad = 0
    worst_cal = 0.0
    for j in range(ks.size):
        seq = total_var[:, j][np.isfinite(total_var[:, j])]
        if seq.size < 2:
            continue
        diffs = np.diff(seq)
        cal_checks += diffs.size
        cal_bad += int((diffs < -1e-12).sum())
        worst_cal = min(worst_cal, float(diffs.min()))

    fly_checks = fly_bad = fly_bad_zero = 0
    worst_fly = 0.0
    has_spread = {"bid", "ask"}.issubset(surface.columns)
    for tau, group in surface.groupby("tau", sort=True):
        sl = group.sort_values("strike")
        if len(sl) < 3:
            continue
        strikes = sl["strike"].to_numpy(dtype=np.float64)
        # Re-price every strike as a call from its implied vol so convexity is tested on
        # one consistent price function rather than on mixed calls and puts.
        calls = np.asarray(
            bs.price(
                sl["forward"].to_numpy(dtype=np.float64),
                strikes,
                to_float(tau),
                0.0,  # undiscounted: convexity is a property of the forward price
                sl["implied_vol"].to_numpy(dtype=np.float64),
                OptionType.CALL,
                0.0,
            )
        )
        if has_spread:
            spreads = (sl["ask"] - sl["bid"]).to_numpy(dtype=np.float64)
        else:  # pragma: no cover - synthetic surfaces without quotes
            spreads = np.zeros_like(strikes)

        k_lo, k_mid, k_hi = strikes[:-2], strikes[1:-1], strikes[2:]
        lam = (k_hi - k_mid) / (k_hi - k_lo)
        butterfly = lam * calls[:-2] + (1.0 - lam) * calls[2:] - calls[1:-1]
        cost = 0.5 * (lam * spreads[:-2] + spreads[1:-1] + (1.0 - lam) * spreads[2:])

        fly_checks += butterfly.size
        fly_bad_zero += int((butterfly < -1e-12).sum())
        net = butterfly + cost
        fly_bad += int((net < 0.0).sum())
        if net.size:
            worst_fly = min(worst_fly, float(net.min()))

    return ArbitrageReport(
        calendar_violations=cal_bad,
        calendar_checks=cal_checks,
        butterfly_violations=fly_bad,
        butterfly_checks=fly_checks,
        butterfly_violations_zero_tol=fly_bad_zero,
        worst_calendar_drop=worst_cal,
        worst_butterfly=worst_fly,
    )
