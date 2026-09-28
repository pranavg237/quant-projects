r"""Put-call parity on a real chain: what can and cannot be tested.

For European options on an asset with forward :math:`F` and discount factor :math:`D`,

.. math:: C(K) - P(K) = D\,(F - K).

**The circularity.** The surface pipeline does not know :math:`F`: it *infers* it from
this very relation (:func:`optpricing.data.implied_forward_curve` takes the median of
:math:`K + (C-P)/D` over strikes within 10% of spot). Checking parity against that
forward cannot test the *level* of parity -- the median residual inside the window is
zero by construction. What it can test is the **shape**: one forward per expiry must
explain every strike, so the residuals must be flat in :math:`K`, and they must sit inside
each pair's bid-ask range. That is a real test with one fitted parameter against 30-230
pairs per expiry, and strikes outside the 10% window are genuinely out of sample.

The level needs data this repo does not have (a dividend forecast and the dealers'
funding rate). Two checks need neither: :func:`american_upper_bound` tests the American
parity bound that depends only on spot and the discount rate, and
:func:`stale_quote_checks` looks for single quotes that are arbitrageable on their own.

**American exercise.** SPY options are American. With rates near 4%, a deep in-the-money
American put is worth roughly its intrinsic value, while the European put is worth
:math:`Ke^{-r\tau} - Se^{-q\tau}` -- less by about :math:`rK\tau`. So for American
options :math:`C - P` falls *faster* than :math:`D(F-K)` once the put is in the money,
and European parity is violated systematically on the high-strike side.
:func:`early_exercise_premia` prices that effect on a binomial tree (American minus
European on the same lattice, so the discretisation error cancels) and
:func:`american_adjusted_forward` re-solves parity with it removed:

.. math:: \hat F_i = K_i + \big(C_i - P_i - (e^C_i - e^P_i)\big)/D.

The tree uses a continuous dividend yield. SPY pays discrete quarterly dividends, which
give deep in-the-money *calls* an exercise premium just before each ex-date; a
continuous yield cannot represent that, so the call premium is understated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import binomial
from . import implied_vol as iv
from .data import ChainSnapshot, RateCurve
from .types import ExerciseStyle, OptionType, to_float

__all__ = [
    "ParityResult",
    "american_adjusted_forward",
    "american_upper_bound",
    "call_put_vol_gap",
    "early_exercise_premia",
    "matched_pairs",
    "parity_residuals",
    "run_parity_analysis",
    "stale_quote_checks",
    "summarise_residuals",
    "violation_distribution",
]

_TOL = 1e-9  # quotes are in cents; this only guards float noise in the comparisons


def matched_pairs(snapshot: ChainSnapshot, clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every (expiry, strike) with a two-sided, uncrossed quote on *both* the call and put.

    Uses the raw snapshot rather than the cleaned chain, because the cleaning filters
    (minimum price, maximum relative spread) remove exactly the deep in-the-money legs on
    which parity is most informative. Whether both legs also survived cleaning is kept as
    the ``in_clean`` column.

    Returns:
        One row per pair with ``expiry, tau, strike, call_bid, call_ask, put_bid,
        put_ask, call_mid, put_mid, call_volume, put_volume, call_open_interest,
        put_open_interest, in_clean``.
    """
    q = snapshot.quotes
    ok = q["bid"].gt(0) & q["ask"].gt(0) & q["ask"].ge(q["bid"])
    two_sided = q.loc[ok]
    values = ["bid", "ask", "volume", "open_interest"]
    wide = two_sided.pivot_table(
        index=["expiry", "tau", "strike"], columns="option_type", values=values, aggfunc="mean"
    )
    wide.columns = [f"{col[1]}_{col[0]}" for col in wide.columns]
    wide = wide.dropna(subset=["call_bid", "call_ask", "put_bid", "put_ask"]).reset_index()
    wide["call_mid"] = 0.5 * (wide["call_bid"] + wide["call_ask"])
    wide["put_mid"] = 0.5 * (wide["put_bid"] + wide["put_ask"])

    if clean is not None:
        kept = clean.groupby(["expiry", "strike"])["option_type"].nunique()
        both = kept[kept == 2].index
        idx = pd.MultiIndex.from_frame(wide[["expiry", "strike"]])
        wide["in_clean"] = idx.isin(both)
    else:
        wide["in_clean"] = False
    return wide.sort_values(["tau", "strike"]).reset_index(drop=True)


def _vol_at(surface: pd.DataFrame, expiry: object, log_moneyness: np.ndarray) -> np.ndarray:
    """Surface implied vol for one expiry, linear in k, flat beyond the quoted range."""
    sl = surface.loc[surface["expiry"] == expiry].sort_values("log_moneyness")
    if sl.empty:
        raise KeyError(f"expiry {expiry} not in the surface")
    k_obs = sl["log_moneyness"].to_numpy(dtype=np.float64)
    vol_obs = sl["implied_vol"].to_numpy(dtype=np.float64)
    return np.asarray(np.interp(log_moneyness, k_obs, vol_obs), dtype=np.float64)


def early_exercise_premia(
    spot: float,
    strikes: np.ndarray,
    tau: float,
    rate: float,
    dividend_yield: float,
    vols: np.ndarray,
    steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """American-minus-European value of the call and the put at each strike.

    Both legs are priced on the same CRR lattice, so the lattice's own discretisation
    error (which is far larger than the premium for short expiries) cancels in the
    difference.

    Returns:
        ``(call_premium, put_premium)``, each the shape of ``strikes``.
    """
    call = np.empty(len(strikes))
    put = np.empty(len(strikes))
    for i, (k, vol) in enumerate(zip(strikes, vols, strict=True)):
        for out, side in ((call, OptionType.CALL), (put, OptionType.PUT)):
            args = (spot, float(k), tau, rate, float(vol), side)
            american = binomial.price(*args, ExerciseStyle.AMERICAN, steps, "crr", dividend_yield)
            european = binomial.price(*args, ExerciseStyle.EUROPEAN, steps, "crr", dividend_yield)
            out[i] = max(american - european, 0.0)
    return call, put


def american_adjusted_forward(
    pairs: pd.DataFrame,
    spot: float,
    rate_curve: RateCurve,
    forward_curve: pd.DataFrame,
    surface: pd.DataFrame,
    moneyness_window: float = 0.10,
    iterations: int = 3,
    steps: int = 200,
) -> pd.DataFrame:
    """Re-solve parity for the forward with the early-exercise premia removed.

    Uses the same pairs as :func:`optpricing.data.implied_forward_curve` (both legs clean,
    strike within ``moneyness_window`` of spot) and the same median estimator, so the only
    difference from the pipeline's forward is the American adjustment. The premia depend
    on the dividend yield, which depends on the forward, so the two are iterated;
    three passes move the forward by well under a cent.

    Args:
        pairs: Output of :func:`matched_pairs` (with ``in_clean`` populated).
        spot: Underlying price.
        rate_curve: Discount curve (the same one the pipeline uses).
        forward_curve: The pipeline's unadjusted forward curve, used as the start point.
        surface: Implied-vol surface supplying the vol for each strike's tree.
        moneyness_window: Strike window for the forward fit, as a fraction of spot.
        iterations: Fixed-point passes.
        steps: Tree steps.

    Returns:
        One row per expiry: ``expiry, tau, forward_raw, forward_adjusted,
        forward_shift_bp, dividend_yield_raw, dividend_yield_adjusted, dispersion_raw,
        dispersion_adjusted`` (dispersion is the interquartile range of the per-strike
        forward estimates, in price units).
    """
    rows: list[dict[str, object]] = []
    for _, f in forward_curve.iterrows():
        g = pairs.loc[
            (pairs["expiry"] == f["expiry"])
            & pairs["in_clean"]
            & pairs["strike"].between(spot * (1 - moneyness_window), spot * (1 + moneyness_window))
        ]
        if g.empty:
            continue
        tau = float(f["tau"])
        rate = float(rate_curve.rate(tau)[0])
        disc = float(f["discount"])
        strikes = g["strike"].to_numpy(dtype=np.float64)
        y = (g["call_mid"] - g["put_mid"]).to_numpy(dtype=np.float64)
        per_strike_raw = strikes + y / disc
        forward = float(f["forward"])
        per_strike = per_strike_raw
        for _ in range(iterations):
            q = rate - np.log(forward / spot) / tau
            vols = _vol_at(surface, f["expiry"], np.log(strikes / forward))
            ee_c, ee_p = early_exercise_premia(spot, strikes, tau, rate, q, vols, steps)
            per_strike = strikes + (y - (ee_c - ee_p)) / disc
            forward = float(np.median(per_strike))
        rows.append(
            {
                "expiry": f["expiry"],
                "tau": tau,
                "n_pairs": len(g),
                "forward_raw": float(f["forward"]),
                "forward_adjusted": forward,
                "forward_shift_bp": 1e4 * (forward / float(f["forward"]) - 1.0),
                "dividend_yield_raw": rate - np.log(float(f["forward"]) / spot) / tau,
                "dividend_yield_adjusted": rate - np.log(forward / spot) / tau,
                "dispersion_raw": float(np.subtract(*np.percentile(per_strike_raw, [75, 25]))),
                "dispersion_adjusted": float(np.subtract(*np.percentile(per_strike, [75, 25]))),
            }
        )
    return pd.DataFrame(rows)


def parity_residuals(
    pairs: pd.DataFrame,
    spot: float,
    rate_curve: RateCurve,
    forward_curve: pd.DataFrame,
    adjusted_forward: pd.DataFrame | None = None,
    surface: pd.DataFrame | None = None,
    moneyness_window: float = 0.10,
    steps: int = 200,
) -> pd.DataFrame:
    r"""Parity residuals for every matched pair, European and (optionally) American.

    Columns added:

    * ``theo_european`` -- :math:`D(F-K)` with the pipeline's forward.
    * ``residual`` -- :math:`C_{mid} - P_{mid} -` ``theo_european``.
    * ``lower`` / ``upper`` -- the tradeable range of the synthetic,
      :math:`C_{bid} - P_{ask}` and :math:`C_{ask} - P_{bid}`.
    * ``excess`` -- how far ``theo_european`` lies outside ``[lower, upper]``, in dollars
      (0 inside). A pair *violates parity beyond the spread* when this is positive: you
      could trade the synthetic against the forward at the quoted prices.
    * ``in_window`` -- whether the strike was used to fit the forward (in-sample).

    With ``adjusted_forward`` and ``surface`` the same columns are produced with suffix
    ``_american``, where the theoretical value is
    :math:`D(F_{adj} - K) + e^C - e^P`.
    """
    fc = forward_curve[["expiry", "forward", "discount"]]
    df = pairs.merge(fc, on="expiry", how="inner")
    df["log_moneyness"] = np.log(df["strike"] / df["forward"])
    df["in_window"] = df["strike"].between(
        spot * (1 - moneyness_window), spot * (1 + moneyness_window)
    )
    df["synthetic_mid"] = df["call_mid"] - df["put_mid"]
    df["lower"] = df["call_bid"] - df["put_ask"]
    df["upper"] = df["call_ask"] - df["put_bid"]
    df["half_spread"] = 0.5 * (df["upper"] - df["lower"])

    def excess(theo: pd.Series) -> pd.Series:
        return (df["lower"] - theo).combine(theo - df["upper"], max).clip(lower=0.0)

    df["theo_european"] = df["discount"] * (df["forward"] - df["strike"])
    df["residual"] = df["synthetic_mid"] - df["theo_european"]
    df["excess"] = excess(df["theo_european"])
    df["violation"] = df["excess"] > _TOL

    if adjusted_forward is not None and surface is not None:
        adj = adjusted_forward[["expiry", "forward_adjusted", "dividend_yield_adjusted"]]
        df = df.merge(adj, on="expiry", how="left")
        ee_call = np.full(len(df), np.nan)
        ee_put = np.full(len(df), np.nan)
        for expiry, g in df.groupby("expiry", sort=False):
            if g["forward_adjusted"].isna().all():
                continue
            tau = float(g["tau"].iloc[0])
            fwd = float(g["forward_adjusted"].iloc[0])
            strikes = g["strike"].to_numpy(dtype=np.float64)
            vols = _vol_at(surface, expiry, np.log(strikes / fwd))
            c, p = early_exercise_premia(
                spot,
                strikes,
                tau,
                float(rate_curve.rate(tau)[0]),
                float(g["dividend_yield_adjusted"].iloc[0]),
                vols,
                steps,
            )
            pos = df.index.get_indexer(g.index)
            ee_call[pos], ee_put[pos] = c, p
        df["ee_call"] = ee_call
        df["ee_put"] = ee_put
        df["theo_american"] = (
            df["discount"] * (df["forward_adjusted"] - df["strike"]) + df["ee_call"] - df["ee_put"]
        )
        df["residual_american"] = df["synthetic_mid"] - df["theo_american"]
        df["excess_american"] = excess(df["theo_american"])
        df["violation_american"] = df["excess_american"] > _TOL
    return df


def summarise_residuals(res: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry violation counts, split in-window (fitted) vs out-of-window (not fitted).

    Needs the ``_american`` columns from :func:`parity_residuals`.
    """
    rows: list[dict[str, object]] = []
    for (expiry, tau), g in res.groupby(["expiry", "tau"], sort=True):
        inside = g.loc[g["in_window"]]
        outside = g.loc[~g["in_window"]]
        rows.append(
            {
                "expiry": expiry,
                "days": to_float(tau) * 365.0,
                "pairs": len(g),
                "in_window": len(inside),
                "median_half_spread_in_window": float(inside["half_spread"].median()),
                "median_abs_residual_european": float(inside["residual"].abs().median()),
                "median_abs_residual_american": float(inside["residual_american"].abs().median()),
                "violations_in_european": int(inside["violation"].sum()),
                "violations_in_american": int(inside["violation_american"].sum()),
                "violations_out_european": int(outside["violation"].sum()),
                "violations_out_american": int(outside["violation_american"].sum()),
            }
        )
    return pd.DataFrame(rows)


def violation_distribution(res: pd.DataFrame) -> pd.DataFrame:
    """How many pairs violate parity beyond the spread, and by how much, in dollars.

    One row per (theory, sample): theory is the European relation with the pipeline's
    forward or the American-adjusted one; the sample is the strikes used to fit the
    forward (in-window) or the rest (out-of-window).
    """
    rows: list[dict[str, object]] = []
    for label, suffix in (("European, pipeline forward", ""), ("American-adjusted", "_american")):
        for sample, mask in (("in-window", res["in_window"]), ("out-of-window", ~res["in_window"])):
            g = res.loc[mask]
            excess = g.loc[g[f"violation{suffix}"], f"excess{suffix}"]
            rows.append(
                {
                    "theory": label,
                    "sample": sample,
                    "pairs": len(g),
                    "violations": len(excess),
                    "violation_rate": len(excess) / len(g) if len(g) else np.nan,
                    "median_excess": float(excess.median()) if len(excess) else 0.0,
                    "p90_excess": float(excess.quantile(0.9)) if len(excess) else 0.0,
                    "max_excess": float(excess.max()) if len(excess) else 0.0,
                    "median_abs_residual": float(g[f"residual{suffix}"].abs().median()),
                }
            )
    return pd.DataFrame(rows)


def call_put_vol_gap(
    res: pd.DataFrame, spot: float, rate_curve: RateCurve, band: float = 0.01
) -> pd.DataFrame:
    r"""Call-minus-put implied vol at strikes within ``band`` of the forward, per expiry.

    Under European parity with the right forward, a call and a put on the same strike
    must imply the same vol, so this gap is a direct read on the forward. It is computed
    twice: with the pipeline's forward on the raw mids, and with the American-adjusted
    forward on *de-Americanised* mids (each leg minus its early-exercise premium).
    Needs the ``_american`` columns from :func:`parity_residuals`.
    """
    rows: list[dict[str, object]] = []
    for expiry, g in res.loc[res["in_clean"]].groupby("expiry", sort=True):
        tau = float(g["tau"].iloc[0])
        rate = float(rate_curve.rate(tau)[0])
        row: dict[str, object] = {"expiry": expiry, "days": tau * 365.0}
        for label, fcol, adjust in (
            ("pipeline", "forward", False),
            ("american", "forward_adjusted", True),
        ):
            fwd = float(g[fcol].iloc[0])
            near = g.loc[np.abs(np.log(g["strike"] / fwd)) <= band]
            if near.empty:
                row[f"gap_{label}"] = np.nan
                continue
            q = rate - np.log(fwd / spot) / tau
            k = near["strike"].to_numpy(dtype=np.float64)
            call = near["call_mid"].to_numpy(dtype=np.float64)
            put = near["put_mid"].to_numpy(dtype=np.float64)
            if adjust:
                call = call - near["ee_call"].to_numpy(dtype=np.float64)
                put = put - near["ee_put"].to_numpy(dtype=np.float64)
            vc = np.asarray(iv.implied_vol(call, spot, k, tau, rate, OptionType.CALL, q))
            vp = np.asarray(iv.implied_vol(put, spot, k, tau, rate, OptionType.PUT, q))
            row[f"gap_{label}"] = float(np.nanmedian(vc - vp) * 100.0)
            row[f"strikes_{label}"] = len(near)
        rows.append(row)
    return pd.DataFrame(rows)


def american_upper_bound(pairs: pd.DataFrame, spot: float, rate_curve: RateCurve) -> pd.DataFrame:
    r"""The one American parity bound that needs neither a forward nor a dividend forecast.

    For American options on a dividend payer (Hull, *Options, Futures and Other
    Derivatives*, ch. 11),

    .. math:: S - \mathrm{PV}(\text{div}) - K \;\le\; C - P \;\le\; S - Ke^{-r\tau}.

    The lower bound needs the dividends, which are not in the data; the upper bound needs
    only spot and the discount rate. A *tradeable* violation is
    :math:`C_{bid} - P_{ask} > S - Ke^{-r\tau}`: sell the call, buy the put and the
    share, borrow :math:`Ke^{-r\tau}`. Spot is a single print with no bid-ask.

    Returns:
        One row per expiry: ``expiry, days, pairs, upper_violations, worst_upper_excess``.
    """
    rows: list[dict[str, object]] = []
    for (expiry, tau), g in pairs.groupby(["expiry", "tau"], sort=True):
        t = to_float(tau)
        disc = float(rate_curve.discount(t)[0])
        k = g["strike"].to_numpy(dtype=np.float64)
        excess = (g["call_bid"] - g["put_ask"]).to_numpy(dtype=np.float64) - (spot - k * disc)
        rows.append(
            {
                "expiry": expiry,
                "days": t * 365.0,
                "pairs": len(g),
                "upper_violations": int((excess > _TOL).sum()),
                "worst_upper_excess": float(max(excess.max(), 0.0)),
            }
        )
    return pd.DataFrame(rows)


def stale_quote_checks(snapshot: ChainSnapshot, tick: float = 0.01) -> pd.DataFrame:
    r"""Quotes that are arbitrageable on their own -- no model, no forward, no pairing.

    Two checks per expiry, on every two-sided uncrossed quote:

    * **Ask below intrinsic.** SPY options are American, so a call offered below
      :math:`S - K` (or a put below :math:`K - S`) could be bought and exercised at once
      for a riskless profit. Reported with the spot that *would* make the quote fair,
      :math:`K + C_{ask}` or :math:`K - P_{ask}`.
    * **Not monotone in strike.** A call must be worth less at a higher strike and a put
      more; a violation is a higher-strike call *bid* above a lower-strike call *ask*
      (adjacent listed strikes), or the mirror image for puts. It is a free vertical
      spread.

    Neither can survive in a live market, so a quote that fails either one is stale or
    indicative. The snapshot carries last-trade times but no quote times, so this is the
    closest the data gets to measuring staleness directly.
    """
    q = snapshot.quotes
    q = q.loc[q["bid"].gt(0) & q["ask"].ge(q["bid"])].copy()
    s = snapshot.spot
    is_call = q["option_type"] == "call"
    q["intrinsic"] = np.where(is_call, s - q["strike"], q["strike"] - s)
    q["implied_spot"] = np.where(is_call, q["strike"] + q["ask"], q["strike"] - q["ask"])
    q["below_intrinsic"] = q["ask"] < q["intrinsic"] - tick / 2

    rows: list[dict[str, object]] = []
    for (expiry, tau), g in q.groupby(["expiry", "tau"], sort=True):
        bad = g.loc[g["below_intrinsic"]]
        monotone = 0
        for side, sign in (("call", 1.0), ("put", -1.0)):
            leg = g.loc[g["option_type"] == side].sort_values("strike")
            bid = leg["bid"].to_numpy(dtype=np.float64)
            ask = leg["ask"].to_numpy(dtype=np.float64)
            if sign > 0:  # call: bid(K_hi) > ask(K_lo)
                monotone += int((bid[1:] > ask[:-1] + tick / 2).sum())
            else:  # put: bid(K_lo) > ask(K_hi)
                monotone += int((bid[:-1] > ask[1:] + tick / 2).sum())
        rows.append(
            {
                "expiry": expiry,
                "days": to_float(tau) * 365.0,
                "quotes": len(g),
                "ask_below_intrinsic": len(bad),
                "implied_spot_min": float(bad["implied_spot"].min()) if len(bad) else np.nan,
                "implied_spot_max": float(bad["implied_spot"].max()) if len(bad) else np.nan,
                "strike_monotonicity_violations": monotone,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ParityResult:
    """Everything :func:`run_parity_analysis` produces."""

    residuals: pd.DataFrame
    by_expiry: pd.DataFrame
    distribution: pd.DataFrame
    forwards: pd.DataFrame
    vol_gap: pd.DataFrame
    upper_bound: pd.DataFrame
    stale: pd.DataFrame


def run_parity_analysis(
    snapshot: ChainSnapshot,
    clean: pd.DataFrame,
    rate_curve: RateCurve,
    forward_curve: pd.DataFrame,
    surface: pd.DataFrame,
    steps: int = 200,
) -> ParityResult:
    """Matched pairs -> European residuals -> American adjustment -> model-free checks."""
    pairs = matched_pairs(snapshot, clean)
    forwards = american_adjusted_forward(
        pairs, snapshot.spot, rate_curve, forward_curve, surface, steps=steps
    )
    residuals = parity_residuals(
        pairs, snapshot.spot, rate_curve, forward_curve, forwards, surface, steps=steps
    )
    return ParityResult(
        residuals=residuals,
        by_expiry=summarise_residuals(residuals),
        distribution=violation_distribution(residuals),
        forwards=forwards,
        vol_gap=call_put_vol_gap(residuals, snapshot.spot, rate_curve),
        upper_bound=american_upper_bound(pairs, snapshot.spot, rate_curve),
        stale=stale_quote_checks(snapshot),
    )
