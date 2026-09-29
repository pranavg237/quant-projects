r"""Option-chain download, local caching and cleaning.

Market data is pulled from Yahoo Finance via ``yfinance``. Every download is written to a
CSV cache keyed by ticker and as-of date, and :func:`load_chain` prefers the cache, so a
rerun never touches the network and results stay reproducible after the quotes have moved.

Three things here are less obvious than they look:

**Time to expiry.** Listed US equity options expire at 16:00 New York time on the expiry
date. Using a naive ``(expiry - today).days / 365`` is wrong by up to a full day, which
matters enormously for the front expiry: a 1-day option mispriced by half a day of
calendar time has its implied vol wrong by ~40%. This module uses an ACT/365 year fraction
measured to 16:00 ET.

**The forward is implied from the quotes; the discount factor comes from the rates
market.** Put-call parity is model-free:

.. math:: C(K) - P(K) = D(F - K), \qquad D = e^{-r\tau},\ F = S_0e^{(r-q)\tau}.

It is tempting to regress :math:`C-P` on :math:`K` and read off *both* :math:`D` (minus
the slope) and :math:`F` (intercept over :math:`D`). Do not: that fit is badly
conditioned at short maturities. Over a +/-10% strike window a 1% error in the slope is a
1% error in :math:`D`, which at :math:`\tau = 0.06` implies a **17 percentage point**
error in :math:`r`. On live SPY quotes the two-parameter fit returns discount factors
above 1 and implied rates of -33% at three weeks, with an :math:`R^2` of 0.9998 -- the
line fits beautifully and its slope is still meaningless. :func:`implied_forward_curve`
keeps both methods so the failure is visible, but defaults to the one desks actually use:
take :math:`D` from the Treasury curve and solve parity for :math:`F` strike by strike,

.. math:: \hat F_i = K_i + (C_i - P_i)/D,

then take a robust centre of the :math:`\hat F_i`. That single-parameter estimate is
well conditioned, and it still absorbs borrow costs, hard-to-borrow premia and discrete
dividend timing into the forward -- which is where they belong. Using it guarantees calls
and puts imply the *same* volatility, so the surface has no spurious kink at the forward.

**Cleaning throws away a lot, on purpose.** Deep-wing quotes on Yahoo are frequently
stale, one-sided or crossed. Keeping them produces a spectacular-looking but meaningless
volatility surface. The filters and how much each removes are reported by
:func:`clean_chain`.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .types import Numeric

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ChainSnapshot",
    "CleaningReport",
    "RateCurve",
    "clean_chain",
    "download_chain",
    "implied_forward_curve",
    "load_chain",
    "load_rate_curve",
    "parity_pairs",
    "year_fraction",
]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
_NY = ZoneInfo("America/New_York")
_EXPIRY_HOUR = 16  # US listed equity options expire at the 16:00 ET close
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def year_fraction(asof: dt.datetime, expiry: dt.date) -> float:
    """ACT/365 year fraction from ``asof`` to the 16:00 ET close on ``expiry``.

    Args:
        asof: Snapshot timestamp. Naive timestamps are assumed to be New York time.
        expiry: Expiry date.

    Returns:
        Year fraction, floored at 0.
    """
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=_NY)
    expiry_dt = dt.datetime.combine(expiry, dt.time(_EXPIRY_HOUR), tzinfo=_NY)
    return max((expiry_dt - asof).total_seconds(), 0.0) / _SECONDS_PER_YEAR


@dataclass
class CleaningReport:
    """How many rows each cleaning filter removed."""

    initial: int
    removed: dict[str, int] = field(default_factory=dict)
    final: int = 0

    def to_frame(self) -> pd.DataFrame:
        """Tabular summary, newest filter last."""
        rows = [{"stage": "downloaded", "rows_removed": 0, "rows_remaining": self.initial}]
        remaining = self.initial
        for name, n in self.removed.items():
            remaining -= n
            rows.append({"stage": name, "rows_removed": n, "rows_remaining": remaining})
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        kept = 100.0 * self.final / self.initial if self.initial else 0.0
        lines = [f"Cleaning: {self.initial} -> {self.final} rows ({kept:.1f}% kept)"]
        lines += [f"  -{n:>6}  {name}" for name, n in self.removed.items()]
        return "\n".join(lines)


@dataclass
class ChainSnapshot:
    """An option chain as of one moment, plus the spot that goes with it.

    Attributes:
        ticker: Underlying symbol.
        asof: Timestamp of the snapshot (New York time).
        spot: Underlying price at ``asof``.
        quotes: One row per listed contract.
    """

    ticker: str
    asof: dt.datetime
    spot: float
    quotes: pd.DataFrame

    def to_csv(self, path: Path) -> None:
        """Persist to a single self-describing CSV (spot/asof ride along as columns)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.quotes.copy()
        out["ticker"] = self.ticker
        out["asof"] = self.asof.isoformat()
        out["spot"] = self.spot
        out.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Path) -> ChainSnapshot:
        """Load a snapshot written by :meth:`to_csv`."""
        df = pd.read_csv(path, parse_dates=["expiry"])
        asof = dt.datetime.fromisoformat(str(df["asof"].iloc[0]))
        ticker = str(df["ticker"].iloc[0])
        spot = float(df["spot"].iloc[0])
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        return cls(ticker=ticker, asof=asof, spot=spot, quotes=df)

    def expiries(self) -> list[dt.date]:
        """Sorted unique expiry dates present in the snapshot."""
        return sorted(set(self.quotes["expiry"]))


def _cache_path(cache_dir: Path, ticker: str, asof_date: dt.date) -> Path:
    return cache_dir / "raw" / f"{ticker.upper()}_{asof_date.isoformat()}.csv"


def download_chain(
    ticker: str,
    max_expiries: int = 12,
    min_tau_days: float = 5.0,
    max_tau_days: float = 800.0,
    cache_dir: Path | None = None,
) -> ChainSnapshot:
    """Download a full option chain from Yahoo Finance and cache it.

    Args:
        ticker: Underlying symbol, e.g. ``"SPY"``.
        max_expiries: Cap on the number of expiries pulled (each is a separate request).
            Expiries are selected to span the available term structure roughly evenly in
            log-time rather than taking the ``n`` nearest, so the surface has a real term
            structure instead of ten weeklies.
        min_tau_days: Skip expiries closer than this. Sub-week options are dominated by
            microstructure and pin risk.
        max_tau_days: Skip expiries further out than this (LEAPS quotes are very wide).
        cache_dir: Root of the data cache. Defaults to ``<repo>/data``.

    Returns:
        A :class:`ChainSnapshot`.

    Raises:
        RuntimeError: if the download yields no usable quotes.
    """
    import yfinance as yf  # noqa: PLC0415  (local: keeps the network dep off the import path)

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    tk = yf.Ticker(ticker)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = tk.history(period="5d", auto_adjust=False)
    if history.empty:
        raise RuntimeError(f"no price history returned for {ticker!r}")
    spot = float(history["Close"].iloc[-1])
    asof = dt.datetime.now(tz=_NY)

    all_expiries = [dt.date.fromisoformat(s) for s in tk.options]
    taus = np.array([year_fraction(asof, e) for e in all_expiries])
    eligible = [
        e
        for e, t in zip(all_expiries, taus, strict=True)
        if min_tau_days / 365.0 <= t <= max_tau_days / 365.0
    ]
    if not eligible:
        raise RuntimeError(f"no expiries for {ticker!r} within the requested tau window")
    chosen = _spread_expiries(eligible, asof, max_expiries)

    frames: list[pd.DataFrame] = []
    for expiry in chosen:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chain = tk.option_chain(expiry.isoformat())
        except Exception as exc:
            warnings.warn(f"skipping expiry {expiry}: {exc}", stacklevel=2)
            continue
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            sub = frame.copy()
            sub["option_type"] = side
            sub["expiry"] = expiry
            frames.append(sub)

    if not frames:
        raise RuntimeError(f"no option quotes returned for {ticker!r}")

    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.rename(
        columns={
            "lastPrice": "last",
            "openInterest": "open_interest",
            "impliedVolatility": "yf_implied_vol",
            "lastTradeDate": "last_trade",
            "contractSymbol": "contract",
        }
    )
    keep = [
        "contract",
        "expiry",
        "option_type",
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "yf_implied_vol",
        "last_trade",
    ]
    quotes = quotes[[c for c in keep if c in quotes.columns]].copy()
    quotes["volume"] = pd.to_numeric(quotes["volume"], errors="coerce").fillna(0.0)
    quotes["open_interest"] = pd.to_numeric(quotes["open_interest"], errors="coerce").fillna(0.0)
    quotes["tau"] = [year_fraction(asof, e) for e in quotes["expiry"]]
    quotes["mid"] = 0.5 * (quotes["bid"] + quotes["ask"])

    snapshot = ChainSnapshot(ticker=ticker.upper(), asof=asof, spot=spot, quotes=quotes)
    snapshot.to_csv(_cache_path(cache_dir, ticker, asof.date()))
    return snapshot


def _spread_expiries(expiries: list[dt.date], asof: dt.datetime, n: int) -> list[dt.date]:
    """Pick ``n`` expiries spread roughly evenly in log-time.

    Taking the ``n`` nearest expiries on a name like SPY returns ten weeklies inside a
    month, which produces a "term structure" with no term in it.
    """
    if len(expiries) <= n:
        return expiries
    taus = np.array([year_fraction(asof, e) for e in expiries])
    targets = np.exp(np.linspace(np.log(taus.min()), np.log(taus.max()), n))
    chosen_idx = sorted({int(np.abs(taus - t).argmin()) for t in targets})
    return [expiries[i] for i in chosen_idx]


def load_chain(
    ticker: str,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    **download_kwargs: Any,
) -> ChainSnapshot:
    """Return a chain snapshot, preferring the local cache over the network.

    Unless ``force_refresh`` is set, the most recent cached snapshot for this ticker is
    returned (a download from today counts), and the network is only used when nothing is
    cached. This is what makes the analysis reproducible: rerunning it next month uses the
    same committed snapshot rather than whatever the market looks like at the time. If a
    requested download fails and any cache exists, the cache is used with a warning.

    Args:
        ticker: Underlying symbol.
        cache_dir: Root of the data cache.
        force_refresh: Ignore the cache and download.
        **download_kwargs: Forwarded to :func:`download_chain`.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    candidates = (
        sorted((cache_dir / "raw").glob(f"{ticker.upper()}_*.csv")) if cache_dir.exists() else []
    )
    candidates += sorted((cache_dir / "snapshots").glob(f"{ticker.upper()}_*.csv"))
    candidates = sorted(candidates, key=lambda p: p.stem.split("_")[-1])

    if not force_refresh and candidates:
        return ChainSnapshot.from_csv(candidates[-1])

    try:
        return download_chain(ticker, cache_dir=cache_dir, **download_kwargs)
    except Exception as exc:
        if candidates:
            warnings.warn(
                f"download failed ({exc}); falling back to cached snapshot {candidates[-1].name}",
                stacklevel=2,
            )
            return ChainSnapshot.from_csv(candidates[-1])
        raise


def clean_chain(
    snapshot: ChainSnapshot,
    max_relative_spread: float = 0.40,
    min_price: float = 0.05,
    require_activity: bool = True,
    min_tau: float = 5.0 / 365.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Filter a raw chain down to quotes worth inverting.

    Filters, in order:

    1. two-sided quote present (``bid > 0`` and ``ask > 0``);
    2. not crossed (``ask >= bid``);
    3. mid at least ``min_price`` -- below this the tick size dominates the vol;
    4. relative spread ``(ask-bid)/mid`` at most ``max_relative_spread``;
    5. some sign of life (``volume > 0`` or ``open_interest > 0``) if ``require_activity``;
    6. ``tau >= min_tau``.

    Args:
        snapshot: Raw chain.
        max_relative_spread: Widest full spread-to-mid ratio, ``(ask - bid) / mid``, accepted.
        min_price: Minimum mid price in currency units.
        require_activity: Require non-zero volume or open interest.
        min_tau: Minimum year fraction.

    Returns:
        ``(cleaned_frame, report)``.
    """
    df = snapshot.quotes.copy()
    report = CleaningReport(initial=len(df))

    def drop(mask: pd.Series, name: str) -> pd.DataFrame:
        nonlocal df
        n_before = len(df)
        df = df.loc[mask].copy()
        report.removed[name] = n_before - len(df)
        return df

    drop(
        (df["bid"] > 0) & (df["ask"] > 0) & df["bid"].notna() & df["ask"].notna(), "one-sided quote"
    )
    drop(df["ask"] >= df["bid"], "crossed quote")
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    drop(df["mid"] >= min_price, f"mid < {min_price}")
    df["relative_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    drop(df["relative_spread"] <= max_relative_spread, f"spread > {max_relative_spread:.0%} of mid")
    if require_activity:
        drop((df["volume"] > 0) | (df["open_interest"] > 0), "no volume and no open interest")
    drop(df["tau"] >= min_tau, f"tau < {min_tau * 365:.0f} days")

    report.final = len(df)
    return df.reset_index(drop=True), report


class RateCurve:
    r"""A piecewise log-linear zero curve, quoted as continuously compounded rates.

    Built from Treasury constant-maturity yields (13-week, 5-year, 10-year, 30-year bills
    and notes, which Yahoo publishes as ``^IRX``, ``^FVX``, ``^TNX`` and ``^TYX``). These
    are bond-equivalent yields, not OIS, and they are converted to continuous compounding
    with :math:`r_c = \ln(1 + y)`. For option maturities under two years the difference
    between the Treasury and OIS curves is a few basis points, which moves an implied vol
    by far less than the bid-ask spread.

    Outside the quoted tenors the curve is held flat rather than extrapolated.
    """

    def __init__(self, tenors: Numeric, rates: Numeric) -> None:
        """Args:
        tenors: Maturities in years, strictly increasing.
        rates: Continuously compounded zero rates at those tenors.
        """
        t = np.atleast_1d(np.asarray(tenors, dtype=np.float64))
        r = np.atleast_1d(np.asarray(rates, dtype=np.float64))
        if t.size != r.size or t.size == 0:
            raise ValueError("tenors and rates must be non-empty and the same length")
        order = np.argsort(t)
        self.tenors = t[order]
        self.rates = r[order]

    def rate(self, tau: Numeric) -> np.ndarray:
        """Continuously compounded zero rate at ``tau`` (flat outside the quoted tenors)."""
        x = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        return np.interp(x, self.tenors, self.rates)

    def discount(self, tau: Numeric) -> np.ndarray:
        r"""Discount factor :math:`e^{-r(\tau)\tau}`."""
        x = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        return np.asarray(np.exp(-self.rate(x) * x), dtype=np.float64)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{t:.2f}y={r:.2%}" for t, r in zip(self.tenors, self.rates, strict=True))
        return f"RateCurve({pairs})"


#: Fallback zero curve used when Yahoo is unreachable and nothing is cached.
#: Roughly the US Treasury curve in late 2026; only used to keep offline runs working,
#: and the surface is insensitive to a few tens of basis points here.
FALLBACK_RATES: dict[float, float] = {0.25: 0.040, 5.0: 0.038, 10.0: 0.042, 30.0: 0.046}


def load_rate_curve(
    cache_dir: Path | None = None, force_refresh: bool = False, asof: dt.date | None = None
) -> RateCurve:
    """Load the Treasury zero curve, preferring the local cache.

    Unless ``force_refresh`` is set, a cached curve is used: the one dated ``asof`` if
    given (so an option chain is priced with the rates from the same day), otherwise the
    most recent. Falls back to :data:`FALLBACK_RATES` (with a warning) if Yahoo is
    unreachable and nothing is cached, so an offline run degrades rather than crashes.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    today = dt.datetime.now(tz=_NY).date()
    cache_file = cache_dir / "raw" / f"ratecurve_{today.isoformat()}.csv"
    older = (
        sorted((cache_dir / "raw").glob("ratecurve_*.csv")) if (cache_dir / "raw").exists() else []
    )
    older += sorted((cache_dir / "snapshots").glob("ratecurve_*.csv"))
    older = sorted(older, key=lambda p: p.stem.split("_")[-1])

    if not force_refresh and older:
        dated = [p for p in older if asof is not None and p.stem.endswith(asof.isoformat())]
        if asof is not None and not dated:
            warnings.warn(f"no rate curve cached for {asof}; using {older[-1].name}", stacklevel=2)
        df = pd.read_csv((dated or older)[-1])
        return RateCurve(df["tenor"].to_numpy(), df["rate"].to_numpy())

    symbols = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}
    tenors: list[float] = []
    rates: list[float] = []
    try:
        import yfinance as yf  # noqa: PLC0415

        for symbol, tenor in symbols.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
            if hist.empty:
                continue
            simple = float(hist["Close"].iloc[-1]) / 100.0
            if not np.isfinite(simple) or simple <= -0.5:
                continue
            tenors.append(tenor)
            rates.append(float(np.log1p(simple)))  # bond-equivalent -> continuous
    except Exception as exc:
        warnings.warn(f"rate curve download failed ({exc})", stacklevel=2)

    if not tenors:
        if older:
            df = pd.read_csv(older[-1])
            warnings.warn(f"using cached rate curve {older[-1].name}", stacklevel=2)
            return RateCurve(df["tenor"].to_numpy(), df["rate"].to_numpy())
        warnings.warn("no rate data available; using the hard-coded fallback curve", stacklevel=2)
        tenors = list(FALLBACK_RATES)
        rates = [float(np.log1p(v)) for v in FALLBACK_RATES.values()]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tenor": tenors, "rate": rates}).to_csv(cache_file, index=False)
    return RateCurve(tenors, rates)


def parity_pairs(group: pd.DataFrame, spot: float, moneyness_window: float = 0.10) -> pd.DataFrame:
    """Strikes of one expiry with both a call and a put mid, within the forward window.

    Returns a frame indexed by strike with ``call`` and ``put`` mid columns (empty if the
    expiry lacks either leg). Shared by :func:`implied_forward_curve` and the
    American-adjusted forward so the two always fit the same pairs.
    """
    wide = group.pivot_table(index="strike", columns="option_type", values="mid", aggfunc="mean")
    if not {"call", "put"}.issubset(wide.columns):
        return pd.DataFrame(columns=["call", "put"], dtype=np.float64)
    pairs = wide[["call", "put"]].dropna()
    lo, hi = spot * (1.0 - moneyness_window), spot * (1.0 + moneyness_window)
    return pairs.loc[(pairs.index >= lo) & (pairs.index <= hi)]


def implied_forward_curve(
    quotes: pd.DataFrame,
    spot: float,
    rate_curve: RateCurve | None = None,
    moneyness_window: float = 0.10,
    min_pairs: int = 4,
    method: str = "fixed-rate",
) -> pd.DataFrame:
    r"""Extract the forward (and implied dividend yield) for each expiry from parity.

    Args:
        quotes: Cleaned chain with ``expiry``, ``tau``, ``strike``, ``option_type``, ``mid``.
        spot: Underlying price.
        rate_curve: Discount curve. Required for ``method="fixed-rate"``; ignored by
            ``method="regression"``.
        moneyness_window: Half-width of the strike window as a fraction of spot. Away from
            the money one parity leg is nearly worthless and quote noise dominates.
        min_pairs: Minimum matched call/put strikes needed to fit an expiry.
        method: ``"fixed-rate"`` takes :math:`D` from ``rate_curve`` and solves parity for
            :math:`F` strike by strike, taking the median (robust to one stale pair).
            ``"regression"`` fits :math:`D` and :math:`F` jointly by OLS. The regression is
            included so its instability at short maturities can be shown rather than
            asserted -- see the module docstring.

    Returns:
        One row per expiry with ``expiry, tau, discount, forward, rate, dividend_yield,
        n_pairs, forward_dispersion`` (the interquartile range of the per-strike forward
        estimates, in price units -- a direct read on quote quality) and ``r_squared``
        for the regression method.

    Raises:
        ValueError: on an unknown ``method``, or if ``rate_curve`` is missing when needed.
    """
    if method not in {"fixed-rate", "regression"}:
        raise ValueError(f"unknown method {method!r}; expected 'fixed-rate' or 'regression'")
    if method == "fixed-rate" and rate_curve is None:
        raise ValueError("method='fixed-rate' requires a rate_curve")

    rows: list[dict[str, object]] = []
    for expiry, group in quotes.groupby("expiry", sort=True):
        tau = float(group["tau"].iloc[0])
        pairs = parity_pairs(group, spot, moneyness_window)
        if len(pairs) < min_pairs:
            continue

        strikes = pairs.index.to_numpy(dtype=np.float64)
        y = (pairs["call"] - pairs["put"]).to_numpy(dtype=np.float64)
        row: dict[str, object] = {"expiry": expiry, "tau": tau, "n_pairs": len(pairs)}

        if method == "fixed-rate":
            assert rate_curve is not None
            discount = float(rate_curve.discount(tau)[0])
            per_strike_forward = strikes + y / discount
            forward = float(np.median(per_strike_forward))
            q75, q25 = np.percentile(per_strike_forward, [75, 25])
            row["forward_dispersion"] = float(q75 - q25)
            row["r_squared"] = np.nan
        else:
            design = np.column_stack([np.ones_like(strikes), strikes])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            intercept, slope = float(coef[0]), float(coef[1])
            if slope >= 0.0:  # pragma: no cover - parity cannot slope up in K
                continue
            discount = -slope
            forward = intercept / discount
            resid = y - design @ coef
            ss_tot = float(((y - y.mean()) ** 2).sum())
            row["r_squared"] = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
            row["forward_dispersion"] = np.nan

        rate = -np.log(discount) / tau if tau > 0 and discount > 0 else np.nan
        div_yield = rate - np.log(forward / spot) / tau if tau > 0 and forward > 0 else np.nan
        row.update(
            {
                "discount": discount,
                "forward": forward,
                "rate": float(rate),
                "dividend_yield": float(div_yield),
            }
        )
        rows.append(row)

    columns = [
        "expiry",
        "tau",
        "discount",
        "forward",
        "rate",
        "dividend_yield",
        "n_pairs",
        "forward_dispersion",
        "r_squared",
    ]
    out = pd.DataFrame(rows)
    return out.reindex(columns=columns) if not out.empty else pd.DataFrame(columns=columns)
