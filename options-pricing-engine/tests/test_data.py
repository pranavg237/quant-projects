"""Chain download, caching, cleaning and forward extraction.

The network is never touched: everything runs against the synthetic chain from
``conftest`` or the committed SPY snapshot.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from conftest import TRUE_DIVIDEND, TRUE_RATE, TRUE_SPOT
from optpricing import data as data_mod

NY = ZoneInfo("America/New_York")


def test_year_fraction_uses_the_1600_et_close() -> None:
    asof = dt.datetime(2026, 9, 18, 16, 0, tzinfo=NY)
    assert data_mod.year_fraction(asof, dt.date(2026, 9, 19)) == pytest.approx(1 / 365)
    assert data_mod.year_fraction(asof, dt.date(2026, 9, 18)) == 0.0
    # Same-day morning: a fraction of one day, not zero and not a whole day.
    morning = dt.datetime(2026, 9, 18, 10, 0, tzinfo=NY)
    frac = data_mod.year_fraction(morning, dt.date(2026, 9, 18))
    assert 0.0 < frac < 1 / 365
    assert frac == pytest.approx(6 / 24 / 365)
    # Already expired.
    assert data_mod.year_fraction(asof, dt.date(2026, 9, 17)) == 0.0
    # A naive timestamp is read as New York time.
    assert data_mod.year_fraction(
        dt.datetime(2026, 9, 18, 16, 0), dt.date(2026, 9, 19)
    ) == pytest.approx(1 / 365)


def test_naive_day_count_would_be_materially_wrong() -> None:
    """The reason the 16:00 convention matters: it is a 50% error on a one-day option."""
    morning = dt.datetime(2026, 9, 18, 10, 0, tzinfo=NY)
    precise = data_mod.year_fraction(morning, dt.date(2026, 9, 19))
    naive = 1.0 / 365.0
    assert abs(precise - naive) / naive > 0.2


def test_rate_curve_interpolates_and_holds_flat_outside() -> None:
    curve = data_mod.RateCurve([0.25, 5.0, 10.0], [0.04, 0.045, 0.05])
    assert float(curve.rate(0.25)[0]) == pytest.approx(0.04)
    assert float(curve.rate(2.625)[0]) == pytest.approx(0.0425)
    assert float(curve.rate(0.01)[0]) == pytest.approx(0.04)  # flat below the first tenor
    assert float(curve.rate(50.0)[0]) == pytest.approx(0.05)  # and above the last
    assert float(curve.discount(2.0)[0]) == pytest.approx(np.exp(-float(curve.rate(2.0)[0]) * 2.0))
    assert "RateCurve" in repr(curve)
    # The constructor sorts, so input order does not matter.
    shuffled = data_mod.RateCurve([10.0, 0.25, 5.0], [0.05, 0.04, 0.045])
    assert np.allclose(shuffled.rate([0.1, 1.0, 7.0, 99.0]), curve.rate([0.1, 1.0, 7.0, 99.0]))


def test_rate_curve_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        data_mod.RateCurve([], [])
    with pytest.raises(ValueError, match="same length"):
        data_mod.RateCurve([1.0, 2.0], [0.04])


def test_snapshot_csv_round_trip(
    synthetic_snapshot: data_mod.ChainSnapshot, tmp_path: Path
) -> None:
    path = tmp_path / "snap.csv"
    synthetic_snapshot.to_csv(path)
    restored = data_mod.ChainSnapshot.from_csv(path)
    assert restored.ticker == synthetic_snapshot.ticker
    assert restored.spot == pytest.approx(synthetic_snapshot.spot)
    assert restored.asof == synthetic_snapshot.asof
    assert restored.expiries() == synthetic_snapshot.expiries()
    pd.testing.assert_series_equal(
        restored.quotes["mid"], synthetic_snapshot.quotes["mid"], check_exact=False
    )


def test_cleaning_keeps_a_perfect_chain(synthetic_snapshot: data_mod.ChainSnapshot) -> None:
    clean, report = data_mod.clean_chain(synthetic_snapshot)
    assert report.initial == len(synthetic_snapshot.quotes)
    assert report.final == len(clean)
    # The synthetic chain is deliberately spotless apart from sub-tick deep wings.
    assert report.final / report.initial > 0.85
    assert (clean["bid"] > 0).all()
    assert (clean["ask"] >= clean["bid"]).all()
    assert "relative_spread" in clean.columns


def test_each_cleaning_filter_fires(synthetic_snapshot: data_mod.ChainSnapshot) -> None:
    """Corrupt one row per rule and check the report attributes the loss correctly."""
    dirty = synthetic_snapshot.quotes.copy()
    dirty.loc[0, "bid"] = 0.0  # one-sided
    dirty.loc[1, ["bid", "ask"]] = [5.0, 4.0]  # crossed
    dirty.loc[2, ["bid", "ask"]] = [0.01, 0.02]  # sub-min-price
    dirty.loc[3, ["bid", "ask"]] = [1.0, 4.0]  # 120% spread
    dirty.loc[4, ["volume", "open_interest"]] = [0.0, 0.0]  # inactive
    snapshot = data_mod.ChainSnapshot(
        synthetic_snapshot.ticker, synthetic_snapshot.asof, synthetic_snapshot.spot, dirty
    )
    _, report = data_mod.clean_chain(snapshot)
    assert report.removed["one-sided quote"] >= 1
    assert report.removed["crossed quote"] >= 1
    assert report.removed["mid < 0.05"] >= 1
    assert report.removed["spread > 40% of mid"] >= 1
    assert report.removed["no volume and no open interest"] >= 1

    frame = report.to_frame()
    assert frame["rows_remaining"].iloc[-1] == report.final
    assert "Cleaning" in str(report)


def test_min_tau_filter() -> None:
    asof = dt.datetime(2026, 9, 18, 10, 0, tzinfo=NY)
    quotes = pd.DataFrame(
        {
            "expiry": [dt.date(2026, 9, 19)] * 2,
            "option_type": ["call", "put"],
            "strike": [100.0, 100.0],
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
            "volume": [10.0, 10.0],
            "open_interest": [10.0, 10.0],
            "tau": [1 / 365, 1 / 365],
            "mid": [1.05, 1.05],
        }
    )
    snapshot = data_mod.ChainSnapshot("X", asof, 100.0, quotes)
    clean, report = data_mod.clean_chain(snapshot)
    assert len(clean) == 0
    assert report.removed["tau < 5 days"] == 2


def test_fixed_rate_parity_recovers_the_forward_exactly(
    synthetic_snapshot: data_mod.ChainSnapshot, flat_rate_curve: data_mod.RateCurve
) -> None:
    """The headline correctness test for the forward extraction."""
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    curve = data_mod.implied_forward_curve(clean, synthetic_snapshot.spot, flat_rate_curve)
    assert len(curve) == 5
    expected = TRUE_SPOT * np.exp((TRUE_RATE - TRUE_DIVIDEND) * curve["tau"])
    assert np.max(np.abs(curve["forward"] - expected)) < 1e-6
    assert np.allclose(curve["rate"], TRUE_RATE, atol=1e-9)
    assert np.allclose(curve["dividend_yield"], TRUE_DIVIDEND, atol=1e-8)
    # Perfect quotes => zero dispersion between per-strike forward estimates.
    assert float(curve["forward_dispersion"].max()) < 1e-6


def test_regression_method_also_recovers_the_forward_on_perfect_quotes(
    synthetic_snapshot: data_mod.ChainSnapshot,
) -> None:
    """With noiseless mids the joint regression is fine -- its problem is conditioning."""
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    curve = data_mod.implied_forward_curve(clean, synthetic_snapshot.spot, method="regression")
    expected = TRUE_SPOT * np.exp((TRUE_RATE - TRUE_DIVIDEND) * curve["tau"])
    assert np.max(np.abs(curve["forward"] - expected)) < 1e-4
    assert np.allclose(curve["rate"], TRUE_RATE, atol=1e-5)
    assert float(curve["r_squared"].min()) > 0.9999


def test_regression_method_is_ill_conditioned_under_realistic_quote_noise(
    synthetic_snapshot: data_mod.ChainSnapshot, flat_rate_curve: data_mod.RateCurve
) -> None:
    """Round the mids to the $0.01 tick -- the regression's implied *rate* falls apart.

    Note what does and does not break. Both methods recover the **forward** to within a
    couple of basis points; on Gaussian noise the least-squares fit is if anything
    slightly better than the median estimator. What collapses is the **discount factor**
    read off the slope, and it collapses like ``1/tau``. That is why the default takes the
    discount factor from the Treasury curve and asks parity only for the forward.
    """
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    noisy = clean.copy()
    rng = np.random.default_rng(0)
    noisy["mid"] = np.round(noisy["mid"] + rng.normal(0.0, 0.01, len(noisy)), 2)

    regression = data_mod.implied_forward_curve(noisy, TRUE_SPOT, method="regression")
    fixed = data_mod.implied_forward_curve(noisy, TRUE_SPOT, flat_rate_curve)

    def forward_error(curve: pd.DataFrame) -> float:
        expected = TRUE_SPOT * np.exp((TRUE_RATE - TRUE_DIVIDEND) * curve["tau"])
        return float(np.max(np.abs(curve["forward"] - expected)))

    rate_error = np.abs(regression["rate"].to_numpy() - TRUE_RATE)
    # One cent of noise is already tens of basis points of rate error at two weeks...
    assert rate_error[0] > 0.002
    # ...and it scales like 1/tau, so the front expiry is by far the worst. That blow-up
    # is the whole argument: a conditioning problem, not a bias.
    assert rate_error[0] > 5.0 * rate_error[-1]
    # Meanwhile both methods pin the forward to a couple of basis points of spot.
    assert forward_error(fixed) < 0.001 * TRUE_SPOT
    assert forward_error(regression) < 0.001 * TRUE_SPOT


def test_regression_implies_absurd_rates_on_real_quotes(
    cached_spy_snapshot: data_mod.ChainSnapshot,
) -> None:
    """On live SPY quotes the joint regression returns rates no market ever traded.

    The synthetic test above shows the mechanism with clean Gaussian noise; this one shows
    what it does to real data, which also carries stale quotes and non-simultaneous marks.
    Implied rates come out at tens of percent, positive and negative, while the fit's
    R-squared stays above 0.998 -- the line is beautiful and its slope is meaningless.
    """
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    regression = data_mod.implied_forward_curve(
        clean, cached_spy_snapshot.spot, method="regression"
    )
    front = regression.loc[regression["tau"] < 0.25]
    assert len(front) >= 4
    assert float(front["r_squared"].min()) > 0.99  # the fit looks excellent...
    assert float(front["rate"].min()) < -0.02  # ...and implies materially negative rates
    assert float(front["rate"].max()) - float(front["rate"].min()) > 0.10  # wildly unstable

    fixed = data_mod.implied_forward_curve(
        clean, cached_spy_snapshot.spot, data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    )
    # The default method cannot do this: its rates are the curve's by construction.
    assert float(fixed["rate"].min()) > 0.0
    assert float(fixed["rate"].max()) < 0.10


def test_forward_curve_argument_validation(synthetic_snapshot: data_mod.ChainSnapshot) -> None:
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    with pytest.raises(ValueError, match="unknown method"):
        data_mod.implied_forward_curve(clean, TRUE_SPOT, method="magic")
    with pytest.raises(ValueError, match="requires a rate_curve"):
        data_mod.implied_forward_curve(clean, TRUE_SPOT, method="fixed-rate")


def test_forward_curve_skips_expiries_without_enough_pairs(
    flat_rate_curve: data_mod.RateCurve,
) -> None:
    quotes = pd.DataFrame(
        {
            "expiry": [dt.date(2026, 12, 18)] * 2,
            "tau": [0.25, 0.25],
            "strike": [100.0, 100.0],
            "option_type": ["call", "put"],
            "mid": [5.0, 4.0],
        }
    )
    assert data_mod.implied_forward_curve(quotes, 100.0, flat_rate_curve).empty
    # Calls only: no pairs at all.
    calls_only = quotes.loc[quotes["option_type"] == "call"]
    assert data_mod.implied_forward_curve(calls_only, 100.0, flat_rate_curve).empty


def test_real_spy_snapshot_is_usable(cached_spy_snapshot: data_mod.ChainSnapshot) -> None:
    """Regression guard on the committed real-market snapshot."""
    clean, report = data_mod.clean_chain(cached_spy_snapshot)
    assert report.initial > 1000
    assert 0.5 < report.final / report.initial < 1.0
    curve = data_mod.implied_forward_curve(
        clean, cached_spy_snapshot.spot, data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    )
    assert len(curve) >= 8
    # Forwards rise with maturity (SPY's dividend yield is below the risk-free rate).
    assert curve["forward"].is_monotonic_increasing
    assert (curve["forward"] > cached_spy_snapshot.spot * 0.99).all()


def test_load_chain_prefers_the_cache(
    synthetic_snapshot: data_mod.ChainSnapshot, tmp_path: Path
) -> None:
    """A cache entry dated today must be used without any network call."""
    today = dt.datetime.now(tz=NY).date()
    snapshot = data_mod.ChainSnapshot(
        "CACHED", dt.datetime.now(tz=NY), 123.0, synthetic_snapshot.quotes
    )
    (tmp_path / "raw").mkdir(parents=True)
    snapshot.to_csv(tmp_path / "raw" / f"CACHED_{today.isoformat()}.csv")
    loaded = data_mod.load_chain("CACHED", cache_dir=tmp_path)
    assert loaded.spot == pytest.approx(123.0)


def test_load_chain_raises_when_offline_with_no_cache(tmp_path: Path) -> None:
    with pytest.raises(Exception):  # noqa: B017 - yfinance raises its own types
        data_mod.load_chain("DEFINITELYNOTATICKER", cache_dir=tmp_path, force_refresh=True)
