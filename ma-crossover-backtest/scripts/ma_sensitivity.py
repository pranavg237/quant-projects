"""Sharpe surface of the SPY MA crossover over a dense fast/slow grid (a robustness check).

Every cell is the full-sample (2005-01-03 to 2025-08-29) excess Sharpe, net of 5 bp
slippage and 1 bp commission, with cash earning the Ken French daily T-bill rate. The
window is the same one the walk-forward's out-of-sample series and in-sample-best
comparison cover in RESULTS.md.

THIS SURFACE IS IN-SAMPLE. It is a diagnostic of how sensitive the result is to the
parameters, and nothing is selected from it: the walk-forward in run_strategies.py is the
only place parameters are chosen. Picking the brightest cell here and reporting it would
be exactly the overfitting the rest of the project is built to avoid.

Writes reports/sensitivity/ma_sharpe_grid.csv (the numbers behind the figure),
ma_sharpe_points.csv (plateau statistics for the points of interest) and
ma_sharpe_heatmap.png.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from quantbt import metrics
from quantbt.data import load_yahoo
from quantbt.research.runner import load_rf
from quantbt.research.sensitivity import ma_sharpe_grid, surface_summary
from quantbt.validation import sharpe_std_error
from quantbt.vectorized import backtest_ma_crossover

FAST = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150]
SLOW = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
WF_DIR = Path("reports/strategies/ma_crossover")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--out", default="reports/sensitivity")
    args = parser.parse_args()
    t0 = time.time()

    data_start = "1998-01-01"  # enough warm-up for a 300-day MA before 2005
    data = load_yahoo("SPY", start=data_start, end=args.end)
    rf = load_rf(data_start, args.end)
    table = ma_sharpe_grid(data, FAST, SLOW, start=args.start, end=args.end, rf=rf)

    bench = data.single("close").pct_change().loc[args.start : args.end].copy()
    bench.iloc[0] = 0.0
    bench_sharpe = metrics.sharpe(bench, rf=rf.reindex(bench.index).fillna(0.0))

    # Cross-check against the event-driven engine's full-sample grid for the 17 points
    # the walk-forward searches (reports/strategies/ma_crossover/grid.csv). On identical
    # data the two implementations agree to ~1e-15; the committed grid.csv was built from
    # an earlier download of the same dates, and Yahoo's re-adjustments move Sharpe in
    # the 6th decimal, hence the tolerance.
    engine = pd.read_csv(WF_DIR / "grid.csv").rename(columns={"short": "fast", "long": "slow"})
    merged = engine.merge(table, on=["fast", "slow"], suffixes=("_engine", "_vec"))
    diff = float((merged["sharpe_engine"] - merged["sharpe_vec"]).abs().max())
    print(f"engine vs vectorised Sharpe on the {len(merged)} shared points: max |diff| {diff:.1e}")
    if len(merged) != len(engine) or diff > 1e-5:
        raise SystemExit("vectorised grid does not reproduce the engine's grid.csv")

    folds = pd.read_csv(WF_DIR / "folds.csv")
    chosen = Counter(zip(folds["short"], folds["long"], strict=True))
    is_best = engine.iloc[int(engine["sharpe"].to_numpy().argmax())]
    grid_best = table.iloc[int(table["sharpe"].to_numpy().argmax())]
    points = {
        (int(is_best["fast"]), int(is_best["slow"])): "in-sample best of the 17-point grid",
        (50, 200): "the textbook 50/200",
        (int(grid_best["fast"]), int(grid_best["slow"])): "best cell of this dense grid",
    }
    for (f, s), n in chosen.most_common():
        points.setdefault((int(f), int(s)), f"chosen by walk-forward in {n} folds")
    summary = surface_summary(table, bench_sharpe, points)
    summary.insert(0, "label", list(points.values()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ma_sharpe_grid.csv", index=False, float_format="%.6f")
    summary.to_csv(out / "ma_sharpe_points.csv", index=False, float_format="%.6f")

    s = table["sharpe"]
    pd.set_option("display.width", 200)
    print(f"SPY MA crossover, {args.start} to {args.end}: {len(table)} valid cells")
    print(f"buy-and-hold excess Sharpe over the same dates: {bench_sharpe:.3f}")
    print(
        f"cell Sharpe: min {s.min():.3f}, median {s.median():.3f}, max {s.max():.3f}; "
        f"{(s > bench_sharpe).sum()} of {len(s)} cells beat buy-and-hold"
    )
    # How big is the spread across cells compared with the noise in any one of them?
    one = backtest_ma_crossover(
        data, int(is_best["fast"]), int(is_best["slow"]), start=args.start, end=args.end, rf=rf
    )
    excess = one.returns - rf.reindex(one.returns.index).fillna(0.0)
    se = sharpe_std_error(excess) * np.sqrt(metrics.periods_per_year(excess.index))
    print(
        f"standard error of one cell's Sharpe: {se:.3f}; spread max - min = "
        f"{s.max() - s.min():.3f} ({(s.max() - s.min()) / se:.1f} standard errors), "
        f"interquartile range {s.quantile(0.75) - s.quantile(0.25):.3f}"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    _plot(table, bench_sharpe, chosen, args, out / "ma_sharpe_heatmap.png")
    print(f"written to {out}/ in {time.time() - t0:.1f}s")


def _plot(
    table: pd.DataFrame,
    bench_sharpe: float,
    chosen: Counter[tuple[int, int]],
    args: argparse.Namespace,
    path: Path,
) -> None:
    pivot = table.pivot(index="fast", columns="slow", values="sharpe").reindex(
        index=FAST, columns=SLOW
    )
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    span = max(abs(finite.max() - bench_sharpe), abs(finite.min() - bench_sharpe))
    norm = TwoSlopeNorm(vcenter=bench_sharpe, vmin=bench_sharpe - span, vmax=bench_sharpe + span)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f2f2f2")

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(SLOW)), [str(s) for s in SLOW])
    ax.set_yticks(range(len(FAST)), [str(f) for f in FAST])
    ax.set_xlabel("Slow moving average (days)")
    ax.set_ylabel("Fast moving average (days)")
    for i in range(len(FAST)):
        for j in range(len(SLOW)):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            dark = abs(norm(v) - 0.5) > 0.3
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if dark else "#222222",
            )  # fmt: skip
            n = chosen.get((FAST[i], SLOW[j]), 0)
            if n:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.0, ec="#111111"))
                ax.text(j + 0.45, i - 0.42, f"{n}", ha="right", va="top", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"Excess Sharpe, net of costs (white = SPY buy-and-hold, {bench_sharpe:.2f})")
    ax.set_title(
        f"SPY MA crossover: Sharpe by fast/slow window, {args.start[:4]}-{args.end[:4]}\n"
        "IN-SAMPLE robustness diagnostic, not a parameter selection\n"
        "Outlined: parameters the walk-forward chose (corner number = folds)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
