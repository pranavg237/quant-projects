"""Build an HTML tearsheet for every strategy under reports/strategies/.

Reads the CSVs that scripts/run_strategies.py wrote, so it is fast and offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantbt.report import TearsheetInputs, write_tearsheet

COSTS = "fills at the next open, 5 bps slippage, 1 bp commission"


def _read_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame.iloc[:, 0]


def _read_frame(path: Path, index_col: int | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def build_one(directory: Path, out_dir: Path) -> Path:
    spec = json.loads((directory / "spec.json").read_text())
    returns = _read_series(directory / "oos_returns.csv")
    bench = _read_series(directory / "benchmark_returns.csv")
    rf = _read_series(directory / "rf.csv")
    weights = pd.read_csv(directory / "oos_weights.csv", index_col=0, parse_dates=True)
    factors = _read_frame(directory / "factors.csv", index_col=0)
    overfit_frame = _read_frame(directory / "overfit_oos.csv", index_col=0)
    overfit = overfit_frame.iloc[:, 0] if overfit_frame is not None else None
    folds = _read_frame(directory / "folds.csv")
    summary_path = directory / "summary.json"
    caveats = list(spec.get("caveats", []))
    if summary_path.exists():
        row = json.loads(summary_path.read_text())
        caveats.append(
            f"Parameters were re-chosen every {spec['test_years']:g} year(s) on the "
            f"preceding {spec['train_years']:g} years ({int(row['folds'])} folds)."
        )

    inputs = TearsheetInputs(
        name=spec["name"],
        description=spec.get("description", ""),
        returns=returns,
        benchmark_returns=bench,
        benchmark_name=spec.get("benchmark", "benchmark"),
        rf=rf,
        weights=weights,
        factors=factors,
        overfit=overfit,
        folds=folds,
        caveats=tuple(caveats),
        spec=spec,
        costs=COSTS,
    )
    return write_tearsheet(inputs, out_dir / f"{spec['name']}.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/strategies")
    parser.add_argument("--out", default="reports/tearsheets")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    directories = sorted(d for d in root.iterdir() if (d / "spec.json").exists())
    if not directories:
        raise SystemExit(f"no strategy output under {root}; run scripts/run_strategies.py first")
    written = [build_one(d, out_dir) for d in directories]
    index = _write_index(written, out_dir, root)
    for path in written:
        print(f"wrote {path}")
    print(f"wrote {index}")


def _write_index(written: list[Path], out_dir: Path, root: Path) -> Path:
    results = root / "results.csv"
    rows = ""
    if results.exists():
        table = pd.read_csv(results).set_index("strategy")
        for name in table.index:
            row = table.loc[name]
            rows += (
                f"<tr><td><a href='{name}.html'>{name}</a></td>"
                f"<td>{row['sharpe']:.2f}</td><td>{row['bench_sharpe']:.2f}</td>"
                f"<td>{row['cagr'] * 100:.1f}%</td><td>{row['max_drawdown'] * 100:.1f}%</td>"
                f"<td>{row['ff5_alpha'] * 100:+.1f}% (t={row['ff5_alpha_t']:.2f})</td></tr>"
            )
    html = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>quantbt tearsheets</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
         padding: 40px 16px; background: light-dark(#f9f9f7, #0d0d0d);
         color: light-dark(#0b0b0b, #ffffff); }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }}
  th, td {{ text-align: right; padding: 8px 10px;
            border-bottom: 1px solid light-dark(#e1e0d9, #2c2c2a);
            font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: light-dark(#898781, #898781); font-weight: 500; }}
  a {{ color: light-dark(#2a78d6, #3987e5); }}
  p {{ color: light-dark(#52514e, #c3c2b7); }}
</style></head>
<body><div class='wrap'>
<h1>Strategy tearsheets</h1>
<p>Walk-forward out-of-sample results. Parameters were chosen on training windows only;
every number is net of commission and slippage.</p>
<table><thead><tr><th>Strategy</th><th>Sharpe</th><th>Benchmark Sharpe</th><th>CAGR</th>
<th>Max drawdown</th><th>FF5 alpha</th></tr></thead><tbody>{rows}</tbody></table>
</div></body></html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
