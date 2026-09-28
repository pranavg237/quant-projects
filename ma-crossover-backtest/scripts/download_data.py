"""Download every symbol the research scripts use into data/cache/ (always live).

Then run scripts/build_snapshot.py to freeze the cache into a new committed snapshot.
"""

from __future__ import annotations

import sys

from quantbt.data import load_yahoo, use_live_data
from quantbt.research.universes import ALL_SYMBOLS


def main() -> None:
    use_live_data(True)
    failed: list[str] = []
    for symbol in ALL_SYMBOLS:
        try:
            data = load_yahoo(symbol)
            print(
                f"{symbol:6s} {data.index[0].date()} .. {data.index[-1].date()} ({len(data)} bars)"
            )
        except Exception as exc:
            failed.append(symbol)
            print(f"{symbol:6s} FAILED: {exc}", file=sys.stderr)
    if failed:
        print(f"failed: {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
