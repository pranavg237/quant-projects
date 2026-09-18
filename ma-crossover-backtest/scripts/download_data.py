"""Populate the data cache for every symbol used by the research scripts."""

from __future__ import annotations

import sys

from quantbt.data import load_yahoo
from quantbt.research.universes import ALL_SYMBOLS


def main() -> None:
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
