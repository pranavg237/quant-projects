"""quantbt: an event-driven backtesting framework with bias safeguards.

The package grew out of a 60-line moving-average crossover script. Its design rules:

* a signal computed from bar ``t`` can only be filled at bar ``t + 1`` or later,
* every fill pays commission and slippage,
* P&L uses total-return prices, level-based signals use split-adjusted prices,
* every index is a tz-naive ``DatetimeIndex`` of exchange dates,
* universes are point-in-time so delisted names are not silently dropped.
"""

from __future__ import annotations

from quantbt.data import PriceData, load_csv, load_yahoo
from quantbt.metrics import summary

__all__ = ["PriceData", "load_csv", "load_yahoo", "summary"]
__version__ = "0.2.0"
