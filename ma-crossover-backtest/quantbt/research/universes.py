"""Symbol lists used by the research scripts.

Every list here is a *current* list of instruments that exist today, chosen by hand.
That is selection bias by construction and, for the stock list, survivorship bias too:
nothing that was delisted, acquired or went bankrupt between 2000 and today is here.
The framework's point-in-time universe lets a name enter when its data starts, which
removes look-ahead on listing dates but does nothing for the missing failures. Treat
every cross-sectional stock result as an upper bound. See README.md.
"""

from __future__ import annotations

ASSET_CLASS_ETFS: list[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "VNQ",  # equity / real estate
    "TLT",
    "IEF",
    "LQD",
    "HYG",  # rates / credit
    "GLD",
    "SLV",
    "DBC",  # commodities
]

SECTOR_ETFS: list[str] = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

LARGE_CAP_STOCKS: list[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "UNH",
    "HD",
    "MA",
    "DIS",
    "BAC",
    "XOM",
    "CVX",
    "PFE",
    "KO",
    "PEP",
    "WMT",
    "CSCO",
    "INTC",
    "MRK",
    "ABT",
    "T",
    "VZ",
    "ORCL",
    "IBM",
    "MCD",
    "NKE",
    "HON",
    "MMM",
    "CAT",
    "BA",
    "GE",
    "GS",
    "MS",
    "C",
    "WFC",
    "AXP",
    "LOW",
    "TGT",
    "COST",
    "AMGN",
    "GILD",
    "MDT",
    "LLY",
    "BMY",
    "QCOM",
    "TXN",
    "ADBE",
    "CRM",
    "NFLX",
    "SBUX",
    "UPS",
    "UNP",
    "LMT",
    "RTX",
    "DE",
    "CVS",
    "MO",
    "PM",
    "DUK",
    "SO",
    "NEE",
    "AMT",
    "SPG",
]

PAIR_CANDIDATES: list[tuple[str, str]] = [
    ("XOM", "CVX"),
    ("KO", "PEP"),
    ("WMT", "TGT"),
    ("HD", "LOW"),
    ("JPM", "BAC"),
    ("V", "MA"),
    ("MS", "GS"),
    ("UPS", "FDX"),
    ("VZ", "T"),
    ("PFE", "MRK"),
    ("DUK", "SO"),
    ("EWA", "EWC"),
    ("XLE", "XOP"),
    ("GDX", "GDXJ"),
    ("IEF", "TLT"),
    ("XLU", "VPU"),
    ("XLF", "KBE"),
]

# ("CVS", "WBA") was in the first draft of this list. Walgreens was taken private in
# 2025 and Yahoo stopped serving its history, so the pair had to be dropped -- a live
# demonstration of why a universe built from today's tradable names is biased: the
# failures are not merely unprofitable in the sample, they are absent from it.
PAIR_SYMBOLS: list[str] = sorted({s for pair in PAIR_CANDIDATES for s in pair})

ALL_SYMBOLS: list[str] = sorted(
    set(ASSET_CLASS_ETFS) | set(SECTOR_ETFS) | set(LARGE_CAP_STOCKS) | set(PAIR_SYMBOLS)
)
