"""optpricing: a compact but complete European/American option pricing engine.

Modules
-------
``blackscholes``   Analytic Black-Scholes-Merton price and Greeks (fully vectorised).
``binomial``       Cox-Ross-Rubinstein / Jarrow-Rudd trees for European and American options.
``american``       Early-exercise premia and de-Americanising American quotes.
``montecarlo``     Monte Carlo pricing with antithetic and control variates.
``implied_vol``    Newton-Raphson implied volatility with a bracketed bisection fallback.
``heston``         Heston (1993) stochastic volatility pricing via the characteristic function.
``calibration``    Least-squares calibration of Heston to a quoted implied-volatility surface.
``data``           Cached yfinance option-chain download and cleaning.
``surface``        Implied-volatility surface construction from a cleaned chain.
``plotting``       Charts for the smile, the term structure and the 3-D surface.
"""

from __future__ import annotations

from . import american, binomial, blackscholes, data, heston, implied_vol, montecarlo, surface
from .types import ExerciseStyle, FloatArray, OptionType

__all__ = [
    "ExerciseStyle",
    "FloatArray",
    "OptionType",
    "american",
    "binomial",
    "blackscholes",
    "data",
    "heston",
    "implied_vol",
    "montecarlo",
    "surface",
]

__version__ = "1.0.0"
