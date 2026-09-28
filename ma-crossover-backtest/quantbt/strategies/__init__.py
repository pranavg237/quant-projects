"""Strategies implemented on the framework."""

from quantbt.strategies.buy_and_hold import BuyAndHold
from quantbt.strategies.ma_crossover import MACrossover
from quantbt.strategies.mean_reversion import MeanReversion
from quantbt.strategies.pairs import PairsTrading, fit_pair
from quantbt.strategies.tsmom import TimeSeriesMomentum
from quantbt.strategies.xsmom import CrossSectionalMomentum

__all__ = [
    "BuyAndHold",
    "CrossSectionalMomentum",
    "MACrossover",
    "MeanReversion",
    "PairsTrading",
    "TimeSeriesMomentum",
    "fit_pair",
]
