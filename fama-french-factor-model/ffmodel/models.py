"""Factor model specifications."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Canonical factor order, used when laying out tables across models.
ALL_FACTORS: Tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM")


@dataclass(frozen=True)
class FactorModel:
    name: str
    label: str
    factors: Tuple[str, ...]
    # Which French factor file supplies the base factors. SMB differs between
    # the 3-factor file and the 5-factor (2x3) file, so each model uses its own.
    source: str


MODELS: Dict[str, FactorModel] = {
    m.name: m
    for m in (
        FactorModel("capm", "CAPM", ("Mkt-RF",), "ff3"),
        FactorModel("ff3", "Fama-French 3-factor", ("Mkt-RF", "SMB", "HML"), "ff3"),
        FactorModel("carhart", "Carhart 4-factor", ("Mkt-RF", "SMB", "HML", "MOM"), "ff3"),
        FactorModel("ff5", "Fama-French 5-factor", ("Mkt-RF", "SMB", "HML", "RMW", "CMA"), "ff5"),
        FactorModel("ff6", "Fama-French 6-factor", ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"), "ff5"),
    )
}


def get_model(name: str) -> FactorModel:
    try:
        return MODELS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}") from None
