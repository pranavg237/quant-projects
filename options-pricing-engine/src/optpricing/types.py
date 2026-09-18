"""Shared types, enums and small numerical helpers used across the pricing engine."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""Canonical float array type. Every public pricer returns one of these."""

Numeric: TypeAlias = float | int | npt.ArrayLike
"""Anything a caller may pass in for a market parameter."""


class OptionType(StrEnum):
    """Vanilla option right.

    A :class:`~enum.StrEnum`, so ``OptionType.CALL == "call"`` is ``True`` and callers can
    pass plain strings anywhere an ``OptionType`` is accepted.
    """

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        """+1 for a call, -1 for a put.

        Lets most Black-Scholes formulae be written once instead of twice: a vanilla
        payoff is ``max(phi * (S - K), 0)`` with ``phi`` this sign.
        """
        return 1.0 if self is OptionType.CALL else -1.0


class ExerciseStyle(StrEnum):
    """When the holder may exercise."""

    EUROPEAN = "european"
    AMERICAN = "american"


def as_array(x: Numeric) -> FloatArray:
    """Coerce a scalar or array-like to a float64 ndarray (0-d for scalars)."""
    return np.asarray(x, dtype=np.float64)


def to_option_type(value: OptionType | str) -> OptionType:
    """Normalise a string or enum member to an :class:`OptionType`.

    Raises:
        ValueError: if ``value`` is not one of ``"call"`` / ``"put"`` (case-insensitive).
    """
    if isinstance(value, OptionType):
        return value
    try:
        return OptionType(str(value).lower())
    except ValueError as exc:  # pragma: no cover - message formatting only
        raise ValueError(f"unknown option type {value!r}; expected 'call' or 'put'") from exc


def to_float(value: Any) -> float:
    """Coerce a value to ``float``.

    Exists because ``pandas`` types a ``groupby`` key as ``Hashable``, so a bare
    ``float(key)`` fails type checking even where the key is obviously numeric.
    """
    return float(cast(float, value))


def to_exercise_style(value: ExerciseStyle | str) -> ExerciseStyle:
    """Normalise a string or enum member to an :class:`ExerciseStyle`."""
    if isinstance(value, ExerciseStyle):
        return value
    try:
        return ExerciseStyle(str(value).lower())
    except ValueError as exc:  # pragma: no cover - message formatting only
        raise ValueError(
            f"unknown exercise style {value!r}; expected 'european' or 'american'"
        ) from exc
