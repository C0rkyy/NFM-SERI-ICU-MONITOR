"""
Shared dashboard utilities.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safely cast *value* to float, returning *default* on failure.

    Handles ``None``, non-numeric strings, ``TypeError``, ``ValueError``,
    and ``NaN`` / ``inf`` without raising.
    """
    if value is None:
        return default
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(value_f):
        return default
    return value_f
