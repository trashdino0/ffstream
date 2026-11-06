"""Fantasy scoring for kickers and team defenses.

The nflverse player-week table does not include kicking or team-defense fantasy
points in its standard ``fantasy_points`` column, so we compute both from the
underlying stats. Every rule is configurable in ``config/scoring.json``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def _tier_points(values: pd.Series, tiers: Iterable[Iterable[float]]) -> pd.Series:
    """Map a continuous value to points using inclusive [min, max, points] tiers."""
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    for low, high, points in tiers:
        mask = values.between(float(low), float(high), inclusive="both")
        out.loc[mask] = float(points)
    return out


def add_kicker_points(df: pd.DataFrame, scoring: dict[str, Any]) -> pd.Series:
    """Return kicker fantasy points for each row of a player-week frame."""
    cfg = scoring["kicker"]

    fg_0_39 = (
        _num(df, "fg_made_0_19") + _num(df, "fg_made_20_29") + _num(df, "fg_made_30_39")
    )
    fg_40_49 = _num(df, "fg_made_40_49")
    fg_50_59 = _num(df, "fg_made_50_59")
    fg_60 = _num(df, "fg_made_60_")

    points = (
        fg_0_39 * float(cfg["fg_0_39"])
        + fg_40_49 * float(cfg["fg_40_49"])
        + fg_50_59 * float(cfg["fg_50_59"])
        + fg_60 * float(cfg["fg_60_plus"])
        + _num(df, "pat_made") * float(cfg["xp"])
    )

    # Anything not made is a miss, including blocks. This keeps the rule robust
    # to seasons where the blocked/missed columns are split differently.
    fg_misses = (_num(df, "fg_att") - _num(df, "fg_made")).clip(lower=0)
    xp_misses = (_num(df, "pat_att") - _num(df, "pat_made")).clip(lower=0)
    points = points + fg_misses * float(cfg.get("fg_miss", 0.0))
    points = points + xp_misses * float(cfg.get("xp_miss", 0.0))
    return points.astype("float64")


def add_dst_points(df: pd.DataFrame, scoring: dict[str, Any]) -> pd.Series:
    """Return team-defense fantasy points for each row of a team-game frame."""
    cfg = scoring["dst"]

    points = (
        _num(df, "def_sacks") * float(cfg["sack"])
        + _num(df, "def_ints") * float(cfg["interception"])
        + _num(df, "def_fumble_rec") * float(cfg["fumble_recovery"])
        + _num(df, "def_safeties") * float(cfg["safety"])
        + _num(df, "def_tds") * float(cfg["touchdown"])
        + _num(df, "def_blocked_kicks") * float(cfg["blocked_kick"])
        + _num(df, "def_two_pt") * float(cfg["two_point_return"])
    )

    points_allowed = pd.to_numeric(df.get("def_points_allowed"), errors="coerce")
    yards_allowed = pd.to_numeric(df.get("def_yards_allowed"), errors="coerce")
    points = points + _tier_points(points_allowed, cfg["points_allowed"])
    points = points + _tier_points(yards_allowed, cfg["yards_allowed"])
    return points.astype("float64")


def kicker_points(row: pd.Series | dict[str, Any], scoring: dict[str, Any]) -> float:
    """Single-row convenience wrapper used by tests and ad-hoc analysis."""
    frame = pd.DataFrame([row])
    return float(add_kicker_points(frame, scoring).iloc[0])


def dst_points(row: pd.Series | dict[str, Any], scoring: dict[str, Any]) -> float:
    """Single-row convenience wrapper used by tests and ad-hoc analysis."""
    frame = pd.DataFrame([row])
    return float(add_dst_points(frame, scoring).iloc[0])
