"""Roster-aware kicker identity for the weekly board.

The training rows always describe the kicker who actually played. Before a
future week the relevant question is who is on the roster *now*, which can
differ from the last kicker with a recorded game. The nflverse weekly roster
release gives us that snapshot.

Lookups are restricted to the target week or earlier, so a Thursday result
cannot leak into a Sunday board.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import kicker_history_feature_columns

ACTIVE_STATUS = "ACT"


def _clean_name(value: object, fallback: str) -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() != "nan" else fallback


def latest_roster_snapshot(
    rosters: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """All kickers in each team's latest roster snapshot at or before a week."""
    if rosters is None or rosters.empty:
        return pd.DataFrame()

    frame = rosters.copy()
    frame["position"] = frame["position"].fillna("").astype(str).str.upper()
    frame = frame[frame["position"] == "K"]
    if "game_type" in frame.columns:
        game_type = frame["game_type"].fillna("REG").astype(str).str.upper()
        frame = frame[game_type == "REG"]

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["season", "week", "team"])
    frame = frame[
        (frame["season"] < season)
        | ((frame["season"] == season) & (frame["week"] <= week))
    ]
    if frame.empty:
        return frame

    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["_week_key"] = frame["season"] * 100 + frame["week"]
    latest_key = frame.groupby("team")["_week_key"].transform("max")
    return frame[frame["_week_key"] == latest_key].drop(columns="_week_key")


def _prior_player_rows(
    kicker_features: pd.DataFrame,
    player_ids: list[str],
    season: int,
    week: int,
) -> pd.DataFrame:
    if (
        kicker_features is None
        or kicker_features.empty
        or "player_id" not in kicker_features.columns
        or not player_ids
    ):
        return pd.DataFrame()
    mask = kicker_features["player_id"].astype(str).isin(player_ids)
    before = (kicker_features["season"] < season) | (
        (kicker_features["season"] == season) & (kicker_features["week"] < week)
    )
    history = kicker_features[mask & before].copy()
    if history.empty:
        return history
    sort_columns = [c for c in ["season", "week", "gameday"] if c in history.columns]
    return history.sort_values(sort_columns)


def apply_roster_overrides(
    rows: pd.DataFrame,
    kicker_features: pd.DataFrame,
    rosters: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Replace stale kickers with the team's active kicker, with status flags.

    Columns added:
    - ``roster_status``: confirmed, replaced, competition, unconfirmed, or
      unavailable.
    - ``roster_note``: human-readable reason, including the old and new names.
    """
    rows = rows.copy()
    rows["roster_status"] = "unavailable"
    rows["roster_note"] = ""
    if rows.empty:
        return rows
    if "player_id" not in rows.columns:
        rows["player_id"] = np.nan
    if "player_display_name" not in rows.columns:
        rows["player_display_name"] = np.nan

    snapshot = latest_roster_snapshot(rosters, season, week)
    if snapshot.empty:
        rows["roster_note"] = "no weekly roster data"
        return rows

    feature_columns = [
        c for c in kicker_history_feature_columns(kicker_features) if c in rows.columns
    ]

    for idx, row in rows.iterrows():
        team = str(row.get("team", ""))
        board_id = row.get("player_id")
        board_key = str(board_id) if pd.notna(board_id) else ""
        board_name = _clean_name(row.get("player_display_name"), f"{team} K")

        team_snapshot = snapshot[snapshot["team"] == team]
        if team_snapshot.empty:
            rows.at[idx, "roster_status"] = "unavailable"
            rows.at[idx, "roster_note"] = "team not in roster snapshot"
            continue

        active = team_snapshot[
            team_snapshot["status"].fillna("").astype(str).str.upper() == ACTIVE_STATUS
        ].drop_duplicates(subset=["gsis_id"])
        active_ids = [str(v) for v in active["gsis_id"].dropna().tolist()]

        if board_key:
            board_entry = team_snapshot[
                team_snapshot["gsis_id"].astype(str) == board_key
            ]
        else:
            board_entry = team_snapshot.iloc[0:0]
        board_status = (
            str(board_entry["status"].iloc[0]).upper()
            if not board_entry.empty
            else "NOT_ON_ROSTER"
        )

        if board_key and board_key in active_ids:
            if len(active_ids) == 1:
                rows.at[idx, "roster_status"] = "confirmed"
            else:
                rows.at[idx, "roster_status"] = "competition"
                rows.at[idx, "roster_note"] = f"{len(active_ids)} active kickers listed"
            continue

        replacement = None
        if len(active_ids) == 1:
            replacement = active_ids[0]
        elif len(active_ids) > 1:
            history = _prior_player_rows(kicker_features, active_ids, season, week)
            if not history.empty:
                replacement = str(history.iloc[-1]["player_id"])
            else:
                replacement = sorted(active_ids)[0]

        if replacement is None:
            rows.at[idx, "roster_status"] = "unconfirmed"
            rows.at[idx, "roster_note"] = (
                f"{board_name} listed as {board_status}; no active kicker in roster"
            )
            continue

        replacement_entry = active[active["gsis_id"].astype(str) == replacement].iloc[0]
        replacement_name = _clean_name(replacement_entry.get("full_name"), f"{team} K")
        rows.at[idx, "player_id"] = replacement
        rows.at[idx, "player_display_name"] = replacement_name

        history = _prior_player_rows(kicker_features, [replacement], season, week)
        if not history.empty:
            latest_row = history.iloc[-1]
            for column in feature_columns:
                rows.at[idx, column] = latest_row[column]
        else:
            for column in feature_columns:
                rows.at[idx, column] = np.nan
            if "kicker_games" in rows.columns:
                rows.at[idx, "kicker_games"] = 0
            if "kicker_is_new" in rows.columns:
                rows.at[idx, "kicker_is_new"] = 1

        rows.at[idx, "roster_status"] = "replaced"
        rows.at[idx, "roster_note"] = (
            f"{board_name} ({board_status}) -> {replacement_name}"
        )

    return rows
