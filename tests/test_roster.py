"""Tests for roster-aware kicker identity."""

from __future__ import annotations

import pandas as pd
import pytest

from ffstream.roster import apply_roster_overrides, latest_roster_snapshot


def _rosters(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    defaults = {
        "season": 2021,
        "week": 10,
        "game_type": "REG",
        "position": "K",
        "team": "A",
        "status": "ACT",
        "full_name": "Kicker",
        "gsis_id": "K1",
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
        else:
            frame[column] = frame[column].fillna(value)
    return frame


def _kicker_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "K1",
                "player_display_name": "Old Kicker",
                "season": 2021,
                "week": 5,
                "gameday": "2021-10-10",
                "team": "A",
                "kicker_kicker_pts_last4": 11.0,
                "kicker_pts_std_last8": 2.5,
                "kicker_games": 4,
                "kicker_is_new": 0,
            },
            {
                "player_id": "K2",
                "player_display_name": "New Kicker",
                "season": 2021,
                "week": 6,
                "gameday": "2021-10-17",
                "team": "B",
                "kicker_kicker_pts_last4": 7.5,
                "kicker_pts_std_last8": 1.5,
                "kicker_games": 2,
                "kicker_is_new": 0,
            },
        ]
    )


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "player": "Old Kicker",
                "player_id": "K1",
                "player_display_name": "Old Kicker",
                "team": "A",
                "opponent": "B",
                "proj_mean": 8.0,
                "kicker_kicker_pts_last4": 11.0,
                "kicker_pts_std_last8": 2.5,
                "kicker_games": 4,
                "kicker_is_new": 0,
            }
        ]
    )


def test_snapshot_ignores_weeks_after_the_target() -> None:
    rosters = _rosters(
        [
            {"week": 10, "team": "A", "gsis_id": "K1", "full_name": "Old"},
            {"week": 12, "team": "A", "gsis_id": "K3", "full_name": "Future"},
        ]
    )
    snapshot = latest_roster_snapshot(rosters, season=2021, week=11)

    assert snapshot["gsis_id"].tolist() == ["K1"]
    assert snapshot["full_name"].tolist() == ["Old"]


def test_snapshot_keeps_latest_per_team_and_filters_positions() -> None:
    rosters = _rosters(
        [
            {"week": 9, "team": "A", "gsis_id": "K1", "full_name": "Old"},
            {"week": 10, "team": "A", "gsis_id": "K1", "full_name": "Old"},
            {"week": 10, "team": "A", "gsis_id": "P1", "position": "P"},
            {"week": 10, "team": "B", "gsis_id": "K2", "full_name": "Other"},
        ]
    )
    snapshot = latest_roster_snapshot(rosters, season=2021, week=11)

    assert sorted(snapshot["gsis_id"]) == ["K1", "K2"]
    assert len(snapshot) == 2


def test_confirmed_kicker_is_untouched() -> None:
    rows = apply_roster_overrides(
        _board(), _kicker_features(), _rosters([{}]), season=2021, week=11
    )

    assert rows.loc[0, "roster_status"] == "confirmed"
    assert rows.loc[0, "player_id"] == "K1"
    assert rows.loc[0, "kicker_kicker_pts_last4"] == pytest.approx(11.0)


def test_cut_kicker_is_replaced_and_flagged() -> None:
    rosters = _rosters(
        [
            {"gsis_id": "K1", "full_name": "Old Kicker", "status": "CUT"},
            {"gsis_id": "K2", "full_name": "New Kicker", "status": "ACT"},
        ]
    )
    rows = apply_roster_overrides(_board(), _kicker_features(), rosters, 2021, 11)

    assert rows.loc[0, "roster_status"] == "replaced"
    assert rows.loc[0, "player_id"] == "K2"
    assert rows.loc[0, "player_display_name"] == "New Kicker"
    assert "Old Kicker (CUT) -> New Kicker" in rows.loc[0, "roster_note"]
    # K2 last played in week 6, so its history should carry over.
    assert rows.loc[0, "kicker_kicker_pts_last4"] == pytest.approx(7.5)
    assert rows.loc[0, "kicker_games"] == 2


def test_replacement_without_history_is_treated_as_new() -> None:
    rosters = _rosters(
        [
            {"gsis_id": "K1", "full_name": "Old Kicker", "status": "CUT"},
            {"gsis_id": "K9", "full_name": "Rookie Kicker", "status": "ACT"},
        ]
    )
    rows = apply_roster_overrides(_board(), _kicker_features(), rosters, 2021, 11)

    assert rows.loc[0, "player_id"] == "K9"
    assert rows.loc[0, "kicker_games"] == 0
    assert rows.loc[0, "kicker_is_new"] == 1
    assert pd.isna(rows.loc[0, "kicker_kicker_pts_last4"])


def test_two_active_kickers_is_competition_not_a_swap() -> None:
    rosters = _rosters(
        [
            {"gsis_id": "K1", "full_name": "Old Kicker", "status": "ACT"},
            {"gsis_id": "K2", "full_name": "New Kicker", "status": "ACT"},
        ]
    )
    rows = apply_roster_overrides(_board(), _kicker_features(), rosters, 2021, 11)

    assert rows.loc[0, "roster_status"] == "competition"
    assert rows.loc[0, "player_id"] == "K1"
    assert rows.loc[0, "roster_note"] == "2 active kickers listed"


def test_no_active_kicker_is_unconfirmed() -> None:
    rosters = _rosters(
        [
            {"gsis_id": "K1", "full_name": "Old Kicker", "status": "CUT"},
            {"gsis_id": "K2", "full_name": "Practice Kicker", "status": "DEV"},
        ]
    )
    rows = apply_roster_overrides(_board(), _kicker_features(), rosters, 2021, 11)

    assert rows.loc[0, "roster_status"] == "unconfirmed"
    assert rows.loc[0, "player_id"] == "K1"
    assert "no active kicker" in rows.loc[0, "roster_note"]


def test_missing_roster_data_marks_rows_unavailable() -> None:
    rows = apply_roster_overrides(
        _board(), _kicker_features(), pd.DataFrame(), 2021, 11
    )

    assert rows.loc[0, "roster_status"] == "unavailable"
    assert rows.loc[0, "roster_note"] == "no weekly roster data"
    assert rows.loc[0, "player_id"] == "K1"


def test_empty_board_is_returned_with_columns() -> None:
    board = _board().iloc[0:0]
    rows = apply_roster_overrides(board, _kicker_features(), _rosters([{}]), 2021, 11)

    assert rows.empty
    assert "roster_status" in rows.columns
    assert "roster_note" in rows.columns
