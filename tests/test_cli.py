"""CLI parsing and slate-filtering tests (no data downloads)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffstream.cli import _filter_to_slate, _parse_seasons, build_parser


def test_parse_seasons_range() -> None:
    assert _parse_seasons("2021-2024") == [2021, 2022, 2023, 2024]


def test_parse_seasons_list() -> None:
    assert _parse_seasons("2021, 2023,2025") == [2021, 2023, 2025]


def test_parse_seasons_default_is_full_history() -> None:
    seasons = _parse_seasons(None)
    assert seasons[0] == 2014
    assert seasons[-1] == 2026


def test_week_parser_defaults() -> None:
    args = build_parser().parse_args(["week"])
    assert args.season is None
    assert args.week is None
    assert args.top == 15
    assert args.rank_by == "auto"
    assert args.include_completed is False


def test_backtest_parser_defaults() -> None:
    args = build_parser().parse_args(["backtest"])
    assert args.seasons == "2021-2025"
    assert args.no_2026 is False


def _slate_frames():
    rows = pd.DataFrame({"game_id": ["g1", "g2"], "team": ["A", "B"]})
    team_features = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [9, 9],
            "game_id": ["g1", "g2"],
            "team": ["A", "B"],
            "team_score": [20.0, np.nan],
        }
    )
    return rows, team_features


def test_filter_to_slate_drops_completed_games() -> None:
    rows, team_features = _slate_frames()
    filtered = _filter_to_slate(rows, team_features, 2026, 9, include_completed=False)
    assert list(filtered["game_id"]) == ["g2"]


def test_filter_to_slate_keeps_everything_when_requested() -> None:
    rows, team_features = _slate_frames()
    filtered = _filter_to_slate(rows, team_features, 2026, 9, include_completed=True)
    assert list(filtered["game_id"]) == ["g1", "g2"]


def test_filter_to_slate_falls_back_when_all_games_played() -> None:
    rows, team_features = _slate_frames()
    team_features["team_score"] = 20.0
    filtered = _filter_to_slate(rows, team_features, 2026, 9, include_completed=False)
    assert list(filtered["game_id"]) == ["g1", "g2"]
