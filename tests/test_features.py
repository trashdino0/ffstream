"""Feature-engineering tests, with emphasis on pre-kickoff leakage safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffstream.features import (
    _add_kicker_rolling,
    build_dst_features,
    build_team_features,
    dst_feature_columns,
    kicker_history_feature_columns,
    kicker_model_features,
    latest_kicker_rows,
    make_kicker_prediction_rows,
)

SCHEDULE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "gameday",
    "team",
    "opponent",
    "is_home",
    "team_score",
    "opp_score",
    "spread_line",
    "total_line",
    "roof",
    "surface",
    "temp",
    "wind",
    "rest",
    "opp_rest",
    "div_game",
    "stadium_id",
    "stadium",
]


def _team_games() -> pd.DataFrame:
    """A tiny two-team, three-season schedule."""
    rows = [
        # 2020: A hosts B
        dict(
            season=2020,
            week=1,
            game_id="2020_01_A_B",
            gameday="2020-09-10",
            team="A",
            opponent="B",
            is_home=1,
            team_score=27,
            opp_score=20,
            spread_line=-3.0,
            total_line=44.0,
            roof="outdoors",
            surface="grass",
            temp=70,
            wind=5,
            rest=7,
            opp_rest=7,
            div_game=0,
            stadium_id="AAA",
            stadium="Alpha",
            off_points=27,
            off_plays=60,
            def_points_allowed=20,
            def_yards_allowed=310,
            dst_pts=3.0,
        ),
        dict(
            season=2020,
            week=1,
            game_id="2020_01_A_B",
            gameday="2020-09-10",
            team="B",
            opponent="A",
            is_home=0,
            team_score=20,
            opp_score=27,
            spread_line=-3.0,
            total_line=44.0,
            roof="outdoors",
            surface="grass",
            temp=70,
            wind=5,
            rest=7,
            opp_rest=7,
            div_game=0,
            stadium_id="AAA",
            stadium="Alpha",
            off_points=20,
            off_plays=58,
            def_points_allowed=27,
            def_yards_allowed=350,
        ),
        # 2021 week 1: A hosts B
        dict(
            season=2021,
            week=1,
            game_id="2021_01_A_B",
            gameday="2021-09-09",
            team="A",
            opponent="B",
            is_home=1,
            team_score=30,
            opp_score=17,
            spread_line=-6.0,
            total_line=44.0,
            roof="dome",
            surface="turf",
            temp=70,
            wind=0,
            rest=7,
            opp_rest=7,
            div_game=1,
            stadium_id="AAA",
            stadium="Alpha",
            off_points=30,
            off_plays=62,
            def_points_allowed=17,
            def_yards_allowed=290,
            dst_pts=9.0,
        ),
        dict(
            season=2021,
            week=1,
            game_id="2021_01_A_B",
            gameday="2021-09-09",
            team="B",
            opponent="A",
            is_home=0,
            team_score=17,
            opp_score=30,
            spread_line=-6.0,
            total_line=44.0,
            roof="dome",
            surface="turf",
            temp=70,
            wind=0,
            rest=7,
            opp_rest=7,
            div_game=1,
            stadium_id="AAA",
            stadium="Alpha",
            off_points=17,
            off_plays=55,
            def_points_allowed=30,
            def_yards_allowed=380,
            dst_pts=3.0,
        ),
        # 2021 week 2: B hosts A
        dict(
            season=2021,
            week=2,
            game_id="2021_02_B_A",
            gameday="2021-09-16",
            team="A",
            opponent="B",
            is_home=0,
            team_score=12,
            opp_score=24,
            spread_line=-3.0,
            total_line=41.0,
            roof="outdoors",
            surface="grass",
            temp=55,
            wind=12,
            rest=7,
            opp_rest=7,
            div_game=1,
            stadium_id="BBB",
            stadium="Beta",
            off_points=12,
            off_plays=51,
            def_points_allowed=24,
            def_yards_allowed=340,
            dst_pts=4.0,
        ),
        dict(
            season=2021,
            week=2,
            game_id="2021_02_B_A",
            gameday="2021-09-16",
            team="B",
            opponent="A",
            is_home=1,
            team_score=24,
            opp_score=12,
            spread_line=-3.0,
            total_line=41.0,
            roof="outdoors",
            surface="grass",
            temp=55,
            wind=12,
            rest=7,
            opp_rest=7,
            div_game=1,
            stadium_id="BBB",
            stadium="Beta",
            off_points=24,
            off_plays=59,
            def_points_allowed=12,
            def_yards_allowed=270,
            dst_pts=11.0,
        ),
        # 2021 week 3 is scheduled but unplayed.
        dict(
            season=2021,
            week=3,
            game_id="2021_03_A_B",
            gameday="2021-09-23",
            team="A",
            opponent="B",
            is_home=1,
            team_score=np.nan,
            opp_score=np.nan,
            spread_line=-2.0,
            total_line=43.0,
            roof="outdoors",
            surface="grass",
            temp=60,
            wind=8,
            rest=7,
            opp_rest=7,
            div_game=0,
            stadium_id="AAA",
            stadium="Alpha",
        ),
        dict(
            season=2021,
            week=3,
            game_id="2021_03_A_B",
            gameday="2021-09-23",
            team="B",
            opponent="A",
            is_home=0,
            team_score=np.nan,
            opp_score=np.nan,
            spread_line=-2.0,
            total_line=43.0,
            roof="outdoors",
            surface="grass",
            temp=60,
            wind=8,
            rest=7,
            opp_rest=7,
            div_game=0,
            stadium_id="AAA",
            stadium="Alpha",
        ),
    ]
    return pd.DataFrame(rows)


def test_rolling_team_features_shift_one_game() -> None:
    feats = build_team_features(_team_games())
    a = (
        feats[feats["team"] == "A"]
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )

    week1 = a[(a["season"] == 2021) & (a["week"] == 1)].iloc[0]
    week2 = a[(a["season"] == 2021) & (a["week"] == 2)].iloc[0]
    week3 = a[(a["season"] == 2021) & (a["week"] == 3)].iloc[0]

    # The window carries across seasons, but only includes prior games.
    assert week1["off_points_last4"] == pytest.approx(27.0)
    # Week 2 sees the 2020 game and week 1, never its own 12-point game.
    assert week2["off_points_last4"] == pytest.approx(28.5)
    assert week3["off_points_last4"] == pytest.approx(23.0)


def test_opponent_rolling_features_use_opponent_history() -> None:
    feats = build_team_features(_team_games())
    a_week2 = feats[
        (feats["team"] == "A") & (feats["season"] == 2021) & (feats["week"] == 2)
    ].iloc[0]
    # B scored 20 in 2020 and 17 in week 1; B's 24-point week-2 game is excluded.
    assert a_week2["opp_off_points_last4"] == pytest.approx(18.5)

    a_week1 = feats[
        (feats["team"] == "A") & (feats["season"] == 2021) & (feats["week"] == 1)
    ].iloc[0]
    assert a_week1["opp_off_points_last4"] == pytest.approx(20.0)


def test_prior_season_features_are_previous_year_only() -> None:
    feats = build_team_features(_team_games())
    a_2021 = feats[(feats["team"] == "A") & (feats["season"] == 2021)]
    assert np.allclose(a_2021["off_points_prior"], 27.0)


def test_vegas_implied_points_split() -> None:
    feats = build_team_features(_team_games())
    a_week1 = feats[
        (feats["team"] == "A") & (feats["season"] == 2021) & (feats["week"] == 1)
    ].iloc[0]
    b_week1 = feats[
        (feats["team"] == "B") & (feats["season"] == 2021) & (feats["week"] == 1)
    ].iloc[0]
    # total 44, spread -6 => 25 / 19
    assert a_week1["team_implied"] == pytest.approx(25.0)
    assert a_week1["opp_implied"] == pytest.approx(19.0)
    assert b_week1["team_implied"] == pytest.approx(19.0)
    assert b_week1["implied_margin"] == pytest.approx(-6.0)


def _kicker_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "K1",
                "player_display_name": "Kicker One",
                "season": 2021,
                "week": 1,
                "gameday": "2021-09-09",
                "team": "A",
                "kicker_pts": 12.0,
                "fg_att": 3,
                "fg_made": 3,
                "pat_att": 3,
                "pat_made": 3,
                "fg_long": 48,
                "fg_made_40_49": 1,
                "fg_missed_40_49": 0,
            },
            {
                "player_id": "K1",
                "player_display_name": "Kicker One",
                "season": 2021,
                "week": 2,
                "gameday": "2021-09-16",
                "team": "A",
                "kicker_pts": 5.0,
                "fg_att": 1,
                "fg_made": 0,
                "pat_att": 1,
                "pat_made": 1,
                "fg_long": 0,
                "fg_made_40_49": 0,
                "fg_missed_40_49": 1,
            },
            {
                "player_id": "K2",
                "player_display_name": "Kicker Two",
                "season": 2021,
                "week": 3,
                "gameday": "2021-09-23",
                "team": "A",
                "kicker_pts": 9.0,
                "fg_att": 2,
                "fg_made": 2,
                "pat_att": 3,
                "pat_made": 3,
                "fg_long": 41,
                "fg_made_40_49": 1,
                "fg_missed_40_49": 0,
            },
            {
                "player_id": "K3",
                "player_display_name": "Kicker Three",
                "season": 2021,
                "week": 1,
                "gameday": "2021-09-09",
                "team": "B",
                "kicker_pts": 4.0,
                "fg_att": 1,
                "fg_made": 1,
                "pat_att": 1,
                "pat_made": 1,
                "fg_long": 33,
                "fg_made_40_49": 0,
                "fg_missed_40_49": 0,
            },
        ]
    )


def test_kicker_rolling_stats_shift_one_game() -> None:
    rolled = _add_kicker_rolling(_kicker_history())
    k1 = rolled[rolled["player_id"] == "K1"].sort_values("week").reset_index(drop=True)

    assert pd.isna(k1.loc[0, "kicker_kicker_pts_last4"])
    assert k1.loc[1, "kicker_kicker_pts_last4"] == pytest.approx(12.0)
    assert k1.loc[1, "kicker_fg_pct_last4"] == pytest.approx(1.0)
    assert k1.loc[1, "kicker_games"] == 1
    assert k1.loc[0, "kicker_is_new"] == 1
    assert k1.loc[1, "kicker_is_new"] == 0


def test_kicker_feature_lists_exclude_raw_targets() -> None:
    rolled = _add_kicker_rolling(_kicker_history())
    history_cols = kicker_history_feature_columns(rolled)
    assert "kicker_pts" not in history_cols
    assert "kicker_kicker_pts_last4" in history_cols

    model_cols = kicker_model_features(rolled)
    assert "kicker_pts" not in model_cols
    assert "fg_att" not in model_cols
    assert "kicker_fg_att_last4" in model_cols


def test_latest_kicker_before_week_ignores_target_week_stats() -> None:
    rolled = _add_kicker_rolling(_kicker_history())
    latest = latest_kicker_rows(rolled, before_season=2021, before_week=3)
    a = latest[latest["team"] == "A"].iloc[0]
    # K2's week-3 row must not be treated as pregame information for week 3.
    assert a["player_id"] == "K1"
    assert a["player_display_name"] == "Kicker One"


def test_prediction_rows_join_last_known_kicker() -> None:
    rolled = _add_kicker_rolling(_kicker_history())
    team_features = _team_games()
    rows = make_kicker_prediction_rows(team_features, rolled, 2021, 3)

    assert len(rows) == 2
    a = rows[rows["team"] == "A"].iloc[0]
    assert a["player_id"] == "K1"
    assert a["player_display_name"] == "Kicker One"
    # The week-3 schedule is present, but the week-3 stats are not.
    assert pd.isna(a["team_score"])


def test_dst_features_are_rolling_and_prior_only() -> None:
    team_features = build_team_features(_team_games())
    dst = build_dst_features(team_features)
    a = dst[dst["team"] == "A"].sort_values(["season", "week"]).reset_index(drop=True)
    a_2021 = a[a["season"] == 2021].reset_index(drop=True)

    assert a_2021.loc[0, "dst_pts_last4"] == pytest.approx(3.0)
    assert a_2021.loc[1, "dst_pts_last4"] == pytest.approx(6.0)
    assert a_2021.loc[2, "dst_pts_last4"] == pytest.approx(16.0 / 3.0)
    assert a_2021.loc[0, "dst_pts_prior"] == pytest.approx(3.0)


def test_dst_feature_columns_exclude_raw_stats() -> None:
    team_features = build_team_features(_team_games())
    dst = build_dst_features(team_features)
    cols = dst_feature_columns(dst)
    assert "dst_pts" not in cols
    assert "def_sacks" not in cols
    assert "dst_pts_last4" in cols
    assert any(c.startswith("opp_off_") and c.endswith("_last4") for c in cols)
