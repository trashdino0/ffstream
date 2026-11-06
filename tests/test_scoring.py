"""Scoring-rule tests for kickers and team defenses."""

from __future__ import annotations

import pandas as pd
import pytest

from ffstream.config import load_scoring
from ffstream.scoring import add_dst_points, add_kicker_points

SCORING = {
    "kicker": {
        "fg_0_39": 3.0,
        "fg_40_49": 4.0,
        "fg_50_59": 5.0,
        "fg_60_plus": 5.0,
        "xp": 1.0,
        "fg_miss": -1.0,
        "xp_miss": -1.0,
    },
    "dst": {
        "sack": 1.0,
        "interception": 2.0,
        "fumble_recovery": 2.0,
        "safety": 2.0,
        "touchdown": 6.0,
        "blocked_kick": 2.0,
        "two_point_return": 2.0,
        "points_allowed": [
            [0, 0, 10],
            [1, 6, 7],
            [7, 13, 4],
            [14, 20, 1],
            [21, 27, 0],
            [28, 34, -1],
            [35, 999, -4],
        ],
        "yards_allowed": [
            [0, 99, 5],
            [100, 199, 3],
            [200, 299, 2],
            [300, 399, 0],
            [400, 449, -1],
            [450, 499, -3],
            [500, 999, -5],
        ],
    },
}


def test_repo_scoring_config_has_required_keys() -> None:
    config = load_scoring()
    assert {"fg_0_39", "fg_40_49", "fg_50_59", "fg_60_plus", "xp"} <= set(
        config["kicker"]
    )
    assert {"sack", "interception", "points_allowed", "yards_allowed"} <= set(
        config["dst"]
    )


def test_kicker_points_by_distance_and_xp() -> None:
    row = {
        "fg_att": 4,
        "fg_made": 4,
        "fg_made_0_19": 1,
        "fg_made_40_49": 1,
        "fg_made_50_59": 1,
        "fg_made_60_": 1,
        "pat_att": 2,
        "pat_made": 2,
    }
    frame = pd.DataFrame([row])
    # 3 + 4 + 5 + 5 + 2 XP
    assert add_kicker_points(frame, SCORING).iloc[0] == pytest.approx(19.0)


def test_kicker_misses_are_penalized() -> None:
    frame = pd.DataFrame(
        [
            {
                "fg_att": 3,
                "fg_made": 1,
                "fg_made_30_39": 1,
                "pat_att": 2,
                "pat_made": 1,
            }
        ]
    )
    # 3 points + 1 XP - 2 missed FG - 1 missed XP
    assert add_kicker_points(frame, SCORING).iloc[0] == pytest.approx(1.0)


def test_kicker_missing_columns_default_to_zero() -> None:
    frame = pd.DataFrame([{"fg_att": 0, "fg_made": 0, "pat_att": 0, "pat_made": 0}])
    assert add_kicker_points(frame, SCORING).iloc[0] == pytest.approx(0.0)


def test_dst_points_combine_events_and_allowances() -> None:
    frame = pd.DataFrame(
        [
            {
                "def_sacks": 3,
                "def_ints": 1,
                "def_fumble_rec": 1,
                "def_safeties": 1,
                "def_tds": 1,
                "def_blocked_kicks": 1,
                "def_two_pt": 1,
                "def_points_allowed": 13,
                "def_yards_allowed": 250,
            }
        ]
    )
    # 3 + 2 + 2 + 2 + 6 + 2 + 2 = 19 events, plus 4 (points) + 2 (yards)
    assert add_dst_points(frame, SCORING).iloc[0] == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("points_allowed", "expected"),
    [
        (0, 10.0),
        (6, 7.0),
        (7, 4.0),
        (13, 4.0),
        (14, 1.0),
        (20, 1.0),
        (21, 0.0),
        (27, 0.0),
        (28, -1.0),
        (34, -1.0),
        (35, -4.0),
    ],
)
def test_dst_points_allowed_tiers(points_allowed: int, expected: float) -> None:
    frame = pd.DataFrame(
        [{"def_points_allowed": points_allowed, "def_yards_allowed": 350}]
    )
    assert add_dst_points(frame, SCORING).iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("yards_allowed", "expected"),
    [
        (0, 5.0),
        (99, 5.0),
        (100, 3.0),
        (199, 3.0),
        (299, 2.0),
        (399, 0.0),
        (449, -1.0),
        (499, -3.0),
        (500, -5.0),
    ],
)
def test_dst_yards_allowed_tiers(yards_allowed: int, expected: float) -> None:
    frame = pd.DataFrame(
        [{"def_points_allowed": 21, "def_yards_allowed": yards_allowed}]
    )
    assert add_dst_points(frame, SCORING).iloc[0] == pytest.approx(expected)
