"""Tests for the standalone backtest-review script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "backtest_review", ROOT / "scripts" / "backtest_review.py"
)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": [1] * 6,
            "player_id": list("ABCDEF"),
            "player_display_name": [f"Kicker {i}" for i in range(6)],
            "team": list("ABCDEF"),
            "opponent_team": ["X"] * 6,
            "proj_mean": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            "kicker_pts": [1.0, 12.0, 12.0, 2.0, 20.0, 3.0],
        }
    )


def test_ranks_are_computed_within_each_week() -> None:
    ranked = review._rank_frame(_frame(), "kicker_pts", "K")
    assert ranked["_model_rank"].tolist() == [1, 2, 3, 4, 5, 6]
    # 20, 12, 12, 3, 2, 1 -> ranks 1, 2, 2, 4, 5, 6
    assert ranked["_actual_rank"].tolist() == [6, 2, 2, 5, 1, 4]
    tied = ranked[ranked["player_id"] == "B"].iloc[0]
    assert bool(tied["_actual_tied"]) is True


def test_model_top_is_sorted_by_projection() -> None:
    ranked = review._rank_frame(_frame(), "kicker_pts", "K")
    top = review._top(ranked, "proj_mean", "kicker_pts", 3)
    assert list(top["player_id"]) == ["A", "B", "C"]


def test_compact_table_shows_both_directions() -> None:
    ranked = review._rank_frame(_frame(), "kicker_pts", "K")
    detail = pd.DataFrame(review._detail_rows(ranked, 2026, "K", "kicker_pts", 3))
    table = review._compact_table(detail, 1, "K", 3)
    assert "Model pick" in table
    assert "Actual leader" in table
    assert "Kicker 0" in table
    assert "Kicker 4" in table
