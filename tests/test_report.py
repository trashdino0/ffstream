"""Ranking-table, HTML-report, and file-output tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffstream.report import prepare_table, render_html, write_weekly_outputs


def _rows(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_display_name": [f"Kicker {i:02d}" for i in range(n)],
            "team": [f"T{i:02d}" for i in range(n)],
            "opponent": ["OPP"] * n,
            "team_implied": [24.0] * n,
            "opp_implied": [20.0] * n,
            "total_line": [44.0] * n,
            "is_dome": [0] * n,
            "wind": [5.0] * n,
            "temp": [60.0] * n,
            "opp_def_fg_att_allowed_last4": [3.0] * n,
            "opp_off_sacks_last4": [2.0] * n,
            "game_id": [f"g{i}" for i in range(n)],
            "season": [2026] * n,
            "week": [9] * n,
            "player_id": [f"p{i}" for i in range(n)],
        }
    )


def _pred(n: int = 20) -> pd.DataFrame:
    mean = np.linspace(12.0, 4.0, n)
    return pd.DataFrame(
        {
            "proj_mean": mean,
            "proj_p10": mean - 3.0,
            "proj_p90": mean + 3.0,
            "boom_prob": np.linspace(0.6, 0.05, n),
            "stream_score": mean + 0.5,
        }
    )


def test_prepare_kicker_table_ranks_and_tiers() -> None:
    table = prepare_table(_rows(), _pred(), "kicker_pts", "mean")
    assert list(table["rank"]) == list(range(1, 21))
    assert table.loc[0, "tier"] == "STREAM"
    assert table.loc[5, "tier"] == "SPOT START"
    assert table.loc[10, "tier"] == "DEEP"
    assert table.loc[16, "tier"] == "SIT"
    assert table.loc[0, "player"] == "Kicker 00"
    assert table.loc[0, "projected"] == pytest.approx(12.0)
    assert table.loc[0, "boom_pct"] == pytest.approx(60.0)


def test_prepare_dst_table_ranks_and_tiers() -> None:
    table = prepare_table(_rows(), _pred(), "dst_pts", "mean")
    assert table.loc[0, "tier"] == "START"
    assert table.loc[5, "tier"] == "STREAM"
    assert table.loc[10, "tier"] == "DEPTH"
    assert table.loc[16, "tier"] == "SIT"
    # Defenses fall back to the team abbreviation when no player name exists.
    assert table.loc[0, "player"] == "Kicker 00"


def test_stream_ranking_can_override_mean_ranking() -> None:
    pred = _pred()
    pred["stream_score"] = np.linspace(4.0, 12.0, len(pred))
    table = prepare_table(_rows(), pred, "kicker_pts", "stream")
    assert table.loc[0, "player"] == "Kicker 19"
    assert table.loc[19, "player"] == "Kicker 00"


def test_prepare_table_fills_missing_kicker_name() -> None:
    rows = _rows(3)
    rows["player_display_name"] = [None, "Named Kicker", None]
    table = prepare_table(rows, _pred(3), "kicker_pts", "mean")
    assert table.loc[0, "player"] == "T00 K"
    assert table.loc[1, "player"] == "Named Kicker"


def test_render_html_contains_both_boards_and_escapes_names() -> None:
    rows = _rows(3)
    rows["player_display_name"] = ["A&B <script>alert(1)</script>", "Kicker 01", None]
    kicker = prepare_table(rows, _pred(3), "kicker_pts", "mean")
    dst = prepare_table(_rows(3), _pred(3), "dst_pts", "mean")

    html = render_html(kicker, dst, 2026, 9, meta={"kicker_train_rows": 1000})
    assert "2026 Week 9" in html
    assert "kicker-table" in html
    assert "dst-table" in html
    assert "A&amp;B &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "1,000" in html


def test_write_weekly_outputs_creates_three_files(tmp_path) -> None:
    kicker = prepare_table(_rows(3), _pred(3), "kicker_pts", "mean")
    dst = prepare_table(_rows(3), _pred(3), "dst_pts", "mean")

    paths = write_weekly_outputs(kicker, dst, 2026, 9, out_dir=tmp_path)
    assert set(paths) == {"kickers", "dst", "html"}
    for path in paths.values():
        assert (tmp_path / Path(path).name).exists()
    assert "rank" in pd.read_csv(paths["kickers"]).columns
    assert "<!doctype html>" in (tmp_path / "week_2026_09_report.html").read_text(
        encoding="utf-8"
    )
