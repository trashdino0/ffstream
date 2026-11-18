"""Tests for weekly board grading metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffstream.grade import _metrics, _percentile, _safe_spearman


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "projected": [10.0, 8.0, 6.0],
            "actual": [12.0, 8.0, 4.0],
            "floor": [5.0, 5.0, 5.0],
            "ceiling": [15.0, 15.0, 15.0],
            "boom_pct": [50.0, 0.0, 0.0],
            "rank": [1, 2, 3],
            "actual_rank": [1, 2, 3],
        }
    )


def test_metrics_match_hand_computed_values() -> None:
    metrics = _metrics(_board(), threshold=10.0)

    assert metrics["rows"] == 3
    assert metrics["board_rows"] == 3
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["coverage"] == pytest.approx(2.0 / 3.0)
    assert metrics["top5_avg"] == pytest.approx(8.0)
    assert metrics["top5_vs_league"] == pytest.approx(0.0)
    assert metrics["top5_beating_league"] == 1
    assert metrics["top5_overlap"] == 3
    assert metrics["boom_pred"] == pytest.approx(50.0 / 300.0)
    assert metrics["boom_actual"] == pytest.approx(1.0 / 3.0)


def test_metrics_ignore_unplayed_rows() -> None:
    board = _board()
    board.loc[1, "actual"] = np.nan
    metrics = _metrics(board, threshold=10.0)

    assert metrics["rows"] == 2
    assert metrics["board_rows"] == 3
    assert metrics["mae"] == pytest.approx(2.0)


def test_spearman_is_nan_for_constant_input() -> None:
    series = pd.Series([1.0, 1.0, 1.0])
    assert np.isnan(_safe_spearman(series, pd.Series([1.0, 2.0, 3.0])))


def test_percentile_maps_best_rank_to_100() -> None:
    ranks = pd.Series([1, 2, 3, 4])
    result = _percentile(ranks, 4).to_numpy()

    assert result[0] == pytest.approx(100.0)
    assert result[-1] == pytest.approx(0.0)
    assert np.allclose(result, [100.0, 200.0 / 3.0, 100.0 / 3.0, 0.0])
