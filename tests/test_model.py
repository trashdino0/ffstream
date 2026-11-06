"""Model-training, projection, and ranking-metric tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffstream.model import (
    DST_BLEND,
    KICKER_BLEND,
    baseline_scores,
    feature_importance,
    predict_models,
    score_predictions,
    train_models,
    walk_forward,
)

KICKER_FEATURES = [
    "team_implied",
    "kicker_kicker_pts_last4",
    "kicker_pts_std_last8",
]
DST_FEATURES = [
    "opp_implied",
    "dst_pts_last4",
    "dst_pts_std_last8",
    "def_sacks_last4",
]


def _frame(target: str = "kicker_pts", rows_per_season: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for season in (2021, 2022):
        team_implied = rng.uniform(14.0, 32.0, rows_per_season)
        opp_implied = rng.uniform(14.0, 32.0, rows_per_season)
        form = rng.normal(7.5, 2.2, rows_per_season)
        vol = rng.uniform(1.5, 5.0, rows_per_season)
        sacks = rng.uniform(0.5, 5.0, rows_per_season)
        if target == "kicker_pts":
            y = (
                0.18 * team_implied
                + 0.45 * form
                + rng.normal(0.0, 1.3, rows_per_season)
            )
        else:
            y = (
                -0.15 * opp_implied
                + 0.55 * form
                + 1.1 * sacks
                + rng.normal(0.0, 1.5, rows_per_season)
            )
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "week": np.arange(rows_per_season) // 8 + 1,
                    "team_implied": team_implied,
                    "opp_implied": opp_implied,
                    "kicker_kicker_pts_last4": form,
                    "kicker_pts_std_last8": vol,
                    "dst_pts_last4": form,
                    "dst_pts_std_last8": vol,
                    "def_sacks_last4": sacks,
                    target: y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize(
    ("target", "features", "blend"),
    [
        ("kicker_pts", KICKER_FEATURES, KICKER_BLEND),
        ("dst_pts", DST_FEATURES, DST_BLEND),
    ],
)
def test_train_and_predict_shapes(
    target: str, features: list[str], blend: float
) -> None:
    frame = _frame(target)
    train = frame[frame["season"] == 2021]
    test = frame[frame["season"] == 2022]

    models = train_models(train, target, features, season=2022, week=1)
    assert models["train_rows"] == len(train)
    assert models["features"] == features

    pred = predict_models(models, test)
    assert len(pred) == len(test)
    assert pred["proj_mean"].notna().all()
    assert (pred["proj_sigma"] > 0).all()
    assert (pred["proj_p10"] < pred["proj_mean"]).all()
    assert (pred["proj_mean"] < pred["proj_p90"]).all()
    assert pred["boom_prob"].between(0.0, 1.0).all()

    expected = (
        pred["proj_mean"]
        + 0.12 * (pred["proj_p90"] - pred["proj_mean"]).clip(lower=0)
        + pred["boom_prob"]
    )
    assert np.allclose(pred["stream_score"], expected)

    if blend < 1.0:
        assert not np.allclose(pred["proj_mean"], pred["model_mean"])
    else:
        assert np.allclose(pred["proj_mean"], pred["model_mean"])


def test_predict_handles_missing_features() -> None:
    frame = _frame("kicker_pts")
    models = train_models(frame[frame["season"] == 2021], "kicker_pts", KICKER_FEATURES)
    test = frame[frame["season"] == 2022].head(3).copy()
    test.loc[test.index[0], "kicker_kicker_pts_last4"] = np.nan
    test.loc[test.index[1], "team_implied"] = np.nan

    pred = predict_models(models, test)
    assert pred["proj_mean"].notna().all()


def test_walk_forward_uses_only_prior_seasons() -> None:
    frame = _frame("dst_pts")
    pred, metrics = walk_forward(frame, "dst_pts", DST_FEATURES, [2022])

    assert len(pred) == 40
    assert (pred["season"] == 2022).all()
    assert "prediction" in pred.columns
    assert metrics["n"] == 40
    assert np.isfinite(metrics["spearman"])
    assert metrics["spearman"] > 0.1


def test_score_predictions_rewards_correct_ranking() -> None:
    target = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    pred = pd.DataFrame(
        {
            "season": [2022] * 8,
            "week": [1] * 8,
            "kicker_pts": target,
            "prediction": target,
        }
    )
    metrics = score_predictions(pred, "kicker_pts")
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["top5_hit"] == pytest.approx(1.0)
    assert metrics["top5_gain"] > 0
    assert metrics["mae"] == pytest.approx(0.0)


def test_baselines_are_scored_on_the_same_holdout() -> None:
    frame = _frame("kicker_pts")
    baselines = baseline_scores(frame, "kicker_pts", [2022])
    assert set(baselines["baseline"]) == {"vegas", "form", "vegas_plus_form"}
    assert (baselines["n"] > 0).all()
    assert np.isfinite(baselines["spearman"]).all()


def test_feature_importance_sorted_by_absolute_coefficient() -> None:
    frame = _frame("kicker_pts")
    models = train_models(frame[frame["season"] == 2021], "kicker_pts", KICKER_FEATURES)
    importance = feature_importance(models)
    assert list(importance.columns) == ["feature", "coefficient", "abs_coefficient"]
    assert len(importance) == len(KICKER_FEATURES)
    assert importance["abs_coefficient"].is_monotonic_decreasing
    assert set(importance["feature"]) == set(KICKER_FEATURES)
