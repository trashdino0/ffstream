"""Walk-forward evaluation and backtest artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .build import load_processed
from .config import OUTPUT_DIR, ensure_dirs
from .features import (
    build_dst_features,
    build_kicker_features,
    build_team_features,
    dst_model_features,
    kicker_model_features,
)
from .model import (
    baseline_scores,
    feature_importance,
    score_predictions,
    train_final,
    walk_forward,
    within_season_backtest,
)

SUMMARY_COLUMNS = [
    "target",
    "evaluation",
    "model",
    "n",
    "mae",
    "rmse",
    "spearman",
    "top1_actual",
    "top3_actual",
    "top5_actual",
    "top5_hit",
    "top5_gain",
]


def _row(metrics: dict, target: str, evaluation: str, model: str) -> dict:
    row = {k: metrics.get(k) for k in SUMMARY_COLUMNS}
    row.update({"target": target, "evaluation": evaluation, "model": model})
    return row


def _per_week(pred: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for (season, week), group in pred.groupby(["season", "week"], sort=False):
        group = group.dropna(subset=[target, "prediction"])
        if group.empty:
            continue
        ordered = group.sort_values("prediction", ascending=False)
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "n": len(group),
                "league_avg": group[target].mean(),
                "top1_actual": ordered.head(1)[target].mean(),
                "top3_actual": ordered.head(3)[target].mean(),
                "top5_actual": ordered.head(5)[target].mean(),
                "oracle_top5": group[target].nlargest(5).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_features():
    team_games, kicker_games = load_processed()
    team_features = build_team_features(team_games)
    kicker_features = build_kicker_features(kicker_games, team_features)
    dst_features = build_dst_features(team_features)
    return team_features, kicker_features, dst_features


def run_backtest(
    seasons: Iterable[int] = (2021, 2022, 2023, 2024, 2025),
    include_2026: bool = True,
) -> pd.DataFrame:
    ensure_dirs()
    seasons = sorted(int(s) for s in seasons)
    team_features, kicker_features, dst_features = build_features()
    kicker_feats = kicker_model_features(kicker_features)
    dst_feats = dst_model_features(dst_features)

    summary_rows: list[dict] = []
    for target, frame, features in [
        ("kicker_pts", kicker_features, kicker_feats),
        ("dst_pts", dst_features, dst_feats),
    ]:
        print(f"\n=== {target} ===")
        pred, metrics = walk_forward(frame, target, features, seasons)
        summary_rows.append(
            _row(metrics, target, f"walk-forward {seasons[0]}-{seasons[-1]}", "model")
        )
        stream = pred.copy()
        stream["prediction"] = stream["stream_score"]
        stream_metrics = score_predictions(stream, target)
        summary_rows.append(
            _row(
                stream_metrics,
                target,
                f"walk-forward {seasons[0]}-{seasons[-1]}",
                "model stream-score",
            )
        )

        baselines = baseline_scores(frame, target, seasons)
        for _, row in baselines.iterrows():
            summary_rows.append(
                _row(
                    row.to_dict(),
                    target,
                    f"walk-forward {seasons[0]}-{seasons[-1]}",
                    row["baseline"],
                )
            )

        pred.to_parquet(
            OUTPUT_DIR / f"backtest_predictions_{target}.parquet", index=False
        )
        _per_week(pred, target).to_csv(
            OUTPUT_DIR / f"backtest_weekly_{target}.csv", index=False
        )

        if include_2026:
            holdout, holdout_metrics = walk_forward(frame, target, features, [2026])
            summary_rows.append(_row(holdout_metrics, target, "2026 holdout", "model"))
            holdout_stream = holdout.copy()
            holdout_stream["prediction"] = holdout_stream["stream_score"]
            summary_rows.append(
                _row(
                    score_predictions(holdout_stream, target),
                    target,
                    "2026 holdout",
                    "model stream-score",
                )
            )
            weeks = sorted(
                frame[(frame["season"] == 2026) & frame[target].notna()][
                    "week"
                ].unique()
            )
            weekly, weekly_metrics = within_season_backtest(
                frame, target, features, 2026, weeks
            )
            summary_rows.append(_row(weekly_metrics, target, "2026 weekly", "model"))
            for _, row in baseline_scores(frame, target, [2026]).iterrows():
                summary_rows.append(
                    _row(row.to_dict(), target, "2026 holdout", row["baseline"])
                )
            holdout.to_parquet(
                OUTPUT_DIR / f"backtest_2026_predictions_{target}.parquet", index=False
            )
            weekly.to_parquet(
                OUTPUT_DIR / f"backtest_2026_weekly_predictions_{target}.parquet",
                index=False,
            )

        final = train_final(frame, target, features, 2026, 9)
        importance = feature_importance(final, top_n=30)
        importance.to_csv(OUTPUT_DIR / f"feature_importance_{target}.csv", index=False)
        print(
            f"model MAE {metrics.get('mae', float('nan')):.3f} | "
            f"Spearman {metrics.get('spearman', float('nan')):.3f} | "
            f"top5 {metrics.get('top5_actual', float('nan')):.2f} "
            f"(+{metrics.get('top5_gain', float('nan')):.2f} vs league)"
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary[SUMMARY_COLUMNS]
    summary.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR / 'backtest_summary.csv'}")
    return summary


def main() -> None:
    run_backtest()


if __name__ == "__main__":
    main()
