"""Command line interface for FF Stream Lab."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .build import build_all, load_processed
from .config import OUTPUT_DIR, load_scoring
from .data import current_week, download_seasons, latest_season_with_games
from .features import (
    build_dst_features,
    build_kicker_features,
    build_team_features,
    dst_model_features,
    kicker_model_features,
    make_dst_prediction_rows,
    make_kicker_prediction_rows,
)
from .model import predict_models, save_models, train_final
from .report import prepare_table, write_weekly_outputs

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _parse_seasons(value: str | None) -> list[int]:
    if not value:
        return list(range(2014, 2027))
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _display(table: pd.DataFrame, top: int, title: str) -> None:
    print(f"\n{title}")
    columns = [
        "rank",
        "player",
        "team",
        "opponent",
        "projected",
        "floor",
        "ceiling",
        "boom_pct",
        "tier",
    ]
    columns = [c for c in columns if c in table.columns]
    print(table.head(top)[columns].to_string(index=False))


def _filter_to_slate(
    rows: pd.DataFrame,
    team_features: pd.DataFrame,
    season: int,
    week: int,
    include_completed: bool,
) -> pd.DataFrame:
    if include_completed or rows.empty:
        return rows
    slate = team_features[
        (team_features["season"] == season)
        & (team_features["week"] == week)
        & team_features["team_score"].isna()
    ]
    if slate.empty:
        return rows
    return rows[rows["game_id"].isin(set(slate["game_id"]))].copy()


def run_week(
    season: int | None = None,
    week: int | None = None,
    top: int = 15,
    include_completed: bool = False,
    rank_by: str = "auto",
    out_dir: Path | None = None,
) -> dict[str, str]:
    team_games, kicker_games = load_processed()
    team_features = build_team_features(team_games)
    kicker_features = build_kicker_features(kicker_games, team_features)
    dst_features = build_dst_features(team_features)

    season = int(season or latest_season_with_games())
    week = int(week or current_week(season))
    print(f"Building {season} Week {week} streaming board…")

    kicker_rows = make_kicker_prediction_rows(
        team_features, kicker_features, season, week
    )
    dst_rows = make_dst_prediction_rows(team_features, season, week)
    kicker_rows = _filter_to_slate(
        kicker_rows, team_features, season, week, include_completed
    )
    dst_rows = _filter_to_slate(
        dst_rows, team_features, season, week, include_completed
    )
    if kicker_rows.empty or dst_rows.empty:
        raise SystemExit(
            f"No games found for {season} Week {week}. Check the season/week or run `ffstream refresh`."
        )

    kicker_feats = kicker_model_features(kicker_features)
    dst_feats = dst_model_features(dst_features)
    kicker_models = train_final(
        kicker_features, "kicker_pts", kicker_feats, season, week
    )
    dst_models = train_final(dst_features, "dst_pts", dst_feats, season, week)

    kicker_pred = predict_models(kicker_models, kicker_rows)
    dst_pred = predict_models(dst_models, dst_rows)

    kicker_rank = "stream" if rank_by in {"auto", "stream"} else "mean"
    dst_rank = "stream" if rank_by == "stream" else "mean"
    kicker_table = prepare_table(kicker_rows, kicker_pred, "kicker_pts", kicker_rank)
    dst_table = prepare_table(dst_rows, dst_pred, "dst_pts", dst_rank)

    meta = {
        "kicker_train_rows": kicker_models.get("train_rows", 0),
        "dst_train_rows": dst_models.get("train_rows", 0),
        "notes": "Model trained on all games before the target week.",
    }
    paths = write_weekly_outputs(
        kicker_table, dst_table, season, week, meta=meta, out_dir=out_dir
    )
    save_models(kicker_models, f"kicker_{season}_{week}")
    save_models(dst_models, f"dst_{season}_{week}")

    _display(kicker_table, top, f"{season} Week {week} — Kicker streams")
    _display(dst_table, top, f"{season} Week {week} — Team-defense streams")
    print("\nSaved:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return paths


def run_train(season: int, week: int) -> None:
    team_games, kicker_games = load_processed()
    team_features = build_team_features(team_games)
    kicker_features = build_kicker_features(kicker_games, team_features)
    dst_features = build_dst_features(team_features)
    kicker_models = train_final(
        kicker_features,
        "kicker_pts",
        kicker_model_features(kicker_features),
        season,
        week,
    )
    dst_models = train_final(
        dst_features, "dst_pts", dst_model_features(dst_features), season, week
    )
    print(save_models(kicker_models, f"kicker_{season}_{week}"))
    print(save_models(dst_models, f"dst_{season}_{week}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffstream",
        description="Fantasy football streaming models for kickers and team defenses.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    refresh = sub.add_parser("refresh", help="Download/cache nflverse data.")
    refresh.add_argument("--seasons", default=None, help="e.g. 2014-2026")
    refresh.add_argument("--force", action="store_true")

    build = sub.add_parser("build", help="Build processed team-game and kicker tables.")
    build.add_argument("--seasons", default=None, help="e.g. 2014-2026")

    backtest = sub.add_parser("backtest", help="Run walk-forward validation.")
    backtest.add_argument("--seasons", default="2021-2025", help="e.g. 2021-2025")
    backtest.add_argument("--no-2026", action="store_true")

    week = sub.add_parser("week", help="Generate the weekly streaming board.")
    week.add_argument("--season", type=int, default=None)
    week.add_argument("--week", type=int, default=None)
    week.add_argument("--top", type=int, default=15)
    week.add_argument(
        "--include-completed",
        action="store_true",
        help="Include games that already have a final score.",
    )
    week.add_argument(
        "--rank-by",
        choices=["auto", "mean", "stream"],
        default="auto",
        help="auto uses stream score for kickers and mean for defenses.",
    )
    week.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)

    train = sub.add_parser("train", help="Train and save models for a target week.")
    train.add_argument("--season", type=int, required=True)
    train.add_argument("--week", type=int, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "refresh":
        download_seasons(_parse_seasons(args.seasons), force=args.force)
        return

    if args.command == "build":
        scoring = load_scoring()
        build_all(_parse_seasons(args.seasons), scoring)
        return

    if args.command == "backtest":
        run_backtest(
            _parse_seasons(args.seasons),
            include_2026=not args.no_2026,
        )
        return

    if args.command == "week":
        run_week(
            season=args.season,
            week=args.week,
            top=args.top,
            include_completed=args.include_completed,
            rank_by=args.rank_by,
            out_dir=args.out_dir,
        )
        return

    if args.command == "train":
        run_train(args.season, args.week)
        return

    parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
