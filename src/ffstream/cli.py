"""Command line interface for FF Stream Lab."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .build import build_all, load_processed
from .config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, load_scoring
from .data import (
    current_week,
    download_seasons,
    latest_season_with_games,
    load_weekly_rosters,
)
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
from .roster import apply_roster_overrides

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
    try:
        rosters = load_weekly_rosters([season])
    except Exception as exc:  # noqa: BLE001 - roster data must not break the board
        print(f"warning: roster check skipped ({exc})")
        rosters = pd.DataFrame()
    kicker_rows = apply_roster_overrides(
        kicker_rows, kicker_features, rosters, season, week
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


def _check(label: str, ok: bool, detail: str = "", critical: bool = False) -> bool:
    """Print one check and return True only when it is a blocking failure."""
    status = "OK" if ok else ("FAIL" if critical else "WARN")
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
    return (not ok) and critical


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def run_doctor(season: int | None = None) -> int:
    """Check the environment, data files, and output directory."""
    failed = False

    print("FF Stream Lab doctor\n")

    version = sys.version_info
    failed |= _check(
        "Python",
        version >= (3, 10),
        f"{version.major}.{version.minor}.{version.micro}",
        critical=True,
    )
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    failed |= _check(
        "Virtual environment",
        in_venv,
        sys.prefix,
        critical=False,
    )

    packages = [
        "pandas",
        "numpy",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "joblib",
        "requests",
    ]
    versions = ", ".join(f"{pkg} {_package_version(pkg)}" for pkg in packages)
    missing = [pkg for pkg in packages if _package_version(pkg) == "missing"]
    failed |= _check("Packages", not missing, versions, critical=True)

    try:
        scoring = load_scoring()
        failed |= _check(
            "Scoring config",
            True,
            f"{len(scoring.get('kicker', {}))} kicker rules, "
            f"{len(scoring.get('dst', {}))} DST rules",
        )
    except Exception as exc:  # noqa: BLE001 - doctor should report, not crash
        failed |= _check("Scoring config", False, str(exc), critical=True)

    games_path = RAW_DIR / "games.parquet"
    games_ok = games_path.exists()
    failed |= _check(
        "Raw games",
        games_ok,
        str(games_path) if games_ok else "missing; run: ffstream refresh",
        critical=True,
    )

    if season is None:
        seasons = []
        if games_ok:
            try:
                seasons = pd.read_parquet(games_path, columns=["season"])[
                    "season"
                ].unique()
            except Exception:  # noqa: BLE001
                seasons = []
        season = int(max(seasons)) if len(seasons) else None

    current = None
    if games_ok and season is not None:
        games = pd.read_parquet(games_path)
        regular = games[(games["game_type"] == "REG") & (games["season"] == season)]
        unplayed = regular[regular["home_score"].isna()]
        current = (
            int(unplayed["week"].min())
            if not unplayed.empty
            else int(regular["week"].max())
        )
        completed = regular[regular["home_score"].notna()]
        last_week = int(completed["week"].max()) if not completed.empty else None
        print(
            f"\nSeason {season}: current week {current}, last completed week {last_week}"
        )
        age_days = (
            datetime.now() - datetime.fromtimestamp(games_path.stat().st_mtime)
        ).total_seconds() / 86400
        _check(
            "Schedule freshness",
            age_days <= 3,
            f"{age_days:.1f} days old",
            critical=False,
        )

    raw_files = [
        (
            f"Player stats {season}",
            RAW_DIR / f"stats_player_week_{season}.parquet",
            True,
        ),
        (f"Play-by-play {season}", RAW_DIR / f"play_by_play_{season}.parquet", True),
        (
            f"Weekly rosters {season}",
            RAW_DIR / f"roster_weekly_{season}.parquet",
            False,
        ),
    ]
    for label, path, critical in raw_files:
        failed |= _check(
            label,
            path.exists(),
            path.name if path.exists() else "missing; run: ffstream refresh",
            critical=critical,
        )

    for label, path in [
        ("Processed team games", PROCESSED_DIR / "team_games.parquet"),
        ("Processed kicker games", PROCESSED_DIR / "kicker_games.parquet"),
    ]:
        ok = path.exists()
        detail = path.name
        if ok:
            try:
                frame = pd.read_parquet(path)
                if frame.empty:
                    detail = "0 rows"
                else:
                    latest = frame.sort_values(["season", "week"]).iloc[-1]
                    detail = (
                        f"{len(frame):,} rows, latest {int(latest['season'])} "
                        f"week {int(latest['week'])}"
                    )
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = str(exc)
        failed |= _check(label, ok, detail, critical=True)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = OUTPUT_DIR / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        failed |= _check("Output directory", True, str(OUTPUT_DIR))
    except Exception as exc:  # noqa: BLE001
        failed |= _check("Output directory", False, str(exc), critical=True)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    pip_detail = (result.stdout or result.stderr).strip() or f"exit {result.returncode}"
    failed |= _check("pip check", result.returncode == 0, pip_detail, critical=True)

    print()
    if failed:
        print("Doctor found blocking problems.")
        return 1
    print("Doctor found no blocking problems.")
    return 0


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

    doctor = sub.add_parser(
        "doctor", help="Check Python, data freshness, and output paths."
    )
    doctor.add_argument("--season", type=int, default=None)

    grade = sub.add_parser(
        "grade", help="Grade a saved weekly board against actual results."
    )
    grade.add_argument("--season", type=int, default=None)
    grade_target = grade.add_mutually_exclusive_group()
    grade_target.add_argument("--week", type=int)
    grade_target.add_argument(
        "--latest",
        action="store_true",
        help="Grade the most recent completed week that has a saved board.",
    )
    grade.add_argument("--board-dir", type=Path, default=None)
    grade.add_argument("--out-dir", type=Path, default=None)
    grade.add_argument("--top", type=int, default=5)

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

    if args.command == "doctor":
        raise SystemExit(run_doctor(args.season))

    if args.command == "grade":
        from .grade import run_grade

        season = args.season or latest_season_with_games()
        raise SystemExit(
            run_grade(
                season,
                week=args.week,
                latest=args.latest,
                board_dir=args.board_dir,
                out_dir=args.out_dir,
                top=args.top,
            )
        )

    parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
