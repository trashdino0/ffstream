"""Grade a saved weekly board against actual results.

Reads the board CSVs written by ``ffstream week`` and scores them with the
same metrics used in the backtests: MAE/RMSE, bias, rank correlation, interval
coverage, boom calibration, and top-k performance versus the league average.

Usage:
    python -m ffstream grade --season 2026 --week 10
    python -m ffstream grade --latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .build import load_processed
from .config import OUTPUT_DIR

DEFAULT_BOARD_DIR = OUTPUT_DIR
DEFAULT_OUT_DIR = OUTPUT_DIR

KICKER_START_THRESHOLD = 10.0
DST_START_THRESHOLD = 12.0


def _pct(fraction: float) -> str:
    if not np.isfinite(fraction):
        return "—"
    return f"{100 * fraction:.1f}%"


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _safe_pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def _percentile(rank: pd.Series, count: int) -> pd.Series:
    if count <= 1:
        return pd.Series(100.0, index=rank.index)
    return ((count - rank) / (count - 1) * 100).clip(0, 100)


def _metrics(frame: pd.DataFrame, threshold: float) -> dict:
    scored = frame[frame["actual"].notna()].copy()
    errors = scored["projected"] - scored["actual"]
    league = float(scored["actual"].mean()) if len(scored) else float("nan")
    board = frame.sort_values("rank")
    top = board.head(5)
    covered = (
        (scored["actual"] >= scored["floor"]) & (scored["actual"] <= scored["ceiling"])
    ).mean()
    boom_pred = float(scored["boom_pct"].mean() / 100) if len(scored) else float("nan")
    boom_actual = (
        float((scored["actual"] >= threshold).mean()) if len(scored) else float("nan")
    )
    return {
        "rows": len(scored),
        "board_rows": len(frame),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "bias": float(errors.mean()),
        "spearman": _safe_spearman(scored["projected"], scored["actual"]),
        "pearson": _safe_pearson(scored["projected"], scored["actual"]),
        "league_avg": league,
        "coverage": float(covered),
        "boom_pred": boom_pred,
        "boom_actual": boom_actual,
        "top1_actual": float(top["actual"].dropna().iloc[0])
        if top["actual"].notna().any()
        else float("nan"),
        "top5_avg": float(top["actual"].mean()),
        "top5_vs_league": float(top["actual"].mean() - league),
        "top5_beating_league": int((top["actual"] > league).sum()),
        "top5_overlap": int(top["actual_rank"].le(5).sum())
        if "actual_rank" in top
        else 0,
    }


def _actual_top(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    scored = frame[frame["actual"].notna()].copy()
    top = scored[scored["actual_rank"] <= n].sort_values("actual_rank")
    return top[["name", "team", "opponent", "actual", "rank"]]


def _fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def board_paths(season: int, week: int, board_dir: Path) -> tuple[Path, Path]:
    return (
        board_dir / f"week_{season}_{week:02d}_kickers.csv",
        board_dir / f"week_{season}_{week:02d}_dst.csv",
    )


def latest_completed_week(season: int) -> int | None:
    """Most recent week with actual results in the processed kicker table."""
    _, kicker_games = load_processed()
    played = kicker_games[
        (kicker_games["season"] == season) & (kicker_games["kicker_pts"].notna())
    ]
    if played.empty:
        return None
    return int(played["week"].max())


def latest_gradable_week(
    season: int, board_dir: Path = DEFAULT_BOARD_DIR
) -> int | None:
    """Most recent completed week that has a saved board to compare against."""
    completed = latest_completed_week(season)
    if completed is None:
        return None
    for week in range(completed, 0, -1):
        kicker_csv, dst_csv = board_paths(season, week, board_dir)
        if kicker_csv.exists() and dst_csv.exists():
            return week
    return None


def grade(
    season: int, week: int, board_dir: Path, out_dir: Path, top_n: int
) -> dict[str, Path]:
    team_games, kicker_games = load_processed()
    kickers = kicker_games[
        (kicker_games.season == season) & (kicker_games.week == week)
    ].copy()
    teams = team_games[(team_games.season == season) & (team_games.week == week)].copy()
    teams = teams[teams["dst_pts"].notna()]

    kicker_csv, dst_csv = board_paths(season, week, board_dir)
    for path in (kicker_csv, dst_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"Board file not found: {path}. Run `ffstream week --season {season} "
                f"--week {week}` before grading."
            )
    kicker_board = pd.read_csv(kicker_csv)
    dst_board = pd.read_csv(dst_csv)

    # Kickers: match on player_id, then fall back to the team's actual kicker.
    kicker_board = kicker_board.merge(
        kickers[["player_id", "player_display_name", "team", "kicker_pts"]],
        on="player_id",
        how="left",
        suffixes=("", "_actual"),
    )
    used = kickers.set_index("player_id")["kicker_pts"]
    kicker_board["actual"] = kicker_board["player_id"].map(used)
    kicker_board["status"] = np.where(
        kicker_board["actual"].notna(), "matched", "not_in_box_score"
    )
    if "player_display_name" not in kicker_board.columns:
        kicker_board["player_display_name"] = np.nan

    # Team-credited fallback: if the listed kicker did not play, credit whoever
    # actually kicked for that team so the board's slot can still be graded.
    kicker_board["slot_actual"] = kicker_board.apply(
        lambda row: (
            row["actual"]
            if pd.notna(row["actual"])
            else (kickers.loc[kickers["team"] == row["team"], "kicker_pts"].max())
        ),
        axis=1,
    )
    kicker_board["actual_kicker"] = kicker_board.apply(
        lambda row: (
            row["player"]
            if pd.notna(row["actual"])
            else (
                kickers.loc[kickers["team"] == row["team"], "player_display_name"].max()
                if (kickers["team"] == row["team"]).any()
                else ""
            )
        ),
        axis=1,
    )
    rank_counts = kickers["kicker_pts"].rank(ascending=False, method="min")
    rank_map = dict(zip(kickers["player_id"], rank_counts))
    kicker_board["actual_rank"] = kicker_board["player_id"].map(rank_map)
    kicker_board["percentile"] = _percentile(kicker_board["actual_rank"], len(kickers))
    kicker_board["in_interval"] = (kicker_board["actual"] >= kicker_board["floor"]) & (
        kicker_board["actual"] <= kicker_board["ceiling"]
    )
    kicker_board["error"] = kicker_board["projected"] - kicker_board["actual"]
    kicker_board["position"] = "K"
    kicker_board["name"] = kicker_board["player"]

    dst_board = dst_board.merge(
        teams[["team", "opponent", "dst_pts"]],
        on=["team", "opponent"],
        how="left",
        suffixes=("", "_actual"),
    )
    dst_board["actual"] = dst_board["dst_pts"]
    dst_board["status"] = np.where(dst_board["actual"].notna(), "matched", "missing")
    dst_board["slot_actual"] = dst_board["actual"]
    dst_board["actual_kicker"] = ""
    dst_rank = teams["dst_pts"].rank(ascending=False, method="min")
    rank_map = dict(zip(teams["team"], dst_rank))
    dst_board["actual_rank"] = dst_board["team"].map(rank_map)
    dst_board["percentile"] = _percentile(dst_board["actual_rank"], len(teams))
    dst_board["in_interval"] = (dst_board["actual"] >= dst_board["floor"]) & (
        dst_board["actual"] <= dst_board["ceiling"]
    )
    dst_board["error"] = dst_board["projected"] - dst_board["actual"]
    dst_board["position"] = "DST"
    dst_board["name"] = dst_board["team"]

    columns = [
        "position",
        "rank",
        "name",
        "team",
        "opponent",
        "projected",
        "proj_mean",
        "floor",
        "ceiling",
        "boom_pct",
        "tier",
        "actual",
        "error",
        "actual_rank",
        "percentile",
        "in_interval",
        "status",
        "actual_kicker",
        "slot_actual",
    ]
    evaluation = pd.concat(
        [kicker_board[columns], dst_board[columns]], ignore_index=True
    ).sort_values(["position", "rank"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"week_{season}_{week:02d}_evaluation.csv"
    evaluation.to_csv(csv_path, index=False)

    k_metrics = _metrics(kicker_board, KICKER_START_THRESHOLD)
    d_metrics = _metrics(dst_board, DST_START_THRESHOLD)
    k_metrics_slot = _metrics(
        kicker_board.assign(actual=kicker_board["slot_actual"]),
        KICKER_START_THRESHOLD,
    )

    lines = [
        f"# {season} Week {week} Board Accuracy",
        "",
        "Board generated before the week's games. Kickers are evaluated in the board's "
        "default order (stream score); defenses in projected-points order.",
        "",
        "## Kickers",
        "",
    ]
    not_matched = kicker_board[kicker_board["status"] != "matched"]
    if not not_matched.empty:
        lines.append(
            f"- Rows evaluated: {k_metrics['rows']} of {k_metrics['board_rows']} board picks"
        )
    lines += [
        f"- MAE: {_fmt(k_metrics['mae'])}",
        f"- RMSE: {_fmt(k_metrics['rmse'])}",
        f"- Bias (projected - actual): {_fmt(k_metrics['bias'])}",
        f"- Spearman (projected points): {_fmt(k_metrics['spearman'], 3)}",
        f"- Spearman (board rank): {_fmt(k_metrics['spearman'], 3)}",
        f"- Pearson: {_fmt(k_metrics['pearson'], 3)}",
        f"- League average actual: {_fmt(k_metrics['league_avg'])}",
        f"- Floor/ceiling coverage: {_pct(k_metrics['coverage'])}",
        f"- Boom calibration: predicted {_pct(k_metrics['boom_pred'])} vs actual "
        f"{_pct(k_metrics['boom_actual'])} (threshold {KICKER_START_THRESHOLD:g})",
        f"- Top-1 actual: {_fmt(k_metrics['top1_actual'])}",
        f"- Top-5 average actual: {_fmt(k_metrics['top5_avg'])} vs league "
        f"{_fmt(k_metrics['league_avg'])} ({_fmt(k_metrics['top5_vs_league'], 2)})",
        f"- Top-5 picks beating league average: {k_metrics['top5_beating_league']}/5",
        f"- Top-5 overlap with actual top 5: {k_metrics['top5_overlap']}/5",
        "",
    ]
    if not not_matched.empty:
        lines.append(
            "**Slot-credited view** (a listed kicker who did not play is replaced by "
            "the team's actual kicker):"
        )
        lines.append("")
        lines.append(
            f"- Top-5 average actual: {_fmt(k_metrics_slot['top5_avg'])} vs league "
            f"{_fmt(k_metrics_slot['league_avg'])} "
            f"({_fmt(k_metrics_slot['top5_vs_league'], 2)})"
        )
        lines.append(
            f"- Top-5 picks beating league average: {k_metrics_slot['top5_beating_league']}/5"
        )
        lines.append("")
    lines += [
        "| Pick | Kicker | Team | Opp | Proj | Actual | Actual rank | Pct |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in kicker_board.sort_values("rank").head(top_n).iterrows():
        actual = (
            _fmt(row["actual"])
            if pd.notna(row["actual"])
            else f"({_fmt(row['slot_actual'])})"
        )
        rank = _fmt(row["actual_rank"], 0) if pd.notna(row["actual_rank"]) else "—"
        pctl = _fmt(row["percentile"], 0) if pd.notna(row["percentile"]) else "—"
        lines.append(
            f"| {int(row['rank'])} | {row['player']} | {row['team']} | {row['opponent']} "
            f"| {_fmt(row['projected'])} | {actual} | {rank} | {pctl} |"
        )
    lines += [
        "",
        "**Actual top 5**",
        "",
        "| Player/Team | Actual | Model rank |",
        "| --- | ---: | ---: |",
    ]
    for _, row in _actual_top(kicker_board, 5).iterrows():
        lines.append(f"| {row['name']} | {_fmt(row['actual'])} | {int(row['rank'])} |")
    lines += [
        "",
        "## Team defenses",
        "",
        f"- Rows evaluated: {d_metrics['rows']} of {d_metrics['board_rows']} board picks",
        f"- MAE: {_fmt(d_metrics['mae'])}",
        f"- RMSE: {_fmt(d_metrics['rmse'])}",
        f"- Bias (projected - actual): {_fmt(d_metrics['bias'])}",
        f"- Spearman: {_fmt(d_metrics['spearman'], 3)}",
        f"- Pearson: {_fmt(d_metrics['pearson'], 3)}",
        f"- League average actual: {_fmt(d_metrics['league_avg'])}",
        f"- Floor/ceiling coverage: {_pct(d_metrics['coverage'])}",
        f"- Boom calibration: predicted {_pct(d_metrics['boom_pred'])} vs actual "
        f"{_pct(d_metrics['boom_actual'])} (threshold {DST_START_THRESHOLD:g})",
        f"- Top-1 actual: {_fmt(d_metrics['top1_actual'])}",
        f"- Top-5 average actual: {_fmt(d_metrics['top5_avg'])} vs league "
        f"{_fmt(d_metrics['league_avg'])} ({_fmt(d_metrics['top5_vs_league'], 2)})",
        f"- Top-5 picks beating league average: {d_metrics['top5_beating_league']}/5",
        f"- Top-5 overlap with actual top 5: {d_metrics['top5_overlap']}/5",
        "",
        "| Pick | Defense | Opp | Proj | Actual | Actual rank | Pct |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in dst_board.sort_values("rank").head(top_n).iterrows():
        rank = _fmt(row["actual_rank"], 0) if pd.notna(row["actual_rank"]) else "—"
        pctl = _fmt(row["percentile"], 0) if pd.notna(row["percentile"]) else "—"
        lines.append(
            f"| {int(row['rank'])} | {row['team']} | {row['opponent']} "
            f"| {_fmt(row['projected'])} | {_fmt(row['actual'])} | {rank} | {pctl} |"
        )
    lines += [
        "",
        "**Actual top 5**",
        "",
        "| Player/Team | Actual | Model rank |",
        "| --- | ---: | ---: |",
    ]
    for _, row in _actual_top(dst_board, 5).iterrows():
        lines.append(f"| {row['name']} | {_fmt(row['actual'])} | {int(row['rank'])} |")
    lines += ["", "## Notes", ""]
    if not not_matched.empty:
        for _, row in not_matched.iterrows():
            lines.append(
                f"- {row['player']} ({row['team']}) was on the board at #{int(row['rank'])} "
                f"but did not appear in the Week {week} box score; "
                f"{row['actual_kicker'] or 'the replacement'} kicked and scored "
                f"{_fmt(row['slot_actual'])}."
            )
    if "team_actual" in kicker_board.columns:
        moved = kicker_board[
            (kicker_board["status"] == "matched")
            & (kicker_board["team_actual"].notna())
            & (kicker_board["team_actual"] != kicker_board["team"])
        ]
        for _, row in moved.iterrows():
            lines.append(
                f"- {row['player']} was listed with {row['team']} but recorded the Week {week} "
                f"game with {row['team_actual']}; the player-level score is still graded."
            )
    cleaned: list[str] = []
    for line in lines:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    lines = cleaned + [""]
    lines.append(
        "*Scoring: `config/scoring.json`. Actuals computed from refreshed nflverse "
        "player-week and play-by-play data with the project's `add_kicker_points` / "
        "`add_dst_points`.*"
    )
    md_path = out_dir / f"week_{season}_{week:02d}_accuracy.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"csv": csv_path, "markdown": md_path}


def resolve_week(
    season: int,
    week: int | None,
    latest: bool,
    board_dir: Path = DEFAULT_BOARD_DIR,
) -> int | None:
    if week:
        return week
    if latest:
        return latest_gradable_week(season, board_dir)
    return latest_gradable_week(season, board_dir)


def run_grade(
    season: int,
    week: int | None = None,
    latest: bool = False,
    board_dir: Path | None = None,
    out_dir: Path | None = None,
    top: int = 5,
) -> int:
    board_dir = board_dir or DEFAULT_BOARD_DIR
    out_dir = out_dir or DEFAULT_OUT_DIR
    resolved = resolve_week(season, week, latest, board_dir)
    if resolved is None:
        if latest:
            print(
                f"Nothing to grade for {season}: no completed week has a saved "
                "board yet."
            )
            return 0
        print(f"No completed week found for {season}.")
        return 1

    try:
        paths = grade(season, resolved, board_dir, out_dir, top)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    print(f"Graded {season} Week {resolved}")
    print(f"wrote {paths['csv']}")
    print(f"wrote {paths['markdown']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade a saved weekly board against actual results."
    )
    parser.add_argument("--season", type=int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--week", type=int)
    target.add_argument(
        "--latest",
        action="store_true",
        help="Grade the most recent completed week that has a saved board.",
    )
    parser.add_argument("--board-dir", type=Path, default=DEFAULT_BOARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--top", type=int, default=5, help="Board rows shown per table."
    )
    args = parser.parse_args(argv)

    return run_grade(
        args.season,
        week=args.week,
        latest=args.latest,
        board_dir=args.board_dir,
        out_dir=args.out_dir,
        top=args.top,
    )


if __name__ == "__main__":
    raise SystemExit(main())
