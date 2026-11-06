"""Download and load nflverse data.

Sources
-------
- schedules/games: https://github.com/nflverse/nflverse-data/releases/tag/schedules
- weekly player stats: https://github.com/nflverse/nflverse-data/releases/tag/stats_player
- play-by-play: https://github.com/nflverse/nflverse-data/releases/tag/pbp

Everything is cached as parquet under ``data/raw``. The project never scrapes
live sites, so it stays reproducible and does not depend on an API key.
"""

from __future__ import annotations

import concurrent.futures as futures
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import RAW_DIR, ensure_dirs

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

PBP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "play_type",
    "yards_gained",
    "epa",
    "success",
    "down",
    "ydstogo",
    "yardline_100",
    "pass",
    "rush",
    "sack",
    "qb_hit",
    "interception",
    "fumble",
    "fumble_lost",
    "fumble_recovery_1_team",
    "fumble_recovery_2_team",
    "td_team",
    "touchdown",
    "field_goal_attempt",
    "field_goal_result",
    "extra_point_attempt",
    "extra_point_result",
    "two_point_attempt",
    "two_point_conv_result",
    "defensive_two_point_conv",
    "safety",
    "return_touchdown",
    "punt_blocked",
    "drive",
    "drive_inside20",
    "drive_ended_with_score",
    "fixed_drive_result",
    "drive_time_of_possession",
]

PLAYER_WEEK_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "position_group",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
    "fg_made",
    "fg_att",
    "fg_missed",
    "fg_blocked",
    "fg_long",
    "fg_pct",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed_0_19",
    "fg_missed_20_29",
    "fg_missed_30_39",
    "fg_missed_40_49",
    "fg_missed_50_59",
    "fg_missed_60_",
    "fg_made_distance",
    "fg_missed_distance",
    "pat_made",
    "pat_att",
    "pat_missed",
    "pat_blocked",
    "pat_pct",
    "gwfg_made",
    "gwfg_att",
    "gwfg_distance",
]


def _url(kind: str, filename: str) -> str:
    return f"{BASE}/{kind}/{filename}"


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download one file atomically, returning the cached path."""
    ensure_dirs()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)
    return dest


def download_games(force: bool = False) -> Path:
    return download_file(
        _url("schedules", "games.parquet"), RAW_DIR / "games.parquet", force
    )


def download_player_week(season: int, force: bool = False) -> Path:
    return download_file(
        _url("stats_player", f"stats_player_week_{season}.parquet"),
        RAW_DIR / f"stats_player_week_{season}.parquet",
        force,
    )


def download_pbp(season: int, force: bool = False) -> Path:
    return download_file(
        _url("pbp", f"play_by_play_{season}.parquet"),
        RAW_DIR / f"play_by_play_{season}.parquet",
        force,
    )


def download_seasons(
    seasons: Iterable[int], force: bool = False, workers: int = 4
) -> None:
    """Download schedules plus player weeks and PBP for each season."""
    seasons = sorted(set(int(s) for s in seasons))
    download_games(force=force)
    jobs = []
    for season in seasons:
        jobs.append((download_player_week, season))
        jobs.append((download_pbp, season))

    def run(item):
        fn, season = item
        try:
            return fn(season, force=force)
        except requests.HTTPError as exc:
            # Future/current seasons may not have every asset yet. The pipeline
            # will simply use the seasons that are present.
            return f"skip {season}: {exc}"

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run, jobs):
            if isinstance(result, str):
                print(result)
            else:
                print(f"cached {Path(result).name}")


def load_games(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    path = RAW_DIR / "games.parquet"
    if not path.exists():
        download_games()
    games = pd.read_parquet(path)
    games["gameday"] = pd.to_datetime(games["gameday"], errors="coerce")
    if seasons is not None:
        games = games[games["season"].isin(list(seasons))].copy()
    return games.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def load_player_weeks(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    seasons = sorted(int(s) for s in (seasons or []))
    frames = []
    for season in seasons:
        path = RAW_DIR / f"stats_player_week_{season}.parquet"
        if not path.exists():
            try:
                download_player_week(season)
            except requests.HTTPError:
                continue
        if path.exists():
            frame = pd.read_parquet(path)
            cols = [c for c in PLAYER_WEEK_COLUMNS if c in frame.columns]
            frames.append(frame[cols])
    if not frames:
        return pd.DataFrame(columns=PLAYER_WEEK_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def load_pbp(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    seasons = sorted(int(s) for s in (seasons or []))
    frames = []
    for season in seasons:
        path = RAW_DIR / f"play_by_play_{season}.parquet"
        if not path.exists():
            try:
                download_pbp(season)
            except requests.HTTPError:
                continue
        if path.exists():
            frame = pd.read_parquet(path)
            cols = [c for c in PBP_COLUMNS if c in frame.columns]
            frames.append(frame[cols])
    if not frames:
        return pd.DataFrame(columns=PBP_COLUMNS)
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in PBP_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def available_seasons() -> list[int]:
    """Seasons with both weekly player stats and PBP cached."""
    seasons = []
    for path in RAW_DIR.glob("play_by_play_*.parquet"):
        try:
            season = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if (RAW_DIR / f"stats_player_week_{season}.parquet").exists():
            seasons.append(season)
    return sorted(seasons)


def latest_season_with_games() -> int:
    games = load_games()
    return int(games["season"].max())


def current_week(season: int | None = None) -> int:
    """First week with unplayed games (the current slate), else the last week."""
    games = load_games([season] if season else None)
    games = games[games["game_type"] == "REG"]
    if season is not None:
        games = games[games["season"] == season]
    unplayed = games[games["home_score"].isna()]
    if not unplayed.empty:
        return int(unplayed["week"].min())
    return int(games["week"].max()) if not games.empty else 1
