"""Build clean team-game and kicker-game tables from nflverse raw data."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, ensure_dirs
from .data import load_games, load_pbp, load_player_weeks
from .scoring import add_dst_points, add_kicker_points

TEAM_GAMES_PATH = PROCESSED_DIR / "team_games.parquet"
KICKER_GAMES_PATH = PROCESSED_DIR / "kicker_games.parquet"


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _clock_to_seconds(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.fullmatch(r"(\d+):(\d{2})", text)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    try:
        return float(text)
    except ValueError:
        return np.nan


def _offense_table(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    plays = _numeric(
        plays,
        [
            "yards_gained",
            "epa",
            "success",
            "pass",
            "rush",
            "sack",
            "qb_hit",
            "interception",
            "fumble_lost",
        ],
    )
    plays["off_td"] = (
        (plays["td_team"] == plays["posteam"]) & (plays["touchdown"] == 1)
    ).astype(int)

    grouped = plays.groupby(["game_id", "posteam"], as_index=False).agg(
        off_plays=("play_type", "size"),
        off_yards=("yards_gained", "sum"),
        off_epa=("epa", "mean"),
        off_success=("success", "mean"),
        off_turnovers=("interception", "sum"),
        off_sacks=("sack", "sum"),
        off_qb_hits=("qb_hit", "sum"),
        off_tds=("off_td", "sum"),
        off_explosive_rate=("yards_gained", lambda s: float((s >= 20).mean())),
    )
    plays["fumble_lost"] = plays["fumble_lost"].fillna(0)
    lost = plays.groupby(["game_id", "posteam"], as_index=False)["fumble_lost"].sum()
    grouped = grouped.merge(lost, on=["game_id", "posteam"], how="left")
    grouped["off_turnovers"] = grouped["off_turnovers"].fillna(0) + grouped[
        "fumble_lost"
    ].fillna(0)
    grouped = grouped.drop(columns=["fumble_lost"])

    # Separate pass/rush splits.
    for side, mask in (("pass", plays["pass"] == 1), ("rush", plays["rush"] == 1)):
        split = (
            plays[mask]
            .groupby(["game_id", "posteam"], as_index=False)
            .agg(
                **{
                    f"off_{side}_yards": ("yards_gained", "sum"),
                    f"off_{side}_epa": ("epa", "mean"),
                }
            )
        )
        grouped = grouped.merge(split, on=["game_id", "posteam"], how="left")

    # Tendencies on early downs.
    early = plays[plays["down"].isin([1, 2])]
    early_stats = early.groupby(["game_id", "posteam"], as_index=False).agg(
        off_early_down_plays=("play_type", "size"),
        off_early_down_pass_rate=("pass", "mean"),
    )
    grouped = grouped.merge(early_stats, on=["game_id", "posteam"], how="left")

    # Passing play share (pass attempts including sacks vs. all plays).
    pass_plays = (
        plays[plays["pass"] == 1].groupby(["game_id", "posteam"], as_index=False).size()
    )
    pass_plays = pass_plays.rename(columns={"size": "off_pass_plays"})
    grouped = grouped.merge(pass_plays, on=["game_id", "posteam"], how="left")
    grouped["off_pass_rate"] = grouped["off_pass_plays"].fillna(0) / grouped[
        "off_plays"
    ].clip(lower=1)

    # Kicking-game opportunity.
    kicks = pbp.copy()
    kicks = _numeric(
        kicks,
        ["field_goal_attempt", "extra_point_attempt", "two_point_attempt"],
    )
    kicks["fg_made"] = (kicks["field_goal_result"] == "made").astype(int)
    kicks["xp_made"] = kicks["extra_point_result"].isin(["made", "good"]).astype(int)
    kicks["two_pt"] = (kicks["two_point_conv_result"] == "success").astype(int)
    kick_stats = kicks.groupby(["game_id", "posteam"], as_index=False).agg(
        off_fg_att=("field_goal_attempt", "sum"),
        off_fg_made=("fg_made", "sum"),
        off_xp_att=("extra_point_attempt", "sum"),
        off_xp_made=("xp_made", "sum"),
        off_two_pt=("two_pt", "sum"),
    )
    grouped = grouped.merge(kick_stats, on=["game_id", "posteam"], how="left")
    for column in [
        "off_fg_att",
        "off_fg_made",
        "off_xp_att",
        "off_xp_made",
        "off_two_pt",
    ]:
        grouped[column] = grouped[column].fillna(0)

    grouped["off_points"] = (
        6 * grouped["off_tds"].fillna(0)
        + 3 * grouped["off_fg_made"]
        + grouped["off_xp_made"]
        + 2 * grouped["off_two_pt"]
    )

    # Drive-level red-zone and pace stats.
    drives = pbp.dropna(subset=["posteam", "drive"]).copy()
    drive_stats = drives.groupby(["game_id", "posteam", "drive"], as_index=False).agg(
        inside20=("drive_inside20", "max"),
        result=("fixed_drive_result", "first"),
        top=("drive_time_of_possession", "first"),
        plays=("play_type", "size"),
    )
    drive_stats["inside20"] = drive_stats["inside20"].fillna(0)
    drive_stats["td"] = (drive_stats["result"] == "Touchdown").astype(int)
    drive_stats["rz_td"] = drive_stats["inside20"].astype(int) * drive_stats["td"]
    drive_stats["top_seconds"] = drive_stats["top"].map(_clock_to_seconds)

    team_drives = drive_stats.groupby(["game_id", "posteam"], as_index=False).agg(
        off_drives=("drive", "size"),
        off_red_zone_trips=("inside20", "sum"),
        off_red_zone_tds=("rz_td", "sum"),
        off_top_seconds=("top_seconds", "sum"),
        off_drive_plays=("plays", "sum"),
    )
    grouped = grouped.merge(team_drives, on=["game_id", "posteam"], how="left")
    grouped["off_red_zone_td_rate"] = grouped["off_red_zone_tds"] / grouped[
        "off_red_zone_trips"
    ].replace(0, np.nan)
    grouped["off_points_per_drive"] = grouped["off_points"] / grouped[
        "off_drives"
    ].replace(0, np.nan)
    grouped["off_plays_per_drive"] = grouped["off_plays"] / grouped[
        "off_drives"
    ].replace(0, np.nan)
    grouped["off_td_rate"] = grouped["off_tds"] / grouped["off_drives"].replace(
        0, np.nan
    )
    grouped["off_fg_rate"] = grouped["off_fg_att"] / grouped["off_drives"].replace(
        0, np.nan
    )
    grouped["off_sec_per_play"] = grouped["off_top_seconds"] / grouped[
        "off_plays"
    ].replace(0, np.nan)
    grouped["off_points_per_play"] = grouped["off_points"] / grouped[
        "off_plays"
    ].replace(0, np.nan)
    grouped["off_turnover_rate"] = grouped["off_turnovers"] / grouped[
        "off_plays"
    ].replace(0, np.nan)
    grouped["off_sack_rate"] = grouped["off_sacks"] / grouped["off_plays"].replace(
        0, np.nan
    )
    grouped["off_fg_per_rz_trip"] = grouped["off_fg_att"] / grouped[
        "off_red_zone_trips"
    ].replace(0, np.nan)

    return grouped


def _defense_table(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.copy()
    plays = _numeric(
        plays,
        [
            "sack",
            "interception",
            "fumble_lost",
            "safety",
            "touchdown",
            "punt_blocked",
            "defensive_two_point_conv",
            "qb_hit",
            "field_goal_attempt",
            "extra_point_attempt",
        ],
    )
    plays["is_def_td"] = (
        (plays["td_team"] == plays["defteam"])
        & (plays["td_team"] != plays["posteam"])
        & (plays["touchdown"] == 1)
    ).astype(int)
    blocked = (
        (plays.get("punt_blocked", 0) == 1)
        | (plays.get("field_goal_result", "").astype(str) == "blocked")
        | (plays.get("extra_point_result", "").astype(str) == "blocked")
    )
    plays["is_blocked"] = (blocked & (plays["is_def_td"] == 0)).astype(int)
    plays["is_two_pt_return"] = (
        pd.to_numeric(plays.get("defensive_two_point_conv", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    grouped = (
        plays.dropna(subset=["defteam"])
        .groupby(["game_id", "defteam"], as_index=False)
        .agg(
            def_sacks=("sack", "sum"),
            def_ints=("interception", "sum"),
            def_fumble_rec=("fumble_lost", "sum"),
            def_safeties=("safety", "sum"),
            def_tds=("is_def_td", "sum"),
            def_blocked_kicks=("is_blocked", "sum"),
            def_two_pt=("is_two_pt_return", "sum"),
            def_qb_hits=("qb_hit", "sum"),
        )
    )
    grouped["def_takeaways"] = grouped["def_ints"] + grouped["def_fumble_rec"]
    return grouped


def _skeleton(games: pd.DataFrame) -> pd.DataFrame:
    home = games.assign(
        team=games["home_team"],
        opponent=games["away_team"],
        is_home=1,
        team_score=games["home_score"],
        opp_score=games["away_score"],
        rest=games["home_rest"],
        opp_rest=games["away_rest"],
        qb_name=games["home_qb_name"],
        opp_qb_name=games["away_qb_name"],
        coach=games["home_coach"],
        opp_coach=games["away_coach"],
    )
    away = games.assign(
        team=games["away_team"],
        opponent=games["home_team"],
        is_home=0,
        team_score=games["away_score"],
        opp_score=games["home_score"],
        rest=games["away_rest"],
        opp_rest=games["home_rest"],
        qb_name=games["away_qb_name"],
        opp_qb_name=games["home_qb_name"],
        coach=games["away_coach"],
        opp_coach=games["home_coach"],
    )
    columns = [
        "season",
        "week",
        "game_id",
        "gameday",
        "game_type",
        "team",
        "opponent",
        "is_home",
        "team_score",
        "opp_score",
        "spread_line",
        "total_line",
        "roof",
        "surface",
        "temp",
        "wind",
        "rest",
        "opp_rest",
        "div_game",
        "stadium_id",
        "stadium",
        "qb_name",
        "opp_qb_name",
        "coach",
        "opp_coach",
        "home_team",
        "away_team",
    ]
    return pd.concat([home[columns], away[columns]], ignore_index=True)


def build_team_games(
    seasons: Iterable[int],
    games: pd.DataFrame | None = None,
    pbp: pd.DataFrame | None = None,
    scoring: dict[str, Any] | None = None,
) -> pd.DataFrame:
    seasons = sorted(int(s) for s in seasons)
    games_all = games if games is not None else load_games(seasons)
    games_reg = games_all[
        (games_all["season"].isin(seasons)) & (games_all["game_type"] == "REG")
    ].copy()
    pbp_all = pbp if pbp is not None else load_pbp(seasons)
    pbp_reg = pbp_all[pbp_all["game_id"].isin(set(games_reg["game_id"]))].copy()

    skeleton = _skeleton(games_reg)
    if pbp_reg.empty:
        for column in [
            "off_plays",
            "off_yards",
            "off_epa",
            "off_success",
            "off_turnovers",
            "off_sacks",
            "off_qb_hits",
            "off_tds",
            "off_explosive_rate",
            "off_pass_yards",
            "off_pass_epa",
            "off_rush_yards",
            "off_rush_epa",
            "off_early_down_plays",
            "off_early_down_pass_rate",
            "off_pass_plays",
            "off_pass_rate",
            "off_fg_att",
            "off_fg_made",
            "off_xp_att",
            "off_xp_made",
            "off_two_pt",
            "off_points",
            "off_drives",
            "off_red_zone_trips",
            "off_red_zone_tds",
            "off_top_seconds",
            "off_drive_plays",
            "off_red_zone_td_rate",
            "off_points_per_drive",
            "off_plays_per_drive",
            "off_td_rate",
            "off_fg_rate",
            "off_sec_per_play",
        ]:
            skeleton[column] = np.nan
        skeleton["dst_pts"] = np.nan
        return skeleton

    offense = _offense_table(pbp_reg)
    defense = _defense_table(pbp_reg)

    team = skeleton.merge(
        offense.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left"
    )
    team = team.merge(
        defense.rename(columns={"defteam": "team"}), on=["game_id", "team"], how="left"
    )

    # The opponent's offensive production is what this team's defense allowed.
    # Prefix every opponent column first so no duplicate ``off_*`` names leak
    # into the team row and get pandas _x/_y suffixes.
    opponent_offense = offense.rename(columns={"posteam": "opponent"})
    opponent_offense = opponent_offense.rename(
        columns={
            c: f"opp_{c}"
            for c in opponent_offense.columns
            if c not in {"game_id", "opponent"}
        }
    )
    team = team.merge(opponent_offense, on=["game_id", "opponent"], how="left")

    allowed_aliases = {
        "opp_off_points": "def_points_allowed",
        "opp_off_yards": "def_yards_allowed",
        "opp_off_epa": "def_epa_allowed",
        "opp_off_success": "def_success_allowed",
        "opp_off_pass_yards": "def_pass_yards_allowed",
        "opp_off_pass_epa": "def_pass_epa_allowed",
        "opp_off_rush_yards": "def_rush_yards_allowed",
        "opp_off_rush_epa": "def_rush_epa_allowed",
        "opp_off_turnovers": "def_off_turnovers_allowed",
        "opp_off_sacks": "def_sacks_allowed",
        "opp_off_qb_hits": "def_qb_hits_allowed",
        "opp_off_plays": "def_plays_allowed",
        "opp_off_pass_rate": "def_pass_rate_allowed",
        "opp_off_early_down_pass_rate": "def_early_down_pass_rate_allowed",
        "opp_off_explosive_rate": "def_explosive_rate_allowed",
        "opp_off_red_zone_trips": "def_red_zone_trips_allowed",
        "opp_off_red_zone_td_rate": "def_red_zone_td_rate_allowed",
        "opp_off_fg_att": "def_fg_att_allowed",
        "opp_off_fg_made": "def_fg_made_allowed",
        "opp_off_points_per_drive": "def_points_per_drive_allowed",
        "opp_off_plays_per_drive": "def_plays_per_drive_allowed",
        "opp_off_td_rate": "def_td_rate_allowed",
        "opp_off_sec_per_play": "def_sec_per_play_allowed",
    }
    for source, target in allowed_aliases.items():
        if source in team.columns:
            team[target] = team[source]
    team = team.drop(columns=[c for c in team.columns if c.startswith("opp_off_")])

    for column in [
        "def_sacks",
        "def_ints",
        "def_fumble_rec",
        "def_safeties",
        "def_tds",
        "def_blocked_kicks",
        "def_two_pt",
        "def_qb_hits",
        "def_takeaways",
    ]:
        if column not in team.columns:
            team[column] = 0.0
        team[column] = team[column].fillna(0.0)

    team["def_points_allowed_per_play"] = team["def_points_allowed"] / team[
        "def_plays_allowed"
    ].replace(0, np.nan)
    team["def_pressure_rate"] = (team["def_sacks"] + team["def_qb_hits"]) / team[
        "def_plays_allowed"
    ].replace(0, np.nan)
    team["def_takeaway_rate"] = team["def_takeaways"] / team[
        "def_plays_allowed"
    ].replace(0, np.nan)

    if scoring is not None:
        team["dst_pts"] = add_dst_points(team, scoring)
    else:
        team["dst_pts"] = np.nan
    return team.sort_values(["season", "week", "gameday", "team"]).reset_index(
        drop=True
    )


def build_kicker_games(
    seasons: Iterable[int],
    games: pd.DataFrame | None = None,
    player_weeks: pd.DataFrame | None = None,
    scoring: dict[str, Any] | None = None,
) -> pd.DataFrame:
    seasons = sorted(int(s) for s in seasons)
    players = player_weeks if player_weeks is not None else load_player_weeks(seasons)
    games_all = games if games is not None else load_games(seasons)
    games_reg = games_all[
        (games_all["season"].isin(seasons)) & (games_all["game_type"] == "REG")
    ].copy()

    kickers = players[
        (players["position"] == "K")
        & (players["season"].isin(seasons))
        & (players["season_type"] == "REG")
    ].copy()
    if kickers.empty or scoring is None:
        return pd.DataFrame()

    kickers["kicker_pts"] = add_kicker_points(kickers, scoring)
    kickers["kicker_attempts"] = pd.to_numeric(
        kickers["fg_att"], errors="coerce"
    ).fillna(0) + pd.to_numeric(kickers["pat_att"], errors="coerce").fillna(0)
    kickers = kickers.sort_values(
        ["season", "week", "team", "kicker_attempts"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["game_id", "team"], keep="first")

    game_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "spread_line",
        "total_line",
        "roof",
        "surface",
        "temp",
        "wind",
        "home_rest",
        "away_rest",
        "div_game",
        "stadium_id",
        "stadium",
        "home_qb_name",
        "away_qb_name",
    ]
    kickers = kickers.merge(
        games_reg[game_cols], on=["game_id", "season", "week"], how="left"
    )
    kickers["is_home"] = (kickers["team"] == kickers["home_team"]).astype(int)
    kickers["opponent"] = np.where(
        kickers["is_home"] == 1, kickers["away_team"], kickers["home_team"]
    )
    return kickers.sort_values(["season", "week", "gameday", "team"]).reset_index(
        drop=True
    )


def build_all(
    seasons: Iterable[int], scoring: dict[str, Any], force: bool = False
) -> None:
    ensure_dirs()
    seasons = sorted(int(s) for s in seasons)
    games = load_games(seasons)
    pbp = load_pbp(seasons)
    players = load_player_weeks(seasons)
    team_games = build_team_games(seasons, games=games, pbp=pbp, scoring=scoring)
    kicker_games = build_kicker_games(
        seasons, games=games, player_weeks=players, scoring=scoring
    )
    team_games.to_parquet(TEAM_GAMES_PATH, index=False)
    kicker_games.to_parquet(KICKER_GAMES_PATH, index=False)
    print(f"wrote {TEAM_GAMES_PATH} ({len(team_games):,} rows)")
    print(f"wrote {KICKER_GAMES_PATH} ({len(kicker_games):,} rows)")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TEAM_GAMES_PATH.exists() or not KICKER_GAMES_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `ffstream build` or `python -m ffstream build` first."
        )
    return pd.read_parquet(TEAM_GAMES_PATH), pd.read_parquet(KICKER_GAMES_PATH)
