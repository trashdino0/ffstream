"""Pregame feature engineering for kicker and team-defense models.

All rolling features are shifted by one game within each team (or kicker), so a
row only ever sees prior games. Future scheduled games are kept in the team
table with missing stats; their rolling features therefore represent the
information available before that game.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

OFFENSE_METRICS = [
    "off_points",
    "off_yards",
    "off_plays",
    "off_epa",
    "off_success",
    "off_pass_epa",
    "off_rush_epa",
    "off_pass_rate",
    "off_early_down_pass_rate",
    "off_turnovers",
    "off_sacks",
    "off_qb_hits",
    "off_explosive_rate",
    "off_red_zone_trips",
    "off_red_zone_td_rate",
    "off_fg_att",
    "off_xp_att",
    "off_points_per_drive",
    "off_plays_per_drive",
    "off_td_rate",
    "off_fg_rate",
    "off_sec_per_play",
    "off_points_per_play",
    "off_turnover_rate",
    "off_sack_rate",
    "off_fg_per_rz_trip",
]

DEFENSE_METRICS = [
    "def_points_allowed",
    "def_yards_allowed",
    "def_epa_allowed",
    "def_success_allowed",
    "def_pass_epa_allowed",
    "def_rush_epa_allowed",
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_tds",
    "def_safeties",
    "def_takeaways",
    "def_qb_hits",
    "def_explosive_rate_allowed",
    "def_red_zone_td_rate_allowed",
    "def_fg_att_allowed",
    "def_pass_rate_allowed",
    "def_points_per_drive_allowed",
    "def_plays_per_drive_allowed",
    "def_points_allowed_per_play",
    "def_pressure_rate",
    "def_takeaway_rate",
]

KICKER_METRICS = [
    "kicker_pts",
    "fg_att",
    "fg_made",
    "fg_att_40plus",
    "fg_made_40plus",
    "fg_att_50plus",
    "fg_made_50plus",
    "pat_att",
    "pat_made",
    "fg_long",
    "kicker_attempts",
]

SCHEDULE_FEATURES = [
    "team_implied",
    "opp_implied",
    "implied_margin",
    "total_line",
    "spread_line",
    "is_home",
    "abs_implied_margin",
    "is_favorite",
    "is_close_game",
    "is_high_total",
    "is_low_total",
    "rest",
    "opp_rest",
    "rest_diff",
    "short_week",
    "is_division",
    "is_dome",
    "is_turf",
    "is_windy",
    "is_cold",
    "altitude",
    "temp",
    "wind",
    "team_implied_rank",
    "opp_implied_rank",
]


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _numeric(
        df,
        ["total_line", "spread_line", "rest", "opp_rest", "temp", "wind", "div_game"],
    )
    total = df["total_line"]
    spread = df["spread_line"]
    home_implied = total / 2.0 - spread / 2.0
    away_implied = total / 2.0 + spread / 2.0
    df["team_implied"] = np.where(df["is_home"] == 1, home_implied, away_implied)
    df["opp_implied"] = np.where(df["is_home"] == 1, away_implied, home_implied)
    df["implied_margin"] = df["team_implied"] - df["opp_implied"]
    df["abs_implied_margin"] = df["implied_margin"].abs()
    df["is_favorite"] = (df["team_implied"] > df["opp_implied"]).astype(int)
    df["is_close_game"] = (df["abs_implied_margin"] <= 3).astype(int)
    df["is_high_total"] = (df["total_line"] >= 47).astype(int)
    df["is_low_total"] = (df["total_line"] <= 41).astype(int)

    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    surface = df["surface"].fillna("").astype(str).str.lower()
    df["is_turf"] = surface.str.contains("turf|matrix|sport", regex=True).astype(int)
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df.loc[df["is_dome"] == 1, "wind"] = 0.0
    df.loc[df["is_dome"] == 1, "temp"] = 70.0
    df["is_windy"] = (df["wind"].fillna(0) >= 15).astype(int)
    df["is_cold"] = (df["temp"].fillna(70) <= 32).astype(int)

    stadium = df["stadium_id"].fillna("").astype(str).str.upper()
    df["altitude"] = np.select(
        [stadium.str.startswith("DEN"), stadium.str.contains("MEX")],
        [5280.0, 7380.0],
        default=0.0,
    )
    df["is_division"] = df["div_game"].fillna(0).astype(int)
    df["rest"] = df["rest"].fillna(7)
    df["opp_rest"] = df["opp_rest"].fillna(7)
    df["rest_diff"] = df["rest"] - df["opp_rest"]
    df["short_week"] = (df["rest"] <= 4).astype(int)

    grouped = df.groupby(["season", "week"])
    df["team_implied_rank"] = grouped["team_implied"].rank(
        ascending=False, method="min"
    )
    df["opp_implied_rank"] = grouped["opp_implied"].rank(ascending=True, method="min")

    # Keep the model honest about missing lines; a missing total is different
    # from a genuinely low total.
    df["line_missing"] = df["team_implied"].isna().astype(int)
    return df


def _add_rolling_metrics(
    df: pd.DataFrame,
    metrics: Iterable[str],
    windows: tuple[int, ...] = (4, 8),
) -> pd.DataFrame:
    for metric in metrics:
        if metric not in df.columns:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        grouped = df.groupby("team", sort=False)[metric]
        shifted = grouped.shift(1)
        for window in windows:
            # transform keeps the original index, which makes this safe to
            # assign back into the sorted frame.
            df[f"{metric}_last{window}"] = shifted.groupby(df["team"]).transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
        df[f"{metric}_season"] = df.groupby(["team", "season"], sort=False)[
            metric
        ].transform(lambda s: s.shift(1).expanding().mean())

    # Prior-season averages, shifted forward one season.
    season_means = df.groupby(["team", "season"], as_index=False)[list(metrics)].mean()
    season_means["season"] = season_means["season"] + 1
    prior = season_means.rename(columns={m: f"{m}_prior" for m in metrics})
    df = df.merge(prior, on=["team", "season"], how="left")
    return df


def _feature_suffixes(windows: tuple[int, ...]) -> list[str]:
    return [f"_last{w}" for w in windows] + ["_season", "_prior"]


def _is_rolling(column: str) -> bool:
    return column.endswith(("_last4", "_last8", "_season", "_prior"))


def kicker_history_feature_columns(df: pd.DataFrame) -> list[str]:
    """Only the kicker's own pregame rolling features, no team context."""
    columns = [
        c
        for c in df.columns
        if c.startswith("kicker_") and (_is_rolling(c) or c.endswith("_career"))
    ]
    columns.extend(
        c
        for c in ["kicker_pts_std_last8", "kicker_games", "kicker_is_new"]
        if c in df.columns
    )
    return [c for c in dict.fromkeys(columns) if c in df.columns]


def build_team_features(
    team_games: pd.DataFrame,
    windows: tuple[int, ...] = (4, 8),
) -> pd.DataFrame:
    """One row per team-game with pregame form, opponent form, and schedule."""
    df = team_games.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.sort_values(["season", "week", "gameday", "team"]).reset_index(drop=True)
    df = _schedule_features(df)

    metrics = OFFENSE_METRICS + DEFENSE_METRICS
    df = _numeric(df, metrics)
    df = _add_rolling_metrics(df, metrics, windows=windows)

    df["_played"] = df["off_plays"].notna().astype(int)
    df["games_played_season"] = df.groupby(["team", "season"], sort=False)[
        "_played"
    ].transform(lambda s: s.shift(1).cumsum())
    df = df.drop(columns=["_played"])

    # Opponent's pregame form, prefixed with opp_. The opponent row is the
    # opponent's team-game row in this same game.
    feature_cols = []
    for metric in metrics:
        feature_cols.extend(
            [f"{metric}{suffix}" for suffix in _feature_suffixes(windows)]
        )
    opponent = df[["game_id", "team"] + feature_cols].copy()
    opponent = opponent.rename(
        columns={"team": "opponent_key", **{c: f"opp_{c}" for c in feature_cols}}
    )
    df = df.merge(
        opponent,
        left_on=["game_id", "opponent"],
        right_on=["game_id", "opponent_key"],
        how="left",
    )
    df = df.drop(columns=["opponent_key"])
    return df


def _sum_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def _add_kicker_rolling(kickers: pd.DataFrame) -> pd.DataFrame:
    df = kickers.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.sort_values(["player_id", "season", "week", "gameday"]).reset_index(
        drop=True
    )

    df["fg_att_40plus"] = _sum_columns(
        df,
        [
            "fg_made_40_49",
            "fg_made_50_59",
            "fg_made_60_",
            "fg_missed_40_49",
            "fg_missed_50_59",
            "fg_missed_60_",
        ],
    )
    df["fg_made_40plus"] = _sum_columns(
        df, ["fg_made_40_49", "fg_made_50_59", "fg_made_60_"]
    )
    df["fg_att_50plus"] = _sum_columns(
        df, ["fg_made_50_59", "fg_made_60_", "fg_missed_50_59", "fg_missed_60_"]
    )
    df["fg_made_50plus"] = _sum_columns(df, ["fg_made_50_59", "fg_made_60_"])
    df = _numeric(df, KICKER_METRICS + ["fg_att", "fg_made", "pat_att", "pat_made"])

    for metric in KICKER_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
        shifted = df.groupby("player_id", sort=False)[metric].shift(1)
        for window in (4, 8):
            df[f"kicker_{metric}_last{window}"] = shifted.groupby(
                df["player_id"]
            ).transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        df[f"kicker_{metric}_season"] = df.groupby(["player_id", "season"], sort=False)[
            metric
        ].transform(lambda s: s.shift(1).expanding().mean())
        df[f"kicker_{metric}_career"] = df.groupby("player_id", sort=False)[
            metric
        ].transform(lambda s: s.shift(1).expanding().mean())

    season_means = df.groupby(["player_id", "season"], as_index=False)[
        KICKER_METRICS
    ].mean()
    season_means["season"] += 1
    prior = season_means.rename(
        columns={m: f"kicker_{m}_prior" for m in KICKER_METRICS}
    )
    df = df.merge(prior, on=["player_id", "season"], how="left")

    # Accuracy is better measured as made / attempts over the window than as a
    # mean of game-level percentages.
    for window in (4, 8):
        made = df.groupby("player_id", sort=False)["fg_made"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum()
        )
        att = df.groupby("player_id", sort=False)["fg_att"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum()
        )
        df[f"kicker_fg_pct_last{window}"] = made / att.replace(0, np.nan)

        made_long = df.groupby("player_id", sort=False)["fg_made_40plus"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum()
        )
        att_long = df.groupby("player_id", sort=False)["fg_att_40plus"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum()
        )
        df[f"kicker_fg_pct_40plus_last{window}"] = made_long / att_long.replace(
            0, np.nan
        )

    df["kicker_pts_std_last8"] = df.groupby("player_id", sort=False)[
        "kicker_pts"
    ].transform(lambda s: s.shift(1).rolling(8, min_periods=2).std())
    df["kicker_games"] = df.groupby("player_id", sort=False).cumcount()
    df["kicker_is_new"] = (df["kicker_games"] == 0).astype(int)
    return df


def build_kicker_features(
    kicker_games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Kicker rows joined to their own offense and the opposing defense."""
    kickers = _add_kicker_rolling(kicker_games)

    team_cols = [
        "game_id",
        "team",
        *SCHEDULE_FEATURES,
        "line_missing",
        "games_played_season",
        *[f"{m}{s}" for m in OFFENSE_METRICS for s in _feature_suffixes((4, 8))],
        *[f"opp_{m}{s}" for m in DEFENSE_METRICS for s in _feature_suffixes((4, 8))],
    ]
    team_cols = [c for c in dict.fromkeys(team_cols) if c in team_features.columns]
    kickers = kickers.drop(columns=["opponent"], errors="ignore")
    kickers = kickers.merge(
        team_features[team_cols], on=["game_id", "team"], how="left"
    )
    return kickers


def build_dst_features(team_features: pd.DataFrame) -> pd.DataFrame:
    """Team-defense rows: own defense, opponent offense, and game context."""
    dst = team_features.copy()
    dst = dst.sort_values(["season", "week", "gameday", "team"]).reset_index(drop=True)
    dst["dst_boom"] = (dst["dst_pts"] >= 12).astype(int)

    for window in (4, 8):
        dst[f"dst_pts_last{window}"] = dst.groupby("team", sort=False)[
            "dst_pts"
        ].transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
    dst["dst_pts_season"] = dst.groupby(["team", "season"], sort=False)[
        "dst_pts"
    ].transform(lambda s: s.shift(1).expanding().mean())
    dst["dst_pts_std_last8"] = dst.groupby("team", sort=False)["dst_pts"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=2).std()
    )
    season_means = dst.groupby(["team", "season"], as_index=False)["dst_pts"].mean()
    season_means["season"] += 1
    dst = dst.merge(
        season_means.rename(columns={"dst_pts": "dst_pts_prior"}),
        on=["team", "season"],
        how="left",
    )
    return dst


def latest_kicker_rows(
    kicker_features: pd.DataFrame, before_season: int, before_week: int
) -> pd.DataFrame:
    """Latest known kicker per team before the target week.

    This is the pre-week roster view. It deliberately ignores the target week's
    stats, so a Thursday game cannot leak into Sunday predictions.
    """
    history = kicker_features[
        (kicker_features["season"] < before_season)
        | (
            (kicker_features["season"] == before_season)
            & (kicker_features["week"] < before_week)
        )
    ].copy()
    if history.empty:
        return history
    history = history.sort_values(["season", "week", "gameday"])
    latest = history.groupby("team", as_index=False).tail(1).copy()
    return latest


def make_kicker_prediction_rows(
    team_features: pd.DataFrame,
    kicker_features: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Create one kicker row per scheduled team-game for a future week."""
    week_rows = team_features[
        (team_features["season"] == season) & (team_features["week"] == week)
    ].copy()
    if week_rows.empty:
        return week_rows

    latest = latest_kicker_rows(kicker_features, season, week)
    latest_cols = [
        "team",
        "player_id",
        "player_display_name",
        *kicker_history_feature_columns(latest),
    ]
    latest_cols = [c for c in dict.fromkeys(latest_cols) if c in latest.columns]
    rows = week_rows.merge(latest[latest_cols], on="team", how="left")
    rows["player_display_name"] = rows["player_display_name"].fillna(
        rows["team"] + " K"
    )
    return rows


def make_dst_prediction_rows(
    team_features: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    return team_features[
        (team_features["season"] == season) & (team_features["week"] == week)
    ].copy()


CURATED_KICKER_FEATURES = [
    # Game environment / market
    "team_implied",
    "opp_implied",
    "implied_margin",
    "abs_implied_margin",
    "is_favorite",
    "is_close_game",
    "is_high_total",
    "total_line",
    "spread_line",
    "is_home",
    "rest_diff",
    "is_dome",
    "is_windy",
    "is_cold",
    "altitude",
    "team_implied_rank",
    "opp_implied_rank",
    "line_missing",
    # Kicker form and volume
    "kicker_kicker_pts_last4",
    "kicker_kicker_pts_last8",
    "kicker_kicker_pts_season",
    "kicker_kicker_pts_career",
    "kicker_fg_att_last4",
    "kicker_fg_att_last8",
    "kicker_fg_att_season",
    "kicker_fg_att_career",
    "kicker_fg_made_last4",
    "kicker_fg_made_last8",
    "kicker_fg_made_season",
    "kicker_fg_att_40plus_last4",
    "kicker_fg_made_40plus_last4",
    "kicker_fg_att_50plus_last4",
    "kicker_fg_made_50plus_last4",
    "kicker_pat_att_last4",
    "kicker_pat_att_last8",
    "kicker_pat_att_season",
    "kicker_fg_pct_last4",
    "kicker_fg_pct_last8",
    "kicker_fg_pct_40plus_last4",
    "kicker_fg_long_last4",
    "kicker_pts_std_last8",
    "kicker_games",
    "kicker_is_new",
    # Team offense: scoring, stalls, pace
    "off_points_last4",
    "off_points_last8",
    "off_points_season",
    "off_epa_last4",
    "off_success_last4",
    "off_points_per_play_last4",
    "off_red_zone_trips_last4",
    "off_red_zone_td_rate_last4",
    "off_red_zone_td_rate_last8",
    "off_fg_att_last4",
    "off_fg_att_last8",
    "off_fg_att_season",
    "off_fg_rate_last4",
    "off_fg_rate_last8",
    "off_fg_per_rz_trip_last4",
    "off_points_per_drive_last4",
    "off_plays_per_drive_last4",
    "off_pass_rate_last4",
    "off_early_down_pass_rate_last4",
    "off_turnovers_last4",
    "off_sacks_last4",
    "off_sack_rate_last4",
    # Opponent defense: FG opportunity and stop rate
    "opp_def_points_allowed_last4",
    "opp_def_points_allowed_last8",
    "opp_def_points_allowed_season",
    "opp_def_epa_allowed_last4",
    "opp_def_success_allowed_last4",
    "opp_def_fg_att_allowed_last4",
    "opp_def_fg_att_allowed_last8",
    "opp_def_fg_att_allowed_season",
    "opp_def_red_zone_td_rate_allowed_last4",
    "opp_def_red_zone_td_rate_allowed_last8",
    "opp_def_sacks_last4",
    "opp_def_takeaways_last4",
    "opp_def_yards_allowed_last4",
    "opp_def_pressure_rate_last4",
    "opp_def_points_allowed_per_play_last4",
]


def kicker_model_features(df: pd.DataFrame) -> list[str]:
    return [c for c in CURATED_KICKER_FEATURES if c in df.columns]


def dst_model_features(df: pd.DataFrame) -> list[str]:
    return dst_feature_columns(df)


def kicker_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    columns.extend(
        [
            c
            for c in SCHEDULE_FEATURES + ["line_missing", "games_played_season"]
            if c in df.columns
        ]
    )
    # Only rolling team features. Raw current-game stats would leak the target.
    columns.extend(
        [
            c
            for c in df.columns
            if c.startswith("off_") and not c.startswith("opp_off_") and _is_rolling(c)
        ]
    )
    columns.extend(
        [c for c in df.columns if c.startswith("opp_def_") and _is_rolling(c)]
    )
    columns.extend(kicker_history_feature_columns(df))
    return [c for c in dict.fromkeys(columns) if c in df.columns]


def dst_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    columns.extend(
        [
            c
            for c in SCHEDULE_FEATURES + ["line_missing", "games_played_season"]
            if c in df.columns
        ]
    )
    columns.extend(
        [
            c
            for c in df.columns
            if c.startswith("def_") and not c.startswith("opp_def_") and _is_rolling(c)
        ]
    )
    columns.extend(
        [c for c in df.columns if c.startswith("opp_off_") and _is_rolling(c)]
    )
    columns.extend(
        [
            c
            for c in [
                "dst_pts_last4",
                "dst_pts_last8",
                "dst_pts_season",
                "dst_pts_prior",
                "dst_pts_std_last8",
            ]
            if c in df.columns
        ]
    )
    # Own offense matters indirectly: a good offense creates game script and
    # keeps the defense fresh; turnovers can put the defense in bad spots.
    columns.extend(
        [
            c
            for c in df.columns
            if c.startswith("off_")
            and not c.startswith("opp_off_")
            and _is_rolling(c)
            and any(
                key in c
                for key in [
                    "off_points",
                    "off_turnovers",
                    "off_plays_per_drive",
                    "off_sec_per_play",
                ]
            )
        ]
    )
    return [c for c in dict.fromkeys(columns) if c in df.columns]
