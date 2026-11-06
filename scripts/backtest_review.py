"""Human-readable review of the 2026 weekly backtest.

Reads the week-by-week backtest predictions and answers two questions for every
week:

1. Who did the model rank in its top 5, and what did they actually score?
2. Who actually scored the most, and where did the model rank them?

Usage:
    python scripts/backtest_review.py --season 2026 --weeks 1-8 --top 5
"""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

POSITIONS = {
    "K": {
        "target": "kicker_pts",
        "key": "player_id",
        "label": "Kicker",
        "file": "backtest_{season}_weekly_predictions_kicker_pts.parquet",
    },
    "DST": {
        "target": "dst_pts",
        "key": "team",
        "label": "Team defense",
        "file": "backtest_{season}_weekly_predictions_dst_pts.parquet",
    },
}


def _parse_range(value: str | None, default: tuple[int, int]) -> list[int]:
    if not value:
        return list(range(default[0], default[1] + 1))
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.{digits}f}"


def _display_name(row: pd.Series, position: str) -> str:
    if position == "K":
        name = row.get("player_display_name")
        if pd.isna(name) or not str(name).strip():
            return f"{row['team']} K"
        return str(name)
    return str(row["team"])


def _opponent(row: pd.Series) -> str:
    for column in ("opponent_team", "opponent", "opp"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return "—"


def _rank_label(rank: int, tied: bool) -> str:
    return f"T{rank}" if tied else str(rank)


def _rank_frame(df: pd.DataFrame, target: str, position: str) -> pd.DataFrame:
    out = df.copy()
    out["_model_rank"] = (
        out.groupby("week")["proj_mean"].rank(ascending=False, method="min").astype(int)
    )
    out["_actual_rank"] = (
        out.groupby("week")[target].rank(ascending=False, method="min").astype(int)
    )
    out["_model_tied"] = (
        out.groupby(["week", "_model_rank"])["_model_rank"].transform("size") > 1
    )
    out["_actual_tied"] = (
        out.groupby(["week", "_actual_rank"])["_actual_rank"].transform("size") > 1
    )
    out["_name"] = out.apply(lambda r: _display_name(r, position), axis=1)
    out["_opponent"] = out.apply(_opponent, axis=1)
    return out


def _top(df: pd.DataFrame, column: str, secondary: str, top: int) -> pd.DataFrame:
    return df.sort_values(
        [column, secondary, "_name"],
        ascending=[False, False, True],
    ).head(top)


def _summary_rows(
    ranked: pd.DataFrame, target: str, key: str, top: int
) -> pd.DataFrame:
    rows = []
    for week, week_df in ranked.groupby("week", sort=True):
        model_top = _top(week_df, "proj_mean", target, top)
        actual_top = _top(week_df, target, "proj_mean", top)
        rho = spearmanr(week_df["proj_mean"], week_df[target]).statistic
        rows.append(
            {
                "week": int(week),
                "n": int(len(week_df)),
                "model_top_avg": float(model_top[target].mean()),
                "actual_top_avg": float(actual_top[target].mean()),
                "league_avg": float(week_df[target].mean()),
                "model_top1": float(model_top[target].iloc[0]),
                "actual_top1": float(actual_top[target].iloc[0]),
                "overlap": int(len(set(model_top[key]) & set(actual_top[key]))),
                "spearman": float(rho),
            }
        )
    return pd.DataFrame(rows)


def _detail_rows(
    ranked: pd.DataFrame,
    season: int,
    position: str,
    target: str,
    top: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for week, week_df in ranked.groupby("week", sort=True):
        model_top = _top(week_df, "proj_mean", target, top)
        actual_top = week_df[week_df["_actual_rank"] <= top].sort_values(
            ["_actual_rank", "proj_mean", "_name"],
            ascending=[True, False, True],
        )

        for _, row in model_top.iterrows():
            rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "position": position,
                    "view": "model",
                    "rank": int(row["_model_rank"]),
                    "rank_label": _rank_label(
                        int(row["_model_rank"]), bool(row["_model_tied"])
                    ),
                    "player": row["_name"],
                    "team": row["team"],
                    "opponent": row["_opponent"],
                    "projected": float(row["proj_mean"]),
                    "actual": float(row[target]),
                    "other_rank": int(row["_actual_rank"]),
                    "other_rank_label": _rank_label(
                        int(row["_actual_rank"]), bool(row["_actual_tied"])
                    ),
                }
            )

        for _, row in actual_top.iterrows():
            rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "position": position,
                    "view": "actual",
                    "rank": int(row["_actual_rank"]),
                    "rank_label": _rank_label(
                        int(row["_actual_rank"]), bool(row["_actual_tied"])
                    ),
                    "player": row["_name"],
                    "team": row["team"],
                    "opponent": row["_opponent"],
                    "projected": float(row["proj_mean"]),
                    "actual": float(row[target]),
                    "other_rank": int(row["_model_rank"]),
                    "other_rank_label": _rank_label(
                        int(row["_model_rank"]), bool(row["_model_tied"])
                    ),
                }
            )
    return rows


def build_review(
    season: int = 2026,
    weeks: Iterable[int] = range(1, 9),
    top: int = 5,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    weeks = sorted(int(w) for w in weeks)
    detail: list[dict[str, Any]] = []
    summaries: dict[str, pd.DataFrame] = {}
    ranked_frames: dict[str, pd.DataFrame] = {}

    for position, cfg in POSITIONS.items():
        target = cfg["target"]
        path = out_dir / cfg["file"].format(season=season)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m ffstream backtest` first."
            )
        df = pd.read_parquet(path)
        df = df[
            (df["season"] == season) & (df["week"].isin(weeks)) & df[target].notna()
        ].copy()
        if df.empty:
            raise ValueError(f"No completed {season} rows for {target} in {weeks}.")
        ranked = _rank_frame(df, target, position)
        ranked_frames[position] = ranked
        summaries[position] = _summary_rows(ranked, target, cfg["key"], top)
        detail.extend(_detail_rows(ranked, season, position, target, top))

    summary = pd.DataFrame(detail)
    overall: dict[str, dict[str, float]] = {}
    for position, cfg in POSITIONS.items():
        target = cfg["target"]
        ranked = ranked_frames[position]
        weekly = summaries[position]
        overall[position] = {
            "model_top_avg": float(weekly["model_top_avg"].mean()),
            "actual_top_avg": float(weekly["actual_top_avg"].mean()),
            "league_avg": float(weekly["league_avg"].mean()),
            "model_top1": float(weekly["model_top1"].mean()),
            "actual_top1": float(weekly["actual_top1"].mean()),
            "overlap": float(weekly["overlap"].mean()),
            "spearman_mean": float(weekly["spearman"].mean()),
            "spearman_pooled": float(
                spearmanr(ranked["proj_mean"], ranked[target]).statistic
            ),
            "n": int(len(ranked)),
        }

    return {
        "season": season,
        "weeks": weeks,
        "top": top,
        "detail": summary,
        "weekly": summaries,
        "overall": overall,
    }


def _markdown_table(rows: list[dict[str, Any]], position: str, view: str) -> str:
    label = "Kicker" if position == "K" else "Defense"
    title = "model top 5" if view == "model" else "actual top 5"
    lines = [f"#### {label}s — {title}", ""]
    if position == "K" and view == "model":
        lines += [
            "| # | Player | Team | Opp | Proj | Actual | Actual rank |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['rank_label']} | {row['player']} | {row['team']} | "
                f"{row['opponent']} | {_fmt(row['projected'])} | "
                f"{_fmt(row['actual'])} | {row['other_rank_label']} |"
            )
    elif position == "K":
        lines += [
            "| # | Player | Team | Opp | Actual | Proj | Model rank |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['rank_label']} | {row['player']} | {row['team']} | "
                f"{row['opponent']} | {_fmt(row['actual'])} | "
                f"{_fmt(row['projected'])} | {row['other_rank_label']} |"
            )
    elif view == "model":
        lines += [
            "| # | Team | Opp | Proj | Actual | Actual rank |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['rank_label']} | {row['team']} | {row['opponent']} | "
                f"{_fmt(row['projected'])} | {_fmt(row['actual'])} | "
                f"{row['other_rank_label']} |"
            )
    else:
        lines += [
            "| # | Team | Opp | Actual | Proj | Model rank |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['rank_label']} | {row['team']} | {row['opponent']} | "
                f"{_fmt(row['actual'])} | {_fmt(row['projected'])} | "
                f"{row['other_rank_label']} |"
            )
    return "\n".join(lines)


def _compact_table(detail: pd.DataFrame, week: int, position: str, top: int) -> str:
    lines = [
        "| # | Model pick | Proj | Got | Actual # | Actual leader | Actual | Proj | Model # |",
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    position_rows = detail[(detail["week"] == week) & (detail["position"] == position)]
    model_rows = position_rows[position_rows["view"] == "model"].sort_values("rank")
    actual_rows = position_rows[position_rows["view"] == "actual"].sort_values("rank")
    for index in range(min(top, len(model_rows))):
        model = model_rows.iloc[index]
        actual = actual_rows.iloc[index]
        model_label = (
            f"{model['player']} ({model['team']})"
            if position == "K"
            else model["player"]
        )
        actual_label = (
            f"{actual['player']} ({actual['team']})"
            if position == "K"
            else actual["player"]
        )
        lines.append(
            "| {rank} | {mp} | {mproj} | {mact} | {arank} | {ap} | {aact} | {aproj} | {amrank} |".format(
                rank=model["rank_label"],
                mp=model_label,
                mproj=_fmt(model["projected"]),
                mact=_fmt(model["actual"]),
                arank=actual["rank_label"],
                ap=actual_label,
                aact=_fmt(actual["actual"]),
                aproj=_fmt(actual["projected"]),
                amrank=actual["other_rank_label"],
            )
        )
    extras = actual_rows.iloc[top:]
    if len(extras):
        tied = "; ".join(
            f"{row['player']} ({row['team']}) {_fmt(row['actual'])} "
            f"(model #{row['other_rank_label']})"
            for _, row in extras.iterrows()
        )
        lines += ["", f"*Also tied at the cut: {tied}.*"]
    return "\n".join(lines)


def render_markdown(review: dict[str, Any]) -> str:
    season = review["season"]
    weeks = review["weeks"]
    detail = review["detail"]

    lines = [
        f"# {season} Backtest Review — Weeks {weeks[0]}–{weeks[-1]}",
        "",
        "Weekly replay: the model is retrained before each week using only prior games.",
        "",
        "## Summary",
        "",
        "| Position | Model top-5 avg | Actual top-5 avg | League avg | Top-5 vs league | Top-1 avg | Actual top-1 avg | Avg overlap | Pooled Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for position, cfg in POSITIONS.items():
        o = review["overall"][position]
        lines.append(
            "| {label} | {model} | {actual} | {league} | {gain:+.2f} | {top1} | {actual1} | {overlap:.1f}/5 | {rho:.3f} |".format(
                label=cfg["label"],
                model=_fmt(o["model_top_avg"]),
                actual=_fmt(o["actual_top_avg"]),
                league=_fmt(o["league_avg"]),
                gain=o["model_top_avg"] - o["league_avg"],
                top1=_fmt(o["model_top1"]),
                actual1=_fmt(o["actual_top1"]),
                overlap=o["overlap"],
                rho=o["spearman_pooled"],
            )
        )

    lines += [
        "",
        "## Week by week",
        "",
        "Rankings use projected points (`proj_mean`). The model top 5 is exactly five rows; "
        "the actual top 5 includes every player tied at the cutoff, so it can be longer than five. "
        "`T` marks a tie.",
        "",
    ]
    for week in weeks:
        lines += [f"## Week {week}", ""]
        for position, cfg in POSITIONS.items():
            position_rows = detail[
                (detail["week"] == week) & (detail["position"] == position)
            ]
            lines += [
                _markdown_table(
                    position_rows[position_rows["view"] == "model"].to_dict("records"),
                    position,
                    "model",
                ),
                "",
                _markdown_table(
                    position_rows[position_rows["view"] == "actual"].to_dict("records"),
                    position,
                    "actual",
                ),
                "",
            ]
        week_summary = review["weekly"]
        k = week_summary["K"][week_summary["K"]["week"] == week].iloc[0]
        d = week_summary["DST"][week_summary["DST"]["week"] == week].iloc[0]
        lines += [
            "**Week totals:** "
            f"K model top-5 {_fmt(k['model_top_avg'])} / actual top-5 {_fmt(k['actual_top_avg'])} / league {_fmt(k['league_avg'])}; "
            f"DST model top-5 {_fmt(d['model_top_avg'])} / actual top-5 {_fmt(d['actual_top_avg'])} / league {_fmt(d['league_avg'])}.",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Weekly replay retrains before each week, so it is the closest thing to how the live board is used.",
        "- Kicker projections are inherently noisy; the top-5 group can beat the league while the single top pick does not.",
        "- Weekly summary averages use five players per week; ties at the cutoff are broken by projection.",
        "- The actual top-5 tables include all players tied at the cutoff, so they can show six or seven rows.",
        "- Late kicker changes and injuries are not modeled. This replay scores the kicker who actually played, which can be more forgiving than a real lineup decision made before kickoff.",
        "",
    ]
    return "\n".join(lines)


def _html_table(rows: list[dict[str, Any]], position: str, view: str) -> str:
    body = []
    player_header = "Player" if position == "K" else "Team"
    if view == "model":
        header = (
            f"<thead><tr><th>#</th><th>{player_header}</th><th>Proj</th>"
            "<th>Actual</th><th>Actual rank</th></tr></thead>"
        )
    else:
        header = (
            f"<thead><tr><th>#</th><th>{player_header}</th><th>Actual</th>"
            "<th>Proj</th><th>Model rank</th></tr></thead>"
        )
    for row in rows:
        css = "model" if view == "model" else "actual"
        if position == "K":
            player_cell = (
                f"<strong>{escape(str(row['player']))}</strong>"
                f"<span class='sub'>{escape(str(row['team']))} vs "
                f"{escape(str(row['opponent']))}</span>"
            )
        else:
            player_cell = (
                f"<strong>{escape(str(row['team']))}</strong>"
                f"<span class='sub'>vs {escape(str(row['opponent']))}</span>"
            )
        rank = escape(str(row["rank_label"]))
        other = escape(str(row["other_rank_label"]))
        proj = _fmt(row["projected"])
        actual = _fmt(row["actual"])
        if view == "model":
            body.append(
                f"<tr class='{css}'>"
                f"<td class='num'>{rank}</td>"
                f"<td class='player'>{player_cell}</td>"
                f"<td class='num'>{proj}</td>"
                f"<td class='num strong'>{actual}</td>"
                f"<td class='num muted'>{other}</td>"
                "</tr>"
            )
        else:
            body.append(
                f"<tr class='{css}'>"
                f"<td class='num'>{rank}</td>"
                f"<td class='player'>{player_cell}</td>"
                f"<td class='num strong'>{actual}</td>"
                f"<td class='num'>{proj}</td>"
                f"<td class='num muted'>{other}</td>"
                "</tr>"
            )
    return (
        "<table class='review'>"
        + header
        + "<tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_html(review: dict[str, Any]) -> str:
    season = review["season"]
    weeks = review["weeks"]
    detail = review["detail"]
    overall = review["overall"]
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    k = overall["K"]
    d = overall["DST"]

    cards = [
        (
            "Kicker top-5 vs league",
            f"{k['model_top_avg']:.2f}",
            f"league {k['league_avg']:.2f} ({k['model_top_avg'] - k['league_avg']:+.2f})",
        ),
        (
            "Defense top-5 vs league",
            f"{d['model_top_avg']:.2f}",
            f"league {d['league_avg']:.2f} ({d['model_top_avg'] - d['league_avg']:+.2f})",
        ),
        (
            "Pooled Spearman",
            f"K {k['spearman_pooled']:.3f}",
            f"DST {d['spearman_pooled']:.3f}",
        ),
    ]
    card_html = "".join(
        f"<div class='card'><div class='label'>{escape(label)}</div>"
        f"<div class='value'>{escape(value)}</div>"
        f"<div class='detail'>{escape(det)}</div></div>"
        for label, value, det in cards
    )

    summary_rows = []
    for position, cfg in POSITIONS.items():
        o = overall[position]
        summary_rows.append(
            "<tr>"
            f"<td>{escape(cfg['label'])}</td>"
            f"<td class='num'>{o['model_top_avg']:.2f}</td>"
            f"<td class='num'>{o['actual_top_avg']:.2f}</td>"
            f"<td class='num'>{o['league_avg']:.2f}</td>"
            f"<td class='num'>{o['model_top_avg'] - o['league_avg']:+.2f}</td>"
            f"<td class='num'>{o['model_top1']:.2f}</td>"
            f"<td class='num'>{o['actual_top1']:.2f}</td>"
            f"<td class='num'>{o['overlap']:.1f}/5</td>"
            f"<td class='num'>{o['spearman_pooled']:.3f}</td>"
            "</tr>"
        )

    week_html = []
    for week in weeks:
        week_html.append(
            f"<section class='week'><div class='week-head'><h2>Week {week}</h2>"
        )
        week_html.append(
            "<p>Weekly replay: retrained before this week. Model top 5 and actual top 5.</p></div>"
        )
        for position, cfg in POSITIONS.items():
            position_rows = detail[
                (detail["week"] == week) & (detail["position"] == position)
            ]
            model_rows = position_rows[position_rows["view"] == "model"].to_dict(
                "records"
            )
            actual_rows = position_rows[position_rows["view"] == "actual"].to_dict(
                "records"
            )
            week_html.append(
                f"<div class='panel'><div class='panel-head'><h3>{escape(cfg['label'])}s — model top 5</h3>"
                f"<span class='hint'>projection → actual</span></div>"
                f"{_html_table(model_rows, position, 'model')}</div>"
                f"<div class='panel'><div class='panel-head'><h3>{escape(cfg['label'])}s — actual top 5</h3>"
                f"<span class='hint'>actual → model projection</span></div>"
                f"{_html_table(actual_rows, position, 'actual')}</div>"
            )
        ks = review["weekly"]["K"]
        ds = review["weekly"]["DST"]
        kw = ks[ks["week"] == week].iloc[0]
        dw = ds[ds["week"] == week].iloc[0]
        week_html.append(
            "<div class='week-totals'>"
            f"K: model top-5 {kw['model_top_avg']:.2f}, actual top-5 {kw['actual_top_avg']:.2f}, league {kw['league_avg']:.2f} "
            f"(overlap {int(kw['overlap'])}/5, rho {kw['spearman']:.3f}) · "
            f"DST: model top-5 {dw['model_top_avg']:.2f}, actual top-5 {dw['actual_top_avg']:.2f}, league {dw['league_avg']:.2f} "
            f"(overlap {int(dw['overlap'])}/5, rho {dw['spearman']:.3f})"
            "</div></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FF Stream Lab — {season} Backtest Review</title>
<style>
  :root {{
    --bg:#0b0f14; --panel:#121820; --panel2:#0f151d; --line:#243041;
    --text:#e7edf5; --muted:#8fa0b5; --accent:#34d399; --accent2:#60a5fa;
    --warn:#fbbf24; --bad:#fb7185;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:28px 20px 60px; }}
  header {{ border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:22px; }}
  .eyebrow {{ color:var(--accent); letter-spacing:.14em; text-transform:uppercase;
    font-size:12px; font-weight:700; }}
  h1 {{ margin:6px 0 0; font-size:34px; line-height:1.1; letter-spacing:-.02em; }}
  h2 {{ margin:0; font-size:22px; letter-spacing:-.01em; }}
  h3 {{ margin:0; font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; }}
  .sub {{ color:var(--muted); font-size:14px; margin-top:8px; }}
  .stamp {{ color:var(--muted); font-size:12px; margin-top:8px; }}
  .cards {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin:22px 0; }}
  .card {{ background:linear-gradient(180deg, var(--panel), var(--panel2));
    border:1px solid var(--line); border-radius:14px; padding:16px; }}
  .card .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; }}
  .card .value {{ font-size:26px; font-weight:750; margin-top:6px; letter-spacing:-.02em; }}
  .card .detail {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  .summary {{ background:var(--panel); border:1px solid var(--line); border-radius:16px;
    padding:18px; margin-bottom:24px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em;
    text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); }}
  td {{ padding:10px 8px; border-bottom:1px solid #1a2431; font-size:13px; vertical-align:middle; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.muted {{ color:var(--muted); }}
  .week {{ background:var(--panel); border:1px solid var(--line); border-radius:16px;
    padding:18px; margin-bottom:20px; }}
  .week-head {{ margin-bottom:12px; }}
  .week-head p {{ color:var(--muted); font-size:12px; margin:6px 0 12px; }}
  .panel {{ margin-top:16px; }}
  .panel-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:8px; }}
  .hint {{ color:var(--muted); font-size:11px; }}
  table.review {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  table.review tr.model td {{ background:#101b18; }}
  table.review tr.actual td {{ background:#0f151d; }}
  .view {{ display:inline-block; padding:3px 7px; border-radius:999px; font-size:10px;
    font-weight:750; letter-spacing:.05em; }}
  tr.model .view {{ background:rgba(52,211,153,.14); color:#6ee7b7; border:1px solid rgba(52,211,153,.35); }}
  tr.actual .view {{ background:rgba(96,165,250,.14); color:#93c5fd; border:1px solid rgba(96,165,250,.35); }}
  .player strong {{ display:block; font-size:13px; }}
  .player .sub {{ display:block; color:var(--muted); font-size:11px; margin-top:2px; }}
  .strong {{ font-weight:750; }}
  .week-totals {{ color:#b6c3d4; font-size:12px; margin-top:12px; padding-top:10px;
    border-top:1px solid var(--line); line-height:1.6; }}
  .method {{ margin-top:24px; background:var(--panel); border:1px solid var(--line);
    border-radius:16px; padding:20px; }}
  .method p, .method li {{ color:#b6c3d4; font-size:13px; line-height:1.65; }}
  @media (max-width: 800px) {{
    .cards {{ grid-template-columns:1fr; }}
    h1 {{ font-size:28px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">FF Stream Lab</div>
    <h1>{season} Backtest Review — Weeks {weeks[0]}–{weeks[-1]}</h1>
    <div class="sub">Weekly replay: retrained before each week using only prior games. Model top 5 vs actual top 5 for kickers and team defenses.</div>
    <div class="stamp">Generated {generated}</div>
  </header>

  <section class="cards">{card_html}</section>

  <section class="summary">
    <h2>8-week summary</h2>
    <table style="margin-top:12px">
      <thead><tr><th>Position</th><th class="num">Model top-5 avg</th><th class="num">Actual top-5 avg</th>
      <th class="num">League avg</th><th class="num">Top-5 vs league</th><th class="num">Model top-1 avg</th>
      <th class="num">Actual top-1 avg</th><th class="num">Avg overlap</th><th class="num">Pooled rho</th></tr></thead>
      <tbody>{"".join(summary_rows)}</tbody>
    </table>
  </section>

  {"".join(week_html)}

  <section class="method">
    <h2>How to read this</h2>
    <p><strong>Model top 5</strong> is ranked by projected points. The <strong>Actual</strong> column is what that player scored, and <strong>Actual rank</strong> is where that performance finished that week.</p>
    <p><strong>Actual top 5</strong> is ranked by real points. The <strong>Proj</strong> column is what the model projected for that player, and <strong>Model rank</strong> is where the model had him.</p>
    <p><strong>Overlap</strong> is how many of the model's top 5 were also in the actual top 5. <strong>Rho</strong> is the weekly Spearman correlation across every kicker or defense that played. <strong>T</strong> marks a tie; tied players at the cutoff are included in the full tables.</p>
    <p>Kicker scoring is extremely noisy: the model's top-5 group can beat the league average while the single top pick does not. Defense has a stronger and more stable signal. Late kicker changes and injuries are not modeled; this replay scores the kicker who actually played.</p>
  </section>
</div>
</body>
</html>
"""


def render_compact(review: dict[str, Any]) -> str:
    season = review["season"]
    weeks = review["weeks"]
    detail = review["detail"]
    top = review["top"]
    lines = [
        f"# {season} Backtest Review — Weeks {weeks[0]}–{weeks[-1]} (compact)",
        "",
        "Model top 5 vs actual top 5. `Got` = actual points for the model pick. "
        "`Actual #` / `Model #` are that player's rank in the other column.",
        "",
    ]
    for week in weeks:
        lines += [f"## Week {week}", "", "**Kickers**", ""]
        lines.append(_compact_table(detail, week, "K", top))
        lines += ["", "**Defenses**", ""]
        lines.append(_compact_table(detail, week, "DST", top))
        lines.append("")
    return "\n".join(lines)


def render_combined(review: dict[str, Any]) -> str:
    season = review["season"]
    weeks = review["weeks"]
    detail = review["detail"]
    top = review["top"]
    lines = [
        f"# {season} Backtest Review — Weeks {weeks[0]}–{weeks[-1]}",
        "",
        "Model top 5 vs actual top 5. `Proj` and `Actual` are fantasy points. "
        "`Actual #` is the model pick's actual rank; `Model #` is the actual leader's model rank.",
        "",
    ]
    for week in weeks:
        lines += [
            f"## Week {week}",
            "",
            "| Pos | # | Model pick | Proj | Got | Actual # | Actual leader | Actual | Proj | Model # |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        notes = []
        for position in ("K", "DST"):
            table = _compact_table(detail, week, position, top)
            for line in table.splitlines():
                if line.startswith("| #") or line.startswith("| ---"):
                    continue
                if line.startswith("| "):
                    lines.append(f"| {position} |" + line[1:])
                elif line.startswith("*Also"):
                    notes.append(f"**{position}:** {line}")
        if notes:
            lines += [""] + notes
        lines.append("")
    return "\n".join(lines)


def write_review(review: dict[str, Any], out_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    season = review["season"]
    weeks = review["weeks"]
    suffix = f"{season}_{weeks[0]:02d}_{weeks[-1]:02d}"
    csv_path = out_dir / f"backtest_review_{suffix}.csv"
    md_path = out_dir / f"backtest_review_{suffix}.md"
    compact_path = out_dir / f"backtest_review_{suffix}_compact.md"
    combined_path = out_dir / f"backtest_review_{suffix}_combined.md"
    html_path = out_dir / f"backtest_review_{suffix}.html"
    review["detail"].to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(review), encoding="utf-8")
    compact_path.write_text(render_compact(review), encoding="utf-8")
    combined_path.write_text(render_combined(review), encoding="utf-8")
    html_path.write_text(render_html(review), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "compact": str(compact_path),
        "combined": str(combined_path),
        "html": str(html_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a weekly backtest.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--weeks", default="1-8", help="e.g. 1-8")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    weeks = _parse_range(args.weeks, (1, 8))
    review = build_review(season=args.season, weeks=weeks, top=args.top)
    paths = write_review(review)

    for position, cfg in POSITIONS.items():
        o = review["overall"][position]
        print(
            f"{cfg['label']}: model top-5 {o['model_top_avg']:.2f} vs league "
            f"{o['league_avg']:.2f} ({o['model_top_avg'] - o['league_avg']:+.2f}); "
            f"pooled rho {o['spearman_pooled']:.3f}; overlap {o['overlap']:.1f}/5"
        )
    print("\nSaved:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
