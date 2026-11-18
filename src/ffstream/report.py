"""Weekly ranking tables, CSV output, and a self-contained HTML report."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.{digits}f}{suffix}"


def _weather(row: pd.Series) -> str:
    if int(row.get("is_dome", 0) or 0) == 1:
        return "Dome"
    wind = row.get("wind")
    temp = row.get("temp")
    if pd.notna(wind) and float(wind) >= 15:
        return f"Wind {float(wind):.0f} mph"
    if pd.notna(temp) and float(temp) <= 32:
        return f"Cold {float(temp):.0f}°F"
    return "Outdoors"


def _matchup_note(row: pd.Series, target: str) -> str:
    weather = _weather(row)
    if target == "kicker_pts":
        return (
            f"Implied {_fmt(row.get('team_implied'))} | "
            f"opp FGA allowed {_fmt(row.get('opp_def_fg_att_allowed_last4'))}/g | "
            f"{weather}"
        )
    return (
        f"Opp implied {_fmt(row.get('opp_implied'))} | "
        f"opp sacks allowed {_fmt(row.get('opp_off_sacks_last4'))}/g | "
        f"{weather}"
    )


def _tier(rank: int, target: str) -> str:
    if target == "kicker_pts":
        if rank <= 5:
            return "STREAM"
        if rank <= 10:
            return "SPOT START"
        if rank <= 16:
            return "DEEP"
        return "SIT"
    if rank <= 5:
        return "START"
    if rank <= 10:
        return "STREAM"
    if rank <= 16:
        return "DEPTH"
    return "SIT"


def prepare_table(
    rows: pd.DataFrame,
    pred: pd.DataFrame,
    target: str,
    rank_by: str = "mean",
) -> pd.DataFrame:
    table = rows.copy()
    for column in pred.columns:
        table[column] = pred[column]

    table["player"] = table.get("player_display_name", table.get("team", "Team"))
    table["player"] = table["player"].fillna(
        table["team"] + " K" if target == "kicker_pts" else table["team"]
    )
    table["projected"] = table["proj_mean"].round(2)
    table["floor"] = table["proj_p10"].round(2)
    table["ceiling"] = table["proj_p90"].round(2)
    table["boom_pct"] = (table["boom_prob"] * 100).round(1)

    sort_column = "stream_score" if rank_by == "stream" else "proj_mean"
    table = table.sort_values(sort_column, ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["tier"] = table["rank"].map(lambda r: _tier(int(r), target))
    table["matchup"] = table.apply(lambda r: _matchup_note(r, target), axis=1)

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
        "matchup",
        "team_implied",
        "opp_implied",
        "total_line",
        "stream_score",
        "proj_mean",
        "stream_score",
        "player_id",
        "roster_status",
        "roster_note",
        "game_id",
        "season",
        "week",
    ]
    columns = [c for c in dict.fromkeys(columns) if c in table.columns]
    return table[columns].copy()


def _badge_class(tier: str) -> str:
    return {
        "STREAM": "stream",
        "START": "start",
        "SPOT START": "spot",
        "DEEP": "deep",
        "DEPTH": "deep",
        "SIT": "sit",
    }.get(tier, "sit")


def _roster_badge(row: pd.Series) -> str:
    status = str(row.get("roster_status", "confirmed") or "confirmed").lower()
    if status == "confirmed":
        return ""
    labels = {
        "replaced": "ROSTER SWAP",
        "competition": "COMPETITION",
        "unconfirmed": "CHECK ROSTER",
        "unavailable": "NO ROSTER DATA",
    }
    label = labels.get(status, status.upper())
    note = escape(str(row.get("roster_note", "")))
    return (
        f"<span class='roster {escape(status)}' title='{note}'>{escape(label)}</span>"
    )


def _roster_alerts(table: pd.DataFrame) -> str:
    if "roster_status" not in table.columns:
        return ""
    status = table["roster_status"].fillna("confirmed").astype(str)
    flagged = table[status != "confirmed"]
    if flagged.empty:
        return ""

    grouped: dict[str, list[str]] = {}
    for _, row in flagged.sort_values(["roster_status", "rank"]).iterrows():
        note = str(row.get("roster_note", "") or "needs a lineup check")
        grouped.setdefault(note, []).append(str(row["player"]))

    items = []
    for note, players in grouped.items():
        if len(players) == 1:
            label = f"<strong>{escape(players[0])}</strong>"
        else:
            label = f"<strong>{len(players)} kickers</strong>"
        items.append(f"<li>{label}: {escape(note)}</li>")
    return (
        "<div class='alerts'><div class='alerts-title'>Roster checks</div>"
        f"<ul>{''.join(items)}</ul></div>"
    )


def _html_table(table: pd.DataFrame, table_id: str) -> str:
    rows = []
    for _, row in table.head(32).iterrows():
        tier = escape(str(row["tier"]))
        roster = _roster_badge(row)
        rows.append(
            "<tr>"
            f"<td class='rank' data-sort='{int(row['rank'])}'>{int(row['rank'])}</td>"
            f"<td class='player'><strong>{escape(str(row['player']))}</strong>"
            f"<span class='team'>{escape(str(row.get('team', '')))} vs {escape(str(row.get('opponent', '')))}</span>"
            f"{roster}</td>"
            f"<td data-sort='{row['projected']}'><strong>{_fmt(row['projected'])}</strong></td>"
            f"<td data-sort='{row['floor']}' class='muted'>{_fmt(row['floor'])}</td>"
            f"<td data-sort='{row['ceiling']}' class='muted'>{_fmt(row['ceiling'])}</td>"
            f"<td data-sort='{row['boom_pct']}'>{_fmt(row['boom_pct'], 0, '%')}</td>"
            f"<td><span class='badge {_badge_class(tier)}'>{tier}</span></td>"
            f"<td class='note'>{escape(str(row['matchup']))}</td>"
            "</tr>"
        )
    return (
        f"<table id='{table_id}' class='rankings'>"
        "<thead><tr>"
        "<th>#</th><th>Player</th><th>Proj</th><th>Floor</th><th>Ceiling</th>"
        "<th>Boom</th><th>Tier</th><th>Matchup</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_html(
    kicker: pd.DataFrame,
    dst: pd.DataFrame,
    season: int,
    week: int,
    meta: dict[str, Any] | None = None,
) -> str:
    meta = meta or {}
    top_k = kicker.iloc[0] if not kicker.empty else None
    top_d = dst.iloc[0] if not dst.empty else None
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    k_name = escape(str(top_k["player"])) if top_k is not None else "—"
    k_proj = _fmt(top_k["projected"]) if top_k is not None else "—"
    d_name = escape(str(top_d["team"])) if top_d is not None else "—"
    d_proj = _fmt(top_d["projected"]) if top_d is not None else "—"

    notes = meta.get("notes", "")
    rows_k = int(meta.get("kicker_train_rows", 0))
    rows_d = int(meta.get("dst_train_rows", 0))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FF Stream Lab — {season} Week {week}</title>
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
  header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end;
    border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:22px; }}
  .eyebrow {{ color:var(--accent); letter-spacing:.14em; text-transform:uppercase;
    font-size:12px; font-weight:700; }}
  h1 {{ margin:6px 0 0; font-size:34px; line-height:1.1; letter-spacing:-.02em; }}
  h2 {{ margin:0 0 12px; font-size:19px; letter-spacing:-.01em; }}
  h3 {{ margin:0 0 8px; font-size:14px; color:var(--muted); text-transform:uppercase;
    letter-spacing:.1em; }}
  .sub {{ color:var(--muted); font-size:14px; margin-top:8px; }}
  .stamp {{ color:var(--muted); font-size:12px; text-align:right; white-space:nowrap; }}
  .cards {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin-bottom:24px; }}
  .card {{ background:linear-gradient(180deg, var(--panel), var(--panel2));
    border:1px solid var(--line); border-radius:14px; padding:16px; }}
  .card .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; }}
  .card .value {{ font-size:24px; font-weight:750; margin-top:6px; letter-spacing:-.02em; }}
  .card .detail {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:16px;
    padding:18px; overflow:hidden; }}
  .panel-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline;
    margin-bottom:10px; }}
  .hint {{ color:var(--muted); font-size:12px; }}
  .toolbar {{ display:flex; gap:10px; align-items:center; margin:18px 0 12px; }}
  input[type=search] {{ width:100%; max-width:360px; background:var(--panel); color:var(--text);
    border:1px solid var(--line); border-radius:10px; padding:10px 12px; outline:none; }}
  input[type=search]:focus {{ border-color:var(--accent); }}
  .table-wrap {{ overflow:auto; border-radius:10px; border:1px solid var(--line); }}
  table {{ width:100%; border-collapse:collapse; min-width:600px; }}
  th {{ position:sticky; top:0; background:#0e141c; color:var(--muted); font-size:11px;
    text-transform:uppercase; letter-spacing:.08em; text-align:left; padding:10px 8px;
    cursor:pointer; user-select:none; border-bottom:1px solid var(--line); }}
  th:hover {{ color:var(--text); }}
  td {{ padding:10px 8px; border-bottom:1px solid #1a2431; font-size:13px; vertical-align:middle; }}
  tbody tr:hover {{ background:#151f2b; }}
  td.rank {{ color:var(--muted); width:34px; }}
  td.player strong {{ display:block; font-size:13px; }}
  td.player .team {{ display:block; color:var(--muted); font-size:11px; margin-top:2px; }}
  .muted {{ color:var(--muted); }}
  .note {{ color:#b6c3d4; font-size:12px; }}
  .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:10px;
    font-weight:750; letter-spacing:.05em; white-space:nowrap; }}
  .badge.stream, .badge.start {{ background:rgba(52,211,153,.14); color:#6ee7b7; border:1px solid rgba(52,211,153,.35); }}
  .badge.spot {{ background:rgba(96,165,250,.14); color:#93c5fd; border:1px solid rgba(96,165,250,.35); }}
  .badge.deep {{ background:rgba(251,191,36,.12); color:#fcd34d; border:1px solid rgba(251,191,36,.3); }}
  .badge.sit {{ background:rgba(251,113,133,.1); color:#fda4af; border:1px solid rgba(251,113,133,.28); }}
  .roster {{ display:inline-block; margin-top:4px; padding:2px 6px; border-radius:999px;
    font-size:9px; font-weight:750; letter-spacing:.06em; white-space:nowrap; }}
  .roster.replaced {{ background:rgba(251,191,36,.15); color:#fcd34d; border:1px solid rgba(251,191,36,.35); }}
  .roster.competition {{ background:rgba(96,165,250,.14); color:#93c5fd; border:1px solid rgba(96,165,250,.35); }}
  .roster.unconfirmed {{ background:rgba(251,113,133,.12); color:#fda4af; border:1px solid rgba(251,113,133,.3); }}
  .roster.unavailable {{ background:rgba(148,163,184,.12); color:#cbd5e1; border:1px solid rgba(148,163,184,.3); }}
  .alerts {{ margin:0 0 12px; padding:10px 12px; border-radius:10px;
    background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.28); }}
  .alerts-title {{ font-size:11px; font-weight:750; letter-spacing:.06em;
    text-transform:uppercase; color:#fcd34d; margin-bottom:4px; }}
  .alerts ul {{ margin:0; padding-left:16px; color:#e8eef6; font-size:12px; }}
  .alerts li {{ margin:2px 0; }}
  .method {{ margin-top:24px; background:var(--panel); border:1px solid var(--line);
    border-radius:16px; padding:20px; }}
  .method p, .method li {{ color:#b6c3d4; font-size:13px; line-height:1.65; }}
  .method code {{ color:#93c5fd; background:#0b1118; padding:2px 5px; border-radius:5px; }}
  .foot {{ color:var(--muted); font-size:11px; margin-top:18px; line-height:1.6; }}
  @media (max-width: 980px) {{
    .grid {{ grid-template-columns:1fr; }}
    .cards {{ grid-template-columns:1fr; }}
    header {{ flex-direction:column; align-items:flex-start; }}
    .stamp {{ text-align:left; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="eyebrow">FF Stream Lab</div>
      <h1>{season} Week {week} — Streaming Board</h1>
      <div class="sub">Kicker and team-defense projections built from pregame form, Vegas lines, matchup tendencies, rest, and weather.</div>
    </div>
    <div class="stamp">Generated {generated}<br>Model: regularized regression + market blend</div>
  </header>

  <section class="cards">
    <div class="card">
      <div class="label">Top kicker stream</div>
      <div class="value">{k_name}</div>
      <div class="detail">{k_proj} projected points · {escape(str(top_k["team"])) if top_k is not None else "—"}</div>
    </div>
    <div class="card">
      <div class="label">Top defense stream</div>
      <div class="value">{d_name}</div>
      <div class="detail">{d_proj} projected points · {escape(str(top_d["opponent"])) if top_d is not None else "—"}</div>
    </div>
    <div class="card">
      <div class="label">Training sample</div>
      <div class="value">{rows_k:,} / {rows_d:,}</div>
      <div class="detail">kicker games / team-defense games</div>
    </div>
  </section>

  <div class="toolbar">
    <input id="search" type="search" placeholder="Filter player, team, or matchup…" aria-label="Filter rankings">
    <span class="hint">Click a column header to sort. Proj = expected points. Ceiling = 90th percentile.</span>
  </div>

  <section class="grid">
    <div class="panel">
      <div class="panel-head"><h2>Kickers</h2><span class="hint">ranked by stream score</span></div>
      {_roster_alerts(kicker)}
      <div class="table-wrap">{_html_table(kicker, "kicker-table")}</div>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>Team defenses</h2><span class="hint">ranked by projected points</span></div>
      <div class="table-wrap">{_html_table(dst, "dst-table")}</div>
    </div>
  </section>

  <section class="method">
    <h2>How the board is built</h2>
    <p><strong>Kickers.</strong> The model uses Vegas implied team total and game script, the kicker's recent attempts and accuracy, team red-zone touchdown rate (more TDs means fewer field goals), opponent field-goal attempts allowed, dome/weather, rest, and recent scoring. The final projection blends the statistical model with a market-implied baseline so early-season or thin-history kickers do not drift too far from the game line.</p>
    <p><strong>Defenses.</strong> The model uses the opponent's implied total, offensive EPA, sack and turnover rates, red-zone efficiency, pace, the defense's own pressure/takeaway form, and game environment. Streaming defenses are driven by pressure and turnover upside as much as by points allowed.</p>
    <p><strong>Reading the columns.</strong> <code>Floor</code> and <code>Ceiling</code> are the 10th and 90th percentile outcomes. <code>Boom</code> is the estimated probability of a start-worthy week (kicker ≥ 10, defense ≥ 12). For kickers, the board is ordered by a stream score that gives a small tilt to ceiling; use <code>Proj</code> if you want pure expected points.</p>
    <p><strong>Data.</strong> nflverse play-by-play, weekly player stats, schedules, weekly rosters, closing spreads/totals, rest, roof, surface, temperature, and wind. Every rolling feature is shifted one game, so the model only sees information available before kickoff.</p>
    <p class="foot">Kicker names are checked against the nflverse weekly roster snapshot; a <code>ROSTER SWAP</code>, <code>COMPETITION</code>, or <code>CHECK ROSTER</code> badge means the lineup needs a manual check. Game-day inactives and injury reports are still not ingested. Kicker scoring is inherently high variance; treat the projections as a ranking aid, not a guarantee. {escape(notes)}</p>
  </section>
</div>

<script>
const search = document.getElementById('search');
search.addEventListener('input', () => {{
  const q = search.value.toLowerCase();
  document.querySelectorAll('.rankings tbody tr').forEach(tr => {{
    tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
  }});
}});

document.querySelectorAll('.rankings').forEach(table => {{
  table.querySelectorAll('th').forEach((th, index) => {{
    th.addEventListener('click', () => {{
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const dir = th.dataset.dir === 'asc' ? -1 : 1;
      table.querySelectorAll('th').forEach(x => delete x.dataset.dir);
      th.dataset.dir = dir === 1 ? 'desc' : 'asc';
      rows.sort((a, b) => {{
        const ac = a.children[index], bc = b.children[index];
        const av = ac.dataset.sort ?? ac.innerText;
        const bv = bc.dataset.sort ?? bc.innerText;
        const an = parseFloat(av), bn = parseFloat(bv);
        if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * dir;
        return String(av).localeCompare(String(bv)) * dir;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}});
</script>
</body>
</html>
"""


def write_weekly_outputs(
    kicker: pd.DataFrame,
    dst: pd.DataFrame,
    season: int,
    week: int,
    meta: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> dict[str, str]:
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"week_{season}_{week:02d}"
    kicker_path = out_dir / f"{stem}_kickers.csv"
    dst_path = out_dir / f"{stem}_dst.csv"
    html_path = out_dir / f"{stem}_report.html"
    kicker.to_csv(kicker_path, index=False)
    dst.to_csv(dst_path, index=False)
    html_path.write_text(render_html(kicker, dst, season, week, meta), encoding="utf-8")
    return {
        "kickers": str(kicker_path),
        "dst": str(dst_path),
        "html": str(html_path),
    }
