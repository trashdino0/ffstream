# FF Stream Lab

Weekly streaming projections for **fantasy kickers** and **team defenses**.

Most fantasy tools treat kickers and defenses as an afterthought. This project is built around the two positions you actually stream: it trains separate models, uses pregame-only information, and produces a ranked board with floor, ceiling, and boom probability so you can decide who to start this week.

See [learned.md](learned.md) for the full build notes: data sources, leakage rules, model choices, backtest results, what did not work, and transferable lessons for an NFL game-score model.

```
Kickers   -> blended statistical + Vegas-implied projection, ranked by stream score
Defenses  -> regularized regression on matchup, pressure, and game environment
```

## Quick start

Requires Python 3.10+.

On Windows, run `setup.cmd` once (or double-click it). It creates `.venv` and
installs all dependencies. Then `weekly.cmd` handles the weekly refresh, build,
and board.

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# First run: download nflverse data (several GB, takes a while)
python -m ffstream refresh --seasons 2014-2026

# Build processed tables
python -m ffstream build --seasons 2014-2026

# Sanity-check the install (packages, files, freshness)
python -m ffstream doctor

# Current week's board (auto-detects season/week)
python -m ffstream week --top 15
```

On Windows you can also just double-click `weekly.cmd`. It refreshes the
current season, rebuilds the tables, and writes the board to `outputs/`.

Without installing the package, run everything through the source tree:

```powershell
$env:PYTHONPATH="src"
python -m ffstream week --season 2026 --week 9
```

## Weekly routine (after Monday night)

Once the week's games are final, run:

```powershell
.\scripts\weekly.ps1
```

That refreshes the current season, rebuilds every season's processed tables, generates the next board, and then auto-grades the most recent completed week that has a saved board. The script uses the project `.venv` automatically, then falls back to `py`/`python`, and exits nonzero if any step fails. To target a specific week, change the board size, or skip the download:

```powershell
.\scripts\weekly.ps1 -Season 2026 -Week 10 -Top 25
.\scripts\weekly.ps1 -SkipRefresh          # reuse cached raw data
```

`weekly.cmd` is the double-click wrapper around the same script and pauses at the end so the output stays visible. From a terminal, call `scripts\weekly.ps1` directly to avoid the pause.

Manual equivalent:

```powershell
$env:PYTHONPATH="src"
python -m ffstream refresh --seasons 2026 --force
python -m ffstream build
python -m ffstream week --top 20
```

### Grade the previous week

`weekly.ps1` grades automatically at the end of every run, but you can score any
saved board directly:

```powershell
$env:PYTHONPATH="src"
python -m ffstream grade --season 2026 --week 10
python -m ffstream grade --season 2026 --latest   # newest completed week with a board
```

It writes `outputs/week_2026_10_accuracy.md` and
`outputs/week_2026_10_evaluation.csv` with MAE, bias, rank correlation,
floor/ceiling coverage, boom calibration, and top-5 results versus the league
average. Kickers are matched by `player_id`; if a listed kicker did not play,
the report shows the team's actual kicker and a slot-credited top-5 view.

`python scripts/grade_week.py` still works as a compatibility wrapper.

Do **not** run `build --seasons 2026` by itself — it would overwrite the processed tables with only 2026 data and remove the training history. Run `build` with no season filter to rebuild 2014-2026.

The `week` command retrains automatically on every game before the target week, so no separate `train` step is needed.

## Commands

| Command | What it does |
| --- | --- |
| `ffstream refresh` | Download/cache nflverse schedules, weekly rosters, weekly player stats, and play-by-play under `data/raw`. Skips files already cached unless `--force`. |
| `ffstream build` | Recompute `data/processed/team_games.parquet` and `kicker_games.parquet`, including fantasy scoring. |
| `ffstream week` | Train on every game before the target week and write the kicker/defense board plus an HTML report. |
| `ffstream grade` | Score a saved board against actual results (`--week N` or `--latest`). |
| `ffstream doctor` | Check Python, packages, raw/processed files, data freshness, and output paths. |
| `ffstream backtest` | Walk-forward validation for 2021-2025, a 2026 holdout, and a realistic week-by-week 2026 replay. Writes metrics and predictions to `outputs/`. |
| `ffstream train` | Train and save joblib models for a specific season/week under `models/`. |

Useful `week` options:

```powershell
python -m ffstream week --season 2026 --week 9 --top 20
python -m ffstream week --rank-by mean          # rank kickers by expected points
python -m ffstream week --include-completed     # keep Thursday games already played
python -m ffstream week --out-dir my_boards
```

### Programmatic use

The pipeline is importable, not just a CLI:

```python
from ffstream.cli import run_week

paths = run_week(season=2026, week=9, top=20)
print(paths["html"])
```

Lower-level pieces (`build_team_features`, `build_kicker_features`, `train_final`, `predict_models`, `prepare_table`) can be composed for custom boards or other positions.

## Outputs

`ffstream week` writes three files per slate:

- `outputs/week_2026_09_kickers.csv`
- `outputs/week_2026_09_dst.csv`
- `outputs/week_2026_09_report.html` — self-contained, searchable/sortable board

Key columns:

| Column | Meaning |
| --- | --- |
| `projected` | Expected fantasy points (`proj_mean`). |
| `floor` / `ceiling` | 10th / 90th percentile outcome. |
| `boom_pct` | Estimated probability of a start-worthy week (kicker >= 10, defense >= 12). |
| `stream_score` | `mean + 0.12 * upside + boom_prob`. Kickers are ranked by this by default. |
| `tier` | Kicker: STREAM / SPOT START / DEEP / SIT. Defense: START / STREAM / DEPTH / SIT. |
| `matchup` | Implied total, opponent tendency, and weather context. |
| `roster_status` | Kicker only. `confirmed`, `replaced`, `competition`, `unconfirmed`, or `unavailable`. `replaced` means the board switched to the team's current kicker; the other flags need a manual check. |

`ffstream backtest` writes:

- `outputs/backtest_summary.csv` — model vs. Vegas, form, and Vegas+form baselines
- `outputs/backtest_predictions_*.parquet` — 2021-2025 walk-forward predictions
- `outputs/backtest_2026_predictions_*.parquet` — 2026 holdout
- `outputs/backtest_2026_weekly_predictions_*.parquet` — train-before-each-week replay
- `outputs/backtest_weekly_*.csv` — top-1/3/5 actual scoring by week
- `outputs/feature_importance_*.csv` — largest standardized coefficients

For a readable week-by-week review of the 2026 replay:

```powershell
python scripts/backtest_review.py --season 2026 --weeks 1-8 --top 5
```

This writes a CSV, a full markdown report, a compact markdown view, and a self-contained HTML report under `outputs/`.

## Scoring

Fantasy points are computed from raw stats because nflverse's `fantasy_points` column excludes kickers and team defenses. Rules live in [`config/scoring.json`](config/scoring.json):

- Kicker: 3 / 4 / 5 / 5 points by field-goal distance, 1 point per XP, -1 per missed FG or XP.
- Defense: sacks, interceptions, fumble recoveries, safeties, touchdowns, blocked kicks, two-point returns, plus tiered points-allowed and yards-allowed bonuses.

**If you edit scoring, rerun `ffstream build` before `ffstream week`.** Scoring is applied while building the processed tables.

## How it works

### Data

All data comes from [nflverse](https://github.com/nflverse/nflverse-data) GitHub releases — no API keys and no scraping:

- `games.parquet` — schedules, closing spread/total, roof, surface, rest, weather, officials
- `stats_player_week_{season}.parquet` — kicker attempts, makes, distances, and XP
- `play_by_play_{season}.parquet` — offense/defense efficiency, pace, red-zone, pressure, turnovers
- `roster_weekly_{season}.parquet` — weekly active/reserve/cut status used to keep kicker names current

The pipeline aggregates play-by-play into one row per team-game, then derives kicker rows and team-defense rows.

### Leakage protection

Every rolling feature is shifted one game before it is used, so a row only sees games that kicked off earlier. Opponent features are joined from the opponent's own pregame row. Prior-season averages are shifted forward one season. The weekly board takes the **last kicker known before the target week**, then checks the **nflverse weekly roster snapshot at or before that week**; a Thursday game cannot leak into Sunday projections.

This is covered by tests: `tests/test_features.py` checks that a team's current-game stats never appear in its own pregame features.

### Features

The models consume:

- Vegas: team and opponent implied totals, spread, total, implied margin, ranked implied totals
- Form: rolling 4/8-game and season-to-date offense, defense, and kicker volume/efficiency
- Kicker: attempts, makes, 40+/50+ volume, accuracy, long distance, XP volume, scoring volatility, career and prior-season rates
- Matchup: opponent FG attempts allowed, red-zone TD rate allowed, pressure, takeaways, yards allowed
- Situation: rest, short week, division game, home/away, dome, turf, temperature, wind, altitude

### Models

Model choice was data-driven via walk-forward comparison:

| Position | Estimator | Features | Blend |
| --- | --- | --- | --- |
| Kicker | ElasticNet (`alpha=0.1`, `l1_ratio=0.5`) | 77 curated pregame features | 50/50 with a Vegas-implied baseline |
| Defense | Ridge (`alpha=500`) | Full pregame rolling feature set | Model only |

The kicker blend keeps thin-history or early-season kickers anchored to the game environment. Defense has a stronger standalone signal, so it uses the model directly.

Uncertainty uses the model's residual standard deviation, scaled by each player's recent scoring volatility (clipped to 0.6x-1.8x). Floor/ceiling are normal quantiles; boom probability is the normal survival function above the start threshold.

### Backtest snapshot

Walk-forward results from `outputs/backtest_summary.csv` (2021-2025, then 2026 holdout):

| Target | Period | Model MAE | Spearman | Top-5 avg | Gain vs. league | Vegas Spearman | Form Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kicker | 2021-2025 | 3.68 | 0.155 | 8.52 | +0.71 | 0.140 | 0.062 |
| Kicker | 2026 holdout | 3.76 | 0.036 | 8.40 | +0.09 | 0.018 | 0.043 |
| Defense | 2021-2025 | 4.82 | 0.326 | 10.37 | +3.53 | 0.346 | 0.113 |
| Defense | 2026 holdout | 4.45 | 0.388 | 9.74 | +3.36 | 0.373 | 0.116 |

The 2026 week-by-week replay (retraining before each week) is similar: kicker Spearman 0.035, defense 0.383.

**Honest read:** the defense model has a real, repeatable edge. The kicker model is a modest ranking aid: it beats Vegas-only and form-only baselines over 2021-2025, and its top-5 group still outscores the league average in 2026, but week-to-week kicker correlation is close to zero. Kicker scoring is dominated by opportunity and variance. Use the board to break ties and avoid bad environments, not as a guarantee.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```

The suite covers scoring rules and tier boundaries, shifted/no-leakage features, kicker-change handling, roster swaps and roster flags, model outputs and ranking metrics, grading math, report tiers/escaping, and CLI parsing/slate filtering. CI runs the same suite plus `pip check` on every push (`.github/workflows/ci.yml`).

## Project layout

```
config/scoring.json          League scoring rules
src/ffstream/
  data.py                    nflverse download + cache
  build.py                   Play-by-play -> team-game and kicker-game tables
  scoring.py                 Kicker and DST fantasy points
  features.py                Pregame rolling features, opponent joins, prediction rows
  model.py                   Training, blending, uncertainty, walk-forward validation
  backtest.py                Backtest orchestration and artifacts
  roster.py                  Weekly-roster kicker identity and roster flags
  grade.py                   Score a saved board against actual results
  report.py                  Ranked tables, CSV, self-contained HTML
  cli.py                     Command line entry points
scripts/backtest_review.py   Model top-5 vs actual top-5 review for a backtest
scripts/grade_week.py        Legacy wrapper around `ffstream grade`
tests/                       Pytest suite
data/raw/                    Cached nflverse parquet (gitignored)
data/processed/              Processed team-game and kicker tables (gitignored)
models/                      Saved joblib models (gitignored)
outputs/                     Weekly boards and backtests (gitignored)
```

## Limitations

- **Roster check is a snapshot, not injury news.** The board swaps to the team's current kicker using the nflverse weekly roster and flags `COMPETITION` / `CHECK ROSTER`, but it does not know game-day inactives or practice-report status. Confirm the active kicker on game day.
- **Weather is game-time, not forecast.** The weekly board uses the weather recorded in the schedule file, which may be missing or stale for future games.
- **Kicker signal is weak.** Treat projections as a ranking aid. The model is intentionally conservative and blends toward the market.
- **No depth-chart or coaching-change awareness** beyond what is embedded in recent play-by-play.
- **Backtests are not a promise.** Walk-forward validation avoids lookahead, but roster rules, scoring, and NFL behavior change.

## Possible next steps

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for the prioritized backlog with effort
estimates. The short list:

- Ingest injuries and game-day inactives (weekly roster swaps are handled)
- Add a live weather forecast source
- Model team kicker opportunity explicitly (red-zone stalls and 4th-down aggressiveness)
- Extend the same framework to streaming QB/TE
- Calibrate boom probabilities and add quantile regression for ceilings
- Package a weekly GitHub Action that publishes the HTML board

## Data credit

Data provided by [nflverse](https://github.com/nflverse) under their open-data terms. This project is for personal analysis; verify scoring against your league settings.
