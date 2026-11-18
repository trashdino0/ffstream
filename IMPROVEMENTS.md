# Improvement backlog

Last updated: 2026-11-17

A prioritized list of things worth doing next. Effort is rough: **S** < 2 hours,
**M** = half a day to a day, **L** = multi-day.

## Current baseline

| Target | Period | MAE | Spearman | Top-5 avg | Top-5 gain |
| --- | --- | ---: | ---: | ---: | ---: |
| Kicker | 2021-2025 | 3.68 | 0.155 | 8.52 | +0.71 |
| Kicker | 2026 replay | 3.76 | 0.035 | 8.60 | +0.29 |
| Defense | 2021-2025 | 4.82 | 0.326 | 10.37 | +3.53 |
| Defense | 2026 replay | 4.45 | 0.383 | 9.54 | +3.16 |

Week 10 (2026) was a good illustration of the split: kicker Spearman -0.27,
defense +0.54. The defense edge is real; the kicker board is a tie-breaker.
Data and operations are likely worth more than model complexity at this point.

---

## P0 — Trust and correctness

These are small and prevent the board from being silently wrong.

- [x] **Roster-aware kicker identity** (M). Done: `ffstream refresh` now pulls
  `weekly_rosters`, and `src/ffstream/roster.py` swaps in the team's current
  active kicker using the snapshot at or before the target week. The board adds
  `roster_status` (`confirmed`, `replaced`, `competition`, `unconfirmed`,
  `unavailable`) plus a human-readable note, and the HTML report shows a
  "Roster checks" panel. Covered by `tests/test_roster.py`.
- [ ] **Market-feature invariant tests** (S). Assert `team_implied + opp_implied
  == total_line`, the favorite has the higher implied total, implied ranks are
  monotone within a week, and home/away assignment is correct. This would have
  caught the `spread_line` sign bug that was just fixed.
- [x] **`ffstream doctor`** (S-M). Done: checks Python/venv, package versions,
  scoring config, raw/processed files, schedule freshness, current/last week,
  output directory, and `pip check`. It still does not verify model artifacts
  or missing lines; those remain candidates.
- [ ] **Data freshness guard** (S). Warn or fail if `games.parquet` or the
  current season's stats are stale, if the target week has no posted lines, or
  if the previous week is not fully final. `doctor` reports freshness but the
  `week` command does not enforce it.
- [ ] **Kickoff-aware board cutoff** (M). Today the board excludes games with a
  final score. A Thursday game in progress can still be included, and partial
  stats can leak into "prior" features. Add an `--as-of` timestamp using
  `gameday`/`gametime` and exclude anything that has kicked off.
- [ ] **Prediction manifest** (S). Write a JSON alongside each board with
  `generated_at`, `as_of`, season/week, package versions, data max week, model
  parameters, and a config hash. Makes later grading reproducible.
- [x] **Move grading into the CLI** (S). Done: `ffstream grade --season 2026
  --week 10` or `--latest`. The grading math is covered by
  `tests/test_grade.py`; `scripts/grade_week.py` remains as a wrapper.
- [ ] **Dependency lock** (S). Pin the current working set (`pip freeze` or a
  `uv.lock`) and run `pip check` in CI. The new PC setup exposed how easy it is
  to run against the wrong interpreter.
- [ ] **Team abbreviation normalization** (S). `learned.md` flags LA/LAR, WAS,
  and LV style changes across seasons. Verify every cross-season join and add a
  canonical team map before the next data refresh.

## P1 — Biggest expected accuracy gains

- [ ] **Injuries and inactives** (M-L). Ingest nflverse injury reports and add a
  manual `config/inactives.json` override for late scratches. Display a status
  column on the board. This is the single most-cited limitation in `learned.md`.
- [ ] **Weather forecast, not game-time weather** (M). Historical `games.parquet`
  has actual temperature/wind; future rows are missing or stale. Use a free
  forecast API (Open-Meteo) keyed by stadium coordinates and kickoff time, or
  explicitly null weather for future rows so the model does not train on
  information it will not have.
- [ ] **Line-timing audit** (M). The backtest trains on `spread_line` /
  `total_line` from the schedule file, which are closing (or latest) lines. If
  predictions are made before close, that is lookahead. Capture opening lines,
  or document the assumption and evaluate with lines available at prediction
  time. This affects how much the backtest can be trusted.
- [ ] **Two-stage kicker model** (M-L). Model expected FG attempts
  (opportunity: red-zone stalls, 4th-down aggressiveness, pace, opponent)
  separately from points per attempt (distance mix, accuracy), then combine.
  Direct points regression is noisy; opportunity is the more learnable part.
- [ ] **DST component model** (M-L). Model points allowed, yards allowed,
  sacks, takeaways, and touchdowns separately, then sum with scoring rules.
  Direct fantasy-point regression mixes very different processes and makes
  touchdowns (rare, noisy) dominate.
- [ ] **Calibrated intervals and boom probabilities** (M). Current sigma is
  residual std scaled by recent volatility, and boom probability is normal.
  Check reliability curves/PIT by week and position; use empirical quantiles,
  isotonic calibration, or conformal prediction.
- [ ] **Recency weighting** (S-M). Add sample weights that decay with age; the
  current model treats 2014 and 2025 equally. Tune the half-life by
  walk-forward, not on the reported test window.
- [ ] **Sign priors / monotonic constraints** (M). Higher team implied total
  should help kickers; higher opponent implied total should hurt defenses.
  Enforce these directions (bounded coefficients, or monotonic constraints in a
  tree model) to reduce unstable fits.
- [ ] **Evaluate stream-score vs mean ordering systematically** (S). Week 9
  showed stream-score ordering hurt the kicker top-5 (9.00) versus mean
  ordering (11.40). Decide the default from the full walk-forward sample, not
  one week.
- [ ] **Nested tuning** (M). `KICKER_BLEND = 0.5`, `alpha=0.1`, `l1_ratio=0.5`,
  and `Ridge(alpha=500)` were chosen with visibility into the evaluation
  window. Use an inner validation split or nested walk-forward before reporting
  final numbers.

## P2 — Evaluation and weekly workflow

- [x] **Auto-grade the previous week in `weekly.cmd`** (S). Done: after writing
  the board, `weekly.ps1` runs `ffstream grade --latest`, which grades the most
  recent completed week that has a saved board and exits 0 when there is
  nothing to grade yet.
- [ ] **Bootstrap confidence intervals** (S-M). Report 95% intervals for
  Spearman and top-5 gain by resampling weeks or players. A 0.02 correlation
  difference over five seasons is not a result.
- [ ] **Streaming simulation** (M). Each week, pick the top available kicker and
  defense, track season total, league average, and regret versus the best
  available option. This is closer to the real decision than MAE.
- [ ] **Rolling model health in the report** (S). Show the last 4-8 weeks of
  MAE/Spearman and a confidence badge; if the kicker model is below a threshold,
  say "tie-breaker only" on the page itself.
- [ ] **Per-week review dashboard** (M). Turn the backtest review into a
  persistent HTML dashboard: model top-5 vs actual top-5, misses, and weekly
  trend.
- [ ] **Closing-line value / market comparison** (M). For defenses especially,
  compare the model's top-5 to the Vegas-implied top-5 each week. If the model
  is not beating the market baseline, do not present it as an edge.

## P3 — Product and reporting

- [ ] **Rest-of-season streamer planner** (M). The schedule already has future
  spreads/totals. Rank teams by the next 2-4 weeks of matchups and show a
  "stash next week" column. High user value, mostly reporting.
- [ ] **Playoff-week emphasis** (S-M). Add a weeks 15-17 view and a strength-of-
  schedule score for defenses and kickers.
- [ ] **Richer board columns** (M): injury status, weather forecast, last-4 game
  log, opponent rank vs position, model-vs-market disagreement, and a one-line
  "why this pick" from the largest coefficients.
- [ ] **Printable cheat sheet** (S). A one-page HTML/PDF view with the top 10
  per position, tiers, and byes.
- [ ] **Ownership / availability filter** (M). Let the user paste their league's
  rostered players or connect an API so the board shows only streamers who are
  actually available.
- [ ] **Scoring presets** (S). Support ESPN/Yahoo/Sleeper DST presets via
  `--scoring`, plus a league-specific config path.
- [ ] **QB/TE streaming** (L). Reuse the team features and nflverse player
  stats; QB needs opponent pass defense, implied total, and pressure; TE needs
  target share and red-zone usage.
- [ ] **Game-score model** (L). `learned.md` already has the blueprint: predict
  margin and total, benchmark against the closing line, and derive implied
  scores. Useful even if it never beats the market.

## P4 — Engineering hygiene

- [x] **CI** (S-M). Added `.github/workflows/ci.yml`: installs `.[dev]`, runs
  `pip check`, and runs `pytest` on push/PR. Ruff is still pending because it is
  not in the dev dependencies yet.
- [ ] **Ruff config + formatting** (S). `.ruff_cache` exists but ruff is not in
  the dev dependencies. Add it to `pyproject.toml` and enforce in CI.
- [ ] **Type checking** (M). Add mypy or pyright for `src/ffstream`; the model
  artifact dicts are a good place to start with a `TypedDict`/dataclass.
- [ ] **Config refactor** (S-M). Move model constants (`KICKER_ALPHA`,
  `KICKER_BLEND`, tier thresholds, boom thresholds) into one config module or
  JSON file so experiments do not require code edits.
- [ ] **Fix pandas fragmentation warnings** (S). Feature building inserts many
  columns one at a time; build blocks and `pd.concat` once. Mostly a speed and
  cleanliness win.
- [ ] **Incremental build/cache** (M). Rebuilding 2014-2026 every week is slow.
  Cache processed tables and features per season and invalidate only the
  current season/week.
- [ ] **Tests for scripts** (S). Grading math and roster swaps are now covered;
  still missing an end-to-end fixture board plus a synthetic week of results
  for `grade()` and the CLI grade path.
- [ ] **Data validation** (S). Assert row counts, unique keys, no duplicate
  team-games, one kicker row per team-week unless a mid-game change occurred.
- [ ] **Archive old one-off outputs** (S). `week_2026_10_tnf_grade.md` is
  superseded by the full-week report; move historical one-offs to
  `outputs/archive/` so the directory stays readable.
- [ ] **Fix git ownership** (S). `git` currently refuses to operate on the repo
  because the directory is owned by a different Windows account. Add the
  `safe.directory` exception or fix ownership before relying on history/CI.

## Explicitly deprioritized

From `learned.md`, these were tested and did not help. Do not redo them without
a genuinely new angle:

- Global LightGBM/boosting for kicker regression.
- Blending kickers with opponent implied total, recent form, dome/weather flags.
- Blending defenses with opponent implied total, total line, or recent form.
- Ranking kickers purely by ceiling.

A new angle would be something like a ranking objective (LambdaRank), monotonic
constraints, or a two-stage opportunity model — not another unconstrained GBM.

## Recommended next five

1. **Injuries and game-day inactives** — roster swaps are handled; this closes
   the remaining "who actually plays" gap.
2. **Line/weather timing audit + `--as-of` cutoff** — protects the backtest and
   the board from lookahead.
3. **Two-stage kicker model or calibrated intervals** — the most plausible path
   to a better kicker signal.
4. **Rest-of-season planner** — highest user value for the least modeling risk.
5. **Bootstrap CIs + streaming simulation** — prove whether the defense edge and
   the kicker top-5 gain survive resampling and a start/sit simulation.
