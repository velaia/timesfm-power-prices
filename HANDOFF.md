# Handoff: make `price-forecast` presentable on GitHub

You are picking up a small, working experiment and turning it into a polished public GitHub
project. It should read as a **TimesFM 3.0 experiment, investigation and MVP**: a reader should
understand within 10 seconds what was tried, how well it worked, and why it is interesting.

Read this whole file first, then `README.md`, `docs/RESULTS.md` and the code in
`src/price_forecast/`. This file is the only context you get from the earlier session.

---

## 1. What the project is

Zero-shot forecasting of **German day-ahead electricity prices** (bidding zone DE-LU, 96
quarter-hour prices per day) with Google's **TimesFM 3.0** foundation model
(`google/timesfm-3.0-pytorch`). The model is used without any training or fine-tuning.

- **Forecast setup:** each forecast for day D is issued at **10:00 local time on D-1**, before the
  ~12:00 EPEX day-ahead auction. That is the moment a forecast is still useful, for example for
  planning a dishwasher run, EV charging or a heat pump.
- **Model inputs:**
  - 112 days of price history.
  - Optionally the prices of 8 coupled neighbour markets as extra targets.
  - Covariates known in advance:
    - weather forecasts (Open-Meteo),
    - official German TSO wind/solar generation forecasts (SMARD.de),
    - weekend and holiday flags.
- **API keys:** none. Every data source is free and keyless.

The owner (Daniel, GitHub user `velaia`) also has a separate project,
`~/git/energiepreis-vattenfall-api`. It fetches the actual published prices, charts them and
pushes them to Home Assistant. This project's JSON output deliberately uses the same
`{date: {HHMM: ct/kWh}}` shape. Don't modify that other repo.

## 2. Current state (2026-10-04)

- **Git:** repo at `~/git/timesfm`, branch `main`, no remote. Two commits:
  - `d3882ea`: data pipeline, backtest, README, `docs/RESULTS.md` with two PNGs.
  - `ae38c8e`: the `price-forecast tomorrow` command.
- **Stack:** uv project, Python 3.14, package `price_forecast`, CLI entry point `price-forecast`.
- **Git-ignored and rebuildable:**
  - `data/`: API cache and `dataset.parquet`.
  - `outputs/`: backtest parquet/CSV/PNGs and `forecast_<date>.json/.png`.
  - `models/`: the 1.2 GB checkpoint.
- **Exception:** `data/tso_availability.csv` is **not** rebuildable. It logs, per live run, how many
  of the TSO forecast slots for D were already published. Don't delete it.

### Commands (all work)

```bash
uv sync
uv run price-forecast fetch        # SMARD + Open-Meteo -> data/dataset.parquet (2025-06-01 .. tomorrow), ~1 min uncached
uv run price-forecast backtest     # 90 days x 5 TimesFM configs + 2 naive baselines, ~8 min on Apple MPS
uv run price-forecast tomorrow     # live forecast of tomorrow; --date YYYY-MM-DD for any day
```

The backtest results are deterministic: rerunning gives identical numbers.

### Code map (`src/price_forecast/`)

| File | Role |
|---|---|
| `__init__.py` | argparse CLI: `fetch`, `backtest`, `tomorrow` |
| `smard.py` | SMARD chart-data client (weekly chunks, disk cache) |
| `weather.py` | Open-Meteo Previous Runs client; 25 hand-picked points → 4 German-wide features |
| `calendar_features.py` | `is_weekend`, `holiday_share` (share of the 16 states with a holiday) |
| `dataset.py` | Aligns everything on a 15-min UTC grid; DST days have 92/100 local slots |
| `model.py` | Loads TimesFM 3.0 and calls `predict_batch`; sets `HF_HUB_CACHE=models/hf` |
| `backtest.py` | Configs, rolling-origin backtest, scoring (`PREDICT_KWARGS` shared with live runs) |
| `forecast.py` | Live forecast of one day; TSO fallback; JSON/console output |
| `plots.py` | matplotlib charts; palette from the `dataviz` skill (light mode) |
| `metrics.py` | MAE/RMSE/sMAPE/MASE/coverage/pinball (copied from `~/git/timesfm-3.0`) |

## 3. Results to present (do not invent numbers)

These come from 90 forecast days (2026-07-08 .. 2026-10-05) with 112 days of context. Errors are
DE-LU in EUR/MWh; divide by 10 for ct/kWh. Full tables are in `docs/RESULTS.md` and
`outputs/backtest_ctx*_summary.csv`.

| Config | MAE | vs "same as yesterday" |
|---|---|---|
| same as yesterday (`naive_d1`) | 33.5 | — |
| same as last week (`naive_d7`) | 41.7 | −24% |
| TimesFM, DE price only | 24.2 | +28% |
| TimesFM, 9 coupled markets | 23.7 | +29% |
| + weather + calendar | 16.1 | +52% |
| + SMARD TSO forecasts | 13.8 | +59% |
| DE only + all covariates (best) | 13.2 | +61% |

### Story beats worth telling

1. **Zero-shot works.** With the right covariates a general foundation model roughly halves the
   error of the naive benchmark and beats it on 91% of days. There is no training at all, and one
   forecast takes about 1 s on a MacBook GPU.
2. **Covariates matter more than the multivariate story.** TimesFM 3's headline feature is
   multivariate forecasting, but co-forecasting 8 neighbour markets barely helps (24.2 → 23.7).
   Weather plus generation forecasts do almost all the work.
3. **Longer context helps,** with diminishing returns. MAE for the best config: 7 days 20.2,
   14 days 16.3, 28 days 14.7, 56 days 13.8, 112 days 13.2.
4. **Honest limitations:**
   - Spikes get damped. On 2026-09-14 the evening peak was ~740 EUR/MWh and the forecast peaked
     at ~480 (MAE that day 39.5 vs 90.4 naive).
   - Only 57% of negative-price slots are caught, although forecast negatives are 88% precise.
   - The q10–q90 band covers ~78% of actual prices (target 80%), so it is roughly calibrated.
5. **Leakage caveat (state it plainly).** SMARD keeps only the final vintage of the TSO forecasts.
   It is **not yet verified** that they are published by 10:00 on D-1. The weather-only config
   (16.1, +52%) is the guaranteed-honest number. The live command falls back to it automatically
   and logs TSO availability to settle the question.
6. **Practical metric.** The forecast's cheapest 3-hour window is within 30 minutes of the true
   cheapest window on ~86% of days, against 66% for "same as yesterday".

### Live forecast on record

A forecast for **Tue 2026-10-06** was issued on Sun 2026-10-04 at 22:58 with the weather-only
fallback:

- daily mean 19.0 ct/kWh
- cheapest window 11:45–14:45 at ~9.5 ct/kWh
- evening peak ~36 ct/kWh at 19:30
- files: `outputs/forecast_2026-10-06.json/.png`

Once the auction result is published (Mon 2026-10-05, ~13:00), you can compare it to the actual
prices with `uv run price-forecast fetch` and then `uv run price-forecast tomorrow --date 2026-10-06`.

**Caution:** rerunning that command overwrites the original forecast files and uses newer weather
data. Copy `outputs/forecast_2026-10-06.*` aside first if you want a true "forecast on record vs
actual" exhibit.

## 4. Gotchas you must know

- **Clamped forecasts.** `TimesFM3Evaluator.predict_batch` defaults to `make_positive=True`, which
  clamps forecasts at 0 and would erase negative prices. `backtest.PREDICT_KWARGS` turns it off.
  Keep it that way.
- **Horizon padding.** The horizon is padded to the 64-step output patch (96 → 128).
  `padding_mode="edge"` handles the covariates.
- **Model cache.** `~/.cache/huggingface/hub` is a symlink to an external drive that is usually
  not mounted. The project downloads the checkpoint to `models/hf` (the user chose this). Don't
  "fix" it by touching the symlink.
- **Open-Meteo quirk.** For times whose day-1 model run doesn't exist yet, `*_previous_day1`
  silently returns the latest forecast (verified). Live runs issued earlier than 10:00 D-1
  therefore use longer-lead weather.
- **Pre-switch prices.** Before 2025-10-01, SMARD's quarter-hour price series repeats each hourly
  price 4×, because the 15-min day-ahead market started on 2025-10-01. It is used only as context.
- **Weights license.** The TimesFM 3.0 weights are under the **TimesFM Non-Commercial License
  v1.0**. The README must say so. The repo itself needs its own license; ask the user which one
  (MIT is a reasonable suggestion for the code).
- **Small known wart.** In the backtest summary, `naive_d1` shows `days beat d-1 % = 0.00`, which
  compares the baseline with itself. It should show as n/a.

## 5. What to deliver

### Must have

1. **README rewrite**, presentation first:
   - A title, a one-line pitch and an **"At a glance" / TL;DR block at the top** (3–5 bullets with
     the headline numbers above).
   - A **hero graphic** directly below the summary (see below).
   - Short sections:
     - Why (the 10:00 D-1 use case).
     - How it works, with a simple diagram of data sources → aligned grid → TimesFM → JSON/chart.
       Mermaid renders on GitHub.
     - Results table.
     - What I learned (the story beats).
     - Limitations and caveats.
     - Quickstart.
     - Data sources with attribution: SMARD.de / Bundesnetzagentur, Open-Meteo, `holidays`.
     - License note.
   - Keep the existing detailed reference material (CLI flags, configs, data columns) lower down
     or move it into `docs/`.
2. **At least one beautiful graphic**, generated by code in the repo (not hand-made) and
   committed under `docs/`. Suggested hero: one striking forecast day showing actual price, the
   TimesFM median, the q10–q90 band and the naive baseline, plus a compact companion panel with
   MAE by config (an ablation bar chart, sorted, with the naive baseline as a reference line).
   Good candidate days:
   - 2026-09-05: deep negative prices, MAE 10.4 vs 47.7 naive.
   - 2026-10-05: clean, 8.3 vs 22.0.

   These are 112-day-context numbers from `outputs/backtest_ctx112d.parquet`.

   Load the **`dataviz` skill before writing any chart code** and follow it: one axis per chart,
   fixed colour per entity (the existing palette: actual = near-black ink, TimesFM = blue
   `#2a78d6`, naive = dashed orange `#eb6834`, extra = aqua `#1baf7a`), thin marks, legend plus
   direct labels, recessive grid. Then **render it and look at it** before committing.
   Consider adding a `price-forecast figures` subcommand, or a `scripts/make_figures.py`, so the
   graphics are reproducible from `outputs/backtest_ctx112d.parquet`.
3. **Make `docs/RESULTS.md` consistent** with the README and regenerate its charts if the
   styling changes.
4. **Repo hygiene:**
   - Decide on a LICENSE file with the user.
   - Fill in the `pyproject.toml` metadata (keywords, URLs once a remote exists).
   - Keep `.gitignore`.
   - Remove `HANDOFF.md` or move it to `docs/` at the end, as the user prefers.

### Nice to have (ask before doing big ones)

- Fix the `naive_d1` "days beat" wart.
- A small test suite (pytest) for the pure parts: `metrics`, `dataset.grid` DST handling,
  `forecast.interval_key` / `by_interval`, `weather.wind_capacity_factor`. Avoid network and model
  downloads in tests.
- A GitHub Actions workflow running ruff + those tests. No model download in CI.
- A "forecast on record vs actual" exhibit for Tue 2026-10-06 (see the caution in section 3).
- Social-preview image (1280×640) derived from the hero graphic.

## 6. Rules of engagement

- **Don't create the GitHub repo, add a remote, push or make anything public without the user's
  explicit go-ahead.** Committing locally on a branch is fine; ask before rewriting history.
- **Every number in the README must trace back to a file in `outputs/` or a command you ran.** If
  you rerun the backtest, update all numbers consistently. It is deterministic for the same data,
  but `fetch` on a later date shifts the 90-day window. To reproduce the published table, pass
  `--end 2026-10-05`.
- **Don't break the CLI** or change the JSON output shape. The Vattenfall project's format is
  intentional.
- **Keep the tone of an honest experiment write-up,** not marketing. The caveats are part of what
  makes it credible.
- **Match the existing code style:** small modules, docstrings explaining *why*, no
  over-abstraction.
- **Commit messages** end with the attribution line the harness provides.

## 7. Questions to ask the user early

1. Which license for the code? (Suggest MIT, noting the non-commercial weights.)
2. Repo name and visibility, for example `velaia/timesfm-power-prices`, plus any preferred
   wording or language. The README is English; the user is German-speaking.
3. Which hero day do they like: deep negatives (2026-09-05), clean typical (2026-10-05), or the
   Tue 2026-10-06 forecast-on-record once its actual prices are known?
4. Should `HANDOFF.md` be removed before publishing?
