# Usage and reference

## Commands

```bash
uv sync
uv run price-forecast fetch            # 2025-06-01 .. tomorrow -> data/dataset.parquet (~1 min uncached)
uv run price-forecast fetch --start 2025-10-01 --end 2026-09-30
uv run price-forecast backtest         # 90 days x 5 TimesFM configs + 2 baselines (~8 min on Apple MPS)
uv run price-forecast tomorrow         # live forecast of tomorrow
uv run price-forecast figures          # rebuild the charts in docs/ from outputs/backtest_ctx*d files
```

The model checkpoint (~1.2 GB) is downloaded to `models/hf` (git-ignored) unless `HF_HUB_CACHE`
is set. Raw API responses are cached under `data/raw/`. Only chunks older than two days are
cached, so reruns refresh recent data.

### Forecast tomorrow

```bash
uv run price-forecast tomorrow                     # run at ~10:00, forecasts tomorrow
uv run price-forecast tomorrow --date 2026-10-05   # a past day also prints the actual error
```

The command refreshes only the data window it needs (context plus target day), then forecasts
with `tfm_de_wx_tso`. If SMARD hasn't published the TSO forecasts for D yet, it falls back to
`tfm_multi_wx` (weather + calendar). You can also pick a config with `--config`.

It prints an hourly table in ct/kWh with the cheapest 3-hour window and writes:

- `outputs/forecast_<date>.json`: median, q10 and q90 as `{date: {HHMM: ct/kWh}}`, plus the
  cheapest 3-hour window and the slots forecast negative. The shape matches the
  energiepreis-vattenfall-api project, so the output can feed the same Home Assistant setup.
- `outputs/forecast_<date>.png`: the forecast band next to today's actual prices.

Live runs append to `data/tso_availability.csv` how many TSO forecast slots for D were already
published. This file is the only one under `data/` that can't be rebuilt.

### Backtest

```bash
uv run price-forecast backtest                      # last 90 days, 112 days of context
uv run price-forecast backtest --end 2026-10-05     # reproduce the published window
uv run price-forecast backtest --context-days 28 --configs tfm_multi tfm_de_wx_tso
```

Each day D is forecast from what is known at 10:00 on D-1:

- **Context:** prices up to D-1 23:45. These come out of the D-2 auction, so they are known at issue time.
- **Covariates:** past-and-future series running through the end of D.

Results land in `outputs/backtest_ctx<N>d*`: a per-slot parquet, a summary CSV and two PNGs.
Rerunning on the same data gives identical numbers.

| Config | Targets | Covariates |
| --- | --- | --- |
| `naive_d1` / `naive_d7` | — | copy of yesterday / last week |
| `tfm_de` | DE-LU | — |
| `tfm_multi` | 9 markets | — |
| `tfm_multi_wx` | 9 markets | weather + calendar |
| `tfm_multi_wx_tso` | 9 markets | weather + calendar + SMARD TSO forecasts |
| `tfm_de_wx_tso` | DE-LU | weather + calendar + SMARD TSO forecasts |

## Data columns

Everything is stored on a 15-minute UTC grid. DST days have 92 or 100 local quarter hours.

| Source | Columns | Notes |
| --- | --- | --- |
| [SMARD.de](https://www.smard.de) chart-data API | `price_<zone>` (EUR/MWh): DE_LU, FR, NL, BE, AT, CH, PL, CZ, DK1 | 15-min since 2025-10-01; earlier, each hourly price is repeated 4× |
| SMARD.de | `fc_solar_gw`, `fc_wind_onshore_gw`, `fc_wind_offshore_gw`, `fc_total_gw` | TSO day-ahead generation forecast for DE |
| [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) | `wx_irradiance`, `wx_wind_onshore_cf`, `wx_wind_offshore_cf`, `wx_temperature` | `*_previous_day1`, the forecast made the day before, averaged over 25 German solar, wind and population points |
| `holidays` package | `is_weekend`, `holiday_share` | share of the 16 states with a public holiday |

## Inference settings

- **`make_positive=False`:** `TimesFM3Evaluator.predict_batch` clamps forecasts at 0 by default,
  which would erase negative prices.
- **`padding_mode="edge"`:** the 96-slot horizon is padded to the model's 128-step output patch,
  and covariates are edge-padded to match.
- **Symmetric averaging** stays on (the evaluator default).

## Open-Meteo quirk

If the day-1 model run for a time doesn't exist yet, `*_previous_day1` silently returns the
latest forecast instead. Live runs issued earlier than 10:00 on D-1 therefore use weather
forecasts with a longer lead time.

## Layout

| File | Role |
|---|---|
| `__init__.py` | argparse CLI: `fetch`, `backtest`, `tomorrow`, `figures` |
| `smard.py` | SMARD chart-data client (weekly chunks, disk cache) |
| `weather.py` | Open-Meteo Previous Runs client; 25 points → 4 German-wide features |
| `calendar_features.py` | `is_weekend`, `holiday_share` |
| `dataset.py` | Aligns everything on a 15-min UTC grid |
| `model.py` | Loads TimesFM 3.0 and calls `predict_batch` |
| `backtest.py` | Configs, rolling-origin backtest, scoring |
| `forecast.py` | Live forecast of one day; TSO fallback; JSON/console output |
| `plots.py` / `figures.py` | Backtest/forecast charts; README figures |
| `metrics.py` | MAE, RMSE, coverage, pinball loss |
