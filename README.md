# price-forecast

Zero-shot forecasts of German day-ahead electricity prices (DE-LU, 96 quarter hours) with
[`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch).
The forecast for day D is issued at 10:00 CET on D-1, before the day-ahead auction.
Backtest results: [docs/RESULTS.md](docs/RESULTS.md).

## Data (no API keys)

| Source | Columns | Notes |
| --- | --- | --- |
| [SMARD.de](https://www.smard.de) chart-data API | `price_<zone>` (EUR/MWh): DE_LU, FR, NL, BE, AT, CH, PL, CZ, DK1 | 15-min since 2025-10-01; earlier hours are repeated 4x |
| SMARD.de | `fc_solar_gw`, `fc_wind_onshore_gw`, `fc_wind_offshore_gw`, `fc_total_gw` | TSO day-ahead generation forecast for DE |
| [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) | `wx_irradiance`, `wx_wind_onshore_cf`, `wx_wind_offshore_cf`, `wx_temperature` | `*_previous_day1` = forecast made the day before, averaged over German solar / wind / population points |
| `holidays` package | `is_weekend`, `holiday_share` | share of the 16 states with a public holiday |

## Usage

```bash
uv sync
uv run price-forecast fetch            # 2025-06-01 .. tomorrow -> data/dataset.parquet
uv run price-forecast fetch --start 2025-10-01 --end 2026-09-30
```

### Backtest

```bash
uv run price-forecast backtest                      # last 90 days, 112 days of context
uv run price-forecast backtest --context-days 28 --configs tfm_multi tfm_de_wx_tso
```

Each day D is forecast from what is known at 10:00 on D-1: prices up to D-1 23:45 as context,
covariates as past-and-future series through D. Results land in `outputs/backtest_ctx<N>d*`
(per-slot parquet, summary CSV, two PNGs). Configs:

| Config | Targets | Covariates |
| --- | --- | --- |
| `naive_d1` / `naive_d7` | — | copy of yesterday / last week |
| `tfm_de` | DE-LU | — |
| `tfm_multi` | 9 markets | — |
| `tfm_multi_wx` | 9 markets | weather + calendar |
| `tfm_multi_wx_tso` | 9 markets | weather + calendar + SMARD TSO forecasts |
| `tfm_de_wx_tso` | DE-LU | weather + calendar + SMARD TSO forecasts |

`make_positive` is switched off (the evaluator clamps to ≥ 0 by default, which would erase
negative prices) and covariates are edge-padded from the 96-slot day to the model's 128-step
output patch.

The model checkpoint is downloaded to `models/hf` (git-ignored) unless `HF_HUB_CACHE` is set.

Raw API responses are cached under `data/raw/`; only chunks older than two days are cached, so
reruns refresh recent data. Everything is stored on a 15-minute UTC grid; DST days have 92 / 100
local quarter hours.

TimesFM 3.0 weights are under the **TimesFM Non-Commercial License v1.0**.
