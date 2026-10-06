# Backtest results — 2026-10-04

TimesFM 3.0 (`google/timesfm-3.0-pytorch`), zero-shot, forecasting the 96 quarter-hour DE-LU
day-ahead prices of day D as of **10:00 on D-1**.

- **Period:** 90 forecast days, 2026-07-08 .. 2026-10-05
- **Context:** 112 days (10,752 quarter hours)
- **Device:** Apple-silicon GPU (MPS)
- **Prices in the period:** mean 131.7 EUR/MWh, std 73.7, 551 of 8,640 quarter hours negative

Reproduce with `uv run price-forecast fetch && uv run price-forecast backtest --end 2026-10-05`
(and `--context-days 7|14|28|56` for the context table). Rebuild the charts below with
`uv run price-forecast figures`.

Back to the [README](../README.md) · CLI and data reference: [USAGE.md](USAGE.md) · Speed: [PERFORMANCE.md](PERFORMANCE.md)
· Follow-up: [more history, coarser steps and a full year](#more-history-and-coarser-steps)
([interactive charts](https://claude.ai/artifact/AFtV1Mv1Q3uaZB8NB8xvia))

## Main table (context 112 days)

| Config | What | MAE | RMSE | vs d-1 | days beating d-1 | 3h regret | 3h hit | neg recall | neg precision | 80% band coverage | pinball | s/day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `naive_d1` | same as yesterday | 33.54 | 53.06 | — | — | 2.07 | 66% | 43% | 42% | — | — | — |
| `naive_d7` | same as last week | 41.70 | 62.20 | −24% | 39% | 3.39 | 54% | 54% | 44% | — | — | — |
| `tfm_de` | DE price only | 24.21 | 38.01 | +28% | 74% | 1.04 | 72% | 34% | 83% | 77% | 9.52 | 0.16 |
| `tfm_multi` | 9 coupled markets | 23.71 | 37.30 | +29% | 76% | 1.06 | 72% | 36% | 86% | 76% | 9.32 | 0.82 |
| `tfm_multi_wx` | + weather + calendar | 16.06 | 25.88 | +52% | 86% | 0.60 | 83% | 44% | 86% | 77% | 6.33 | 1.40 |
| `tfm_multi_wx_tso` | + SMARD TSO forecasts | 13.75 | 21.97 | +59% | 90% | 0.52 | 88% | 52% | 86% | 77% | 5.46 | 1.74 |
| `tfm_de_wx_tso` | DE only, all covariates | **13.22** | **20.90** | **+61%** | **91%** | 0.57 | 86% | 57% | 88% | 78% | **5.24** | 1.04 |

Errors in EUR/MWh.

- **vs d-1:** reduction in mean daily MAE compared with `naive_d1`.
- **3h regret:** how much more expensive (mean price, EUR/MWh) the forecast's cheapest 3-hour window
  really was than the true cheapest window.
- **3h hit:** share of days where the forecast's cheapest window starts within 30 minutes of the true one.
- **neg recall / precision:** for slots with price < 0.

## Context length (MAE, EUR/MWh)

| Context days | `tfm_multi_wx` | `tfm_multi_wx_tso` | `tfm_de_wx_tso` |
|---|---|---|---|
| 7 | 23.84 | 21.07 | 20.20 |
| 14 | 19.60 | 17.14 | 16.27 |
| 28 | 17.40 | 15.42 | 14.71 |
| 56 | 16.34 | 14.27 | 13.78 |
| 112 | 16.06 | 13.75 | 13.22 |
| 160 | 15.97 | — | 13.40 |

With 28 days of context, `tfm_de` scored 25.44 and `tfm_multi` 24.37. 160 days is the most
TimesFM 3 accepts at 15-minute steps; only two configs were run there.

## Findings

1. **Covariates matter far more than co-forecasting the neighbours.** Adding the 8 coupled
   markets barely helps (24.2 → 23.7), while weather + calendar cut the error by a third and the
   TSO generation forecasts cut it further. With all covariates, DE alone beats the 9-market setup.
2. **`tfm_multi_wx` (16.1) is the safe number for a 10:00 run.** Its weather inputs are
   previous-day Open-Meteo runs, which exist at issue time. SMARD keeps only the final vintage of the
   TSO forecasts, so it is not yet verified that they are published by 10:00 on D-1.
3. **Longer context keeps helping,** with diminishing returns beyond ~56 days. At 160 days, the
   limit, it stops: no better than 112 (see [below](#more-history-and-coarser-steps)).
4. **Price spikes and deep negatives get damped.** On the most volatile day (2026-09-14) the evening
   peak reached ~740 EUR/MWh; the forecast peaked at ~480 (with 28 days of context only ~310). Slots predicted negative are right 88% of
   the time, but only 57% of negative slots are caught.
5. **The quantile band is roughly calibrated:** the q10–q90 band covers 76–78% of the actual prices,
   against a target of 80%.

![MAE by context length](context_length.png)

![Example days](backtest_ctx112d_days.png)

![Daily MAE](backtest_ctx112d_daily_mae.png)

## More history and coarser steps

Follow-up backtests from 2026-10-05 asked two questions. Would more history help? And does the model hold up
over a full year? The charts are on an [interactive page](https://claude.ai/artifact/AFtV1Mv1Q3uaZB8NB8xvia)
(offline copy: [context_study.html](context_study.html), open it in a browser).

**Context is capped.** TimesFM 3 accepts at most 15,360 steps (`_MAX_CONTEXT_LENGTH`). That is 160 days of quarter
hours, 320 of half hours or 640 of hours. Longer inputs are cut without a warning.

### Coarser steps for longer history

Same 90 days as above. Prices and covariates are averaged into UTC half hours or hours and forecast at that
resolution. The forecast is then spread back to quarter hours, held flat or interpolated through the bucket centres,
and scored against the real 15-minute prices. MAE in EUR/MWh.

| Steps | Context days | Spread to 15 min | `tfm_multi_wx` | `tfm_de_wx_tso` |
|---|---|---|---|---|
| 15 min | 112 | — | **16.06** | **13.22** |
| 15 min | 160 | — | **15.97** | 13.40 |
| 30 min | 112 | interpolated | 17.50 | 15.13 |
| 30 min | 160 | interpolated | 17.10 | 14.69 |
| 30 min | 320 | interpolated | 17.08 | 14.58 |
| 1 hour | 112 | interpolated | 18.87 | 16.04 |
| 1 hour | 160 | interpolated | 17.91 | 15.68 |
| 1 hour | 320 | interpolated | 17.56 | 14.93 |
| 1 hour | 480 | interpolated | 16.48 | 14.18 |
| 1 hour | 640 | interpolated | 16.76 | 14.58 |

Holding each half hour or hour flat is worse at every context length, by about 0.3 for half hours and 1.3–1.7 for
hours. The flat-step rows are in `outputs/coarse_*_90d_summary.csv`.

1. **At 15-minute steps, 112 days is enough.** The 160-day limit changes the result by less than 0.2.
2. **Coarser steps do improve with more history,** for example hourly from 16.04 (112 days) to 14.18 (480 days),
   but none catches up with 15-minute steps. Hourly at 480 days is worse than 15-minute steps at 56 days (13.78).
3. **The quarter-hour shape costs more than the extra history wins.** A *perfect* hourly forecast, spread back to
   quarter hours, still scores 5.6 when interpolated and 9.0 when held flat. For half hours it is 3.3 and 4.9.
4. **A blend may help a little.** 0.75 × the 15-minute forecast + 0.25 × the 480-day hourly forecast scores 12.98
   (`tfm_multi_wx`: 15.66). The weight was picked on these same 90 days, so this needs confirming on the full year.

### Full-year backtest

365 forecast days, 2025-10-06 .. 2026-10-05, 15-minute steps, 112 days of context. Prices averaged
105.7 EUR/MWh (std 63.4), and 1,903 of the 35,040 quarter hours were negative.

| Config | What | MAE | RMSE | vs d-1 | days beating d-1 | 3h regret | 3h hit | neg recall | neg precision | 80% band coverage | pinball | MAE Oct–Mar | MAE Apr–Oct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `naive_d1` | same as yesterday | 30.20 | 48.75 | — | — | 5.83 | 48% | 51% | 49% | — | — | 27.00 | 33.22 |
| `naive_d7` | same as last week | 37.69 | 58.76 | −25% | 38% | 7.18 | 41% | 39% | 36% | — | — | 33.01 | 42.10 |
| `tfm_multi_wx` | + weather + calendar | 14.44 | 26.62 | +52% | 87% | 1.57 | 66% | 60% | 88% | 79% | 5.77 | 12.44 | 16.31 |
| `tfm_de_wx_tso` | DE only, all covariates | **12.41** | **23.49** | **+59%** | **91%** | **0.88** | 74% | 68% | 87% | 81% | **4.92** | 10.79 | 13.95 |

1. **The 90-day window was harder than average.** Over the year, both TimesFM configs score better than in the
   main table, and their lead over "same as yesterday" stays about the same (+52% and +59%).
2. **Winter is easiest, and the error follows price volatility.** December and February score under 10; June and
   September score 16–21.
3. **The cheapest-3-hour hit rate drops** to 74% and 66%. The misses are mostly on flat winter days, where several
   windows cost about the same. The regret stays small: the forecast's window costs only 0.9 EUR/MWh more than the
   true cheapest, against 5.8 for "same as yesterday".
4. Forecast days before about 2026-01-21 have context that partly predates 15-minute trading (hourly prices
   repeated 4×). That is what a live run would have seen at the time.

### Reproduce

```bash
# 15-minute steps at the 160-day limit, same 90 days
uv run price-forecast backtest --end 2026-10-05 --context-days 160 --configs tfm_multi_wx tfm_de_wx_tso
# Full year, written to outputs/backtest_ctx112d_365d* so the 90-day files stay
uv run price-forecast backtest --end 2026-10-05 --days 365 --configs tfm_multi_wx tfm_de_wx_tso --out backtest_ctx112d_365d
# Half-hourly and hourly steps need data from 2024-03-01 (the first complete Open-Meteo previous-day runs)
uv run price-forecast fetch --start 2024-03-01
uv run python experiments/context_study/coarse_resolution.py 1h 112 160 320 480 640
uv run python experiments/context_study/coarse_resolution.py 30min 112 160 320
# Rebuild docs/context_study.html from those outputs
uv run python experiments/context_study/build_page.py
```

On an M1 Max this took about 19 minutes for the full year and 23 minutes for both coarse sweeps (model time).

## Setup notes

- **Data:** SMARD.de for prices (9 zones) and TSO generation forecasts; Open-Meteo Previous Runs
  (`*_previous_day1`) at 25 points for irradiance, wind power-curve proxy and temperature;
  `holidays` for calendar flags. See the README.
- **Negative prices:** `TimesFM3Evaluator.predict_batch` defaults to `make_positive=True`, which
  clamps forecasts at 0. It is switched off here.
- **Inference settings:** symmetric averaging stays on (the evaluator default). Covariates are
  edge-padded because the 96-slot horizon is rounded up to the 128-step output patch.
