"""Backtest at 30-minute or hourly resolution, spread back to quarter hours, scored on 15-minute prices.

TimesFM 3 accepts at most 15,360 context steps: 160 days of quarter hours, 320 of half hours or
640 of hours. Coarser steps buy a longer history at the cost of the shape within each hour.

Long contexts need older data, so fetch from 2024-03-01 first (the earliest complete Open-Meteo
previous-day runs). Run from the repo root:

    uv run price-forecast fetch --start 2024-03-01
    uv run python experiments/context_study/coarse_resolution.py 1h 112 160 320 480 640
    uv run python experiments/context_study/coarse_resolution.py 30min 112 160 320

Writes ``outputs/coarse_<freq>_ctx<N>d_<step|linear>_<days>d.parquet`` per context length and
``outputs/coarse_<freq>_<days>d_summary.csv``.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from price_forecast import backtest, model

DATASET = Path("data/dataset.parquet")
OUTPUT_DIR = Path("outputs")
CONFIGS = ["tfm_multi_wx", "tfm_de_wx_tso"]
QUARTERS_PER_STEP = {"30min": 2, "1h": 4}


def baselines(df15: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """The naive baselines on the original 15-minute prices, in ``backtest.run``'s row format."""
    bounds, actual = backtest.day_bounds(df15), df15[backtest.TARGET].to_numpy()
    rows = []
    for name, back in (("naive_d1", 1), ("naive_d7", 7)):
        for d in days:
            s, e = bounds[d]
            rows.append(pd.DataFrame({
                "config": name, "day": pd.Timestamp(d), "slot": np.arange(e - s), "time_utc": df15.index[s:e],
                "actual": actual[s:e], "forecast": backtest.naive(actual, s, e, back, bounds, d), "seconds": 0.0,
            }))
    return pd.concat(rows, ignore_index=True)


def upsample(res: pd.DataFrame, df15: pd.DataFrame, step: int, how: str) -> pd.DataFrame:
    """Coarse per-day forecasts -> quarter hours, held flat or interpolated through the bucket centres."""
    bounds, actual = backtest.day_bounds(df15), df15[backtest.TARGET].to_numpy()
    value_cols = ["forecast"] + [c for c in res if c.startswith("q")]
    out = []
    for (cfg, day), g in res.groupby(["config", "day"], sort=False):
        s, e = bounds[day.date()]
        n = e - s
        assert n == len(g) * step, (day, n, len(g))
        frame = pd.DataFrame({"config": cfg, "day": day, "slot": np.arange(n), "time_utc": df15.index[s:e],
                              "actual": actual[s:e], "seconds": g.seconds.iloc[0]})
        centres = np.arange(len(g)) * step + (step - 1) / 2
        for c in value_cols:
            v = g[c].to_numpy()
            frame[c] = np.repeat(v, step) if how == "step" else np.interp(np.arange(n), centres, v)
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("freq", choices=QUARTERS_PER_STEP)
    parser.add_argument("context_days", type=int, nargs="+")
    parser.add_argument("--days", type=int, default=90, help="number of forecast days (default 90)")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 5), help="last forecast day")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    step = QUARTERS_PER_STEP[args.freq]
    days = [args.end - timedelta(days=i) for i in range(args.days)][::-1]
    configs = [backtest.CONFIG_BY_NAME[n] for n in CONFIGS]
    df15 = pd.read_parquet(DATASET)
    # UTC buckets: CET/CEST offsets are whole hours, so local days still start on a bucket edge.
    coarse = df15.resample(args.freq).mean()
    base = baselines(df15, days)
    backtest.DAY = 96 // step  # steps per day; run() derives the context length from it
    device = model.resolve_device(args.device)
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = []
    for ctx in args.context_days:
        print(f"\n{args.freq}, {ctx} days of context ({ctx * backtest.DAY} steps)", flush=True)
        res = backtest.run(coarse, days, ctx, device, configs)
        res = res[~res.config.isin(list(backtest.BASELINES))]
        for how in ("step", "linear"):
            up = upsample(res, df15, step, how)
            up.to_parquet(OUTPUT_DIR / f"coarse_{args.freq}_ctx{ctx}d_{how}_{args.days}d.parquet")
            table = backtest.score(pd.concat([base, up], ignore_index=True))
            for cfg in CONFIGS:
                row = table.loc[cfg]
                summary.append({"freq": args.freq, "ctx_days": ctx, "upsample": how, "config": cfg, "MAE": row["MAE"],
                                "RMSE": row["RMSE"], "3h hit %": row["3h hit %"], "neg recall %": row["neg recall %"],
                                "80% cov %": row["80% cov %"], "pinball": row["pinball"]})
        print(pd.DataFrame(summary).round(2).to_string(index=False), flush=True)

    path = OUTPUT_DIR / f"coarse_{args.freq}_{args.days}d_summary.csv"
    pd.DataFrame(summary).to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
