"""Zero-shot day-ahead electricity price forecasts for Germany with TimesFM 3.0."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import backtest, dataset

DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "dataset.parquet"
OUTPUT_DIR = Path("outputs")


def cmd_fetch(args: argparse.Namespace) -> None:
    print(f"Building dataset {args.start} .. {args.end}")
    df = dataset.build(args.start, args.end, DATA_DIR)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH)
    print(f"\nWrote {DATASET_PATH}: {len(df):,} quarter hours x {df.shape[1]} columns\n")

    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print(dataset.summary(df))

    local_days = df.index.tz_convert(dataset.LOCAL_TZ).normalize().value_counts()
    odd = local_days[local_days != 96].sort_index()
    if len(odd):
        print("\nDays with != 96 quarter hours (DST switches):")
        for day, n in odd.items():
            print(f"  {day.date()}  {n}")


def cmd_backtest(args: argparse.Namespace) -> None:
    df = pd.read_parquet(DATASET_PATH)
    bounds = backtest.day_bounds(df)
    have_prices = [d for d, (s, e) in bounds.items() if df[backtest.TARGET].iloc[s:e].notna().all()]
    end = args.end or max(have_prices)
    days = [end - timedelta(days=i) for i in range(args.days)][::-1]

    configs = backtest.CONFIGS
    if args.configs:
        configs = [c for c in configs if c.name in args.configs]

    device = backtest.model.resolve_device(args.device)
    print(f"Backtest {days[0]} .. {days[-1]} ({len(days)} days), context {args.context_days} days, {device}")
    results = backtest.run(df, days, args.context_days, device, configs, args.batch_size, not args.no_symmetric)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / f"backtest_ctx{args.context_days}d"
    results.to_parquet(stem.with_suffix(".parquet"))
    table = backtest.score(results)
    table.to_csv(stem.with_name(stem.name + "_summary.csv"))

    labels = {c.name: c.description for c in backtest.CONFIGS} | backtest.BASELINES
    table.insert(0, "what", [labels.get(n, "") for n in table.index])
    actual = results[results.config == "naive_d1"].actual
    print(f"\nDE-LU, EUR/MWh. {len(actual):,} quarter hours, {(actual < 0).sum()} negative, "
          f"mean {actual.mean():.1f}, std {actual.std():.1f}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(table)
    print(f"\nWrote {stem}.parquet and {stem.name}_summary.csv")

    if not args.no_plot:
        from . import plots

        best = min((c.name for c in configs), key=lambda n: table.loc[n, "MAE"])
        plots.example_days(results, best, labels[best], stem.with_name(stem.name + "_days.png"))
        lines = {"naive_d1": labels["naive_d1"]}
        if "tfm_multi" in table.index and best != "tfm_multi":
            lines["tfm_multi"] = "TimesFM, " + labels["tfm_multi"]
        lines[best] = "TimesFM, " + labels[best]
        plots.daily_error(results, lines, stem.with_name(stem.name + "_daily_mae.png"))
        print(f"Wrote {stem.name}_days.png and {stem.name}_daily_mae.png")


def main() -> None:
    parser = argparse.ArgumentParser(prog="price-forecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download SMARD + Open-Meteo data and build data/dataset.parquet")
    fetch.add_argument("--start", type=date.fromisoformat, default=date(2025, 6, 1),
                       help="first local date (default 2025-06-01; 15-min prices start 2025-10-01)")
    fetch.add_argument("--end", type=date.fromisoformat, default=date.today() + timedelta(days=1),
                       help="last local date, inclusive (default tomorrow)")
    fetch.set_defaults(func=cmd_fetch)

    bt = sub.add_parser("backtest", help="rolling day-ahead backtest of TimesFM configs vs naive baselines")
    bt.add_argument("--days", type=int, default=90, help="number of forecast days (default 90)")
    bt.add_argument("--end", type=date.fromisoformat, help="last forecast day (default: last day with prices)")
    bt.add_argument("--context-days", type=int, default=112, help="days of history per forecast (max 160)")
    bt.add_argument("--configs", nargs="+", choices=[c.name for c in backtest.CONFIGS], help="subset of configs")
    bt.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    bt.add_argument("--batch-size", type=int, default=8)
    bt.add_argument("--no-symmetric", action="store_true", help="disable symmetric averaging (halves compute)")
    bt.add_argument("--no-plot", action="store_true", help="skip writing PNGs to outputs/")
    bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)
