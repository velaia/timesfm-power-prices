"""Build docs/context_study.html from the saved backtests (no model, no network).

Expects in outputs/ (see docs/RESULTS.md for the commands that make them):
  backtest_ctx{7,14,28,56,112,160}d_summary.csv   15-min context sweep, 90 days
  backtest_ctx112d.parquet                         15-min, 90 days (example day)
  backtest_ctx112d_365d.parquet / _summary.csv     15-min, full year
  coarse_{30min,1h}_90d_summary.csv + parquets     from coarse_resolution.py

    uv run python experiments/context_study/build_page.py [--fragment PATH]

``--fragment`` also writes the page without the document wrapper, for publishing elsewhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUTPUTS = Path("outputs")
DATASET = Path("data/dataset.parquet")
TARGET_PAGE = Path("docs/context_study.html")
CONFIGS = ["tfm_multi_wx", "tfm_de_wx_tso"]
LOCAL_TZ = "Europe/Berlin"
CONTEXT_DAYS_15MIN = [7, 14, 28, 56, 112, 160]
# Strong quarter-hour swings (8th-largest of the 90 days) without a spike, and a 15-min error close
# to average. The day with the largest swings, 2026-09-14, is the 740 EUR/MWh spike that flattens the chart.
EXAMPLE_DAY = pd.Timestamp("2026-09-25")
EXAMPLE_CONFIG = "tfm_de_wx_tso"
WINDOW_90D = ("2026-07-07 22:00", "2026-10-05 21:45")  # UTC edges of the 90 local forecast days
WINDOW_YEAR = ("2025-10-05 22:00", "2026-10-05 21:45")


def r(x, n: int = 2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def mae(frame: pd.DataFrame) -> pd.Series:
    return (frame.actual - frame.forecast).abs()


def context_sweep() -> list[dict]:
    rows = []
    for ctx in CONTEXT_DAYS_15MIN:
        table = pd.read_csv(OUTPUTS / f"backtest_ctx{ctx}d_summary.csv", index_col=0)
        rows += [{"freq": "15min", "ctx": ctx, "upsample": "native", "config": c, **table.loc[c].to_dict()} for c in CONFIGS]
    for freq in ["30min", "1h"]:
        for row in pd.read_csv(OUTPUTS / f"coarse_{freq}_90d_summary.csv").to_dict("records"):
            rows.append({"freq": freq, "ctx": row.pop("ctx_days"), **row})
    return [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in rows]


def resolution_floor() -> list[dict]:
    """MAE of a perfect half-hourly / hourly forecast spread back to quarter hours."""
    prices = pd.read_parquet(DATASET)["price_DE_LU"]
    out = []
    for name, (lo, hi) in {"90d": WINDOW_90D, "year": WINDOW_YEAR}.items():
        a = prices[lo:hi]
        for freq in ["30min", "1h"]:
            flat = a.resample(freq).transform("mean")
            centres = a.resample(freq).mean()
            centres.index = centres.index + pd.Timedelta(freq) / 2 - pd.Timedelta("7.5min")
            linear = centres.reindex(centres.index.union(a.index)).interpolate("time").reindex(a.index).ffill().bfill()
            out.append({"window": name, "freq": freq, "step": r((a - flat).abs().mean()), "linear": r((a - linear).abs().mean())})
    return out


def full_year() -> dict:
    y = pd.read_parquet(OUTPUTS / "backtest_ctx112d_365d.parquet")
    y["abs"] = mae(y)
    y["month"] = y.day.dt.strftime("%Y-%m")
    naive = y[y.config == "naive_d1"]
    months = []
    for month, g in y.groupby("month"):
        b = naive[naive.month == month]
        row = {"month": month, "days": int(b.day.nunique()), "mean_price": r(b.actual.mean(), 1),
               "std_price": r(b.actual.std(), 1), "neg_share": r(100 * (b.actual < 0).mean(), 1)}
        row |= {c: r(gc["abs"].mean()) for c, gc in g.groupby("config")}
        months.append(row)
    summary = pd.read_csv(OUTPUTS / "backtest_ctx112d_365d_summary.csv", index_col=0)
    halves = {}
    for name, lo, hi in [("winter", "2025-10-06", "2026-03-31"), ("summer", "2026-04-01", "2026-10-05")]:
        g = y[(y.day >= lo) & (y.day <= hi)]
        halves[name] = {c: r(gc["abs"].mean()) for c, gc in g.groupby("config")}
    return {
        "year_months": months,
        "year_summary": {c: {k: r(v) for k, v in summary.loc[c].to_dict().items()} for c in summary.index},
        "year_halves": halves,
    }


def example_day() -> dict:
    q = pd.read_parquet(OUTPUTS / "backtest_ctx112d.parquet").query("config == @EXAMPLE_CONFIG")

    def hourly_mae(path: Path) -> float:
        return float(mae(pd.read_parquet(path).query("config == @EXAMPLE_CONFIG")).mean())

    best = min(OUTPUTS.glob("coarse_1h_ctx*d_linear_90d.parquet"), key=hourly_mae)
    h = pd.read_parquet(best).query("config == @EXAMPLE_CONFIG")
    qq, hh = q[q.day == EXAMPLE_DAY], h[h.day == EXAMPLE_DAY]
    return {
        "example_hourly_ctx": int(best.name.split("ctx")[1].split("d")[0]),
        "example": {
            "day": str(EXAMPLE_DAY.date()), "config": EXAMPLE_CONFIG,
            "time": qq.time_utc.dt.tz_convert(LOCAL_TZ).dt.strftime("%H:%M").tolist(),
            "actual": [r(v, 1) for v in qq.actual], "fc15": [r(v, 1) for v in qq.forecast],
            "fc1h": [r(v, 1) for v in hh.forecast],
            "mae15": r(mae(qq).mean()), "mae1h": r(mae(hh).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragment", type=Path, help="also write the page without <html>/<head>/<body> here")
    args = parser.parse_args()

    naive = pd.read_csv(OUTPUTS / "backtest_ctx112d_summary.csv", index_col=0)
    data = {
        "sweep": context_sweep(),
        "naive90": {"d1": r(naive.loc["naive_d1", "MAE"]), "d7": r(naive.loc["naive_d7", "MAE"])},
        "floor": resolution_floor(),
        **full_year(),
        **example_day(),
    }
    page = (HERE / "page.html").read_text()
    page = page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))).replace("__DATE__", date.today().isoformat())
    if args.fragment:
        args.fragment.write_text(page)
        print(f"Wrote {args.fragment}")

    # The page itself is a fragment (head content, then body content); wrap it into a standalone document.
    head, body = page.split('<div class="wrap">', 1)
    TARGET_PAGE.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head.strip()}\n<style>body {{ margin: 0; }} [hidden] {{ display: none !important; }}</style>\n"
        f'</head>\n<body>\n<div class="wrap">{body.rstrip()}\n</body>\n</html>\n'
    )
    print(f"Wrote {TARGET_PAGE} ({TARGET_PAGE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
