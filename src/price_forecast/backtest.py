"""Rolling-origin backtest: forecast day D from what is known at 10:00 on D-1.

At issue time all of D-1's prices are known (they came out of the D-2 auction),
so the target context ends at D-1 23:45 local and the horizon is exactly the
quarter hours of local day D (92 / 96 / 100 on DST days).

Covariates are passed as past-and-future series covering context + day D.
The weather features are previous-day forecasts and so are available at issue
time; the SMARD TSO forecasts are the final day-ahead vintage, whose
publication time we can't verify — hence separate configs with and without them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import metrics, model
from .calendar_features import LOCAL_TZ

TARGET = "price_DE_LU"
PRICES = [TARGET, "price_FR", "price_NL", "price_BE", "price_AT", "price_CH", "price_PL", "price_CZ", "price_DK1"]
WEATHER = ["wx_irradiance", "wx_wind_onshore_cf", "wx_wind_offshore_cf", "wx_temperature"]
CALENDAR = ["is_weekend", "holiday_share"]
TSO = ["fc_solar_gw", "fc_wind_onshore_gw", "fc_wind_offshore_gw", "fc_total_gw"]

DAY = 96
WINDOW = 12  # 3 hours of quarter hours, for the cheapest-window check


@dataclass(frozen=True)
class Config:
    name: str
    targets: list[str]
    covariates: list[str] = field(default_factory=list)
    description: str = ""


CONFIGS = [
    Config("tfm_de", [TARGET], [], "DE price only"),
    Config("tfm_multi", PRICES, [], "9 coupled markets"),
    Config("tfm_multi_wx", PRICES, WEATHER + CALENDAR, "+ weather + calendar"),
    Config("tfm_multi_wx_tso", PRICES, WEATHER + CALENDAR + TSO, "+ SMARD TSO forecasts"),
    Config("tfm_de_wx_tso", [TARGET], WEATHER + CALENDAR + TSO, "DE only, all covariates"),
]
BASELINES = {"naive_d1": "same as yesterday", "naive_d7": "same as last week"}


def day_bounds(df: pd.DataFrame) -> dict[date, tuple[int, int]]:
    """Row range [start, stop) of each local day."""
    local_days = df.index.tz_convert(LOCAL_TZ).date
    bounds: dict[date, tuple[int, int]] = {}
    for i, d in enumerate(local_days):
        start, _ = bounds.get(d, (i, i))
        bounds[d] = (start, i + 1)
    return bounds


def naive(prices: np.ndarray, start: int, stop: int, days_back: int, bounds, day: date) -> np.ndarray:
    """Copy the price profile from ``days_back`` local days earlier, stretched to this day's length."""
    s, e = bounds[day - timedelta(days=days_back)]
    src = prices[s:e]
    pos = np.linspace(0, len(src) - 1, stop - start)
    return np.interp(pos, np.arange(len(src)), src)


def run(
    df: pd.DataFrame,
    days: list[date],
    context_days: int,
    device: str,
    configs: list[Config] = CONFIGS,
    batch_size: int = 8,
    symmetric: bool = True,
) -> pd.DataFrame:
    """Forecast every day in ``days`` with every config; return one row per (config, day, slot)."""
    bounds = day_bounds(df)
    ctx_len = context_days * DAY
    actual_all = df[TARGET].to_numpy()
    rows: list[pd.DataFrame] = []

    def emit(name: str, day: date, start: int, stop: int, fc: np.ndarray, q: np.ndarray | None, secs: float):
        frame = pd.DataFrame({
            "config": name,
            "day": pd.Timestamp(day),
            "slot": np.arange(stop - start),
            "time_utc": df.index[start:stop],
            "actual": actual_all[start:stop],
            "forecast": fc,
            "seconds": secs,
        })
        if q is not None:
            for k, level in enumerate(model.QUANTILE_LEVELS):
                frame[f"q{round(level * 100):02d}"] = q[:, k]
        rows.append(frame)

    for day in days:
        start, stop = bounds[day]
        emit("naive_d1", day, start, stop, naive(actual_all, start, stop, 1, bounds, day), None, 0.0)
        emit("naive_d7", day, start, stop, naive(actual_all, start, stop, 7, bounds, day), None, 0.0)

    forecaster = model.load_forecaster(device, batch_size)

    # One predict_batch call per (config, horizon length); DST days get their own call.
    by_horizon: dict[int, list[date]] = {}
    for day in days:
        start, stop = bounds[day]
        by_horizon.setdefault(stop - start, []).append(day)

    for cfg in configs:
        target_arr = df[cfg.targets].to_numpy(dtype=np.float32).T
        cov_arr = df[cfg.covariates].to_numpy(dtype=np.float32).T if cfg.covariates else None
        t_cfg = time.perf_counter()

        for horizon, group in by_horizon.items():
            contexts, covs = [], []
            for day in group:
                start, _ = bounds[day]
                contexts.append(target_arr[:, start - ctx_len:start])
                if cov_arr is not None:
                    covs.append(cov_arr[:, start - ctx_len:start + horizon])

            outputs, secs = model.predict(
                forecaster, contexts, horizon,
                past_future_covariates=covs if covs else None,
                make_positive=False,  # prices go negative
                padding_mode="edge",  # horizon is padded to the 64-step output patch
                use_symmetric_averaging=symmetric,
            )
            for day, out in zip(group, outputs):
                start, stop = bounds[day]
                emit(cfg.name, day, start, stop, out.forecast[0, :horizon], out.quantiles[0, :horizon], secs / len(group))

        print(f"  {cfg.name:18s} {len(days)} days in {time.perf_counter() - t_cfg:6.1f}s")

    return pd.concat(rows, ignore_index=True)


def _cheapest_window(prices: np.ndarray) -> int:
    sums = np.convolve(prices, np.ones(WINDOW), mode="valid")
    return int(np.argmin(sums))


def score(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy per config over all backtest days."""
    naive_mae = results[results.config == "naive_d1"].groupby("day").apply(
        lambda g: metrics.mae(g.actual.to_numpy(), g.forecast.to_numpy()), include_groups=False
    )
    qcols = [c for c in results if c.startswith("q")]
    table = {}
    for name, g in results.groupby("config", sort=False):
        actual, fc = g.actual.to_numpy(), g.forecast.to_numpy()
        daily_mae = g.groupby("day").apply(
            lambda d: metrics.mae(d.actual.to_numpy(), d.forecast.to_numpy()), include_groups=False
        )

        regrets, hits = [], []
        for _, d in g.groupby("day"):
            a, f = d.actual.to_numpy(), d.forecast.to_numpy()
            best, picked = _cheapest_window(a), _cheapest_window(f)
            regrets.append(a[picked:picked + WINDOW].mean() - a[best:best + WINDOW].mean())
            hits.append(abs(picked - best) <= 2)  # within 30 minutes of the true cheapest window

        neg_actual, neg_fc = actual < 0, fc < 0
        row = {
            "MAE": metrics.mae(actual, fc),
            "RMSE": metrics.rmse(actual, fc),
            "vs d-1 %": 100.0 * (1.0 - daily_mae.mean() / naive_mae.mean()),
            "days beat d-1 %": 100.0 * float((daily_mae < naive_mae.reindex(daily_mae.index)).mean()),
            "3h regret": float(np.mean(regrets)),
            "3h hit %": 100.0 * float(np.mean(hits)),
            "neg recall %": 100.0 * float((neg_fc & neg_actual).sum() / max(neg_actual.sum(), 1)),
            "neg prec %": 100.0 * float((neg_fc & neg_actual).sum() / max(neg_fc.sum(), 1)),
        }
        if g[qcols].notna().all().all():
            q = g[qcols].to_numpy()
            row["80% cov %"] = metrics.interval_coverage(actual, q)
            row["pinball"] = metrics.pinball_loss(actual, q, model.QUANTILE_LEVELS)
        else:
            row["80% cov %"] = row["pinball"] = float("nan")
        row["s/day"] = float(g.groupby("day").seconds.first().mean())
        table[name] = row
    return pd.DataFrame(table).T
