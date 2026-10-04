"""Live forecast of one local day D, meant to run at about 10:00 on D-1.

Refreshes the data for exactly the window the model needs, checks which inputs
for D are already published, and falls back to weather-only covariates when the
SMARD TSO generation forecasts for D are not out yet. Every run appends what it
found to ``data/tso_availability.csv`` so we learn when SMARD publishes them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, dataset, model
from .backtest import TARGET, TSO, WEATHER, WINDOW
from .calendar_features import LOCAL_TZ

PRIMARY = "tfm_de_wx_tso"  # best in the backtest, needs TSO forecasts for D
FALLBACK = "tfm_multi_wx"  # weather + calendar only, always available by 10:00


@dataclass
class DayForecast:
    day: date
    config: str
    issued_at: datetime
    time_utc: pd.DatetimeIndex
    median: np.ndarray  # EUR/MWh
    quantiles: np.ndarray  # (slots, 9), EUR/MWh
    actual: np.ndarray | None  # EUR/MWh, only when D's auction result is already known
    previous_day: pd.Series  # actual prices of D-1, EUR/MWh
    tso_slots: dict[str, int]

    @property
    def local_time(self) -> pd.DatetimeIndex:
        return self.time_utc.tz_convert(LOCAL_TZ)

    def cheapest_window(self) -> tuple[int, int]:
        """Slot range [start, stop) of the cheapest forecast 3-hour window."""
        start = int(np.argmin(np.convolve(self.median, np.ones(WINDOW), mode="valid")))
        return start, start + WINDOW


def _log_tso(path: Path, issued_at: datetime, day: date, slots: int, coverage: dict[str, int]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["issued_at", "target_day", "slots", *coverage])
        writer.writerow([issued_at.isoformat(timespec="seconds"), day.isoformat(), slots, *coverage.values()])


def run(
    day: date,
    context_days: int,
    device: str,
    data_dir: Path,
    config_name: str | None = None,
    batch_size: int = 8,
) -> DayForecast:
    issued_at = datetime.now().astimezone()
    print(f"Refreshing data {day - timedelta(days=context_days + 1)} .. {day}")
    df = dataset.build(day - timedelta(days=context_days + 1), day, data_dir)

    bounds = backtest.day_bounds(df)
    start, stop = bounds[day]
    horizon = stop - start
    ctx_len = context_days * backtest.DAY
    if start < ctx_len:
        raise SystemExit(f"Not enough history for {context_days} context days.")

    prev_start, _ = bounds[day - timedelta(days=1)]
    known = df[TARGET].iloc[:start]
    if known.isna().any():
        last = known.last_valid_index().tz_convert(LOCAL_TZ)
        raise SystemExit(
            f"Prices are only published through {last:%Y-%m-%d %H:%M}; "
            f"{day} can be forecast once all of {day - timedelta(days=1)} is known."
        )

    coverage = {c: int(df[c].iloc[start:stop].notna().sum()) for c in TSO}
    if day > issued_at.date():  # only live runs tell us anything about publication times
        _log_tso(data_dir / "tso_availability.csv", issued_at, day, horizon, coverage)
    tso_ready = all(n == horizon for n in coverage.values())

    if config_name is None:
        config_name = PRIMARY if tso_ready else FALLBACK
        if not tso_ready:
            print(f"  SMARD TSO forecasts for {day} not complete yet ({coverage}); using {FALLBACK}")
    cfg = backtest.CONFIG_BY_NAME[config_name]
    if not tso_ready and any(c in TSO for c in cfg.covariates):
        raise SystemExit(f"{config_name} needs SMARD TSO forecasts for {day}, which are not published yet.")

    missing_wx = df[WEATHER].iloc[start:stop].isna().sum()
    if missing_wx.any():
        raise SystemExit(f"Weather forecast for {day} incomplete: {missing_wx[missing_wx > 0].to_dict()}")

    context = df[cfg.targets].to_numpy(dtype=np.float32).T[:, start - ctx_len:start]
    covariates = None
    if cfg.covariates:
        covariates = [df[cfg.covariates].to_numpy(dtype=np.float32).T[:, start - ctx_len:stop]]

    forecaster = model.load_forecaster(device, batch_size)
    outputs, secs = model.predict(
        forecaster, [context], horizon,
        past_future_covariates=covariates,
        use_symmetric_averaging=True,
        **backtest.PREDICT_KWARGS,
    )
    out = outputs[0]
    print(f"  {config_name}: {horizon} slots in {secs:.1f}s")

    actual = df[TARGET].iloc[start:stop]
    return DayForecast(
        day=day,
        config=config_name,
        issued_at=issued_at,
        time_utc=df.index[start:stop],
        median=out.forecast[0, :horizon],
        quantiles=out.quantiles[0, :horizon],
        actual=actual.to_numpy() if actual.notna().all() else None,
        previous_day=df[TARGET].iloc[prev_start:start],
        tso_slots=coverage,
    )


def to_ct(eur_per_mwh: np.ndarray) -> np.ndarray:
    return np.asarray(eur_per_mwh) / 10.0


def interval_key(ts: pd.Timestamp) -> str:
    """HHMM-style key as in the Vattenfall project (e.g. "1345", "0" for midnight)."""
    return str(ts.hour * 100 + ts.minute)


def by_interval(fc: DayForecast, values_eur: np.ndarray) -> dict[str, dict[str, float]]:
    """``{date: {HHMM: ct/kWh}}``; on the DST fall-back hour the second occurrence wins."""
    return {fc.day.isoformat(): {
        interval_key(t): round(float(v), 3) for t, v in zip(fc.local_time, to_ct(values_eur))
    }}


def to_json(fc: DayForecast) -> dict:
    w0, w1 = fc.cheapest_window()
    local = fc.local_time
    return {
        "date": fc.day.isoformat(),
        "issued_at": fc.issued_at.isoformat(timespec="seconds"),
        "model": model.CHECKPOINT,
        "config": fc.config,
        "unit": "ct/kWh",
        "prices": by_interval(fc, fc.median),
        "q10": by_interval(fc, fc.quantiles[:, 0]),
        "q90": by_interval(fc, fc.quantiles[:, 8]),
        "cheapest_3h": {
            "start": f"{local[w0]:%H:%M}",
            "end": f"{local[w1 - 1] + pd.Timedelta(minutes=15):%H:%M}",
            "mean_ct": round(float(to_ct(fc.median[w0:w1]).mean()), 3),
        },
        "negative_slots": [f"{t:%H:%M}" for t, v in zip(local, fc.median) if v < 0],
        "tso_forecast_slots": fc.tso_slots,
    }


def save_json(fc: DayForecast, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(fc), indent=2))


def print_summary(fc: DayForecast) -> None:
    local = fc.local_time
    med, q10, q90 = to_ct(fc.median), to_ct(fc.quantiles[:, 0]), to_ct(fc.quantiles[:, 8])
    w0, w1 = fc.cheapest_window()

    print(f"\nDE-LU day-ahead forecast for {fc.day:%a %Y-%m-%d}  (ct/kWh, net, config {fc.config})")
    print(f"issued {fc.issued_at:%Y-%m-%d %H:%M}\n")
    print("  hour   median   q10–q90 band")
    hours = pd.Series(med, index=local.hour)
    for hour in hours.index.unique():
        sel = local.hour == hour
        slots = np.flatnonzero(sel)
        mark = "  ← cheapest 3h" if ((slots >= w0) & (slots < w1)).any() else ""
        mark += "  (negative)" if med[sel].mean() < 0 else ""
        print(f"  {hour:02d}:00  {med[sel].mean():6.2f}   {q10[sel].mean():6.2f} – {q90[sel].mean():6.2f}{mark}")

    lo, hi = int(np.argmin(med)), int(np.argmax(med))
    print(f"\n  daily mean {med.mean():.2f}, min {med[lo]:.2f} at {local[lo]:%H:%M}, max {med[hi]:.2f} at {local[hi]:%H:%M}")
    end = local[w1 - 1] + pd.Timedelta(minutes=15)
    print(f"  cheapest 3h window {local[w0]:%H:%M}–{end:%H:%M}, mean {med[w0:w1].mean():.2f}")
    negative = int((med < 0).sum())
    if negative:
        print(f"  {negative} quarter hours forecast below zero")
    if fc.actual is not None:
        mae = float(np.mean(np.abs(fc.actual - fc.median)))
        print(f"\n  auction result already known: MAE {mae / 10:.2f} ct/kWh ({mae:.1f} EUR/MWh)")
