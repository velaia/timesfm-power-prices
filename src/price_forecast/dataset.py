"""Download every source and align it on one 15-minute UTC grid."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import calendar_features, smard, weather
from .calendar_features import LOCAL_TZ

FREQ = "15min"
# Short holes in forecast series (a missing quarter hour or two) get interpolated; prices never do.
MAX_FILL_STEPS = 4


def grid(start: date, end: date) -> pd.DatetimeIndex:
    """Quarter hours from local midnight of ``start`` to local midnight after ``end``, as UTC."""
    first = pd.Timestamp(start, tz=LOCAL_TZ)
    stop = pd.Timestamp(end + timedelta(days=1), tz=LOCAL_TZ)
    return pd.date_range(first, stop, freq=FREQ, inclusive="left").tz_convert("UTC")


def build(start: date, end: date, data_dir: Path) -> pd.DataFrame:
    """Return the aligned dataset for local dates ``start`` .. ``end`` (inclusive).

    Columns:
      price_<zone>        day-ahead price, EUR/MWh (targets)
      fc_<type>_gw        TSO day-ahead generation forecast for DE, average GW
      wx_*                Open-Meteo previous-day weather forecast features
      is_weekend, holiday_share
    """
    idx = grid(start, end)
    lo, hi = idx[0], idx[-1] + pd.Timedelta(FREQ)
    raw = data_dir / "raw"

    columns: dict[str, pd.Series] = {}
    for zone, fid in smard.PRICE_SERIES.items():
        print(f"  SMARD price {zone}")
        columns[f"price_{zone}"] = smard.fetch_series(fid, smard.PRICE_REGION, lo, hi, raw / "smard")
    for kind, fid in smard.FORECAST_SERIES.items():
        print(f"  SMARD forecast {kind}")
        mwh_per_quarter = smard.fetch_series(fid, smard.FORECAST_REGION, lo, hi, raw / "smard")
        columns[f"fc_{kind}_gw"] = mwh_per_quarter * 4 / 1000

    df = pd.DataFrame({name: s.reindex(idx) for name, s in columns.items()})
    fc_cols = [c for c in df if c.startswith("fc_")]
    df[fc_cols] = df[fc_cols].interpolate(limit=MAX_FILL_STEPS, limit_area="inside")

    print("  Open-Meteo weather (previous-day runs)")
    # One extra UTC day on each side so local-midnight edges have values to interpolate from.
    wx = weather.features(start - timedelta(days=1), end + timedelta(days=1), raw / "openmeteo")
    wx = wx.reindex(wx.index.union(idx)).interpolate(method="time", limit_area="inside").reindex(idx)
    df = df.join(wx)

    df = df.join(calendar_features.features(idx))
    df.index.name = "time_utc"
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column coverage: first/last timestamp with data and count of missing quarter hours."""
    local = df.index.tz_convert(LOCAL_TZ)
    rows = {}
    for col in df:
        valid = df[col].notna().to_numpy()
        rows[col] = {
            "first": local[valid.argmax()].strftime("%Y-%m-%d %H:%M") if valid.any() else "-",
            "last": local[len(valid) - 1 - valid[::-1].argmax()].strftime("%Y-%m-%d %H:%M") if valid.any() else "-",
            "missing": int((~valid).sum()),
            "mean": df[col].mean(),
            "min": df[col].min(),
            "max": df[col].max(),
        }
    return pd.DataFrame(rows).T
