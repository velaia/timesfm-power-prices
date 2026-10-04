"""Weather-forecast covariates from the Open-Meteo Previous Runs API — no key needed.

``*_previous_day1`` variables hold what the model run from the *previous day*
predicted for each hour (lead time 24–47 h). For a forecast issued at 10:00 on
D-1 that is roughly the 00 UTC run of D-1 — information that really exists at
issue time, so backtests don't peek at better-than-available weather.

Point values are aggregated into four German-wide features:
irradiance over solar regions, a wind power-curve proxy for onshore and
offshore clusters, and temperature over the big population centres.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = {
    "shortwave_radiation_previous_day1": "irradiance",
    "wind_speed_100m_previous_day1": "wind_speed",
    "temperature_2m_previous_day1": "temperature",
}

# (lat, lon) per role. Chosen by where German capacity / demand sits, not exhaustive.
POINTS = {
    "solar": [
        (48.14, 11.58),  # Munich
        (48.57, 12.60),  # Lower Bavaria
        (49.60, 10.90),  # Franconia
        (48.78, 9.18),   # Stuttgart
        (48.00, 7.85),   # Freiburg
        (51.76, 14.33),  # Cottbus / Lusatia
        (51.34, 12.37),  # Leipzig
        (51.96, 7.63),   # Münster
    ],
    "wind_onshore": [
        (54.50, 9.00),   # Nordfriesland
        (53.40, 7.40),   # Ostfriesland
        (52.70, 7.40),   # Emsland
        (53.90, 12.30),  # Mecklenburg
        (53.20, 13.90),  # Uckermark
        (53.10, 11.90),  # Prignitz
        (52.00, 11.40),  # Magdeburger Börde
        (51.70, 8.80),   # Paderborn plateau
        (49.90, 7.30),   # Hunsrück
    ],
    "wind_offshore": [
        (54.35, 6.20),   # North Sea, BorWin cluster
        (54.05, 6.95),   # North Sea, south of Helgoland cluster
        (54.60, 12.65),  # Baltic 1 / 2
        (54.80, 14.00),  # Arkona / Wikinger
    ],
    "load": [
        (52.52, 13.40),  # Berlin
        (53.55, 10.00),  # Hamburg
        (48.14, 11.58),  # Munich
        (51.45, 7.00),   # Ruhr
        (50.11, 8.68),   # Frankfurt
    ],
}

_RETRY_BACKOFFS = (0, 5, 15, 60)


def wind_capacity_factor(speed_ms: np.ndarray) -> np.ndarray:
    """Generic turbine power curve: cubic between cut-in 3 m/s and rated 12 m/s, off above 25."""
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    cf = (speed_ms**3 - cut_in**3) / (rated**3 - cut_in**3)
    cf = np.clip(cf, 0.0, 1.0)
    return np.where(speed_ms > cut_out, 0.0, cf)


def _all_points() -> list[tuple[float, float]]:
    seen: dict[tuple[float, float], None] = {}
    for pts in POINTS.values():
        for p in pts:
            seen.setdefault(p, None)
    return list(seen)


def _request(points: list[tuple[float, float]], start: date, end: date) -> list[dict]:
    params = {
        "latitude": ",".join(str(lat) for lat, _ in points),
        "longitude": ",".join(str(lon) for _, lon in points),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    for attempt, backoff in enumerate(_RETRY_BACKOFFS):
        if backoff:
            time.sleep(backoff)
        resp = requests.get(API_URL, params=params, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else [data]
        if resp.status_code not in (429, 500, 502, 503, 504) or attempt == len(_RETRY_BACKOFFS) - 1:
            raise RuntimeError(f"Open-Meteo HTTP {resp.status_code}: {resp.text[:300]}")
    raise AssertionError("unreachable")


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur + timedelta(days=32)).replace(day=1)
        chunks.append((max(cur, start), min(nxt - timedelta(days=1), end)))
        cur = nxt
    return chunks


def fetch_points(start: date, end: date, cache_dir: Path) -> dict[tuple[float, float], pd.DataFrame]:
    """Hourly forecast values per point, keyed by (lat, lon), on a UTC index."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    points = _all_points()
    final_before = date.today() - timedelta(days=2)
    frames: dict[tuple[float, float], list[pd.DataFrame]] = {p: [] for p in points}

    for c_start, c_end in _month_chunks(start, end):
        path = cache_dir / f"openmeteo_{c_start.isoformat()}_{c_end.isoformat()}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            payload = _request(points, c_start, c_end)
            if c_end < final_before:
                path.write_text(json.dumps(payload))
            time.sleep(1)  # stay well under the free-tier per-minute limit

        for point, loc in zip(points, payload):
            hourly = loc["hourly"]
            idx = pd.to_datetime(hourly["time"], utc=True)
            frames[point].append(
                pd.DataFrame({name: hourly[var] for var, name in VARIABLES.items()}, index=idx, dtype="float64")
            )

    return {p: pd.concat(f).sort_index() for p, f in frames.items()}


def features(start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    """German-wide hourly weather features (UTC index)."""
    per_point = fetch_points(start, end, cache_dir)

    def mean_over(role: str, column: str) -> pd.Series:
        return pd.concat([per_point[p][column] for p in POINTS[role]], axis=1).mean(axis=1)

    def mean_cf(role: str) -> pd.Series:
        cfs = [
            pd.Series(wind_capacity_factor(per_point[p]["wind_speed"].to_numpy()), index=per_point[p].index)
            for p in POINTS[role]
        ]
        return pd.concat(cfs, axis=1).mean(axis=1)

    # Open-Meteo radiation is the mean over the *preceding* hour; centre it on the half hour
    # so interpolation to quarter hours doesn't lag the sun by 30 minutes.
    irradiance = mean_over("solar", "irradiance")
    irradiance.index = irradiance.index - pd.Timedelta(minutes=30)

    return pd.concat(
        {
            "wx_irradiance": irradiance,
            "wx_wind_onshore_cf": mean_cf("wind_onshore"),
            "wx_wind_offshore_cf": mean_cf("wind_offshore"),
            "wx_temperature": mean_over("load", "temperature"),
        },
        axis=1,
    )
