"""Client for the SMARD.de chart-data API (Bundesnetzagentur) — no key needed.

SMARD serves each series in weekly chunks. An index file lists every chunk's
start (UNIX ms, Monday 00:00 Europe/Berlin); each chunk holds
``[[ts_ms, value | null], ...]`` at the requested resolution.

Before the 15-minute day-ahead market went live (2025-10-01) the
quarter-hour price series simply repeats each hourly price four times.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.smard.de/app/chart_data"
RESOLUTION = "quarterhour"

# Day-ahead prices in EUR/MWh. SMARD files neighbour prices under region DE-LU.
PRICE_SERIES = {
    "DE_LU": 4169, "FR": 254, "NL": 256, "BE": 4996, "AT": 4170,
    "CH": 259, "PL": 257, "CZ": 261, "DK1": 252,
}
PRICE_REGION = "DE-LU"

# TSO day-ahead generation forecasts for Germany, MWh per quarter hour.
FORECAST_SERIES = {"solar": 125, "wind_onshore": 123, "wind_offshore": 3791, "total": 122}
FORECAST_REGION = "DE"

# A chunk is treated as final (and cached for good) once it ended this long ago.
_FINAL_AFTER = timedelta(days=2)
_RETRY_BACKOFFS = (0, 2, 5, 10)

_session = requests.Session()


def _get_json(url: str) -> dict:
    for attempt, backoff in enumerate(_RETRY_BACKOFFS):
        if backoff:
            time.sleep(backoff)
        try:
            resp = _session.get(url, timeout=30)
        except requests.RequestException:
            if attempt == len(_RETRY_BACKOFFS) - 1:
                raise
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
    resp.raise_for_status()
    raise RuntimeError(f"SMARD request kept failing: {url}")


def _chunk(filter_id: int, region: str, start_ms: int, cache_dir: Path) -> list[list]:
    path = cache_dir / f"{filter_id}_{region}_{RESOLUTION}_{start_ms}.json"
    if path.exists():
        return json.loads(path.read_text())

    url = f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{RESOLUTION}_{start_ms}.json"
    series = _get_json(url)["series"]

    chunk_end = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc) + timedelta(days=7)
    if chunk_end < datetime.now(timezone.utc) - _FINAL_AFTER:
        path.write_text(json.dumps(series))
    return series


def fetch_series(
    filter_id: int, region: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path
) -> pd.Series:
    """Return one SMARD series as floats on a UTC index, limited to [start, end)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = _get_json(f"{BASE_URL}/{filter_id}/{region}/index_{RESOLUTION}.json")["timestamps"]

    start_ms = start.value // 1_000_000
    end_ms = end.value // 1_000_000
    week_ms = 7 * 24 * 3600 * 1000
    wanted = [ts for ts in index if ts < end_ms and ts + week_ms > start_ms]

    rows = []
    for ts in wanted:
        rows.extend(_chunk(filter_id, region, ts, cache_dir))

    series = pd.Series(
        [v for _, v in rows],
        index=pd.to_datetime([t for t, _ in rows], unit="ms", utc=True),
        dtype="float64",
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[(series.index >= start) & (series.index < end)]
