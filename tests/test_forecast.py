from datetime import date, datetime

import numpy as np
import pandas as pd

from price_forecast import dataset, forecast


def _fc(day: date, median: np.ndarray) -> forecast.DayForecast:
    idx = dataset.grid(day, day)
    return forecast.DayForecast(
        day=day, config="tfm_multi_wx", issued_at=datetime(2026, 1, 1, 10), time_utc=idx,
        median=median, quantiles=np.tile(median[:, None], 9), actual=None,
        previous_day=pd.Series(dtype=float), tso_slots={},
    )


def test_interval_key_matches_vattenfall_format():
    assert forecast.interval_key(pd.Timestamp("2026-10-06 00:00")) == "0"
    assert forecast.interval_key(pd.Timestamp("2026-10-06 00:45")) == "45"
    assert forecast.interval_key(pd.Timestamp("2026-10-06 13:45")) == "1345"


def test_by_interval_converts_to_ct_per_kwh():
    fc = _fc(date(2026, 6, 1), np.full(96, 123.4))
    out = forecast.by_interval(fc, fc.median)
    assert list(out) == ["2026-06-01"]
    assert len(out["2026-06-01"]) == 96
    assert out["2026-06-01"]["1200"] == 12.34


def test_by_interval_on_fall_back_day_keeps_second_occurrence():
    values = np.arange(100.0)
    fc = _fc(date(2026, 10, 25), values)
    prices = forecast.by_interval(fc, values)["2026-10-25"]
    assert len(prices) == 96  # 02:00-02:45 appears twice locally
    assert prices["200"] == 12 / 10  # slot 12 is the second 02:00 (first one is slot 8)


def test_cheapest_window_finds_the_valley():
    median = np.full(96, 100.0)
    median[40:52] = -5.0
    assert _fc(date(2026, 6, 1), median).cheapest_window() == (40, 52)
