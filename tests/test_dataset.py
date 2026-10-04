from datetime import date

import pandas as pd
import pytest

from price_forecast import backtest, dataset


@pytest.mark.parametrize(("day", "slots"), [
    (date(2026, 3, 29), 92),   # spring forward: 23 local hours
    (date(2026, 6, 1), 96),
    (date(2026, 10, 25), 100),  # fall back: 25 local hours
])
def test_grid_has_dst_aware_day_lengths(day, slots):
    idx = dataset.grid(day, day)
    assert len(idx) == slots
    assert str(idx.tz) == "UTC"
    assert idx.tz_convert(dataset.LOCAL_TZ)[0].hour == 0


def test_day_bounds_cover_the_grid_without_gaps():
    idx = dataset.grid(date(2026, 10, 24), date(2026, 10, 26))
    bounds = backtest.day_bounds(pd.DataFrame(index=idx))
    assert [e - s for s, e in bounds.values()] == [96, 100, 96]
    assert list(bounds.values())[-1][1] == len(idx)
