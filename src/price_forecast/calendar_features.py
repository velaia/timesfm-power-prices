"""Calendar covariates on the local (Europe/Berlin) day."""

from __future__ import annotations

import holidays
import pandas as pd

LOCAL_TZ = "Europe/Berlin"
STATES = ["BB", "BE", "BW", "BY", "HB", "HE", "HH", "MV", "NI", "NW", "RP", "SH", "SL", "SN", "ST", "TH"]


def features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """``is_weekend`` and ``holiday_share`` (fraction of the 16 states off that day) per timestamp."""
    local_dates = index.tz_convert(LOCAL_TZ).normalize().tz_localize(None).date
    years = sorted({d.year for d in local_dates})
    by_state = [holidays.Germany(subdiv=s, years=years) for s in STATES]

    unique_dates = sorted(set(local_dates))
    share = {d: sum(d in h for h in by_state) / len(STATES) for d in unique_dates}

    return pd.DataFrame(
        {
            "is_weekend": [float(d.weekday() >= 5) for d in local_dates],
            "holiday_share": [share[d] for d in local_dates],
        },
        index=index,
    )
