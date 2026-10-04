import numpy as np

from price_forecast import weather


def test_wind_capacity_factor_power_curve():
    cf = weather.wind_capacity_factor(np.array([0.0, 3.0, 7.5, 12.0, 20.0, 25.0, 26.0]))
    assert cf[0] == cf[1] == 0.0
    assert 0.0 < cf[2] < 1.0
    assert cf[3] == cf[4] == cf[5] == 1.0
    assert cf[6] == 0.0  # storm cut-out
