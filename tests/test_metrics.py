import numpy as np
import pytest

from price_forecast import metrics


def test_mae_rmse():
    actual = np.array([0.0, 10.0, -10.0])
    pred = np.array([1.0, 7.0, -6.0])
    assert metrics.mae(actual, pred) == pytest.approx(8 / 3)
    assert metrics.rmse(actual, pred) == pytest.approx(np.sqrt(26 / 3))


def test_interval_coverage_uses_outer_quantiles():
    actual = np.array([0.0, 5.0, 20.0, -1.0])
    q = np.tile(np.linspace(0, 10, 9), (4, 1))  # q10 = 0, q90 = 10 for every slot
    assert metrics.interval_coverage(actual, q) == 50.0


def test_pinball_loss_is_zero_for_perfect_quantiles_and_asymmetric_otherwise():
    levels = np.array([0.1, 0.5, 0.9])
    assert metrics.pinball_loss(np.array([3.0]), np.full((1, 3), 3.0), levels) == 0.0
    # Under-forecasting by 1 costs level, over-forecasting costs (1 - level).
    under = metrics.pinball_loss(np.array([1.0]), np.zeros((1, 3)), levels)
    over = metrics.pinball_loss(np.array([0.0]), np.ones((1, 3)), levels)
    assert under == pytest.approx(0.5)
    assert over == pytest.approx(0.5)


def test_seasonal_naive_repeats_last_season():
    context = np.arange(10.0)
    np.testing.assert_array_equal(metrics.seasonal_naive(context, 5, 3), [7, 8, 9, 7, 8])
