"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE in percent."""
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    return float(100.0 * np.mean(np.abs(actual - pred) / np.maximum(denom, 1e-8)))


def mase(actual: np.ndarray, pred: np.ndarray, context: np.ndarray, season: int) -> float:
    """Mean absolute scaled error against an in-sample seasonal-naive forecast."""
    if context.shape[-1] <= season:
        return float("nan")
    scale = np.mean(np.abs(context[..., season:] - context[..., :-season]))
    return float(mae(actual, pred) / max(scale, 1e-8))


def seasonal_naive(context: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Repeat the last full season forward — the baseline TimesFM has to beat."""
    if context.shape[-1] < season:
        season = context.shape[-1]
    last = context[..., -season:]
    reps = int(np.ceil(horizon / season))
    return np.concatenate([last] * reps, axis=-1)[..., :horizon]


def interval_coverage(
    actual: np.ndarray, quantiles: np.ndarray, low: int = 0, high: int = 8
) -> float:
    """Share of actuals inside the [q_low, q_high] band, in percent.

    ``quantiles`` has the quantile axis last: index 0 is q0.1 ... index 8 is q0.9.
    """
    lo = quantiles[..., low]
    hi = quantiles[..., high]
    inside = (actual >= lo) & (actual <= hi)
    return float(100.0 * np.mean(inside))


def pinball_loss(actual: np.ndarray, quantiles: np.ndarray, levels: np.ndarray) -> float:
    """Mean pinball (quantile) loss averaged over all quantile levels."""
    actual = actual[..., None]
    diff = actual - quantiles
    loss = np.maximum(levels * diff, (levels - 1.0) * diff)
    return float(np.mean(loss))
