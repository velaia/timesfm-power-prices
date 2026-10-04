"""Backtest charts (matplotlib PNGs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .calendar_features import LOCAL_TZ

# Reference palette (dataviz skill), light mode: categorical slots 1-3 + neutral inks.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
})


def _daily_mae(results: pd.DataFrame, config: str) -> pd.Series:
    g = results[results.config == config]
    return (g.forecast - g.actual).abs().groupby(g.day).mean()


def pick_days(results: pd.DataFrame, config: str) -> dict[str, pd.Timestamp]:
    """Four illustrative days: most negative prices, most volatile, a typical day, the latest day."""
    base = results[results.config == config]
    by_day = base.groupby("day")
    mae = _daily_mae(results, config)
    picks = {
        "most negative prices": (by_day.actual.apply(lambda a: (a < 0).sum())).idxmax(),
        "most volatile": by_day.actual.std().idxmax(),
        "typical (median error)": (mae - mae.median()).abs().idxmin(),
        "latest": base.day.max(),
    }
    return picks


def example_days(results: pd.DataFrame, config: str, label: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    for ax, (why, day) in zip(axes.flat, pick_days(results, config).items()):
        model_day = results[(results.config == config) & (results.day == day)]
        naive_day = results[(results.config == "naive_d1") & (results.day == day)]
        t = model_day.time_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

        ax.fill_between(t, model_day.q10, model_day.q90, color=SERIES[0], alpha=0.18, linewidth=0,
                        label="TimesFM 10–90% band")
        ax.plot(t, naive_day.forecast, color=SERIES[1], lw=1.5, ls=(0, (4, 2)), label="same as yesterday")
        ax.plot(t, model_day.forecast, color=SERIES[0], lw=2, label=f"TimesFM ({label})")
        ax.plot(t, model_day.actual, color=INK, lw=2, label="actual")
        ax.axhline(0, color=INK_2, lw=0.8)

        mae_m = (model_day.forecast - model_day.actual).abs().mean()
        mae_n = (naive_day.forecast - naive_day.actual).abs().mean()
        ax.set_title(f"{day:%a %Y-%m-%d} · {why}\nMAE TimesFM {mae_m:.1f} vs yesterday {mae_n:.1f}", loc="left")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.set_ylabel("EUR/MWh")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("DE-LU day-ahead price, forecast issued 10:00 the day before", x=0.01, ha="left",
                 y=1.04, fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def daily_error(results: pd.DataFrame, configs: dict[str, str], path: Path) -> None:
    """7-day rolling mean of daily MAE per config (at most three series).

    Colours match ``example_days``: the last (best) config is blue, the naive baseline is
    dashed orange, anything in between is aqua.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    names = list(configs)
    for config, label in configs.items():
        if config == "naive_d1":
            color, style = SERIES[1], (0, (4, 2))
        else:
            color, style = (SERIES[0] if config == names[-1] else SERIES[2]), "-"
        series = _daily_mae(results, config).rolling(7, min_periods=1).mean()
        ax.plot(series.index, series.values, color=color, lw=2, ls=style, label=label)
        ax.annotate(label, (series.index[-1], series.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_ylabel("MAE, EUR/MWh (7-day mean)")
    ax.set_ylim(bottom=0)
    ax.set_title("Daily forecast error, DE-LU", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
