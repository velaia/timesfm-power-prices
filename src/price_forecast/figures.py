"""README / docs figures, rebuilt from a saved backtest parquet (no model, no network).

The hero graphic comes in a light and a dark variant so the README can serve the one that
matches the reader's GitHub theme via ``<picture>``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest, plots
from .calendar_features import LOCAL_TZ

HERO_DAY = pd.Timestamp("2026-09-05")
HERO_CONFIG = "tfm_de_wx_tso"
HONEST_CONFIG = "tfm_multi_wx"  # weather + calendar only: every input exists at 10:00 D-1
CONTEXT_DAYS = [7, 14, 28, 56, 112]

# Same reference palette as plots.py; dark steps from the dataviz skill's palette.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", blue="#2a78d6", blue_soft="#9ec5f4", orange="#eb6834", aqua="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", blue="#3987e5", blue_soft="#1c5cab", orange="#d95926", aqua="#199e70"),
}

LABELS = {c.name: c.description for c in backtest.CONFIGS} | backtest.BASELINES
TSO_CONFIGS = {c.name for c in backtest.CONFIGS if set(backtest.TSO) <= set(c.covariates)}


def _rc(t: dict) -> dict:
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.titlecolor": t["ink"], "axes.titleweight": "bold",
        "legend.frameon": False, "legend.labelcolor": t["ink2"], "lines.solid_capstyle": "round",
    }


def _day(results: pd.DataFrame, config: str, day: pd.Timestamp) -> pd.DataFrame:
    d = results[(results.config == config) & (results.day == day)].copy()
    local = d.time_utc.dt.tz_convert(LOCAL_TZ)
    d["hour"] = local.dt.hour + local.dt.minute / 60
    return d


def _mae(d: pd.DataFrame) -> float:
    return float((d.forecast - d.actual).abs().mean())


def _day_panel(ax, results: pd.DataFrame, t: dict, day: pd.Timestamp, config: str, compact: bool = False) -> None:
    """Actual vs TimesFM median, its q10–q90 band and the naive 'same as yesterday' line."""
    model, naive = _day(results, config, day), _day(results, "naive_d1", day)
    x = model.hour.to_numpy()

    ax.fill_between(x, model.q10, model.q90, color=t["blue"], alpha=0.16, linewidth=0, label="TimesFM q10–q90")
    ax.plot(x, naive.forecast, color=t["orange"], lw=1.8, ls=(0, (4, 2.5)), label="same as yesterday")
    ax.plot(x, model.forecast, color=t["blue"], lw=2.2, label="TimesFM median")
    ax.plot(x, model.actual, color=t["ink"], lw=2.2, label="actual price")
    ax.axhline(0, color=t["axis"], lw=1)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if compact:
        return

    # Call out the deepest negative price.
    i = int(model.actual.to_numpy().argmin())
    ax.scatter([x[i]], [model.actual.iloc[i]], s=36, color=t["ink"], edgecolor=t["surface"], lw=2, zorder=5)
    ax.annotate(f"low {model.actual.iloc[i]:.0f} EUR/MWh", (x[i], model.actual.iloc[i]), xytext=(0, -12),
                textcoords="offset points", ha="center", va="top", color=t["ink2"], fontsize=9.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi)

    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"{day:%a %d %b %Y}: a day with negative midday prices", loc="left", pad=48)
    ax.text(0, 1.105, f"Forecast issued 10:00 the day before · MAE TimesFM {_mae(model):.1f} vs "
            f"yesterday's prices {_mae(naive):.1f}", transform=ax.transAxes, color=t["ink2"], fontsize=10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9.5, handlelength=2.2,
              borderaxespad=0.2, columnspacing=1.4)


def _ablation_panel(ax, table: pd.DataFrame, t: dict, highlight: str) -> None:
    """MAE per config, best at the top; 'same as yesterday' as the reference line."""
    rows = table.drop(index="naive_d1").MAE.sort_values(ascending=False)
    ref = table.loc["naive_d1", "MAE"]
    y = np.arange(len(rows))
    colors = [t["blue"] if n == highlight else (t["muted"] if n.startswith("naive") else t["blue_soft"])
              for n in rows.index]
    ax.barh(y, rows.to_numpy(), height=0.56, color=colors, linewidth=0)
    for yi, (name, v) in zip(y, rows.items()):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", color=t["ink"], fontsize=10,
                fontweight="bold" if name == highlight else "normal")

    ax.axvline(ref, color=t["orange"], lw=1.8, ls=(0, (4, 2.5)))
    ax.text(ref - 0.6, len(rows) - 0.45, f"same as yesterday {ref:.1f}", ha="right", va="center",
            color=t["ink2"], fontsize=9.5)

    names = [LABELS[n] + (" †" if n in TSO_CONFIGS else "") for n in rows.index]
    ax.set_yticks(y, names)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlim(0, max(rows.max(), ref) * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("MAE, EUR/MWh (lower is better)")
    ax.set_title("90 days, Jul–Oct 2026", loc="left", pad=48)
    ax.text(0, 1.105, "Mean absolute error by configuration", transform=ax.transAxes, color=t["ink2"], fontsize=10)


def hero(results: pd.DataFrame, table: pd.DataFrame, path: Path, theme: str = "light") -> None:
    t = THEMES[theme]
    with plt.rc_context(_rc(t)):
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6), width_ratios=[1.55, 1],
                                          gridspec_kw={"wspace": 0.42})
        _day_panel(left, results, t, HERO_DAY, HERO_CONFIG)
        _ablation_panel(right, table, t, HERO_CONFIG)
        fig.text(0.125, -0.04, "TimesFM 3.0 zero-shot, 112 days of context, DE-LU day-ahead auction. "
                 "† uses SMARD TSO generation forecasts, whose availability by 10:00 D-1 is not yet verified.",
                 color=t["muted"], fontsize=9)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)


def context_length(summaries: dict[int, pd.DataFrame], path: Path) -> None:
    """MAE vs days of context for the best and the leak-free config, naive baseline as reference."""
    t = THEMES["light"]
    with plt.rc_context(_rc(t)):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        days = sorted(summaries)
        lines = [(HERO_CONFIG, t["blue"], "TimesFM, all covariates †"),
                 (HONEST_CONFIG, t["aqua"], "TimesFM, weather + calendar")]
        for name, color, label in lines:
            mae = [summaries[d].loc[name, "MAE"] for d in days]
            ax.plot(days, mae, color=color, lw=2.2, marker="o", ms=7, mec=t["surface"], mew=2, label=label)
            ax.annotate(f"{mae[-1]:.1f}", (days[-1], mae[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=t["ink"], fontsize=10)
            ax.annotate(f"{mae[0]:.1f}", (days[0], mae[0]), xytext=(-8, 0), textcoords="offset points",
                        va="center", ha="right", color=t["ink"], fontsize=10)
        ref = summaries[days[-1]].loc["naive_d1", "MAE"]
        ax.axhline(ref, color=t["orange"], lw=1.8, ls=(0, (4, 2.5)), label="same as yesterday")
        ax.set_xscale("log", base=2)
        ax.set_xticks(days, [str(d) for d in days])
        ax.minorticks_off()
        ax.set_xlim(days[0] / 1.35, days[-1] * 1.35)
        ax.set_ylim(0, ref * 1.12)
        ax.set_xlabel("Days of price history given to the model")
        ax.set_ylabel("MAE, EUR/MWh")
        ax.set_title("More context helps, with diminishing returns", loc="left", pad=22)
        ax.text(0, 1.02, "Same 90 forecast days for every context length", transform=ax.transAxes,
                color=t["ink2"], fontsize=10)
        ax.legend(loc="lower left", fontsize=9.5)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)


def social(results: pd.DataFrame, table: pd.DataFrame, path: Path) -> None:
    """1280×640 GitHub social preview: headline, one forecast day."""
    t = THEMES["light"]
    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
        gain = 100 * (1 - table.loc[HERO_CONFIG, "MAE"] / table.loc["naive_d1", "MAE"])
        fig.text(0.06, 0.86, "Forecasting German power prices with TimesFM 3.0", fontsize=26,
                 fontweight="bold", color=t["ink"])
        fig.text(0.06, 0.785, f"Zero-shot, no training · {gain:.0f}% lower error than “same as yesterday”"
                 " · forecast issued 10:00 the day before", fontsize=15, color=t["ink2"])
        ax = fig.add_axes([0.075, 0.08, 0.87, 0.56])
        _day_panel(ax, results, t, HERO_DAY, HERO_CONFIG, compact=True)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=4, fontsize=12.5)
        ax.tick_params(labelsize=11)
        fig.savefig(path, dpi=100)
        plt.close(fig)


def build(output_dir: Path, docs_dir: Path) -> list[Path]:
    """Render every docs figure from ``outputs/backtest_ctx*d`` files."""
    results = pd.read_parquet(output_dir / "backtest_ctx112d.parquet")
    table = backtest.score(results)
    summaries = {d: pd.read_csv(output_dir / f"backtest_ctx{d}d_summary.csv", index_col=0) for d in CONTEXT_DAYS
                 if (output_dir / f"backtest_ctx{d}d_summary.csv").exists()}

    docs_dir.mkdir(exist_ok=True)
    written = []
    for theme in THEMES:
        hero(results, table, docs_dir / f"hero_{theme}.png", theme)
        written.append(docs_dir / f"hero_{theme}.png")
    context_length(summaries, docs_dir / "context_length.png")
    social(results, table, docs_dir / "social_preview.png")
    plots.example_days(results, HERO_CONFIG, LABELS[HERO_CONFIG], docs_dir / "backtest_ctx112d_days.png")
    plots.daily_error(results, {"naive_d1": LABELS["naive_d1"], "tfm_multi": "TimesFM, " + LABELS["tfm_multi"],
                                HERO_CONFIG: "TimesFM, " + LABELS[HERO_CONFIG]},
                      docs_dir / "backtest_ctx112d_daily_mae.png")
    written += [docs_dir / n for n in ("context_length.png", "social_preview.png",
                                       "backtest_ctx112d_days.png", "backtest_ctx112d_daily_mae.png")]
    return written
