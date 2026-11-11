"""Chart helpers: two chart types, each saved for light and dark backgrounds.

Rules followed: the title states the finding, one message and one axis per
chart, one colour when the marks measure the same thing, thin marks,
hairline gridlines, interval lines on every estimate, and numbers printed
at the marks so nobody has to read across to an axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "blue": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5"},
}
SOURCE = ("Data: NHANES August 2021 to August 2023, US National Center for Health Statistics.  "
          "Analysis: Elisha Abraham.")
DPI = 200

_installed = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (name for name in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans") if name in _installed), "sans-serif")


def _figure(theme: dict, title: str, subtitle: str, height: float):
    fig = plt.figure(figsize=(7.2, height), dpi=DPI)
    fig.patch.set_facecolor(theme["surface"])
    top = 1 - 0.19 / height
    fig.text(0.04, top, title, fontsize=12.5, fontweight="semibold", color=theme["ink"], ha="left", va="top")
    fig.text(0.04, top - 0.30 / height, subtitle, fontsize=9, color=theme["ink2"], ha="left", va="top")
    fig.text(0.04, 0.10 / height, SOURCE, fontsize=6.8, color=theme["muted"], ha="left", va="bottom")
    return fig


def _axes(fig, theme: dict, box: list[float]):
    ax = fig.add_axes(box)
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(theme["axis"])
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=9, colors=theme["muted"], pad=5)
    ax.tick_params(axis="y", length=0, labelsize=9, colors=theme["ink2"], pad=6)
    return ax


def _per_pixel(ax) -> tuple[float, float]:
    """Data units covered by one logical pixel (1/100 inch) on each axis."""
    ax.figure.canvas.draw()
    scale = ax.figure.dpi / 100
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0, 0), (scale, scale)])
    return abs(x1 - x0), abs(y1 - y0)


def _save(fig, folder: Path, name: str, theme_name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{theme_name}.png"
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def interval_bars(rows: list[dict], title: str, subtitle: str, folder: Path, name: str, x_max: float,
                  ticks: list[int], label_width: float = 0.33) -> Path:
    """Horizontal bars for percentages with 95% interval lines.

    rows: dicts with label, estimate, low, high (all in percent) and an optional
    note printed after the value. A row with estimate None draws a group heading.
    """
    height = 1.45 + 0.34 * len(rows)
    for theme_name, theme in THEMES.items():
        fig = _figure(theme, title, subtitle, height)
        ax = _axes(fig, theme, [label_width, 0.42 / height, 0.96 - label_width, 1 - 1.30 / height])
        ax.set_xlim(0, x_max)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xticks(ticks, [f"{t}%" if t else "0" for t in ticks])
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
        dx, dy = _per_pixel(ax)
        for y, (row, tick) in enumerate(zip(rows, ax.get_yticklabels())):
            if row.get("estimate") is None:
                tick.set_color(theme["ink"])
                tick.set_fontweight("semibold")
                continue
            thickness, radius = 14 * dy, 4
            ax.add_patch(FancyBboxPatch((0, y - thickness / 2), row["estimate"], thickness,
                                        boxstyle=f"round,pad=0,rounding_size={radius * dx}", mutation_aspect=dy / dx,
                                        facecolor=theme["blue"], edgecolor="none", zorder=3))
            ax.add_patch(Rectangle((0, y - thickness / 2), max(row["estimate"] - radius * dx, 0), thickness,
                                   facecolor=theme["blue"], edgecolor="none", zorder=3))
            ax.hlines(y, row["low"], row["high"], color=theme["ink2"], linewidth=1, zorder=4)
            text = f"{row['estimate']:.1f}%" + (f"  {row['note']}" if row.get("note") else "")
            ax.text(row["high"] + 6 * dx, y, text, fontsize=9, color=theme["ink"], ha="left", va="center")
        path = _save(fig, folder, name, theme_name)
        if theme_name == "light":
            light = path
    return light


def forest(rows: list[dict], title: str, subtitle: str, folder: Path, name: str, x_ticks: list[float]) -> Path:
    """Odds ratios with 95% intervals on a log axis. A row with odds_ratio None is a group heading;
    a row with reference=True is the comparison group, drawn at 1."""
    height = 1.77 + 0.30 * len(rows)
    for theme_name, theme in THEMES.items():
        fig = _figure(theme, title, subtitle, height)
        ax = _axes(fig, theme, [0.36, 0.74 / height, 0.44, 1 - 1.62 / height])  # room below for the axis title
        ax.spines["left"].set_visible(False)
        ax.set_xscale("log")
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xticks(x_ticks, [f"{t:g}" for t in x_ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
        ax.axvline(1, color=theme["ink2"], linewidth=0.8, zorder=2)
        for y, (row, tick) in enumerate(zip(rows, ax.get_yticklabels())):
            if row.get("odds_ratio") is None and not row.get("reference"):
                tick.set_color(theme["ink"])
                tick.set_fontweight("semibold")
                continue
            if row.get("reference"):
                ax.plot(1, y, "o", markersize=6, markerfacecolor=theme["surface"], markeredgecolor=theme["ink2"],
                        markeredgewidth=1.2, zorder=4)
                label = "1.0  (reference)"
            else:
                ax.hlines(y, max(row["low"], x_ticks[0]), min(row["high"], x_ticks[-1]), color=theme["ink2"], linewidth=1, zorder=3)
                ax.plot(row["odds_ratio"], y, "o", markersize=7.5, color=theme["blue"], markeredgecolor=theme["surface"],
                        markeredgewidth=1.5, zorder=4)
                label = f"{row['odds_ratio']:.1f}  ({row['low']:.1f} to {row['high']:.1f})"
            ax.text(1.04, y, label, transform=ax.get_yaxis_transform(), fontsize=9, color=theme["ink"], ha="left", va="center")
        ax.set_xlabel("Adjusted odds ratio (log scale)", fontsize=9, color=theme["ink2"], labelpad=6)
        path = _save(fig, folder, name, theme_name)
        if theme_name == "light":
            light = path
    return light
