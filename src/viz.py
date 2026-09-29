"""Matplotlib drawings for grids, paths, and comparison charts."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from .grid import Coord, GridWorld

HEURISTIC_COLORS = {
    "Manhattan": "#2563eb",
    "Euclidean": "#16a34a",
    "Chebyshev": "#d97706",
}
HEURISTIC_STYLES = {
    "Manhattan": "-",
    "Euclidean": "--",
    "Chebyshev": ":",
}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def draw_single(
    world: GridWorld,
    start: Coord,
    goal: Coord,
    path: Optional[list[Coord]] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    fig_created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
    else:
        fig = ax.figure

    grid = world.cells.astype(float)
    cmap = ListedColormap(["#f8fafc", "#0f172a"])
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, world.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.height, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if path and len(path) > 1:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="#2563eb", linewidth=2.4, marker="o", markersize=4, zorder=3)

    ax.scatter([start[1]], [start[0]], s=90, c="#16a34a", marker="o", zorder=4, label="Start", edgecolors="white")
    ax.scatter([goal[1]], [goal[0]], s=110, c="#dc2626", marker="*", zorder=4, label="Goal", edgecolors="white")
    ax.set_title(title, fontsize=11, fontweight="medium")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    _style_axes(ax)
    if fig_created:
        fig.tight_layout()
    return fig


def draw_multi(
    world: GridWorld,
    starts: list[Coord],
    goals: list[Coord],
    paths: Optional[list[list[Coord]]] = None,
    title: str = "",
    t: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    fig_created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
    else:
        fig = ax.figure

    grid = world.cells.astype(float)
    cmap = ListedColormap(["#f8fafc", "#0f172a"])
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, world.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.height, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    n = len(starts)
    for i in range(n):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if paths and i < len(paths) and paths[i]:
            ys = [p[0] for p in paths[i]]
            xs = [p[1] for p in paths[i]]
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.85, zorder=3)
            if t is not None:
                idx = min(t, len(paths[i]) - 1)
                ax.scatter(
                    [paths[i][idx][1]],
                    [paths[i][idx][0]],
                    s=70,
                    c=color,
                    zorder=5,
                    edgecolors="white",
                )
        ax.scatter([starts[i][1]], [starts[i][0]], s=70, marker="o", c=color, zorder=4, edgecolors="white")
        ax.scatter([goals[i][1]], [goals[i][0]], s=90, marker="*", c=color, zorder=4, edgecolors="white")

    ax.set_title(title, fontsize=11, fontweight="medium")
    _style_axes(ax)
    if fig_created:
        fig.tight_layout()
    return fig


HEURISTIC_COLORS = {
    "Manhattan": "#2563eb",
    "Euclidean": "#16a34a",
    "Chebyshev": "#d97706",
}


def bar_compare(df, y: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = list(df.index)
    values = list(df[y])
    colors = [HEURISTIC_COLORS.get(str(l), "#7c3aed") for l in labels]
    ax.bar(labels, values, color=colors, width=0.62, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def heuristics_one_graph(raw_df, summary):
    """All three heuristics on one figure: trial overlay + grouped means."""
    order = [h for h in ("Manhattan", "Euclidean", "Chebyshev") if h in set(raw_df["heuristic"])]
    if not order:
        order = list(raw_df["heuristic"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    for name in order:
        part = raw_df[raw_df["heuristic"] == name].sort_values("trial")
        ax.plot(
            part["trial"],
            part["nodes_expanded"],
            color=HEURISTIC_COLORS.get(name, "#64748b"),
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=name,
        )
    ax.set_title("Nodes expanded per trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Nodes expanded")
    ax.legend(framealpha=0.92)
    ax.grid(linestyle="--", alpha=0.4)
    _style_axes(ax)

    ax = axes[1]
    metrics = [
        ("mean_path_cost", "Path cost"),
        ("mean_nodes_expanded", "Nodes expanded"),
        ("mean_runtime_ms", "Runtime"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    for i, name in enumerate(order):
        vals = []
        for col, _ in metrics:
            series = summary[col]
            peak = float(series.max()) if float(series.max()) else 1.0
            raw = float(summary.loc[name, col]) if name in summary.index else 0.0
            vals.append(100.0 * raw / peak)
        ax.bar(
            x + (i - (len(order) - 1) / 2) * width,
            vals,
            width=width,
            color=HEURISTIC_COLORS.get(name, "#64748b"),
            label=name,
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 115)
    ax.set_title("Means relative to the largest value")
    ax.set_ylabel("% of max across heuristics")
    ax.legend(framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _style_axes(ax)

    fig.suptitle("Heuristic comparison — Manhattan vs Euclidean vs Chebyshev", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
