"""Matplotlib drawings for grids and paths."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .grid import Coord, GridWorld

AGENT_COLORS = [
    "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
]


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
    for i, start in enumerate(starts):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if paths and i < len(paths) and paths[i]:
            ys = [p[0] for p in paths[i]]
            xs = [p[1] for p in paths[i]]
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.85, zorder=3)
            if t is not None:
                idx = min(t, len(paths[i]) - 1)
                ax.scatter([paths[i][idx][1]], [paths[i][idx][0]], s=70, c=color, zorder=5, edgecolors="white")
        ax.scatter([start[1]], [start[0]], s=70, marker="o", c=color, zorder=4, edgecolors="white")
        ax.scatter([goals[i][1]], [goals[i][0]], s=90, marker="*", c=color, zorder=4, edgecolors="white")
    ax.set_title(title, fontsize=11, fontweight="medium")
    _style_axes(ax)
    if fig_created:
        fig.tight_layout()
    return fig
