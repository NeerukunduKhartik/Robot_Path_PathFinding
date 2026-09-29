"""CLI: generate CSV summaries used in the report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.astar import compare_heuristics
from src.experiments import save_tables, summarize_q1, summarize_q2
from src.grid import make_multi_instance, make_single_instance
from src.heuristics import HEURISTICS
from src.mapf import run_all_mapf
from src.viz import bar_compare, draw_multi, draw_single, heuristics_one_graph
import pandas as pd


def save_demo_figures(out: Path) -> None:
    rng = np.random.default_rng(42)
    world, start, goal = make_single_instance(16, 16, 0.22, rng)
    results = compare_heuristics(world, start, goal, HEURISTICS)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, res in zip(axes, results):
        draw_single(
            world,
            start,
            goal,
            res.path,
            title=f"{res.heuristic_name}  cost={res.cost:.0f}  exp={res.nodes_expanded}",
            ax=ax,
        )
    fig.tight_layout()
    fig.savefig(out / "q1_live.png", dpi=140)
    plt.close(fig)

    rng = np.random.default_rng(99)
    world, starts, goals = make_multi_instance(14, 14, 0.14, 4, rng)
    mapf = run_all_mapf(world, starts, goals)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, res in zip(axes, mapf):
        label = "OK" if res.collision_free else f"{res.conflicts} conflicts"
        draw_multi(world, starts, goals, res.paths if res.found else None, title=f"{res.name} ({label})", ax=ax)
    fig.tight_layout()
    fig.savefig(out / "q2_live.png", dpi=140)
    plt.close(fig)

    q1 = pd.read_csv(out / "q1_summary.csv", index_col=0)
    q1_raw = pd.read_csv(out / "q1_raw.csv")
    bar_compare(q1, "mean_nodes_expanded", "Q1: mean nodes expanded", "Nodes").savefig(
        out / "q1_nodes.png", dpi=140
    )
    plt.close("all")
    heuristics_one_graph(q1_raw, q1).savefig(out / "q1_all_heuristics.png", dpi=140, bbox_inches="tight")
    plt.close("all")
    q2 = pd.read_csv(out / "q2_summary.csv", index_col=0)
    bar_compare(q2, "collision_free_rate", "Q2: collision-free rate", "Fraction").savefig(
        out / "q2_success.png", dpi=140
    )
    plt.close("all")


if __name__ == "__main__":
    out = Path("results")
    save_tables(out)
    save_demo_figures(out)
    print("Wrote CSVs and PNG figures under results/")
