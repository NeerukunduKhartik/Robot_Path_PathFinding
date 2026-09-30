"""Batch experiments for Q1 (heuristics) and Q2 (MAPF methods)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .astar import compare_heuristics
from .grid import make_multi_instance, make_single_instance
from .heuristics import HEURISTICS
from .mapf import run_all_mapf


def run_q1(n_trials: int = 30, height: int = 20, width: int = 20, obstacle_density: float = 0.22, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        world, start, goal = make_single_instance(height, width, obstacle_density, rng)
        for res in compare_heuristics(world, start, goal, HEURISTICS):
            row = res.as_dict()
            row["trial"] = trial
            row["grid"] = f"{height}x{width}"
            row["density"] = obstacle_density
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_q1(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("heuristic").agg(
        success_rate=("found", "mean"),
        mean_path_cost=("path_cost", "mean"),
        mean_nodes_expanded=("nodes_expanded", "mean"),
        mean_nodes_generated=("nodes_generated", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
    )


def run_q2(n_trials: int = 18, height: int = 16, width: int = 16, obstacle_density: float = 0.16, n_agents: int = 4, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        world, starts, goals = make_multi_instance(height, width, obstacle_density, n_agents, rng)
        for res in run_all_mapf(world, starts, goals):
            row = res.as_dict()
            row["trial"] = trial
            row["grid"] = f"{height}x{width}"
            row["density"] = obstacle_density
            row["n_agents"] = n_agents
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_q2(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("method").agg(
        collision_free_rate=("collision_free", "mean"),
        mean_conflicts=("conflicts", "mean"),
        mean_sum_of_costs=("sum_of_costs", "mean"),
        mean_makespan=("makespan", "mean"),
        mean_nodes_expanded=("nodes_expanded", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
        mean_high_level=("high_level_nodes", "mean"),
    )


def sweep_q2(
    seed: int = 21,
    trials_per_config: int = 8,
    height: int = 18,
    width: int = 18,
    obstacle_density: float = 0.20,
    max_agents: int = 7,
) -> pd.DataFrame:
    """Run the Q2 agent-scalability sweep at one fixed obstacle density.

    The selected obstacle density is held constant for every configuration so
    that the Q2 graphs measure the effect of the number of agents only.
    """
    frames = []
    rng_seed = int(seed)
    max_agents = max(2, int(max_agents))
    height = int(height)
    width = int(width)
    obstacle_density = float(np.clip(obstacle_density, 0.0, 1.0))

    for agents in range(2, max_agents + 1):
        frames.append(
            run_q2(
                n_trials=int(trials_per_config),
                height=height,
                width=width,
                obstacle_density=obstacle_density,
                n_agents=agents,
                seed=rng_seed,
            )
        )
        rng_seed += 1

    return pd.concat(frames, ignore_index=True)


def save_tables(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    q1 = run_q1()
    q1.to_csv(out_dir / "q1_raw.csv", index=False)
    summarize_q1(q1).to_csv(out_dir / "q1_summary.csv")
    q2 = sweep_q2()
    q2.to_csv(out_dir / "q2_raw.csv", index=False)
    summarize_q2(q2).to_csv(out_dir / "q2_summary.csv")
