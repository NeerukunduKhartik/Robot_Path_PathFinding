from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.astar import compare_heuristics
from src.experiments import run_q1, summarize_q1, summarize_q2, sweep_q2
from src.grid import make_multi_instance, make_single_instance
from src.heuristics import HEURISTIC_NOTES, HEURISTICS
from src.mapf import run_all_mapf
from src.viz import draw_multi, draw_single

st.set_page_config(
    page_title="AI project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1280px;}
.metric-card {padding: 0.8rem 1rem; border: 1px solid #dbe3ec; border-radius: 10px; background: #f8fafc;}
</style>
""", unsafe_allow_html=True)

st.title("Robot path finding with A* and CBS")
st.caption("CSMI17 assignment — single-agent heuristics, then multi-robot planning.")


def _close(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _style(ax):
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fbfcfe")


def _q1_study_plots(df: pd.DataFrame, summary: pd.DataFrame):
    """New Q1 study plots: trends/distributions instead of the old bar charts."""
    order = [h for h in ("Manhattan", "Euclidean", "Chebyshev") if h in df["heuristic"].unique()]
    if not order:
        order = list(df["heuristic"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Keep the three heuristics visually distinct even when their values overlap.
    styles = {
        "Manhattan": {"color": "#2563eb", "marker": "o", "linestyle": "-"},
        "Euclidean": {"color": "#f97316", "marker": "s", "linestyle": "--"},
        "Chebyshev": {"color": "#16a34a", "marker": "^", "linestyle": ":"},
    }

    # 1. Search effort trend
    ax = axes[0, 0]
    for h in order:
        part = df[df["heuristic"] == h].sort_values("trial")
        stl = styles.get(h, {"color": "#7c3aed", "marker": "D", "linestyle": "-."})
        ax.plot(
            part["trial"], part["nodes_expanded"],
            color=stl["color"], marker=stl["marker"],
            linestyle=stl["linestyle"], markersize=3.5,
            linewidth=1.8, label=h, alpha=0.95,
        )
    ax.set_title("Search effort across trials")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Nodes expanded")
    ax.legend(frameon=False)
    _style(ax)

    # 2. Runtime trend
    ax = axes[0, 1]
    for h in order:
        part = df[df["heuristic"] == h].sort_values("trial")
        stl = styles.get(h, {"color": "#7c3aed", "marker": "D", "linestyle": "-."})
        ax.plot(
            part["trial"], part["runtime_ms"],
            color=stl["color"], marker=stl["marker"],
            linestyle=stl["linestyle"], markersize=3.5,
            linewidth=1.8, label=h, alpha=0.95,
        )
    ax.set_title("Runtime across trials")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Runtime (ms)")
    ax.legend(frameon=False)
    _style(ax)

    # 3. Path-cost consistency
    ax = axes[1, 0]
    for h in order:
        part = df[df["heuristic"] == h].sort_values("trial")
        stl = styles.get(h, {"color": "#7c3aed", "marker": "D", "linestyle": "-."})
        ax.plot(
            part["trial"], part["path_cost"],
            color=stl["color"], marker=stl["marker"],
            linestyle=stl["linestyle"], markersize=4,
            linewidth=1.8, label=h, alpha=0.95,
        )
    ax.set_title("Path cost consistency")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Path cost")
    ax.legend(frameon=False)
    _style(ax)

    # 4. Success rate as a dot plot (no bar chart)
    ax = axes[1, 1]
    rates = summary.reindex(order)["success_rate"].fillna(0) * 100
    y = np.arange(len(order))
    ax.scatter(rates.values, y, s=90, zorder=3)
    for x, yy in zip(rates.values, y):
        ax.text(x + 1, yy, f"{x:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Successful trials (%)")
    ax.set_title("Heuristic reliability")
    _style(ax)

    fig.suptitle("Q1 — Heuristic behaviour across repeated maps", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _q2_agent_plots(df: pd.DataFrame):
    """New Q2 scalability plots using actual experiment rows."""
    data = df.copy()
    methods = [m for m in ("Independent A*", "Prioritized STA*", "CBS") if m in data["method"].unique()]
    grouped = data.groupby(["n_agents", "method"], dropna=False).agg(
        success=("collision_free", "mean"),
        runtime=("runtime_ms", "mean"),
        soc=("sum_of_costs", "mean"),
        makespan=("makespan", "mean"),
        conflicts=("conflicts", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    metrics = [
        ("success", "Collision-free success (%)", True),
        ("runtime", "Mean runtime (ms)", False),
        ("soc", "Mean sum of costs", False),
        ("makespan", "Mean makespan", False),
        ("conflicts", "Mean conflicts", False),
    ]
    for ax, (metric, ylabel, percent) in zip(axes.flat, metrics):
        for method in methods:
            part = grouped[grouped["method"] == method].sort_values("n_agents")
            vals = part[metric] * 100 if percent else part[metric]
            ax.plot(part["n_agents"], vals, marker="o", linewidth=2, label=method)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.set_title({
            "success": "Scalability of collision-free planning",
            "runtime": "Computational scaling",
            "soc": "Solution cost as team size grows",
            "makespan": "Completion horizon as team size grows",
            "conflicts": "Conflict growth with team size",
        }[metric])
        if percent:
            ax.set_ylim(0, 105)
        _style(ax)

    # Runtime vs agents scatter in the final panel
    ax = axes[1, 2]
    for method in methods:
        part = data[data["method"] == method]
        ax.scatter(part["n_agents"], part["runtime_ms"], alpha=0.55, s=28, label=method)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Per-trial runtime dispersion")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    fig.suptitle("Q2 — Multi-agent scalability and solution quality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _q2_heatmap(df: pd.DataFrame):
    """Success-rate heatmap: method x number of agents."""
    pivot = (df.groupby(["method", "n_agents"])["collision_free"].mean() * 100).unstack()
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Method")
    ax.set_title("Collision-free success matrix (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Success rate (%)")
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("World")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    height = st.slider("Grid height", 1, 50, 18)
    width = st.slider("Grid width", 1, 50, 18)
    density = st.slider("Obstacle density", 0.0, 1.0, 0.20, 0.01, format="%.2f")
    n_agents = st.slider("Agents (Q2)", 2, 8, 4)
    st.markdown("---")
    st.markdown("Green circle = start, red star = goal. Obstacles are black. Multi-robot colours match start/goal/path per agent.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Q1 · Live A*", "Q1 · Heuristic study", "Q2 · Live MAPF", "Q2 · Method study"
])

with tab1:
    st.subheader("Single robot — three heuristics")
    st.write("The same random grid, start, and goal are solved with Manhattan, Euclidean, and Chebyshev heuristics.")
    if st.button("Generate grid and run A*", type="primary", key="q1_live"):
        rng = np.random.default_rng(int(seed))
        try:
            world, start, goal = make_single_instance(int(height), int(width), float(density), rng)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            results = compare_heuristics(world, start, goal, HEURISTICS)
            st.session_state["q1_pack"] = (world, start, goal, results)
    if "q1_pack" in st.session_state:
        world, start, goal, results = st.session_state["q1_pack"]
        cols = st.columns(3)
        for col, res in zip(cols, results):
            with col:
                st.markdown(f"**{res.heuristic_name}**")
                st.caption(HEURISTIC_NOTES[res.heuristic_name])
                fig = draw_single(world, start, goal, res.path if res.found else None,
                                  title=f"{res.heuristic_name} · cost {res.cost:.0f}" if res.found else res.heuristic_name)
                _close(fig)
                st.caption(f"Expanded {res.nodes_expanded} nodes · {res.runtime_ms:.2f} ms" if res.found else "No path")
        st.dataframe(pd.DataFrame([r.as_dict() for r in results]), use_container_width=True)

with tab2:
    st.subheader("Q1 · Heuristic study")
    st.write("Repeated random-grid experiment focused on search effort, runtime behaviour, path-cost consistency, and reliability.")
    trials = st.slider("Trials", 8, 60, 24, key="q1_trials")
    if st.button("Run Q1 experiment", type="primary", key="q1_batch"):
        with st.spinner("Running A* on random maps…"):
            df = run_q1(n_trials=int(trials), height=int(height), width=int(width),
                        obstacle_density=float(density), seed=int(seed))
        st.session_state["q1_df"] = df
    if "q1_df" in st.session_state:
        df = st.session_state["q1_df"]
        summary = summarize_q1(df)
        st.markdown("#### Experiment summary")
        st.dataframe(summary, use_container_width=True)
        _close(_q1_study_plots(df, summary))
        st.markdown("#### Raw Q1 results")
        st.dataframe(df, use_container_width=True, height=280)

with tab3:
    st.subheader("Several robots, same map")
    st.write("Independent A* ignores other robots. Prioritized STA* uses reservations. CBS resolves vertex and edge conflicts with constraint splitting.")
    if st.button("Generate MAPF instance", type="primary", key="q2_live"):
        rng = np.random.default_rng(int(seed) + 99)
        try:
            world, starts, goals = make_multi_instance(int(height), int(width), float(density), int(n_agents), rng)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Planning…"):
                results = run_all_mapf(world, starts, goals)
            st.session_state["q2_pack"] = (world, starts, goals, results)
    if "q2_pack" in st.session_state:
        world, starts, goals, results = st.session_state["q2_pack"]
        st.dataframe(pd.DataFrame([r.as_dict() for r in results]), use_container_width=True)
        cols = st.columns(3)
        for col, res in zip(cols, results):
            with col:
                label = "collision-free" if res.collision_free else f"{res.conflicts} conflicts"
                fig = draw_multi(world, starts, goals, res.paths if res.found else None, title=f"{res.name} · {label}")
                _close(fig)
                st.caption("No plan" if not res.found else f"Sum of costs {res.sum_of_costs} · makespan {res.makespan}")

with tab4:
    st.subheader("Q2 · Method study")
    st.write("Evaluation of scalability, solution quality, and collision handling at the selected obstacle density.")
    trials2 = st.slider("Trials per configuration", 5, 30, 12, key="q2_trials")
    st.caption(f"Obstacle density is fixed at {float(density):.0%} for every Q2 configuration.")
    if st.button("Run Q2 experiment", type="primary", key="q2_batch"):
        with st.spinner("Running MAPF scalability and robustness experiments…"):
            df = sweep_q2(
                seed=int(seed),
                trials_per_config=int(trials2),
                height=int(height),
                width=int(width),
                obstacle_density=float(density),
                max_agents=int(n_agents),
            )
        st.session_state["q2_df"] = df

    if "q2_df" in st.session_state:
        df = st.session_state["q2_df"]
        summary = summarize_q2(df)
        st.markdown("#### Experiment summary")
        st.dataframe(summary, use_container_width=True)

        st.markdown("#### Agent-count scalability")
        _close(_q2_agent_plots(df))

        heatmap = _q2_heatmap(df)
        if heatmap is not None:
            st.markdown("#### Success-rate overview")
            _close(heatmap)

        st.markdown("#### Raw Q2 results")
        st.dataframe(df, use_container_width=True, height=320)
