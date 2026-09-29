"""Interactive demo for CSMI17 robot / multi-robot path finding."""

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
from src.experiments import run_q1, run_q2, summarize_q1, summarize_q2, sweep_q2
from src.grid import make_multi_instance, make_single_instance
from src.heuristics import HEURISTIC_NOTES, HEURISTICS
from src.mapf import run_all_mapf
from src.viz import bar_compare, draw_multi, draw_single, heuristics_one_graph

st.set_page_config(
    page_title="CSMI17 Path Finding",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1280px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Robot path finding with A* and CBS")
st.caption("CSMI17 assignment — single-agent heuristics, then multi-robot planning.")


def _close(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with st.sidebar:
    st.header("World")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    height = st.slider("Grid height", 8, 30, 18)
    width = st.slider("Grid width", 8, 30, 18)
    density = st.slider("Obstacle density", 0.0, 0.40, 0.20, 0.01)
    n_agents = st.slider("Agents (Q2)", 2, 8, 4)
    st.markdown("---")
    st.markdown(
        "Green circle = start, red star = goal. "
        "Obstacles are black. Multi-robot colours match start/goal/path per agent."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Q1 · Live A*",
        "Q1 · Heuristic study",
        "Q2 · Live MAPF",
        "Q2 · Method study",
    ]
)

with tab1:
    st.subheader("Single robot — three heuristics")
    st.write(
        "The same random grid, start, and goal are solved with Manhattan, Euclidean, "
        "and Chebyshev heuristics. All three are admissible on 4-connected unit grids; "
        "Manhattan is the tightest, so A* usually expands the fewest nodes."
    )
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
                fig = draw_single(
                    world,
                    start,
                    goal,
                    res.path if res.found else None,
                    title=f"{res.heuristic_name} · cost {res.cost:.0f}" if res.found else res.heuristic_name,
                )
                _close(fig)
                st.caption(
                    f"Expanded {res.nodes_expanded} nodes · {res.runtime_ms:.2f} ms"
                    if res.found
                    else "No path"
                )
        st.dataframe(pd.DataFrame([r.as_dict() for r in results]), use_container_width=True)

with tab2:
    st.subheader("Repeated random grids")
    st.write(
        "Metrics: **path cost** (optimality), **nodes expanded** (search effort), "
        "**runtime**, and **success rate**. Because all three heuristics are admissible, "
        "A* returns the same optimal cost; they differ in how much of the grid they explore."
    )
    trials = st.slider("Trials", 8, 60, 24, key="q1_trials")
    if st.button("Run Q1 experiment", type="primary", key="q1_batch"):
        with st.spinner("Running A* on random maps…"):
            df = run_q1(
                n_trials=int(trials),
                height=int(height),
                width=int(width),
                obstacle_density=float(density),
                seed=int(seed),
            )
        st.session_state["q1_df"] = df

    if "q1_df" in st.session_state:
        df = st.session_state["q1_df"]
        summary = summarize_q1(df)
        st.dataframe(summary, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            _close(bar_compare(summary, "mean_nodes_expanded", "Search effort", "Mean nodes expanded"))
        with c2:
            _close(bar_compare(summary, "mean_runtime_ms", "Runtime", "Mean runtime (ms)"))
        st.subheader("All three heuristics on one graph")
        _close(heuristics_one_graph(df, summary))
        st.caption(
            "Left: nodes expanded on the same maps, all three heuristics. "
            "Right: each mean scaled to the largest heuristic (100%). "
            "Path cost bars should line up; Manhattan should be lowest on nodes and runtime."
        )

with tab3:
    st.subheader("Several robots, same map")
    st.write(
        "**Independent A*** plans each robot as if it were alone (plain A*). "
        "**Prioritized space-time A*** treats earlier robots as moving obstacles. "
        "**CBS** splits on the first collision and replans only the conflicting agents."
    )
    if st.button("Generate MAPF instance", type="primary", key="q2_live"):
        rng = np.random.default_rng(int(seed) + 99)
        try:
            world, starts, goals = make_multi_instance(
                int(height), int(width), float(density), int(n_agents), rng
            )
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
                fig = draw_multi(
                    world,
                    starts,
                    goals,
                    res.paths if res.found else None,
                    title=f"{res.name} · {label}",
                )
                _close(fig)
                st.caption(
                    "No plan"
                    if not res.found
                    else f"Sum of costs {res.sum_of_costs} · makespan {res.makespan}"
                )

with tab4:
    st.subheader("Many maps, sizes, densities, and team sizes")
    st.write(
        "Success here means a **collision-free** plan. Independent A* often reports paths "
        "that still collide. CBS is the proposed method: it stays complete for these "
        "vertex/edge conflicts and typically matches or beats prioritized planning on cost."
    )
    mode = st.radio("Experiment", ["Single configuration", "Assignment sweep (sizes × agents)"], horizontal=True)
    trials2 = st.slider("Trials per configuration", 5, 30, 12, key="q2_trials")
    if st.button("Run Q2 experiment", type="primary", key="q2_batch"):
        with st.spinner("This can take a minute on larger maps…"):
            if mode.startswith("Single"):
                df = run_q2(
                    n_trials=int(trials2),
                    height=int(height),
                    width=int(width),
                    obstacle_density=float(density),
                    n_agents=int(n_agents),
                    seed=int(seed),
                )
            else:
                df = sweep_q2(seed=int(seed))
        st.session_state["q2_df"] = df

    if "q2_df" in st.session_state:
        df = st.session_state["q2_df"]
        summary = summarize_q2(df)
        st.dataframe(summary, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            _close(
                bar_compare(
                    summary,
                    "collision_free_rate",
                    "Collision-free success",
                    "Fraction of maps",
                )
            )
        with c2:
            _close(bar_compare(summary, "mean_conflicts", "Residual collisions", "Mean conflicts"))
        st.dataframe(df, use_container_width=True, height=280)
