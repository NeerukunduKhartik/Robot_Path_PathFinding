# CSMI17 Assignment Report — Robot Path Finding

## 1. Problem definition

**Single-robot path finding.** A robot lives on a rectangular occupancy grid. Some cells are blocked. The robot starts in a free cell and must reach a free goal cell. Moves are the four cardinal directions (up, down, left, right), each of cost 1. Diagonals are not allowed. The task is to compute a shortest obstacle-avoiding path with **A\***, using at least three different heuristics, and to compare those heuristics experimentally.

**Multi-robot path finding (MAPF).** Several robots share the same grid. Each robot has its own start and goal. Robots cannot occupy the same cell at the same time, and they cannot swap cells through each other in one timestep (an *edge* / *head-on* conflict). Each robot may wait in place. The objective used here is **sum of costs** (total timesteps until each robot reaches its goal), with **makespan** reported as a secondary metric.

## 2. Assumptions and customisations

- Grids are 4-connected; waiting is allowed only in the multi-agent planners (space-time search).
- Obstacle cells are sampled independently with a chosen density. Starts and goals are sampled on free cells. A map is rejected unless every agent has an individual static path (BFS). That keeps “unsolvable maze” noise out of the comparison.
- Starts are distinct; goals are distinct. Two robots may share a start/goal pair across different agents only if sampling allows distinct cells — they are always distinct in this code.
- After a robot finishes, independent A\* *pads* the path by staying on the goal. That is how parked robots create later vertex conflicts — a realistic extra challenge.
- Prioritised planning uses a fixed order (agent index 0, 1, 2, …). It is incomplete: a later robot can be boxed in by earlier reservations.
- CBS uses standard vertex and edge constraints. High-level search is capped so crowded maps finish in reasonable time.
- Time is discrete. Two robots moving into the same cell at the same timestep is a vertex conflict.

## 3. Algorithms and heuristics

### 3.1 A\* (single robot)

A\* expands nodes by \(f = g + h\). \(g\) is the path cost from the start. \(h\) estimates remaining cost to the goal. With a consistent heuristic, the first time the goal is expanded the path is optimal.

**Three heuristics (all admissible on this movement model):**

1. **Manhattan** \(h = |\Delta r| + |\Delta c|\). For 4-moves of cost 1 this is the true shortest path on an empty grid, so it is the most informed of the three.
2. **Euclidean** \(h = \sqrt{(\Delta r)^2 + (\Delta c)^2}\). Never larger than the true 4-connected distance, but it underestimates more, so A\* must expand more cells to prove optimality.
3. **Chebyshev** \(h = \max(|\Delta r|, |\Delta c|)\). Still a lower bound, but the weakest: A\* behaves closer to Dijkstra.

Because all three are admissible, they should return the **same path cost**. They are compared on **nodes expanded**, **nodes generated**, and **runtime**.

### 3.2 Extra challenges with many robots

Independent A\* on each robot (the “plain A\*” baseline) ignores everyone else. Typical failures:

- **Vertex conflicts** — two robots occupy one cell at the same time (including a robot sitting on its goal).
- **Edge / swapping conflicts** — two robots exchange cells in one step.
- **Coupled state space** — a joint A\* state is the product of all robot positions (and time). It grows exponentially with the number of agents.
- **Deadlocks** — greedy local paths can block corridors; a robot that arrived early can trap another.
- **Priority artefacts** — if you plan in a fixed order, a poor first path can make later robots fail even when a joint solution exists.

### 3.3 Proposed method: Conflict-Based Search (CBS)

CBS is a two-level algorithm (Sharon et al., 2015). It is the method we propose instead of coupling all robots into one giant A\*.

- **Low level.** Each robot is planned with **space-time A\***: the state is \((r, c, t)\). The robot may wait. Constraints forbid being in a cell at a time, or taking a given edge at a time.
- **High level.** Nodes of a constraint tree store a full set of paths. If the paths have no conflict, that node is a solution. Otherwise CBS takes the *first* vertex or edge conflict between two agents \(A\) and \(B\) and creates two children: one adds a constraint to \(A\), the other to \(B\). Only the constrained agent is replanned.

CBS searches the *conflict space*, not the joint configuration space. For unit-cost MAPF it returns a sum-of-costs optimal solution when the high-level search is allowed to finish.

**Prioritised space-time A\*** (reservation table) is included as a practical middle ground: faster than CBS, usually collision-free, but not optimal and not complete.

## 4. Experimental setup

Implementation: Python 3, NumPy, Pandas, Matplotlib, Streamlit (`app.py`). Algorithms live in `src/`.

**Q1.** Random maps (default \(20\times 20\), obstacle density \(0.22\)). For each map, all three heuristics run on the same start and goal. Metrics: success, path cost, nodes expanded / generated, runtime (ms).

**Q2.** Random MAPF maps; a sweep over grid size, density, and number of agents (see `src/experiments.py` → `sweep_q2`). Metrics: collision-free rate, residual conflicts, sum of costs, makespan, low-level expansions, high-level CBS nodes, runtime.

**How to reproduce**

```text
pip install -r requirements.txt
streamlit run app.py
python run_experiments.py
```

In the app: **Q1 · Live A\*** draws one map; **Q1 · Heuristic study** averages many maps; **Q2 · Live MAPF** shows the three planners on one instance; **Q2 · Method study** runs the batch / sweep.

Screenshots: use the Streamlit tabs after generating a grid. The live views are the figures for section 4 of the submitted PDF (grid + paths + metrics table).

## 5. Performance comparison

### Q1 — heuristics

Expected pattern (and what the code is designed to show):

| Heuristic | Path cost | Nodes expanded | Runtime |
|-----------|-----------|----------------|---------|
| Manhattan | Optimal (same as others) | Lowest | Lowest |
| Euclidean | Same | Medium | Medium |
| Chebyshev | Same | Highest | Highest |

Manhattan dominates because it is a perfect estimate on empty 4-grids and remains a tight bound with obstacles. Euclidean and Chebyshev waste expansions on cells that Manhattan never opens. Success rate is essentially 1.0 because maps are filtered to be reachable.

If a trial shows different path costs, that would indicate a bug: admissible A\* must agree on cost.

### Q2 — methods

| Method | Collision-free? | Cost quality | Speed | Completeness |
|--------|-----------------|--------------|-------|--------------|
| Independent A\* | Often **no** | Looks short but invalid | Fastest | N/A (wrong problem) |
| Prioritised STA\* | Usually yes | Suboptimal (priority order) | Fast | Incomplete |
| **CBS (proposed)** | Yes when search finishes | Best (optimal SOC) | Slower on dense teams | Complete for this conflict model (within node cap) |

Independent A\* “succeeds” at finding per-robot paths while still leaving **many conflicts** — that is exactly why plain A\* is the wrong MAPF solver. CBS is better in the sense that matters: **collision-free success** and **sum of costs**. Prioritised STA\* is a useful ablation: it already fixes most collisions via reservations, but CBS still wins on cost and on maps where a later agent is blocked by an earlier reservation.

Larger grids, higher density, and more agents all increase CBS high-level nodes and runtime; independent A\* stays cheap and increasingly collision-heavy.

## 6. GitHub link

After you push this folder:

```text
https://github.com/<your-username>/<your-repo>
```

Create the repository, then:

```text
git init
git add .
git commit -m "CSMI17 robot path finding: A* heuristics and CBS MAPF"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Paste the URL into the submitted report.
