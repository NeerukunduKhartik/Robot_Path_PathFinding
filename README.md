# CSMI17 — Robot path finding

Single-agent **A\*** with three heuristics, then multi-robot planning with independent A\*, prioritized space-time A\*, and **Conflict-Based Search (CBS)**.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Batch tables for the report:

```bash
python run_experiments.py
```

## Heuristics (Q1)

| Heuristic | On a 4-connected unit grid |
|-----------|----------------------------|
| Manhattan | Admissible and consistent; tightest bound |
| Euclidean | Admissible; looser, so more expansions |
| Chebyshev | Admissible; weakest of the three |

Compared by path cost, nodes expanded, runtime, and success rate on random maps.

## Multi-robot (Q2)

Independent A\* ignores other robots (collisions). Prioritized STA\* uses a reservation table. CBS is the proposed method: it branches on the first vertex/edge conflict and replans only the agents involved.

See `REPORT.md` for problem definition, assumptions, algorithms, setup, and results.
