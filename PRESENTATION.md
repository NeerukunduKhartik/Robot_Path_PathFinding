# Presentation outline (10 marks)

**Timebox:** ~8–10 minutes plus questions.

1. **Problem (1 min)** — Grid, obstacles, one robot then many. 4-moves, unit cost.
2. **Q1 (2 min)** — A\* + Manhattan / Euclidean / Chebyshev. Same optimal cost; Manhattan expands fewest nodes (live Streamlit screenshot).
3. **Challenges (2 min, 5 marks)** — Vertex + swap conflicts, exponential joint state, robots parked on goals, incomplete priorities.
4. **CBS (3 min, 15 marks novelty)** — High-level constraint tree, low-level space-time A\*. Contrast with independent A\* and reservation-table STA\*.
5. **Results (2 min)** — Q1 bar charts; Q2 collision-free rate vs residual conflicts; CBS wins validity + cost, STA\* is the fast heuristic, independent A\* is invalid.
6. **Repo** — GitHub link, `streamlit run app.py`.

Demo order in class: generate one Q1 grid, then one Q2 instance showing independent A\* colliding and CBS clear.
