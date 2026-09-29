"""Random grid worlds with obstacles, starts, and goals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Coord = tuple[int, int]
MOVES_4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class GridWorld:
    """Occupancy grid: 1 = obstacle, 0 = free."""

    cells: np.ndarray

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    def in_bounds(self, cell: Coord) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Coord) -> bool:
        return self.in_bounds(cell) and self.cells[cell] == 0

    def neighbors(self, cell: Coord) -> list[Coord]:
        r, c = cell
        out: list[Coord] = []
        for dr, dc in MOVES_4:
            nxt = (r + dr, c + dc)
            if self.is_free(nxt):
                out.append(nxt)
        return out

    def copy(self) -> GridWorld:
        return GridWorld(self.cells.copy())


def generate_grid(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
) -> GridWorld:
    """Fill a grid with random obstacles. Density is the fraction of blocked cells."""
    mask = rng.random((height, width)) < obstacle_density
    cells = mask.astype(np.int8)
    return GridWorld(cells)


def sample_free_cells(
    world: GridWorld,
    n: int,
    rng: np.random.Generator,
    forbidden: Iterable[Coord] = (),
) -> list[Coord]:
    blocked = set(forbidden)
    free = [
        (r, c)
        for r in range(world.height)
        for c in range(world.width)
        if world.cells[r, c] == 0 and (r, c) not in blocked
    ]
    if len(free) < n:
        raise ValueError("Not enough free cells to sample starts/goals.")
    idx = rng.choice(len(free), size=n, replace=False)
    return [free[i] for i in idx]


def bfs_reachable(world: GridWorld, start: Coord, goal: Coord) -> bool:
    """True if a 4-connected path exists from start to goal (obstacles only)."""
    if not world.is_free(start) or not world.is_free(goal):
        return False
    if start == goal:
        return True
    from collections import deque

    q: deque[Coord] = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in world.neighbors(cur):
            if nxt in seen:
                continue
            if nxt == goal:
                return True
            seen.add(nxt)
            q.append(nxt)
    return False


def make_single_instance(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
    max_tries: int = 80,
) -> tuple[GridWorld, Coord, Coord]:
    """Random grid + start/goal that are free and mutually reachable."""
    for _ in range(max_tries):
        world = generate_grid(height, width, obstacle_density, rng)
        try:
            start, goal = sample_free_cells(world, 2, rng)
        except ValueError:
            continue
        if bfs_reachable(world, start, goal):
            return world, start, goal
    raise RuntimeError("Could not sample a solvable single-agent instance.")


def make_multi_instance(
    height: int,
    width: int,
    obstacle_density: float,
    n_agents: int,
    rng: np.random.Generator,
    max_tries: int = 120,
) -> tuple[GridWorld, list[Coord], list[Coord]]:
    """Random MAPF instance: distinct starts/goals, each agent individually solvable."""
    for _ in range(max_tries):
        world = generate_grid(height, width, obstacle_density, rng)
        try:
            cells = sample_free_cells(world, 2 * n_agents, rng)
        except ValueError:
            continue
        starts = cells[:n_agents]
        goals = cells[n_agents:]
        if all(bfs_reachable(world, s, g) for s, g in zip(starts, goals)):
            return world, starts, goals
    raise RuntimeError("Could not sample a solvable multi-agent instance.")
