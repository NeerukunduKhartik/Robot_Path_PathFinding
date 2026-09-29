"""A* search on a 4-connected occupancy grid."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

from .grid import Coord, GridWorld
from .heuristics import Heuristic, manhattan


@dataclass
class SearchResult:
    found: bool
    path: list[Coord]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    runtime_ms: float
    heuristic_name: str = ""

    def as_dict(self) -> dict:
        return {
            "heuristic": self.heuristic_name,
            "found": self.found,
            "path_cost": self.cost if self.found else None,
            "path_length": (len(self.path) - 1) if self.found and self.path else None,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "runtime_ms": round(self.runtime_ms, 4),
        }


@dataclass(order=True)
class _Node:
    f: float
    g: float
    tie: int
    cell: Coord = field(compare=False)


def reconstruct(came_from: dict[Coord, Coord], goal: Coord) -> list[Coord]:
    path = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar(
    world: GridWorld,
    start: Coord,
    goal: Coord,
    heuristic: Heuristic = manhattan,
    heuristic_name: str = "Manhattan",
) -> SearchResult:
    """Standard graph-search A* with a consistent/admissible heuristic."""
    t0 = time.perf_counter()
    if not world.is_free(start) or not world.is_free(goal):
        return SearchResult(False, [], 0.0, 0, 0, 0.0, heuristic_name)
    if start == goal:
        dt = (time.perf_counter() - t0) * 1000
        return SearchResult(True, [start], 0.0, 1, 1, dt, heuristic_name)

    open_heap: list[_Node] = []
    g_score: dict[Coord, float] = {start: 0.0}
    came_from: dict[Coord, Coord] = {}
    closed: set[Coord] = set()
    tie = 0
    heapq.heappush(open_heap, _Node(heuristic(start, goal), 0.0, tie, start))
    generated = 1
    expanded = 0

    while open_heap:
        node = heapq.heappop(open_heap)
        cell = node.cell
        if cell in closed:
            continue
        if node.g > g_score.get(cell, float("inf")):
            continue
        expanded += 1
        if cell == goal:
            path = reconstruct(came_from, goal)
            dt = (time.perf_counter() - t0) * 1000
            return SearchResult(True, path, node.g, expanded, generated, dt, heuristic_name)
        closed.add(cell)
        for nxt in world.neighbors(cell):
            tentative = node.g + 1.0
            if tentative >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = cell
            g_score[nxt] = tentative
            tie += 1
            generated += 1
            f = tentative + heuristic(nxt, goal)
            heapq.heappush(open_heap, _Node(f, tentative, tie, nxt))

    dt = (time.perf_counter() - t0) * 1000
    return SearchResult(False, [], 0.0, expanded, generated, dt, heuristic_name)


def compare_heuristics(
    world: GridWorld,
    start: Coord,
    goal: Coord,
    heuristics: dict[str, Heuristic],
) -> list[SearchResult]:
    results = []
    for name, h in heuristics.items():
        results.append(astar(world, start, goal, heuristic=h, heuristic_name=name))
    return results
