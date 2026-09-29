"""Multi-agent path finding: independent A*, prioritized Space-Time A*, and CBS."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

from .grid import Coord, GridWorld
from .heuristics import manhattan
from .astar import astar

WAIT: Coord = (0, 0)
MOVES_ST: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), WAIT)


@dataclass
class MAPFResult:
    name: str
    found: bool
    paths: list[list[Coord]]
    makespan: int
    sum_of_costs: int
    conflicts: int
    nodes_expanded: int
    high_level_nodes: int
    runtime_ms: float
    extra: str = ""

    @property
    def collision_free(self) -> bool:
        return self.found and self.conflicts == 0

    def as_dict(self) -> dict:
        return {
            "method": self.name,
            "found": self.found,
            "collision_free": self.collision_free,
            "makespan": self.makespan if self.found else None,
            "sum_of_costs": self.sum_of_costs if self.found else None,
            "conflicts": self.conflicts,
            "nodes_expanded": self.nodes_expanded,
            "high_level_nodes": self.high_level_nodes,
            "runtime_ms": round(self.runtime_ms, 4),
            "notes": self.extra,
        }


def pad_paths(paths: list[list[Coord]]) -> list[list[Coord]]:
    if not paths:
        return paths
    t_max = max(len(p) for p in paths)
    padded = []
    for p in paths:
        if not p:
            padded.append(p)
            continue
        extra = p + [p[-1]] * (t_max - len(p))
        padded.append(extra)
    return padded


def count_conflicts(paths: list[list[Coord]]) -> int:
    """Vertex + swapping (edge) conflicts after padding agents at their goals."""
    if not paths or any(not p for p in paths):
        return 0
    padded = pad_paths(paths)
    t_max = len(padded[0])
    n = len(padded)
    conflicts = 0
    for t in range(t_max):
        seen: dict[Coord, int] = {}
        for i in range(n):
            cell = padded[i][t]
            if cell in seen:
                conflicts += 1
            else:
                seen[cell] = i
        if t == 0:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a0, a1 = padded[i][t - 1], padded[i][t]
                b0, b1 = padded[j][t - 1], padded[j][t]
                if a0 == b1 and b0 == a1 and a0 != a1:
                    conflicts += 1
    return conflicts


def _soc(paths: list[list[Coord]]) -> int:
    return sum(max(0, len(p) - 1) for p in paths)


def _makespan(paths: list[list[Coord]]) -> int:
    return max((len(p) - 1) for p in paths) if paths else 0


def independent_astar(
    world: GridWorld,
    starts: list[Coord],
    goals: list[Coord],
) -> MAPFResult:
    """Plain A*: each robot ignores the others. Cheap, often colliding."""
    t0 = time.perf_counter()
    paths: list[list[Coord]] = []
    expanded = 0
    for s, g in zip(starts, goals):
        res = astar(world, s, g)
        expanded += res.nodes_expanded
        if not res.found:
            dt = (time.perf_counter() - t0) * 1000
            return MAPFResult(
                "Independent A*",
                False,
                [],
                0,
                0,
                0,
                expanded,
                0,
                dt,
                extra="an agent had no static path",
            )
        paths.append(res.path)
    n_conf = count_conflicts(paths)
    dt = (time.perf_counter() - t0) * 1000
    return MAPFResult(
        "Independent A*",
        True,
        paths,
        _makespan(paths),
        _soc(paths),
        n_conf,
        expanded,
        0,
        dt,
        extra="ignores other robots",
    )


# ---------------------------------------------------------------------------
# Space-Time A* (low-level planner used by prioritized planning and CBS)
# ---------------------------------------------------------------------------

VertexConstraint = tuple[int, int, int]  # row, col, time
EdgeConstraint = tuple[int, int, int, int, int]  # r1,c1,r2,c2,time (arrive time)


@dataclass
class Constraints:
    vertex: set[VertexConstraint] = field(default_factory=set)
    edge: set[EdgeConstraint] = field(default_factory=set)
    last_goal_block: dict[Coord, int] = field(default_factory=dict)

    def blocked_vertex(self, cell: Coord, t: int) -> bool:
        return (cell[0], cell[1], t) in self.vertex

    def blocked_edge(self, a: Coord, b: Coord, t_arrive: int) -> bool:
        return (a[0], a[1], b[0], b[1], t_arrive) in self.edge


@dataclass(order=True)
class _STNode:
    f: float
    g: int
    tie: int
    cell: Coord = field(compare=False)
    t: int = field(compare=False)


def space_time_astar(
    world: GridWorld,
    start: Coord,
    goal: Coord,
    constraints: Constraints,
    reserved_vertex: Optional[set[tuple[int, int, int]]] = None,
    reserved_edge: Optional[set[tuple[int, int, int, int, int]]] = None,
    max_time: int = 400,
) -> tuple[Optional[list[Coord]], int]:
    """A* in (row, col, time). Wait is allowed. Returns path of cells (index = time) and expansions."""
    reserved_vertex = reserved_vertex or set()
    reserved_edge = reserved_edge or set()

    def occupied(cell: Coord, t: int, prev: Coord) -> bool:
        if constraints.blocked_vertex(cell, t):
            return True
        if (cell[0], cell[1], t) in reserved_vertex:
            return True
        if constraints.blocked_edge(prev, cell, t):
            return True
        if (prev[0], prev[1], cell[0], cell[1], t) in reserved_edge:
            return True
        return False

    t0_limit = max_time
    open_heap: list[_STNode] = []
    g_score: dict[tuple[Coord, int], int] = {(start, 0): 0}
    came: dict[tuple[Coord, int], tuple[Coord, int]] = {}
    tie = 0
    heapq.heappush(open_heap, _STNode(manhattan(start, goal), 0, tie, start, 0))
    expanded = 0
    closed: set[tuple[Coord, int]] = set()

    while open_heap:
        node = heapq.heappop(open_heap)
        state = (node.cell, node.t)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1
        if node.t >= t0_limit:
            continue
        if node.cell == goal:
            later_block = False
            for tt in range(node.t + 1, t0_limit + 1):
                if constraints.blocked_vertex(goal, tt) or (goal[0], goal[1], tt) in reserved_vertex:
                    later_block = True
                    break
            if not later_block:
                path: list[Coord] = []
                cur_c, cur_t = node.cell, node.t
                path.append(cur_c)
                while (cur_c, cur_t) in came:
                    cur_c, cur_t = came[(cur_c, cur_t)]
                    path.append(cur_c)
                path.reverse()
                return path, expanded

        for dr, dc in MOVES_ST:
            nxt = (node.cell[0] + dr, node.cell[1] + dc)
            t_next = node.t + 1
            if not world.is_free(nxt):
                continue
            if occupied(nxt, t_next, node.cell):
                continue
            key = (nxt, t_next)
            tentative = node.g + 1
            if tentative >= g_score.get(key, 10**9):
                continue
            g_score[key] = tentative
            came[key] = (node.cell, node.t)
            tie += 1
            f = tentative + manhattan(nxt, goal)
            heapq.heappush(open_heap, _STNode(f, tentative, tie, nxt, t_next))

    return None, expanded


def _reserve_path(
    path: list[Coord],
    vertex: set[tuple[int, int, int]],
    edge: set[tuple[int, int, int, int, int]],
    hold_goal: int,
) -> None:
    for t, cell in enumerate(path):
        vertex.add((cell[0], cell[1], t))
        if t > 0:
            prev = path[t - 1]
            edge.add((prev[0], prev[1], cell[0], cell[1], t))
            # block the reverse edge at the same timestep (no swaps)
            edge.add((cell[0], cell[1], prev[0], prev[1], t))
    if not path:
        return
    goal = path[-1]
    for t in range(len(path), hold_goal + 1):
        vertex.add((goal[0], goal[1], t))


def prioritized_stastar(
    world: GridWorld,
    starts: list[Coord],
    goals: list[Coord],
    max_time: int = 400,
) -> MAPFResult:
    """Plan agents in index order. Later agents treat earlier paths as moving obstacles."""
    t0 = time.perf_counter()
    n = len(starts)
    reserved_v: set[tuple[int, int, int]] = set()
    reserved_e: set[tuple[int, int, int, int, int]] = set()
    paths: list[list[Coord]] = []
    expanded = 0
    empty = Constraints()
    horizon = max(world.height * world.width, max(manhattan(s, g) for s, g in zip(starts, goals))) + 40
    horizon = min(int(horizon) + 8 * n, max_time)

    for i in range(n):
        path, exp = space_time_astar(
            world,
            starts[i],
            goals[i],
            empty,
            reserved_vertex=reserved_v,
            reserved_edge=reserved_e,
            max_time=horizon,
        )
        expanded += exp
        if path is None:
            dt = (time.perf_counter() - t0) * 1000
            return MAPFResult(
                "Prioritized STA*",
                False,
                [],
                0,
                0,
                0,
                expanded,
                i + 1,
                dt,
                extra=f"agent {i} failed under reservations",
            )
        paths.append(path)
        _reserve_path(path, reserved_v, reserved_e, horizon)

    dt = (time.perf_counter() - t0) * 1000
    return MAPFResult(
        "Prioritized STA*",
        True,
        paths,
        _makespan(paths),
        _soc(paths),
        count_conflicts(paths),
        expanded,
        n,
        dt,
        extra="reservation table; priority = agent index",
    )


# ---------------------------------------------------------------------------
# Conflict-Based Search
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    a: int
    b: int
    t: int
    kind: str  # vertex | edge
    cell: Coord
    cell_b: Coord = (0, 0)


def first_conflict(paths: list[list[Coord]]) -> Optional[Conflict]:
    padded = pad_paths(paths)
    t_max = len(padded[0])
    n = len(padded)
    for t in range(t_max):
        loc: dict[Coord, int] = {}
        for i in range(n):
            cell = padded[i][t]
            if cell in loc:
                return Conflict(loc[cell], i, t, "vertex", cell)
            loc[cell] = i
        if t == 0:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a0, a1 = padded[i][t - 1], padded[i][t]
                b0, b1 = padded[j][t - 1], padded[j][t]
                if a0 == b1 and b0 == a1 and a0 != a1:
                    return Conflict(i, j, t, "edge", a0, b0)
    return None


@dataclass(order=True)
class CTNode:
    cost: int
    tie: int
    constraints: list[Constraints] = field(compare=False)
    paths: list[list[Coord]] = field(compare=False)
    expanded: int = field(compare=False, default=0)


def _clone_constraints(cs: list[Constraints]) -> list[Constraints]:
    return [
        Constraints(set(c.vertex), set(c.edge), dict(c.last_goal_block))
        for c in cs
    ]


def conflict_based_search(
    world: GridWorld,
    starts: list[Coord],
    goals: list[Coord],
    max_high_level: int = 400,
    max_time: int = 350,
) -> MAPFResult:
    """Standard CBS: high-level constraint tree, low-level space-time A*."""
    t0 = time.perf_counter()
    n = len(starts)
    horizon = min(
        max_time,
        int(max(world.height * world.width, max(manhattan(s, g) for s, g in zip(starts, goals))))
        + 30
        + 6 * n,
    )
    constraints = [Constraints() for _ in range(n)]
    paths: list[list[Coord]] = []
    low_exp = 0
    for i in range(n):
        path, exp = space_time_astar(world, starts[i], goals[i], constraints[i], max_time=horizon)
        low_exp += exp
        if path is None:
            dt = (time.perf_counter() - t0) * 1000
            return MAPFResult("CBS", False, [], 0, 0, 0, low_exp, 0, dt, extra="root low-level fail")
        paths.append(path)

    tie = 0
    open_hl: list[CTNode] = [CTNode(_soc(paths), tie, constraints, paths, low_exp)]
    hl_expanded = 0

    while open_hl and hl_expanded < max_high_level:
        node = heapq.heappop(open_hl)
        hl_expanded += 1
        conflict = first_conflict(node.paths)
        if conflict is None:
            dt = (time.perf_counter() - t0) * 1000
            return MAPFResult(
                "CBS",
                True,
                node.paths,
                _makespan(node.paths),
                _soc(node.paths),
                0,
                node.expanded,
                hl_expanded,
                dt,
                extra="optimal under unit costs (standard CBS)",
            )

        for agent in (conflict.a, conflict.b):
            child_cs = _clone_constraints(node.constraints)
            if conflict.kind == "vertex":
                r, c = conflict.cell
                child_cs[agent].vertex.add((r, c, conflict.t))
            else:
                if agent == conflict.a:
                    a_from, a_to = conflict.cell, conflict.cell_b
                else:
                    a_from, a_to = conflict.cell_b, conflict.cell
                # edge constraint: cannot traverse a_from -> a_to arriving at t
                child_cs[agent].edge.add((a_from[0], a_from[1], a_to[0], a_to[1], conflict.t))

            new_paths = list(node.paths)
            path, exp = space_time_astar(
                world, starts[agent], goals[agent], child_cs[agent], max_time=horizon
            )
            total_exp = node.expanded + exp
            if path is None:
                continue
            new_paths[agent] = path
            tie += 1
            heapq.heappush(open_hl, CTNode(_soc(new_paths), tie, child_cs, new_paths, total_exp))

    dt = (time.perf_counter() - t0) * 1000
    best = min(open_hl, key=lambda x: x.cost) if open_hl else None
    if best is None:
        return MAPFResult("CBS", False, [], 0, 0, 0, low_exp, hl_expanded, dt, extra="search exhausted")
    conf = count_conflicts(best.paths)
    return MAPFResult(
        "CBS",
        conf == 0,
        best.paths if conf == 0 else best.paths,
        _makespan(best.paths),
        _soc(best.paths),
        conf,
        best.expanded,
        hl_expanded,
        dt,
        extra="hit high-level node cap",
    )


def run_all_mapf(
    world: GridWorld,
    starts: list[Coord],
    goals: list[Coord],
) -> list[MAPFResult]:
    return [
        independent_astar(world, starts, goals),
        prioritized_stastar(world, starts, goals),
        conflict_based_search(world, starts, goals),
    ]
