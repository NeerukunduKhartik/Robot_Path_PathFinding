"""Admissible and inadmissible heuristics for 4-connected grids."""

from __future__ import annotations

import math
from typing import Callable

from .grid import Coord

Heuristic = Callable[[Coord, Coord], float]


def manhattan(a: Coord, b: Coord) -> float:
    """|Δrow| + |Δcol|. Consistent and admissible for unit 4-moves."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean(a: Coord, b: Coord) -> float:
    """Straight-line distance. Admissible on 4-connected grids, but looser than Manhattan."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: Coord, b: Coord) -> float:
    """max(|Δrow|, |Δcol|). Admissible for 4-moves, least informed of the three."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


HEURISTICS: dict[str, Heuristic] = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev,
}

HEURISTIC_NOTES = {
    "Manhattan": "Admissible + consistent. Tightest of the three for 4-connected unit cost.",
    "Euclidean": "Admissible (never overestimates). Weaker than Manhattan, so A* expands more.",
    "Chebyshev": "Admissible but the weakest bound here (true path is at least the max axis).",
}
