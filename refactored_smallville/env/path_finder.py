"""
Pathfinding utilities for agent movement on the SmallVille grid.
Adapted from generative_agents/reverie/backend_server/path_finder.py.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def path_finder(
    maze: List[List[str]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    collision_block_char: str = "32125",
) -> List[Tuple[int, int]]:
    """BFS shortest path. Coordinates are (x, y); internal grid is row-major."""
    s = (start[1], start[0])
    e = (end[1], end[0])
    path = _bfs(maze, s, e, collision_block_char)
    return [(p[1], p[0]) for p in path]


def _bfs(
    maze: List[List[str]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    collision_block_char: str,
) -> List[Tuple[int, int]]:
    rows = len(maze)
    cols = len(maze[0]) if rows else 0
    blocked = [[1 if maze[r][c] == collision_block_char else 0 for c in range(cols)] for r in range(rows)]

    dist = [[0] * cols for _ in range(rows)]
    dist[start[0]][start[1]] = 1

    k = 0
    limit = 200
    while dist[end[0]][end[1]] == 0 and limit > 0:
        k += 1
        limit -= 1
        for r in range(rows):
            for c in range(cols):
                if dist[r][c] == k:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == 0 and blocked[nr][nc] == 0:
                            dist[nr][nc] = k + 1

    if dist[end[0]][end[1]] == 0:
        return [start]

    r, c = end
    val = dist[r][c]
    path = [(r, c)]
    while val > 1:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == val - 1:
                r, c = nr, nc
                path.append((r, c))
                val -= 1
                break
    path.reverse()
    return path


def closest_coordinate(
    curr: Tuple[int, int], targets: set[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    best = None
    best_dist = float("inf")
    a = np.array(curr)
    for t in targets:
        d = float(np.linalg.norm(np.array(t) - a))
        if d < best_dist:
            best_dist = d
            best = t
    return best
