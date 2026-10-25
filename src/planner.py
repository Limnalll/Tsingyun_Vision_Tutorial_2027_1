"""A tiny grid planner for the demo tour (provided).

``plan_path`` runs A* on a coarse version of the occupancy grid with
obstacles inflated by the robot radius, then shortcuts the result with
line-of-sight checks.  It exists so the demo robot can recover from any
start (including the kidnap button) without wedging itself into corners.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

PLAN_STRIDE = 3     # plan on every 3rd grid cell (0.15 m at 0.05 m)
CLEARANCE = 0.32    # robot radius + margin, meters: cells closer than
#                   # this to an obstacle are untraversable
LOS_CLEARANCE = 0.25  # margin for the line-of-sight shortcut, meters


def _planning_grid(world):
    """(free, cell) -- bool grid of traversable cells and the cell size."""
    free = world.distance_field[::PLAN_STRIDE, ::PLAN_STRIDE] >= CLEARANCE
    return free, world.res * PLAN_STRIDE


def _nearest_free(free, ix, iy):
    """BFS to the closest traversable cell (start/goal may sit too close
    to a wall to be valid planning cells themselves)."""
    ny, nx = free.shape
    seen = {(ix, iy)}
    queue = deque([(ix, iy)])
    while queue:
        cx, cy = queue.popleft()
        if free[cy, cx]:
            return cx, cy
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and n not in seen:
                seen.add(n)
                queue.append(n)
    return ix, iy


def clear_line(world, a, b, margin=LOS_CLEARANCE):
    """True when the segment a-b stays ``margin`` away from obstacles."""
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.1) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = a[0] + t * (b[0] - a[0])
    ys = a[1] + t * (b[1] - a[1])
    return bool(np.all(world.obstacle_distance(xs, ys) >= margin))


def plan_path(world, start, goal):
    """Shortest collision-free path from ``start`` to ``goal`` as
    ``[(x, y), ...]`` world-coordinate points (start excluded, goal
    included).  Returns ``None`` when no route exists.
    """
    free, cell = _planning_grid(world)
    ny, nx = free.shape

    sx, sy = _nearest_free(free, int(start[0] / cell), int(start[1] / cell))
    gx, gy = _nearest_free(free, int(goal[0] / cell), int(goal[1] / cell))

    # A* over the coarse grid, 8-connected, diagonals only through open
    # corners (never cut a corner diagonally)
    def h(x, y):
        return np.hypot(x - gx, y - gy)

    open_heap = [(h(sx, sy), 0.0, sx, sy)]
    came = {(sx, sy): None}
    cost = {(sx, sy): 0.0}
    while open_heap:
        _, g, x, y = heapq.heappop(open_heap)
        if (x, y) == (gx, gy):
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if not (0 <= nx_ < nx and 0 <= ny_ < ny) or not free[ny_, nx_]:
                    continue
                if dx != 0 and dy != 0 and not (free[y, nx_] and free[ny_, x]):
                    continue
                step = np.hypot(dx, dy)
                new_cost = g + step
                if new_cost < cost.get((nx_, ny_), np.inf):
                    cost[(nx_, ny_)] = new_cost
                    came[(nx_, ny_)] = (x, y)
                    heapq.heappush(open_heap, (new_cost + h(nx_, ny_), new_cost, nx_, ny_))
        # guard against unreachable goals
        if not open_heap and (gx, gy) not in came:
            return None

    # reconstruct and convert to world coordinates
    pts = []
    cur = came.get((gx, gy))
    while cur is not None:
        pts.append(((cur[0] + 0.5) * cell, (cur[1] + 0.5) * cell))
        cur = came[cur]
    pts.reverse()
    if clear_line(world, start, goal):
        return [goal]
    # shortcut greedily: keep the furthest line-of-sight successor
    out = []
    i = 0
    current = start
    while i < len(pts):
        j = i
        for k in range(len(pts) - 1, i, -1):
            if clear_line(world, current, pts[k]):
                j = k
                break
        out.append(pts[j])
        current = pts[j]
        if j == len(pts) - 1:
            break
        i = j + 1
    # trade the last coarse cell for the exact goal when it is reachable
    if out and clear_line(world, out[-1], goal):
        out[-1] = tuple(goal)
    else:
        out.append(tuple(goal))
    return out
