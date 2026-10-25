"""The arena: an occupancy-grid museum (provided).

The ``World`` class bundles everything the models of chapters 5-7 need
from the environment:

* an occupancy grid built from a hand-designed museum floor plan,
* a vectorized DDA ray caster (used for noise-free laser ranges ``z*``),
* the pre-computed nearest-obstacle distance field (the "likelihood field"
  of section 6.4, figure 6.9b in the book),
* free-space queries for collision checks and the map-based motion model.

Coordinates are in meters; ``x`` grows to the right, ``y`` grows up, and
headings are measured counter-clockwise from the ``+x`` axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config
from .utils import make_rng

# --------------------------------------------------------------------------
# Floor plan: rectangles (x0, y0, x1, y1) and circles (cx, cy, radius), all
# in meters.  This is the museum your robot patrols.
# --------------------------------------------------------------------------

_WALL_RECTS = [
    # outer walls
    (0.0, 0.0, 16.0, 0.2),
    (0.0, 11.8, 16.0, 12.0),
    (0.0, 0.0, 0.2, 12.0),
    (15.8, 0.0, 16.0, 12.0),
    # wall separating the west wing, with two doorways
    (4.8, 0.2, 5.0, 2.4),
    (4.8, 3.4, 5.0, 8.6),
    (4.8, 9.6, 5.0, 11.8),
    # wall separating the east wing; its south doorway sits lower than
    # the west one, and its divider door is wider -- the wings only look
    # alike at a glance
    (11.0, 0.2, 11.2, 1.8),
    (11.0, 2.8, 11.2, 8.6),
    (11.0, 9.6, 11.2, 11.8),
    # west wing divider with a doorway
    (0.2, 5.9, 2.2, 6.1),
    (3.0, 5.9, 4.8, 6.1),
    # east wing divider with a doorway
    (11.2, 5.9, 12.0, 6.1),
    (13.6, 5.9, 15.8, 6.1),
    # a central exhibition block with two stub walls forming a chicane
    (7.6, 5.6, 8.4, 6.4),
    (9.4, 5.6, 10.0, 6.4),
    (6.0, 5.6, 6.6, 6.4),
]

_WALL_CIRCLES = [
    # pillars in the central hall
    (7.0, 3.0, 0.4),
    (9.0, 3.0, 0.4),
    (7.0, 9.0, 0.4),
    (9.0, 9.0, 0.4),
    # twin pillars that make the two top rooms look alike
    (2.6, 9.6, 0.35),
    (13.4, 9.6, 0.35),
]


@dataclass
class World:
    """Occupancy grid and everything derived from it."""

    width: float
    height: float
    res: float
    grid: np.ndarray            # (ny, nx) bool, True = occupied
    distance_field: np.ndarray  # (ny, nx) float, meters to nearest obstacle

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def museum(cls):
        w = config.WORLD
        res = w.resolution
        nx = int(round(w.width / res))
        ny = int(round(w.height / res))
        grid = np.zeros((ny, nx), dtype=bool)

        def fill_rect(x0, y0, x1, y1):
            ix0, iy0 = int(x0 / res), int(y0 / res)
            ix1, iy1 = int(np.ceil(x1 / res)), int(np.ceil(y1 / res))
            grid[iy0:iy1, ix0:ix1] = True

        def fill_circle(cx, cy, r):
            xs = (np.arange(nx) + 0.5) * res
            ys = (np.arange(ny) + 0.5) * res
            inside = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2 <= r**2
            np.logical_or(grid, inside, out=grid)

        for rect in _WALL_RECTS:
            fill_rect(*rect)
        for circ in _WALL_CIRCLES:
            fill_circle(*circ)

        # nearest-obstacle distance in meters (the likelihood field of 6.4)
        distance_field = distance_transform_edt(~grid, sampling=res)
        return cls(w.width, w.height, res, grid, distance_field)

    # ------------------------------------------------------------------
    # grid helpers
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.grid.shape

    def occupied_at(self, x, y):
        """Occupancy lookups for world-coordinate points (nearest cell)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ix = np.clip((x / self.res).astype(int), 0, self.shape[1] - 1)
        iy = np.clip((y / self.res).astype(int), 0, self.shape[0] - 1)
        return self.grid[iy, ix]

    def obstacle_distance(self, x, y):
        """Distance (meters) from world points to the nearest obstacle.

        Points outside the grid are "infinitely far" from every obstacle:
        a pose that projects beam endpoints outside the arena cannot be
        explained by the map at all.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ix = (x / self.res).astype(int)
        iy = (y / self.res).astype(int)
        inside = (
            (ix >= 0) & (ix < self.shape[1]) & (iy >= 0) & (iy < self.shape[0])
        )
        dist = np.full(ix.shape, 1e3)
        dist[inside] = self.distance_field[
            np.clip(iy[inside], 0, self.shape[0] - 1),
            np.clip(ix[inside], 0, self.shape[1] - 1),
        ]
        return dist

    def is_free(self, x, y, radius=0.0):
        """True where a disc of ``radius`` at (x, y) avoids all obstacles."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if radius <= 0.0:
            return ~self.occupied_at(x, y)
        # sample the disc boundary and center on the grid
        ang = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        free = ~self.occupied_at(x, y)
        for a in ang:
            free &= ~self.occupied_at(x + radius * np.cos(a), y + radius * np.sin(a))
        return free

    def sample_free_poses(self, n, rng=None, min_dist_to_wall=0.3):
        """Uniformly sampled poses (n, 3) in free space (uniform headings)."""
        rng = rng or make_rng()
        poses = np.empty((n, 3))
        filled = 0
        while filled < n:
            m = max(n - filled, 100)
            cand = rng.uniform(
                (0.0, 0.0, -np.pi), (self.width, self.height, np.pi), size=(m, 3)
            )
            ok = self.is_free(cand[:, 0], cand[:, 1], min_dist_to_wall)
            cand = cand[ok]
            take = min(len(cand), n - filled)
            poses[filled : filled + take] = cand[:take]
            filled += take
        return poses

    # ------------------------------------------------------------------
    # ray casting (vectorized Amanatides & Woo DDA)
    # ------------------------------------------------------------------

    def raycast(self, origins, angles, max_range):
        """Noise-free ranges along rays for the occupancy grid.

        Parameters
        ----------
        origins : (M, 2) array of ray start points (e.g. robot positions)
        angles  : (K,) beam directions shared by all origins, or (M, K)
                  per-origin beam directions
        max_range : rays that reach this far without hitting return it

        Returns
        -------
        (M, K) float array of ranges (in meters); a hit at the start cell
        returns 0.
        """
        origins = np.atleast_2d(np.asarray(origins, dtype=float))
        angles = np.asarray(angles, dtype=float)
        if angles.ndim == 1:
            # one shared beam set, broadcast to all origins
            ox = np.repeat(origins[:, 0], len(angles))
            oy = np.repeat(origins[:, 1], len(angles))
            th = np.tile(angles, len(origins))
        else:
            ox = np.repeat(origins[:, 0], angles.shape[1])
            oy = np.repeat(origins[:, 1], angles.shape[1])
            th = angles.ravel()
        return self._dda(ox, oy, th, float(max_range)).reshape(len(origins), -1)

    def _dda(self, ox, oy, th, max_range):
        ny, nx = self.shape
        res = self.res
        dx, dy = np.cos(th), np.sin(th)

        ix = np.clip((ox / res).astype(int), 0, nx - 1)
        iy = np.clip((oy / res).astype(int), 0, ny - 1)

        step_x = np.sign(dx).astype(int)
        step_y = np.sign(dy).astype(int)
        safe_dx = np.where(dx == 0.0, 1e-12, dx)
        safe_dy = np.where(dy == 0.0, 1e-12, dy)

        # distance along the ray to the first vertical/horizontal crossing,
        # and the distance between consecutive crossings
        next_boundary_x = (ix + (step_x > 0)) * res
        next_boundary_y = (iy + (step_y > 0)) * res
        t_max_x = (next_boundary_x - ox) / safe_dx
        t_max_y = (next_boundary_y - oy) / safe_dy
        t_delta_x = np.abs(res / safe_dx)
        t_delta_y = np.abs(res / safe_dy)

        ranges = np.full(len(ox), max_range)
        ranges[self.grid[iy, ix]] = 0.0
        active = ~self.grid[iy, ix]

        while active.any():
            hit_x = t_max_x < t_max_y
            # step into the next cell
            ix = np.where(hit_x, ix + step_x, ix)
            iy = np.where(hit_x, iy, iy + step_y)
            t = np.where(hit_x, t_max_x, t_max_y)
            t_max_x = np.where(hit_x, t_max_x + t_delta_x, t_max_x)
            t_max_y = np.where(hit_x, t_max_y, t_max_y + t_delta_y)

            inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            done = (~inside) | (t >= max_range)
            occupied = np.zeros_like(active)
            occupied[inside] = self.grid[
                np.clip(iy[inside], 0, ny - 1), np.clip(ix[inside], 0, nx - 1)
            ]

            newly_hit = active & ~done & occupied
            ranges[newly_hit] = t[newly_hit]
            active = active & ~done & ~occupied

        return ranges

