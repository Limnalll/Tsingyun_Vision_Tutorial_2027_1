"""Task 1 demo -- the two probabilistic motion models of chapter 5.

Run with::

    uv run task1.py        (or: uv run python -m src.tasks.task1_motion)

Draws, for both the velocity and the odometry model, 500-pose sample
clouds -- plain, and conditioned on the arena map (figures 5.4, 5.9 and
5.11 of the book).  Figures are written into ``results/``.
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import config, viz
from ..motion_models import (
    sample_odometry_motion_model,
    sample_odometry_motion_model_with_map,
    sample_velocity_motion_model,
    sample_velocity_motion_model_with_map,
)
from ..utils import make_rng
from ..world import World

N_SAMPLES = 500


def velocity_part(world, rng):
    x0 = (6.2, 4.2, 0.25)
    u = (2.0, 0.55)
    dt = 1.0
    alphas = config.VELOCITY_ALPHAS

    starts = np.tile(x0, (N_SAMPLES, 1))
    samples = sample_velocity_motion_model(u, starts, alphas, dt, rng)
    samples_map = sample_velocity_motion_model_with_map(
        u, starts, world, alphas, dt, rng, clearance=0.2
    )
    viz.sample_clouds(
        world,
        [samples, samples_map],
        ["velocity model: p(x_t | u_t, x_{t-1})",
         "conditioned on the map: p(x_t | u_t, x_{t-1}, m)"],
        path=np.array([x0[:2], samples.mean(axis=0)[:2]]),
        name="task1_velocity_clouds",
    )
    print("velocity model: mean sample offset from the start pose:",
          np.round(samples.mean(axis=0)[:2] - np.array(x0[:2]), 3))


def odometry_part(world, rng):
    x0 = (9.0, 8.3, 0.0)
    # a measured relative motion: turn, drive north through a pillar
    odom_pair = np.array([
        [0.0, 0.0, 0.0],
        [1.4 * np.cos(np.pi / 2), 1.4 * np.sin(np.pi / 2), np.pi / 2],
    ])
    alphas = config.ODOMETRY_ALPHAS

    starts = np.tile(x0, (N_SAMPLES, 1))
    samples = sample_odometry_motion_model(odom_pair, starts, alphas, rng)
    samples_map = sample_odometry_motion_model_with_map(
        odom_pair, starts, world, alphas, rng, clearance=0.2
    )
    viz.sample_clouds(
        world,
        [samples, samples_map],
        ["odometry model: p(x_t | u_t, x_{t-1})",
         "conditioned on the map: p(x_t | u_t, x_{t-1}, m)"],
        path=np.array([x0[:2], samples.mean(axis=0)[:2]]),
        name="task1_odometry_clouds",
    )
    print("odometry model: mean sample offset from the start pose:",
          np.round(samples.mean(axis=0)[:2] - np.array(x0[:2]), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = make_rng(args.seed)
    world = World.museum()
    print("== task 1: motion models ==")
    velocity_part(world, rng)
    odometry_part(world, rng)
    print("figures written to results/ -- done.")


if __name__ == "__main__":
    main()
