"""Task 4 -- EKF localization in the arena (chapter 7).

Implement ``ekf_predict`` and ``ekf_correct`` in ``src/ekf.py`` (Table 7.2)
first -- only then will this demo run.

    uv run task4.py                 live pygame arena
    uv run task4.py --headless      no GUI: one tour + error report/figure

The robot drives a weaving tour through the pillar fields while two EKF
localizers track it, both driven by the velocity motion model: one
corrected with the beam model, one with the likelihood field.  The panel
shows one at a time -- press 1/2 to toggle -- with its uncertainty
ellipse and trail, plus a cloud of velocity-model samples visualizing
the motion uncertainty feeding the filter.
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import config, viz
from ..ekf import EKFLocalizer
from ..robot import Robot
from ..tour import TourDriver
from ..world import World

BEAM_COLOR = (45, 212, 191)       # teal
FIELD_COLOR = (244, 114, 182)     # pink


def headless(seed, steps, save=True):
    world = World.museum()
    robot = Robot(world, start_pose=(5.6, 8.0, np.pi / 2), seed=seed)
    tour = TourDriver(world)
    beam = EKFLocalizer(world, sensor="beam", init_pose=robot.pose, seed=seed + 1)
    field = EKFLocalizer(world, sensor="likelihood", init_pose=robot.pose, seed=seed + 2)
    err_beam, err_field = [], []
    for t in range(steps):
        v, w = tour.control(robot)
        robot.step(v, w)
        beam.predict((v, w))
        field.predict((v, w))
        if t % config.EKF.update_every == 0:
            scan = robot.scan()
            beam.update(scan, robot.laser_angles)
            field.update(scan, robot.laser_angles)
        err_beam.append(beam.error(robot.pose)[0])
        err_field.append(field.error(robot.pose)[0])

    err_beam, err_field = np.asarray(err_beam), np.asarray(err_field)
    print(f"seed={seed}")
    print(f"{'sensor model':<16}{'update cost':>13}{'beams':>7}"
          f"{'final err':>11}{'mean err':>10}{'max err':>9}")
    for name, ekf, err in (("beam", beam, err_beam),
                           ("likelihood", field, err_field)):
        n = 0 if ekf.last_update_info is None else len(ekf.last_update_info["z"])
        print(f"{name:<16}{ekf.mean_update_ms:>10.2f} ms{n:>7}"
              f"{err[-1]:>9.3f} m{err[-200:].mean():>8.3f} m{err.max():>7.2f} m")
    print("the trade-off of chapter 6: the beam model simulates the physics")
    print("ray by ray (slow, sharp); the likelihood field trades physical")
    print("faithfulness for table lookups (fast, a little looser).")
    if save:
        viz.error_curve(
            [err_beam, err_field],
            [f"beam model ({beam.mean_update_ms:.1f} ms/update)",
             f"likelihood field ({field.mean_update_ms:.2f} ms/update)"],
            name="task4_error_ekf",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="simulation speed multiplier (live app)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI: run one guided tour and report errors")
    ap.add_argument("--steps", type=int, default=1400,
                    help="headless run length")
    ap.add_argument("--no-save", action="store_true",
                    help="headless mode: do not write the error figure")
    args = ap.parse_args()

    try:
        if args.headless:
            headless(args.seed, args.steps, save=not args.no_save)
            return
        from ..arena_app import ArenaApp
        app = ArenaApp(seed=args.seed, speed=args.speed)
        app.run()
    except NotImplementedError as exc:
        print(f"\nnot implemented yet: {exc}")
        print("task 4 needs the EKF in src/ekf.py (Table 7.2) -- and the")
        print("task 1 velocity sampler it reuses. Complete the TODOs and rerun.")
        print("(see README.md for the task map)")


if __name__ == "__main__":
    main()
