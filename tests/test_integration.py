"""End-to-end checks: the EKF localizers track the tour with your models.

This is the project's acceptance test -- everything you built in tasks
1-4 working together, exactly like `uv run task4.py --headless`.
"""

import numpy as np

from src import config
from src.ekf import EKFLocalizer
from src.robot import Robot
from src.tour import TOUR, TourDriver
from src.world import World


def run_ekf(sensor, steps=900, seed=10):
    world = World.museum()
    robot = Robot(world, start_pose=(5.6, 8.0, np.pi / 2), seed=seed)
    tour = TourDriver(world)
    ekf = EKFLocalizer(world, sensor=sensor, init_pose=robot.pose, seed=seed + 1)
    errors = []
    for t in range(steps):
        v, w = tour.control(robot)
        robot.step(v, w)
        ekf.predict((v, w))
        if t % config.EKF.update_every == 0:
            ekf.update(robot.scan(), robot.laser_angles)
        errors.append(ekf.error(robot.pose)[0])
    return np.asarray(errors)


def test_beam_ekf_tracks_the_tour():
    errors = run_ekf("beam")
    assert errors[-200:].mean() < 0.3, (
        f"beam EKF mean error was {errors[-200:].mean():.2f} m; the physically "
        "faithful model should track to a few centimeters"
    )


def test_likelihood_ekf_tracks_the_tour():
    errors = run_ekf("likelihood")
    assert errors[-200:].mean() < 0.6, (
        f"likelihood-field EKF mean error was {errors[-200:].mean():.2f} m"
    )


def test_tour_recovers_from_kidnap():
    """The demo autopilot must drive back to the tour from anywhere.

    (This tests provided infrastructure -- the planner-based tour
    driver -- so it passes even before any TODO is implemented.)
    """
    world = World.museum()
    for tp in world.sample_free_poses(8, np.random.default_rng(2026), 0.25):
        robot = Robot(world, start_pose=tuple(tp), seed=1)
        driver = TourDriver(world)
        d = [np.hypot(p[0] - tp[0], p[1] - tp[1]) for p in TOUR]
        driver.wp = int(np.argmin(d))
        reached = False
        for _ in range(900):
            v, w = driver.control(robot)
            robot.step(v, w)
            tg = TOUR[driver.wp]
            if np.hypot(robot.pose[0] - tg[0], robot.pose[1] - tg[1]) < 0.5:
                reached = True
                break
        assert reached, f"robot teleported to {tp[:2]} never reached the tour"
