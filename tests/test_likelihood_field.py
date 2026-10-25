"""Task 3 tests: the likelihood field model (section 6.4)."""

import numpy as np
import pytest

from src.likelihood_field import LF_PARAMS, likelihood_field_model
from src.robot import Robot
from src.world import World


@pytest.fixture(scope="module")
def world():
    return World.museum()


@pytest.fixture(scope="module")
def robot(world):
    return Robot(world, start_pose=(5.6, 8.0, 0.9), seed=11)


class TestLikelihoodField:
    def test_true_pose_beats_random_poses(self, world, robot):
        scan = robot.scan()
        randoms = world.sample_free_poses(300, np.random.default_rng(1))
        p_true = likelihood_field_model(scan, robot.pose, robot.laser_angles, world, LF_PARAMS)
        p_rand = likelihood_field_model(scan, randoms, robot.laser_angles, world, LF_PARAMS)
        assert p_true > 1e3 * p_rand.max()

    def test_true_pose_is_local_maximum(self, world, robot):
        scan = robot.scan()
        offsets = np.array([[d * np.cos(a), d * np.sin(a), 0.0]
                            for d in (0.2, 0.4) for a in np.linspace(0, 2 * np.pi, 8, False)])
        poses = robot.pose + offsets
        p_true = likelihood_field_model(scan, robot.pose, robot.laser_angles, world, LF_PARAMS)
        p_off = likelihood_field_model(scan, poses, robot.laser_angles, world, LF_PARAMS)
        assert np.all(p_true > p_off), (
            "shifting the true pose by 20-40 cm must lower the scan likelihood"
        )

    def test_max_range_beams_are_ignored(self, world, robot):
        """A scan of pure max-range readings cannot discriminate poses.

        Table 6.3 skips beams equal to z_max entirely, so every pose
        scores exactly the same (the empty product, 1).
        """
        scan = np.full_like(robot.scan(), LF_PARAMS.z_max_range)
        randoms = world.sample_free_poses(20, np.random.default_rng(2))
        ps = likelihood_field_model(scan, randoms, robot.laser_angles, world, LF_PARAMS)
        assert np.allclose(ps, 1.0)

    def test_endpoint_projection(self, world, robot):
        """One beam pointing straight at a nearby wall must score at its endpoint."""
        pose = np.array([8.0, 4.0, 0.0])  # east of the central block at x=7.6..8.4
        # beam along -x (west): the block's west face is 0.4 m away
        scan = np.array([0.4])
        angles = np.array([np.pi])
        p_perfect = likelihood_field_model(scan, pose, angles, world, LF_PARAMS)
        p_short = likelihood_field_model(np.array([0.2]), pose, angles, world, LF_PARAMS)
        assert p_perfect > p_short
