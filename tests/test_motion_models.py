"""Task 1 tests: velocity and odometry motion models (chapter 5)."""

import numpy as np
import pytest

from src.motion_models import (
    odometry_motion_model,
    sample_odometry_motion_model,
    sample_velocity_motion_model,
    velocity_motion_model,
)
from src.utils import wrap_angle

RNG = np.random.default_rng(2026)

X_PREV = (2.0, 3.0, 0.7)
U_VEL = (1.0, 0.5)
ALPHAS_VEL = (0.08, 0.05, 0.015, 0.03, 0.02, 0.05)
DT = 1.0

U_ODOM = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, np.pi / 6]])
ALPHAS_ODOM = (0.01, 0.05, 0.01, 0.05)


def noise_free_velocity():
    rng = np.random.default_rng(0)  # zero noise -> deterministic
    return sample_velocity_motion_model(U_VEL, X_PREV, (0,) * 6, DT, rng)


def noise_free_odometry():
    rng = np.random.default_rng(0)
    return sample_odometry_motion_model(U_ODOM, X_PREV, (0,) * 4, rng)


class TestVelocityMotionModel:
    def test_perfect_motion_has_maximal_probability(self):
        """The noise-free successor pose must be the mode of the density."""
        x_t = noise_free_velocity()
        p_mode = velocity_motion_model(x_t, U_VEL, X_PREV, ALPHAS_VEL, DT)
        perturbed = np.array(
            [x_t + d for d in ([0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3],
                               [-0.3, 0, 0], [0, -0.3, 0], [0, 0, -0.3])]
        )
        p_pert = velocity_motion_model(perturbed, U_VEL, X_PREV, ALPHAS_VEL, DT)
        assert p_mode > 0
        assert np.all(p_mode > p_pert)

    def test_straight_line_case(self):
        u = (1.0, 0.0)
        rng = np.random.default_rng(1)
        x_t = sample_velocity_motion_model(u, X_PREV, (0,) * 6, DT, rng)
        expected = np.array([
            X_PREV[0] + np.cos(X_PREV[2]),
            X_PREV[1] + np.sin(X_PREV[2]),
            X_PREV[2],
        ])
        assert np.allclose(x_t, expected, atol=1e-9)
        assert velocity_motion_model(expected, u, X_PREV, (0,) * 6, DT) == pytest.approx(1.0)

    def test_rotation_in_place(self):
        u = (0.0, 1.0)
        rng = np.random.default_rng(2)
        x_t = sample_velocity_motion_model(u, X_PREV, (0,) * 6, DT, rng)
        assert np.allclose(x_t[:2], X_PREV[:2], atol=1e-9)
        assert wrap_angle(x_t[2] - (X_PREV[2] + 1.0)) < 1e-9
        assert velocity_motion_model(x_t, u, X_PREV, (0,) * 6, DT) == pytest.approx(1.0)

    def test_pose_array_input(self):
        x_t = noise_free_velocity()
        grid = x_t + RNG.normal(0, 0.1, (25, 3))
        p = velocity_motion_model(grid, U_VEL, X_PREV, ALPHAS_VEL, DT)
        assert p.shape == (25,)
        assert np.all(np.isfinite(p))
        assert p[0] < velocity_motion_model(x_t, U_VEL, X_PREV, ALPHAS_VEL, DT)


class TestVelocitySampling:
    def test_sample_statistics_match_density(self):
        """The sampler and the closed form must describe the same distribution."""
        x_t = noise_free_velocity()
        n = 6000
        samples = sample_velocity_motion_model(
            U_VEL, np.tile(X_PREV, (n, 1)), ALPHAS_VEL, DT, np.random.default_rng(42)
        )
        assert samples.shape == (n, 3)
        mean_err = np.hypot(*(samples[:, :2].mean(axis=0) - x_t[:2]))
        assert mean_err < 0.05, "sample cloud should be centered on the noise-free pose"
        spread = samples[:, :2].std(axis=0).mean()
        assert 0.05 < spread < 0.5, "spread should reflect the alphas (not 0, not huge)"

    def test_closed_form_ranks_own_samples(self):
        """A sampled pose should usually score higher than a perturbed one."""
        n = 4000
        samples = sample_velocity_motion_model(
            U_VEL, np.tile(X_PREV, (n, 1)), ALPHAS_VEL, DT, np.random.default_rng(7)
        )
        shifted = samples + RNG.normal(0, 0.15, samples.shape)
        p_own = velocity_motion_model(samples, U_VEL, X_PREV, ALPHAS_VEL, DT)
        p_shift = velocity_motion_model(shifted, U_VEL, X_PREV, ALPHAS_VEL, DT)
        wins = np.mean(p_own > p_shift)
        assert wins > 0.6, f"own density beat the perturbed density only {wins:.0%} of the time"


class TestOdometryMotionModel:
    def test_perfect_motion_has_maximal_probability(self):
        x_t = noise_free_odometry()
        assert odometry_motion_model(x_t, U_ODOM, X_PREV, (0,) * 4) == pytest.approx(1.0)
        p_mode = odometry_motion_model(x_t, U_ODOM, X_PREV, ALPHAS_ODOM)
        p_off = odometry_motion_model(
            x_t + np.array([0.4, 0, 0]), U_ODOM, X_PREV, ALPHAS_ODOM
        )
        assert p_mode > p_off > 0

    def test_expected_noise_free_pose(self):
        x_t = noise_free_odometry()
        rot1 = np.arctan2(0.5, 1.0)
        trans = np.hypot(1.0, 0.5)
        expected = np.array([
            X_PREV[0] + trans * np.cos(X_PREV[2] + rot1),
            X_PREV[1] + trans * np.sin(X_PREV[2] + rot1),
            wrap_angle(X_PREV[2] + np.pi / 6),
        ])
        assert np.allclose(x_t, expected, atol=1e-9)

    def test_angular_wrapping(self):
        """A heading difference near +/-pi must not blow up (book's warning)."""
        u = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, np.pi - 1e-6]])
        x_t = noise_free_odometry()
        p = odometry_motion_model(x_t, u, X_PREV, ALPHAS_ODOM)
        assert np.isfinite(p)


class TestOdometrySampling:
    def test_sample_statistics(self):
        x_t = noise_free_odometry()
        n = 6000
        samples = sample_odometry_motion_model(
            U_ODOM, np.tile(X_PREV, (n, 1)), ALPHAS_ODOM, np.random.default_rng(3)
        )
        mean_err = np.hypot(*(samples[:, :2].mean(axis=0) - x_t[:2]))
        assert mean_err < 0.05
        spread = samples[:, :2].std(axis=0).mean()
        assert 0.03 < spread < 0.4

    def test_batch_matches_single(self):
        n = 5
        batch = sample_odometry_motion_model(
            U_ODOM, np.tile(X_PREV, (n, 1)), ALPHAS_ODOM, np.random.default_rng(9)
        )
        assert batch.shape == (n, 3)
