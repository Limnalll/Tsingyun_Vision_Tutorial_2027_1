"""Task 4 tests: the EKF localization steps (chapter 7, Table 7.2)."""

import numpy as np
import pytest

from src import config
from src.ekf import ekf_correct, ekf_predict

MU = np.array([5.0, 6.0, 0.7])
SIGMA = np.diag([0.01, 0.01, 0.005])


class TestPredict:
    def test_mean_matches_the_noise_free_sampler(self):
        """The propagated mean is exactly the task-1.2 sampler with zero
        noise draws (Table 7.2, line 2)."""
        from src.motion_models import sample_velocity_motion_model

        u = (1.0, 0.5)
        mu_bar, _ = ekf_predict(MU, SIGMA, u, config.VELOCITY_ALPHAS, dt=0.1)
        expected = sample_velocity_motion_model(u, MU, (0,) * 6, 0.1,
                                                np.random.default_rng(0))
        assert np.allclose(mu_bar, expected, atol=1e-9)

    def test_straight_line_mean(self):
        u = (1.0, 0.0)
        mu_bar, _ = ekf_predict(MU, SIGMA, u, config.VELOCITY_ALPHAS, dt=0.1)
        assert np.allclose(mu_bar, [MU[0] + 0.1 * np.cos(MU[2]),
                                    MU[1] + 0.1 * np.sin(MU[2]),
                                    MU[2]], atol=1e-9)

    def test_covariance_grows_and_stays_psd(self):
        u = (1.0, 0.5)
        _, Sigma_bar = ekf_predict(MU, SIGMA, u, config.VELOCITY_ALPHAS, dt=0.1)
        assert np.trace(Sigma_bar) > np.trace(SIGMA), (
            "prediction must add motion uncertainty"
        )
        assert np.all(np.linalg.eigvalsh(Sigma_bar) > -1e-9)

    def test_no_noise_leaves_covariance_rotation_only(self):
        """With zero alphas, R_t vanishes and Sigma_bar = G Sigma G^T:
        the trace stays (approximately) the same and the result is PSD."""
        u = (1.0, 0.5)
        _, Sigma_bar = ekf_predict(MU, SIGMA, u, (0,) * 6, dt=0.1)
        assert np.trace(Sigma_bar) == pytest.approx(np.trace(SIGMA), rel=0.05)
        assert np.all(np.linalg.eigvalsh(Sigma_bar) > -1e-9)

    def test_faster_motion_adds_more_uncertainty(self):
        """The alphas scale with |v| and |omega| -- a faster command must
        add more motion noise (chapter 5's noise model in R_t).  Start
        from a near-zero covariance so the added noise dominates."""
        tiny = np.diag([1e-8, 1e-8, 1e-8])
        u_slow, u_fast = (0.3, 0.2), (2.0, 1.0)
        _, s_slow = ekf_predict(MU, tiny, u_slow, config.VELOCITY_ALPHAS, dt=0.1)
        _, s_fast = ekf_predict(MU, tiny, u_fast, config.VELOCITY_ALPHAS, dt=0.1)
        assert np.trace(s_fast) > 3 * np.trace(s_slow)


class TestCorrect:
    def test_perfect_measurement_leaves_mean(self):
        mu_bar = np.array([1.0, 2.0, 0.3])
        Sigma_bar = np.diag([0.5, 0.5, 0.1])
        mu, Sigma = ekf_correct(mu_bar, Sigma_bar,
                                np.zeros(2), np.array([[1.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0]]),
                                np.diag([0.01, 0.01]))
        assert np.allclose(mu, mu_bar)
        assert np.all(np.linalg.eigvalsh(Sigma) > 0)

    def test_correction_moves_toward_the_measurement(self):
        """A position measurement (H = identity on x, y) must pull the
        mean toward the implied pose."""
        mu_bar = np.array([0.0, 0.0, 0.0])
        Sigma_bar = np.diag([1.0, 1.0, 0.5])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        residual = np.array([1.0, 0.5])
        mu, Sigma = ekf_correct(mu_bar, Sigma_bar, residual, H, np.diag([0.01, 0.01]))
        assert mu[0] > 0.9 and 0.4 < mu[1] < 0.6
        assert np.trace(Sigma) < np.trace(Sigma_bar), (
            "an informative measurement must shrink the covariance"
        )

    def test_gain_respects_noise(self):
        """Very noisy measurements barely move the estimate."""
        mu_bar = np.array([0.0, 0.0, 0.0])
        Sigma_bar = np.diag([1.0, 1.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0]])
        mu, _ = ekf_correct(mu_bar, Sigma_bar, [1.0], H, np.array([[100.0]]))
        assert 0.0 < mu[0] < 0.05
