"""Task 2 tests: the beam model mixture and EM calibration (chapter 6.3)."""

import numpy as np
import pytest

from src import config
from src.beam_model import BeamParams, learn_intrinsic_parameters, p_beam

TRUTH = BeamParams.from_laser()


class TestBeamDensity:
    def test_mixture_is_normalized(self):
        """Integral over [0, z_max] = z_hit + z_short + z_rand (max is a point mass)."""
        z = np.linspace(0.0, TRUTH.z_max_range, 400_001)
        for z_star in (0.5, 2.0, 5.0):
            total = np.trapezoid(p_beam(z, z_star, TRUTH), z)
            expected = TRUTH.z_hit + TRUTH.z_short + TRUTH.z_rand
            assert total == pytest.approx(expected, abs=0.02)

    def test_point_mass_at_max_range(self):
        assert p_beam(TRUTH.z_max_range, 2.0, TRUTH) >= TRUTH.z_max
        assert p_beam(TRUTH.z_max_range - 0.01, 2.0, TRUTH) < TRUTH.z_max

    def test_density_zero_outside_support(self):
        # no mechanism can produce a reading beyond z_max
        assert p_beam(TRUTH.z_max_range + 0.5, 2.0, TRUTH) == 0.0

    def test_reading_beyond_true_range_has_no_short_mass(self):
        """Beyond z* only the hit, max, and random terms remain."""
        from src.utils import gaussian_cdf

        z, z_star = 3.0, 2.0
        s = TRUTH.sigma_hit
        eta = 1.0 / (gaussian_cdf((TRUTH.z_max_range - z_star) / s)
                     - gaussian_cdf(-z_star / s))
        hit = TRUTH.z_hit * eta / (s * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * (z - z_star) ** 2 / s**2)
        expected = hit + TRUTH.z_rand / TRUTH.z_max_range
        assert p_beam(z, z_star, TRUTH) == pytest.approx(expected, rel=0.05)

    def test_hit_gaussian_value(self):
        """At z = z* (interior), the density must be dominated by the hit term."""
        val = p_beam(2.0, 2.0, TRUTH)
        hit_only = TRUTH.z_hit / (TRUTH.sigma_hit * np.sqrt(2 * np.pi))
        assert val == pytest.approx(hit_only + TRUTH.z_rand / TRUTH.z_max_range, rel=0.05)


class TestEMCalibration:
    def test_em_recovers_intrinsic_parameters(self):
        data = np.load(config.EM_DATASET_PATH)
        z, z_star = data["z"].astype(float), data["z_star"].astype(float)
        init = BeamParams(z_hit=0.5, z_short=0.2, z_max=0.1, z_rand=0.2,
                          sigma_hit=0.1, lambda_short=0.6)
        learned, history = learn_intrinsic_parameters(z, z_star, init, max_iter=80)

        assert abs(learned.z_hit - TRUTH.z_hit) < 0.05
        assert abs(learned.z_short - TRUTH.z_short) < 0.03
        assert abs(learned.z_max - TRUTH.z_max) < 0.03
        assert abs(learned.z_rand - TRUTH.z_rand) < 0.03
        assert abs(learned.sigma_hit - TRUTH.sigma_hit) < 0.01
        assert 0.2 < learned.lambda_short < 0.6
        assert abs(learned.weights.sum() - 1.0) < 1e-6
        # log-likelihood must be (at least) as good as under the true params
        ll_true = np.sum(np.log(p_beam(z, z_star, TRUTH)))
        assert history["loglik"][-1] >= ll_true - 1e-6
