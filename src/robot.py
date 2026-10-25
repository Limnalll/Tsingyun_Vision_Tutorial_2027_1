"""The simulated robot and its sensors (provided).

``Robot`` is a differential-drive platform with three sensor interfaces:

* :meth:`Robot.scan` -- a 2-D laser range finder.  Its measurements follow
  the beam model mixture of section 6.3 *physically*: every beam picks one
  of the four error mechanisms and is generated accordingly.  This is the
  "real world" your models have to explain.
* :meth:`Robot.odometry_pair` -- the last two poses in the robot-internal
  odometry frame, the control ``u_t = (xbar_{t-1}, xbar_t)`` of chapter 5.4.

The true world pose stays hidden from the filters -- access it only when
you are explicitly evaluating your localization accuracy.
"""

from __future__ import annotations

import numpy as np

from . import config
from .utils import make_rng, wrap_angle


def relative_pose(a, b):
    """Pose of ``b`` expressed in ``a``'s coordinate frame."""
    ax, ay, ath = a
    bx, by, bth = b
    dx, dy = bx - ax, by - ay
    c, s = np.cos(ath), np.sin(ath)
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap_angle(bth - ath)])


def sample_noisy_ranges(z_star, rng):
    """Physically sample laser ranges from true ranges (the "real world").

    Independent of the beam model you implement in ``src.beam_model`` --
    this is what actually generates the data your model has to explain.
    Samples are drawn from exactly the distributions the beam model
    describes: Gaussian hits truncated to ``[0, z_max]``, truncated
    exponential short readings, max-range failures, and uniform noise.
    """
    from .utils import gaussian_cdf, inv_gaussian_cdf

    lc = config.LASER
    z_star = np.asarray(z_star, dtype=float)
    z = rng.uniform(0.0, 1.0, len(z_star))
    out = np.empty_like(z_star)

    # correct hit: Gaussian truncated to [0, z_max] via inverse CDF
    m = z < lc.z_hit
    s = lc.sigma_hit
    lo = gaussian_cdf((0.0 - z_star[m]) / s)
    hi = gaussian_cdf((lc.z_max - z_star[m]) / s)
    u = rng.uniform(0.0, 1.0, int(m.sum()))
    out[m] = z_star[m] + s * inv_gaussian_cdf(lo + u * (hi - lo))

    # unexpected obstacle: exponential truncated to [0, z*]
    m2 = (z >= lc.z_hit) & (z < lc.z_hit + lc.z_short)
    u = rng.uniform(0.0, 1.0, int(m2.sum()))
    out[m2] = -np.log(1.0 - u * (1.0 - np.exp(-lc.lambda_short * z_star[m2]))) / lc.lambda_short

    # sensor failure: maximum range
    m3 = (z >= lc.z_hit + lc.z_short) & (z < lc.z_hit + lc.z_short + lc.z_max_weight)
    out[m3] = lc.z_max

    # random measurement: uniform over [0, z_max)
    m4 = ~(m | m2 | m3)
    out[m4] = rng.uniform(0.0, lc.z_max, int(m4.sum()))
    return out


class Robot:
    """Unicycle robot with execution noise, odometry, laser, and camera."""

    def __init__(self, world, start_pose=(8.0, 2.0, np.pi / 2), seed=None):
        from .world import World  # noqa: F401  (type hint only)

        self.world = world
        self.rng = make_rng(seed)
        rc = config.ROBOT
        lc = config.LASER

        self.pose = np.array(start_pose, dtype=float)   # true pose
        self.radius = rc.radius
        self._start_pose = self.pose.copy()
        self._prev_odom_pose = self.odom_pose()
        self.odom_pose_history = [self._prev_odom_pose.copy()]

        self.laser_angles = np.linspace(-lc.fov / 2.0, lc.fov / 2.0, lc.num_beams)
        self.z_max = lc.z_max

    # ------------------------------------------------------------------
    # motion
    # ------------------------------------------------------------------

    def step(self, v_cmd, w_cmd, dt=config.DT):
        """Execute a velocity command for ``dt`` seconds."""
        rc = config.ROBOT
        v = float(np.clip(v_cmd, -rc.max_speed, rc.max_speed))
        w = float(np.clip(w_cmd, -rc.max_turn_rate, rc.max_turn_rate))
        v_exec = v + self.rng.normal(0.0, rc.exec_noise_v)
        w_exec = w + self.rng.normal(0.0, rc.exec_noise_w)

        x, y, th = self.pose
        if abs(w_exec) < 1e-6:
            nx, ny = x + v_exec * dt * np.cos(th), y + v_exec * dt * np.sin(th)
        else:
            r = v_exec / w_exec
            nx = x - r * np.sin(th) + r * np.sin(th + w_exec * dt)
            ny = y + r * np.cos(th) - r * np.cos(th + w_exec * dt)
        nth = th + w_exec * dt

        # collision: slide along walls, never through them
        if self.world.is_free(nx, ny, self.radius):
            self.pose = np.array([nx, ny, nth])
        elif self.world.is_free(nx, y, self.radius):
            self.pose = np.array([nx, y, nth])
        elif self.world.is_free(x, ny, self.radius):
            self.pose = np.array([x, ny, nth])
        else:
            self.pose = np.array([x, y, nth])

        self._prev_odom_pose = self.odom_pose_history[-1]
        self.odom_pose_history.append(self.odom_pose())

    # ------------------------------------------------------------------
    # odometry (control u_t of the odometry motion model, eq. 5.33)
    # ------------------------------------------------------------------

    def odom_pose(self):
        """Current pose in the robot-internal odometry frame."""
        return relative_pose(self._start_pose, self.pose)

    def odometry_pair(self):
        """``u_t = (xbar_{t-1}, xbar_t)`` -- the two most recent odometry poses."""
        return np.stack([self._prev_odom_pose, self.odom_pose_history[-1]])

    # ------------------------------------------------------------------
    # laser range finder
    # ------------------------------------------------------------------

    def true_ranges(self):
        """Noise-free ranges ``z*`` for the current pose (for evaluation only)."""
        x, y, th = self.pose
        return self.world.raycast(
            [[x, y]], th + self.laser_angles, self.z_max
        )[0]

    def scan(self):
        """A noisy laser scan: array of ``num_beams`` ranges in meters."""
        return sample_noisy_ranges(self.true_ranges(), self.rng)
