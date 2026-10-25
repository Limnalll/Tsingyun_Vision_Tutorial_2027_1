"""Central place for every tunable number in the project.

Feel free to play with these once the tasks are done -- but the unit tests
assume the defaults, so change them back before running ``uv run pytest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldConfig:
    width: float = 16.0            # meters
    height: float = 12.0           # meters
    resolution: float = 0.05       # grid cell size, meters


WORLD = WorldConfig()

# --------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotConfig:
    radius: float = 0.18           # meters
    max_speed: float = 1.2         # m/s
    max_turn_rate: float = 2.0     # rad/s
    # How faithfully the (simulated) hardware executes a velocity command.
    # The *modelled* noise lives in the motion models' alphas below.
    exec_noise_v: float = 0.02     # std of translational execution error
    exec_noise_w: float = 0.05     # std of rotational execution error


ROBOT = RobotConfig()

# --------------------------------------------------------------------------
# Motion models (chapter 5)
# --------------------------------------------------------------------------

# Time a control is held constant; the filter runs at this period.
DT = 0.1

# Velocity motion model noise, alpha1..alpha6 (Table 5.1 / 5.3).
VELOCITY_ALPHAS = (0.08, 0.05, 0.015, 0.03, 0.02, 0.05)

# Odometry motion model noise, alpha1..alpha4 (Table 5.5 / 5.6).
ODOMETRY_ALPHAS = (0.01, 0.05, 0.01, 0.05)

# --------------------------------------------------------------------------
# Laser range finder (chapter 6, beam model)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaserConfig:
    num_beams: int = 61            # beams per full scan
    fov: float = 3.141592653589793 # field of view in radians (180 degrees)
    z_max: float = 8.0             # maximum range in meters

    # Intrinsic parameters of the *simulated* sensor.  Ground truth for the
    # EM exercise (task 2.2); the localization demos default to these.
    z_hit: float = 0.80
    z_short: float = 0.07
    z_max_weight: float = 0.04     # mixture weight of max-range failures
    z_rand: float = 0.09
    sigma_hit: float = 0.06        # meters
    lambda_short: float = 0.35     # 1/meters


LASER = LaserConfig()

# Where the EM exercise writes / the demos read learned parameters.
EM_DATASET_PATH = DATA_DIR / "em_dataset.npz"
LEARNED_PARAMS_PATH = DATA_DIR / "beam_params_learned.json"

# --------------------------------------------------------------------------
# EKF localization (chapter 7, task 4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EKFConfig:
    init_std: tuple = (0.10, 0.10, 0.05)   # start covariance std, (x, y, theta)
    update_every: int = 2                  # measurement update cadence, steps
    beam_gate: float = 0.5                 # range-innovation gate, meters
    field_gate: float = 0.4                # endpoint-distance gate, meters
    beam_subsample: int = 4                # beams per beam-model update
    beam_noise_var: float = 0.04           # inflated per-beam noise, m^2
    beam_jacobian_gate: float = 15.0       # drop beams with wild Jacobians


EKF = EKFConfig()
