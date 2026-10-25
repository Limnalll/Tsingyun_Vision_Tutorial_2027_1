"""Generate the calibration dataset for the EM exercise (task 2).

Records pairs ``(z, z*)`` of measured and noise-free ranges collected from
many poses in the museum, exactly like the data in figure 6.4 of the book.
Run with::

    uv run python -m src.scripts.make_em_dataset
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..robot import sample_noisy_ranges
from ..utils import make_rng
from ..world import World

NUM_POSES = 300
SEED = 20260101


def main():
    rng = make_rng(SEED)
    world = World.museum()
    laser = config.LASER
    angles = np.linspace(-laser.fov / 2.0, laser.fov / 2.0, laser.num_beams)

    poses = world.sample_free_poses(NUM_POSES, make_rng(SEED + 1), min_dist_to_wall=0.5)
    z_star = world.raycast(poses[:, :2], angles, laser.z_max)

    z = np.empty_like(z_star)
    for i in range(len(poses)):
        z[i] = sample_noisy_ranges(z_star[i], rng)

    config.DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        config.EM_DATASET_PATH,
        z=z.ravel().astype(np.float32),
        z_star=z_star.ravel().astype(np.float32),
        z_max_range=laser.z_max,
        seed=SEED,
    )
    n = z.size
    print(f"wrote {config.EM_DATASET_PATH.name}: {n} measurement pairs "
          f"({NUM_POSES} poses x {laser.num_beams} beams)")
    print("  fraction of max-range readings: %.3f" % (z >= laser.z_max).mean())
    print("  fraction of short readings:     %.3f" % (z < z_star * 0.95).mean())


if __name__ == "__main__":
    main()
