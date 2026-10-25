"""Task 2 demo -- beam model calibration with EM.

Run with::

    uv run task2.py        (or: uv run python -m src.tasks.task2_beam)

Recovers the intrinsic parameters of the beam model from the calibration
data in ``data/em_dataset.npz`` (Table 6.2), prints them against the
ground truth, writes the convergence figure into ``results/``, and
stores the learned parameters in ``data/beam_params_learned.json`` for
the task 4 demo to pick up.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config, viz
from ..beam_model import BeamParams, learn_intrinsic_parameters

EM_INIT = BeamParams(
    z_hit=0.5, z_short=0.2, z_max=0.1, z_rand=0.2,
    sigma_hit=0.1, lambda_short=0.6,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=60, help="max EM iterations")
    args = ap.parse_args()

    truth = BeamParams.from_laser()
    print("== task 2: beam model + EM ==")

    data = np.load(config.EM_DATASET_PATH)
    z, z_star = data["z"].astype(float), data["z_star"].astype(float)
    print(f"calibration set: {len(z)} measured/expected range pairs")

    print(f"EM from a naive start: {EM_INIT}")
    learned, history = learn_intrinsic_parameters(z, z_star, EM_INIT, max_iter=args.iters)
    rows = [
        ("z_hit", truth.z_hit, learned.z_hit),
        ("z_short", truth.z_short, learned.z_short),
        ("z_max", truth.z_max, learned.z_max),
        ("z_rand", truth.z_rand, learned.z_rand),
        ("sigma_hit", truth.sigma_hit, learned.sigma_hit),
        ("lambda_short", truth.lambda_short, learned.lambda_short),
    ]
    print(f"{'parameter':<14}{'truth':>8}{'learned':>9}")
    for name, t, l in rows:
        print(f"{name:<14}{t:>8.3f}{l:>9.3f}")
    viz.em_convergence(history, truth, name="task2_em_convergence")

    config.DATA_DIR.mkdir(exist_ok=True)
    payload = {k: round(float(v), 5) for k, v in zip(
        ["z_hit", "z_short", "z_max", "z_rand", "sigma_hit", "lambda_short"],
        [learned.z_hit, learned.z_short, learned.z_max, learned.z_rand,
         learned.sigma_hit, learned.lambda_short])}
    payload["z_max_range"] = float(truth.z_max_range)
    config.LEARNED_PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"learned parameters written to data/{config.LEARNED_PARAMS_PATH.name}")
    print("figure written to results/ -- done.")


if __name__ == "__main__":
    main()
