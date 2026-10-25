"""Task 3 demo -- the pre-computed likelihood field.

Run with::

    uv run task3.py        (or: uv run python -m src.tasks.task3_fields)

Writes the arena's likelihood field (figure 6.9b of the book) into
``results/``.  Your ``likelihood_field_model`` from this task is not
needed to draw the field itself -- it is exercised by the tests and it
drives one of the two EKF localizers in the task 4 arena.
"""

from __future__ import annotations

import argparse

from .. import viz
from ..world import World


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    world = World.museum()
    print("== task 3: the likelihood field ==")
    viz.likelihood_field_map(world, name="task3_likelihood_field")
    print("dark = close to an obstacle: beam endpoints predicted from a")
    print("correct pose land on bright-to-dark boundaries; the task 4")
    print("likelihood-field EKF measures exactly this distance.")
    print("figure written to results/ -- done.")


if __name__ == "__main__":
    main()
