#!/usr/bin/env python3
"""Convenience wrapper -- equivalent to `uv run python -m src.tasks.task1_motion`."""

from src.tasks import run
from src.tasks.task1_motion import main

if __name__ == "__main__":
    run(main)
