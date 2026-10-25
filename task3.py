#!/usr/bin/env python3
"""Convenience wrapper -- equivalent to `uv run python -m src.tasks.task3_fields`."""

from src.tasks import run
from src.tasks.task3_fields import main

if __name__ == "__main__":
    run(main)
