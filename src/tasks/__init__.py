"""Runnable demos for the four tasks (provided)."""

from __future__ import annotations


def run(main_fn):
    """Run a demo, translating a missing TODO into a friendly message."""
    try:
        main_fn()
    except NotImplementedError as exc:
        print(f"\nnot implemented yet: {exc}")
        print("open the file named in the message, complete the TODO block, and rerun.")
        print("(see README.md for the task map)")
