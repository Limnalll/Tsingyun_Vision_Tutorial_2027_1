"""Matplotlib figure helpers for the task demos (provided).

Every helper writes its figure into ``results/`` -- no interactive
windows, so the demos behave identically on machines without a display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config

# a small consistent style for all task figures
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "image.cmap": "magma",
})

WALL_GREY = "#d8d4cc"
INK = "#1b1e24"
ACCENT = "#e07b39"
BLUE = "#3f8fd2"


def _finish(fig, name):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    print(f"saved results/{name}.png")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# motion model figures (chapter 5)
# --------------------------------------------------------------------------


def _map_ax(ax, world):
    """Draw the arena as a light background."""
    if world is None:
        return
    occ = world.grid
    extent = (0.0, world.width, 0.0, world.height)
    ax.imshow(~occ, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1,
              alpha=0.94, interpolation="nearest")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")


def sample_clouds(world, sample_sets, titles, path=None, name="motion_clouds"):
    """Book-style sample clouds (figures 5.4, 5.9, 5.11): one panel per set."""
    n = len(sample_sets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.5, 3.6), squeeze=False)
    for ax, samples, title in zip(axes[0], sample_sets, titles):
        _map_ax(ax, world)
        if path is not None and len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], "-", color=ACCENT, lw=1.6, label="path")
            ax.plot(path[0, 0], path[0, 1], "o", color=ACCENT, ms=5)
        if samples is not None and len(samples):
            ax.plot(samples[:, 0], samples[:, 1], ".", color=BLUE, ms=2.5, alpha=0.8)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("pose samples from the motion model", y=0.99)
    fig.tight_layout()
    return _finish(fig, name)


# --------------------------------------------------------------------------
# beam model figures (chapter 6)
# --------------------------------------------------------------------------


def em_convergence(history, truth, name="em_convergence"):
    """Log-likelihood and parameter traces across EM iterations."""
    it = np.arange(len(history["loglik"]))
    names = ["z_hit", "z_short", "z_max", "z_rand", "sigma_hit", "lambda_short"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(it, history["loglik"], ".-", color=INK, ms=4)
    axes[0].set_xlabel("EM iteration"), axes[0].set_ylabel("log-likelihood")
    axes[0].set_title("EM convergence")
    for k, name_ in enumerate(names):
        vals = [getattr(p, name_) for p in history["params"]]
        axes[1].plot(it, vals, ".-", ms=4, label=name_)
        axes[1].axhline(getattr(truth, name_), ls=":", lw=1, color="grey")
    axes[1].set_xlabel("EM iteration")
    axes[1].set_title("intrinsic parameters (dotted = ground truth)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, name)


# --------------------------------------------------------------------------
# likelihood field (section 6.4)
# --------------------------------------------------------------------------


def likelihood_field_map(world, name="likelihood_field"):
    """The pre-computed likelihood field (figure 6.9b)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(world.distance_field, origin="lower",
                   extent=(0, world.width, 0, world.height), cmap="magma")
    plt.colorbar(im, ax=ax, label="distance to nearest obstacle [m]")
    ax.set_title("pre-computed likelihood field")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    return _finish(fig, name)


# --------------------------------------------------------------------------
# EKF localization (chapter 7)
# --------------------------------------------------------------------------


def error_curve(errors, labels, name="error_curve", threshold=0.5):
    """Localization error over time for one or more filters."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for e, lab in zip(errors, labels):
        ax.plot(np.arange(len(e)) * config.DT, e, lw=1.2, label=lab)
    ax.axhline(threshold, ls=":", color="grey", lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_ylim(0, None)
    ax.legend()
    ax.set_title("EKF localization error")
    fig.tight_layout()
    return _finish(fig, name)
