"""Small shared helpers (provided, not part of the TODOs).

The two noise helpers implement Table 5.2 and Table 5.4 of the book:

* ``prob_normal(a, b)`` is the density of a zero-mean normal distribution
  with variance ``b`` evaluated at ``a`` (the ``prob`` function used by the
  closed-form motion and measurement models).
* ``sample_normal(b)`` draws from a zero-mean normal distribution with
  variance ``b``.  Table 5.4 approximates this with a sum of twelve uniform
  random numbers; the CLT guarantees that approximation converges to exactly
  this distribution, so here we sample it directly.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_angle(a):
    """Wrap an angle (or array of angles) to the interval (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % TWO_PI - np.pi


def prob_normal(a, b):
    """Density of a zero-mean normal with variance ``b`` at ``a``.

    Implements ``prob normal distribution`` from Table 5.2.  Works on
    scalars and numpy arrays.  A variance of exactly zero is treated as a
    Dirac delta: the probability is 1 where ``a == 0`` and 0 elsewhere.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros(np.broadcast(a, b).shape)
    nz = b > 0.0
    out[nz] = np.exp(-0.5 * np.square(a[nz]) / b[nz]) / np.sqrt(TWO_PI * b[nz])
    out[~nz & (a == 0)] = 1.0
    return out if out.ndim else float(out)


def sample_normal(b, rng):
    """Draw from a zero-mean normal distribution with variance ``b``.

    Implements the ``sample`` function used throughout chapter 5 (Table 5.4
    gives the twelve-uniforms approximation of exactly this distribution).
    ``b`` may be an array; the returned samples match its shape.
    """
    b = np.asarray(b, dtype=float)
    return rng.normal(0.0, np.sqrt(b), size=b.shape)


def gaussian_cdf(x):
    """Cumulative distribution function of the standard normal."""
    from scipy.special import erf

    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def inv_gaussian_cdf(p):
    """Inverse of :func:`gaussian_cdf` (the probit function)."""
    from scipy.special import ndtri

    return ndtri(np.asarray(p, dtype=float))


def make_rng(seed=None):
    """A single place to create numpy random generators."""
    return np.random.default_rng(seed)
