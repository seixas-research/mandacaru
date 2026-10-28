"""Python array interface to native screened exchange-hole evaluation.

The Rust engine integrates the exchange hole with erfc screening and returns
analytic derivatives of the same quadrature. The full hybrid also requires
its short-range exact-exchange operator in the electronic solver.
"""
from __future__ import annotations

import numpy as np
from .._native import Context

OMEGA = 0.11
EXACT_FRACTION = 0.25


def short_range_partials(rho, sigma, omega=OMEGA):
    """Return exchange energy density, d/drho and d/dsigma (atomic units)."""
    rho, sigma = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(sigma, dtype=float))
    values = np.ascontiguousarray(np.stack((rho, sigma), axis=-1).reshape(-1, 2))
    result = Context(threads=1).exchange_hole(values, omega=omega)
    return tuple(result[:, i].reshape(rho.shape) for i in range(3))
