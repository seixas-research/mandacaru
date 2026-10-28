"""Array interface to Rust rVV10 nonlocal correlation.

The native engine owns the interpolation mesh, FFT workspace, analytic kernel,
energy and variational derivatives. Input and output fields use atomic units.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .._native import Context, SpatialGrid

C_VV10 = 0.0093
B_PBE = 6.3
B_R2SCAN = 11.95
Q_MIN = 1e-4
Q_CUT = .5
N_Q = 16
DENSITY_CUTOFF = 1e-12
ISOLATED_PADDING = 2


@dataclass(frozen=True)
class NonlocalTerms:
    energy: float
    d_density: np.ndarray
    d_sigma: np.ndarray


def nonlocal_correlation(grid, density, sigma, *, b=B_R2SCAN, C=C_VV10,
                         q_min=Q_MIN, q_cut=Q_CUT, n_q=N_Q):
    """Compute the nonlocal energy and its density/sigma partials in Rust."""
    native = SpatialGrid(list(grid.shape), [0., 0., 0.],
                         np.ascontiguousarray(grid.step, dtype=float),
                         periodic=bool(getattr(grid, "periodic", False)))
    result = Context(threads=1, memory_limit_bytes=2 * 1024**3).nonlocal_correlation(
        native, np.ascontiguousarray(density, dtype=float).ravel(),
        np.ascontiguousarray(sigma, dtype=float).ravel(), b=b, c=C,
        q_min=q_min, q_cut=q_cut, nodes=n_q, padding=ISOLATED_PADDING)
    return NonlocalTerms(**result)
