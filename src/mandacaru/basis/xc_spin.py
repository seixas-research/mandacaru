"""Array interface to Rust spin-resolved exchange-correlation kernels.

Point energies and every density, gradient and kinetic-density derivative are
computed by the native engine. Python only broadcasts and reshapes arrays.
The HSE intermediate uses full-range hole exchange; solvers explicitly add
short-range semilocal subtraction and screened exact exchange.
"""
from __future__ import annotations

import numpy as np

from .._native import Context

FUNCTIONALS = ("lda", "pbe", "r2scan", "hse06")


def spin_partials(functional, rho_up, rho_dn, sigma_uu=None, sigma_ud=None,
                  sigma_dd=None, tau_up=None, tau_dn=None, *, relativistic=False):
    """Return energy density and seven partials, in atomic units."""
    if functional not in FUNCTIONALS:
        raise ValueError(f"expected canonical functional in {FUNCTIONALS}")
    arrays = np.broadcast_arrays(*(np.asarray(0. if x is None else x, dtype=float)
                                  for x in (rho_up, rho_dn, sigma_uu, sigma_ud,
                                            sigma_dd, tau_up, tau_dn)))
    shape = arrays[0].shape
    variables = np.ascontiguousarray(np.stack(arrays, axis=-1).reshape(-1, 7))
    context = Context(threads=1)
    if functional == "hse06":
        if relativistic:
            raise ValueError("hse06 has no relativistic exchange form")
        result = context.hybrid_points(variables, exact_fraction=0.)
    else:
        result = context.xc_points(functional, variables, relativistic=relativistic)
    return tuple(result[:, i].reshape(shape) for i in range(8))


def unpolarized_partials(functional, rho, sigma, tau):
    """Energy and density/sigma/tau derivatives at equal spin populations."""
    up, contraction, kinetic = np.broadcast_arrays(np.asarray(rho, dtype=float)*.5,
                                                   np.asarray(sigma, dtype=float)*.25,
                                                   np.asarray(tau, dtype=float)*.5)
    f, vu, vd, suu, sud, sdd, tu, td = spin_partials(
        functional, up, up, contraction, contraction, contraction, kinetic, kinetic)
    return f, .5*(vu+vd), .25*(suu+sud+sdd), .5*(tu+td)
