# -*- coding: utf-8 -*-
# file: basis/sto_ng.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Native STO-nG Gaussian basis, generated from scratch (no tabulated data).

A minimal Slater-type orbital (STO) for a subshell ``(n, l)`` has the radial form

.. math::

    S_{nl}(r) \propto r^{\,n-1}\, e^{-\zeta r}, \qquad \zeta = \frac{Z - s}{n^*},

with the screening ``s`` and effective principal quantum number ``n^*`` from
Slater's rules (:meth:`~carcara.basis.HydrogenicOrbital.slater_effective_charge`
supplies ``Z - s``).  The **STO-nG** approximation replaces that Slater orbital
by a fixed contraction of ``n`` Gaussians,

.. math::

    S_{nl}(r) \approx \sum_{i=1}^{n} c_i\, g_i(r), \qquad
    g_i(r) = N(\alpha_i, l)\, r^l e^{-\alpha_i r^2},

whose exponents ``{\alpha_i}`` and coefficients ``{c_i}`` are obtained here by a
least-squares fit of the (radially ``r^2 dr``-weighted) Gaussian expansion to the
Slater orbital -- i.e. computed on the fly rather than read from a basis-set
file.  Because the fit is scale-covariant, it is performed once at the reference
exponent ``\zeta = 1`` (cached per ``(n, l, n)``) and the resulting exponents are
rescaled by ``\zeta^2`` for the actual atom, leaving the coefficients unchanged.

The fit reproduces the standard published STO-nG contractions closely (e.g. the
STO-3G hydrogen 1s exponents ``2.22766, 0.40577, 0.10982`` to five figures).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma, pbdv

from ._config import ground_state_config
from .hydrogenic import HydrogenicOrbital

# Effective principal quantum number n* from Slater's rules.
_N_STAR = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2}


def effective_principal_number(n: int) -> float:
    """Slater's effective principal quantum number ``n*`` for shell ``n``."""
    try:
        return _N_STAR[int(n)]
    except KeyError:
        raise ValueError(
            f"no Slater n* for principal quantum number {n}; "
            f"STO-nG is defined here for n = 1..6") from None


def slater_exponent(atomic_number: int, n: int, l: int) -> float:
    r"""Slater orbital exponent ``zeta = (Z - s) / n*`` for subshell ``(n, l)``."""
    z_eff = HydrogenicOrbital.slater_effective_charge(atomic_number, n, l)
    return z_eff / effective_principal_number(n)


def occupied_subshells(atomic_number: int) -> list[tuple[int, int]]:
    """All occupied ``(n, l)`` subshells (core + valence), a minimal basis set."""
    return sorted(ground_state_config(atomic_number))


# --------------------------------------------------------------------------- #
# The reference (zeta = 1) least-squares fit.
# --------------------------------------------------------------------------- #

def _primitive_overlap(alpha: np.ndarray, l: int) -> np.ndarray:
    """Overlap ``<g_i|g_j>`` of unit-normalized primitives ``r^l e^{-a r^2}``."""
    L = l + 1.5
    return (2.0 * np.sqrt(np.outer(alpha, alpha))
            / (alpha[:, None] + alpha[None, :])) ** L


def _slater_moment(alpha: np.ndarray, q: int) -> np.ndarray:
    r"""``\int_0^\infty r^q e^{-\alpha r^2 - r} dr`` in closed form.

    Gradshteyn-Ryzhik 3.462.1 with ``\nu = q + 1``, ``\beta = 1``: the integral
    equals ``(2\alpha)^{-\nu/2} \Gamma(\nu) e^{1/(8\alpha)} D_{-\nu}(1/\sqrt{2\alpha})``
    with ``D`` the parabolic-cylinder function.
    """
    nu = q + 1.0
    z = 1.0 / np.sqrt(2.0 * alpha)
    D = np.array([pbdv(-nu, zi)[0] for zi in np.atleast_1d(z)])
    return (2.0 * alpha) ** (-nu / 2.0) * gamma(nu) * np.exp(1.0 / (8.0 * alpha)) * D


def _slater_projection(alpha: np.ndarray, n: int, l: int) -> np.ndarray:
    r"""Overlaps ``<g_i | S_{nl}>`` of primitives with the (zeta=1) Slater orbital."""
    L = l + 1.5
    norm = np.sqrt(2.0 * (2.0 * alpha) ** L / gamma(L))           # N(alpha, l)
    sto_norm = np.sqrt(2.0 ** (2 * n + 1) / gamma(2 * n + 1))     # ||S_{nl}|| = 1
    integ = _slater_moment(alpha, n + l + 1)                      # r^{(n-1)+l+2}
    return norm * sto_norm * integ


def _neg_overlap_sq(log_alpha: np.ndarray, n: int, l: int) -> float:
    """``-<S|P_g|S>``: minus the squared overlap recovered by the fit."""
    alpha = np.exp(log_alpha)
    S = _primitive_overlap(alpha, l)
    b = _slater_projection(alpha, n, l)
    try:
        c = np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        return 1.0
    return -float(b @ c)


@lru_cache(maxsize=None)
def _fit_reference(n: int, l: int, n_gaussians: int):
    """Fit ``n_gaussians`` Gaussians to the zeta=1 Slater orbital ``S_{nl}``.

    Returns ``(exponents, coefficients, overlap_sq)`` with exponents sorted
    tightest-first; deterministic (multi-start Nelder-Mead, no RNG) and cached.
    """
    if n_gaussians < 1:
        raise ValueError("n_gaussians must be >= 1")
    centre = 1.0 / max(n - 1, 1) ** 2
    idx = np.arange(n_gaussians) - (n_gaussians - 1) / 2.0
    best = None
    for spread in (1.6, 2.2, 3.0):                 # a few geometric start widths
        t0 = np.log(centre * spread ** idx)
        res = minimize(_neg_overlap_sq, t0, args=(n, l), method="Nelder-Mead",
                       options=dict(xatol=1e-9, fatol=1e-12, maxiter=20000))
        if best is None or res.fun < best.fun:
            best = res
    alpha = np.exp(best.x)
    c = np.linalg.solve(_primitive_overlap(alpha, l),
                        _slater_projection(alpha, n, l))
    order = np.argsort(alpha)[::-1]
    return tuple(alpha[order]), tuple(c[order]), -float(best.fun)


def sto_ng_contraction(atomic_number: int, n: int, l: int,
                       n_gaussians: int = 3):
    """Exponents and coefficients of the STO-nG contraction for ``(Z, n, l)``.

    The cached reference fit (``zeta = 1``) is rescaled by ``zeta^2`` for the
    atom's Slater exponent; coefficients (for unit-normalized primitives) are
    scale-invariant and returned unchanged.
    """
    alpha0, coeff, _ = _fit_reference(int(n), int(l), int(n_gaussians))
    zeta = slater_exponent(atomic_number, n, l)
    return np.array(alpha0) * zeta ** 2, np.array(coeff)


def sto_ng_shells(atomic_number: int, n_gaussians: int = 3):
    """STO-nG contractions for every occupied subshell of the atom.

    Returns a list of ``(n, l, exponents, coefficients)``, one per occupied
    ``(n, l)`` (core and valence), ordered by ``(n, l)``.
    """
    shells = []
    for (n, l) in occupied_subshells(atomic_number):
        exps, coeffs = sto_ng_contraction(atomic_number, n, l, n_gaussians)
        shells.append((n, l, exps, coeffs))
    return shells
