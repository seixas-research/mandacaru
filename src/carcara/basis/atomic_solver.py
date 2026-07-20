# -*- coding: utf-8 -*-
# file: basis/atomic_solver.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Self-consistent all-electron radial atom (spherical LDA).

This is the reference calculation a norm-conserving pseudopotential is built
*from*: it supplies the all-electron valence orbitals :math:`R_{nl}(r)`, their
eigenvalues :math:`\varepsilon_{nl}`, and the screening (Hartree + exchange
-correlation) potential that has to be removed again when the pseudopotential is
"unscreened".

The atom is treated as spherically symmetric, so the Kohn-Sham problem collapses
to a set of one-dimensional radial equations for :math:`u_{nl} = r R_{nl}`,

.. math::

    -\tfrac12 u'' + \Big[\frac{l(l+1)}{2r^2} + V_{\text{eff}}(r)\Big] u
        = \varepsilon\, u ,
    \qquad
    V_{\text{eff}} = -\frac{Z}{r} + V_H[\rho] + V_{xc}[\rho],

solved on a uniform radial grid by a tridiagonal eigensolve and iterated to
self-consistency with linear density mixing.

Exchange-correlation is the local density approximation: Slater exchange plus the
Perdew-Zunger (1981) parameterization of the Ceperley-Alder correlation energy.
That is the standard choice for generating pseudopotentials, and it keeps this
module free of any external data -- consistent with Carcará generating every
basis from scratch.

.. note::

   A uniform grid (rather than the logarithmic grid atomic codes usually use) is
   deliberate: it keeps the eigenproblem a plain symmetric tridiagonal matrix, so
   :func:`scipy.linalg.eigh_tridiagonal` solves it directly with no shooting or
   node counting by hand.  The cost is more points -- resolving a :math:`1s`
   orbital of scale :math:`a_0/Z` needs a fine spacing -- but the solve is
   one-dimensional and takes milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh_tridiagonal

from ._config import ground_state_config

#: Default radial grid: points and outer radius (Bohr).
DEFAULT_POINTS = 4000
DEFAULT_R_MAX = 25.0


# --------------------------------------------------------------------------- #
# Exchange-correlation (LDA).
# --------------------------------------------------------------------------- #

def lda_exchange(rho: np.ndarray):
    r"""Slater exchange: returns ``(energy density e_x, potential v_x)``.

    :math:`e_x = -\tfrac34 (3/\pi)^{1/3}\rho^{1/3}` per electron and
    :math:`v_x = \tfrac43 e_x`.
    """
    rho = np.maximum(rho, 1e-30)
    ex = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    return ex, (4.0 / 3.0) * ex


def lda_correlation(rho: np.ndarray):
    r"""Perdew-Zunger (1981) correlation: returns ``(e_c, v_c)``.

    The standard parameterization of the Ceperley-Alder uniform-electron-gas
    correlation energy, in its unpolarized form, split at :math:`r_s = 1`.
    """
    rho = np.maximum(rho, 1e-30)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

    # High-density (rs < 1) logarithmic form.
    a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
    log_rs = np.log(rs)
    ec_high = a * log_rs + b + c * rs * log_rs + d * rs
    vc_high = (a * log_rs + (b - a / 3.0)
               + (2.0 / 3.0) * c * rs * log_rs
               + (2.0 * d - c) * rs / 3.0)

    # Low-density (rs >= 1) Pade form.
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    sqrt_rs = np.sqrt(rs)
    denom = 1.0 + beta1 * sqrt_rs + beta2 * rs
    ec_low = gamma / denom
    vc_low = ec_low * (1.0 + (7.0 / 6.0) * beta1 * sqrt_rs
                       + (4.0 / 3.0) * beta2 * rs) / denom

    high = rs < 1.0
    return (np.where(high, ec_high, ec_low),
            np.where(high, vc_high, vc_low))


def lda_xc(rho: np.ndarray):
    """Total LDA exchange-correlation ``(e_xc, v_xc)`` (Hartree)."""
    ex, vx = lda_exchange(rho)
    ec, vc = lda_correlation(rho)
    return ex + ec, vx + vc


# --------------------------------------------------------------------------- #
# Radial solves.
# --------------------------------------------------------------------------- #

def solve_radial(r: np.ndarray, potential: np.ndarray, l: int, n_nodes: int):
    r"""Bound state of ``-1/2 u'' + [l(l+1)/2r^2 + V] u = eps u``.

    Parameters
    ----------
    r : ndarray
        Uniform radial grid, strictly positive and equally spaced.
    potential : ndarray
        :math:`V_{\text{eff}}(r)` on that grid (Hartree).
    l : int
        Angular momentum.
    n_nodes : int
        Radial nodes wanted, ``n - l - 1``; selects which eigenvalue to return.

    Returns
    -------
    (u, eps) : (ndarray, float)
        :math:`u = rR` normalized so ``int u^2 dr = 1``, and the eigenvalue.
    """
    step = float(r[1] - r[0])
    diag = 1.0 / step ** 2 + potential + l * (l + 1) / (2.0 * r * r)
    offdiag = -0.5 / step ** 2 * np.ones(r.size - 1)
    values, vectors = eigh_tridiagonal(diag, offdiag, select="i",
                                       select_range=(n_nodes, n_nodes))
    u = vectors[:, 0]
    u = u / np.sqrt(np.trapezoid(u * u, r))
    if u[0] < 0:                                # fix the global sign
        u = -u
    return u, float(values[0])


def hartree_potential(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r"""Radial Hartree potential of a spherical density.

    .. math::

        V_H(r) = \frac{4\pi}{r}\int_0^r \rho(r')r'^2\,dr'
                 + 4\pi\int_r^\infty \rho(r')r'\,dr' .
    """
    from scipy.integrate import cumulative_trapezoid

    inner = cumulative_trapezoid(rho * r * r, r, initial=0.0)
    outer_total = np.trapezoid(rho * r, r)
    outer = outer_total - cumulative_trapezoid(rho * r, r, initial=0.0)
    return 4.0 * np.pi * (inner / r + outer)


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #

@dataclass
class AtomicResult:
    """Converged all-electron atom."""

    atomic_number: int
    r: np.ndarray                       # radial grid (Bohr)
    orbitals: dict                      # (n, l) -> u_nl = r * R_nl
    eigenvalues: dict                   # (n, l) -> Hartree
    occupations: dict                   # (n, l) -> electrons
    density: np.ndarray                 # rho(r), spherical
    v_effective: np.ndarray             # -Z/r + V_H + V_xc
    v_hartree: np.ndarray
    v_xc: np.ndarray
    total_energy: float = 0.0
    converged: bool = False
    iterations: int = 0
    details: dict = field(default_factory=dict)

    def radial(self, n: int, l: int) -> np.ndarray:
        """``R_nl(r) = u_nl / r``."""
        return self.orbitals[(n, l)] / self.r

    @property
    def n_electrons(self) -> int:
        return int(round(sum(self.occupations.values())))

    def __repr__(self) -> str:
        return (f"AtomicResult(Z={self.atomic_number}, "
                f"{len(self.orbitals)} subshells, "
                f"E={self.total_energy:.6f} Ha, converged={self.converged})")


# --------------------------------------------------------------------------- #
# Self-consistent field.
# --------------------------------------------------------------------------- #

def solve_atom(atomic_number: int, *, points: int = DEFAULT_POINTS,
               r_max: float = DEFAULT_R_MAX, max_iterations: int = 200,
               tolerance: float = 1e-6, mixing: float = 0.3,
               configuration=None) -> AtomicResult:
    r"""Self-consistent spherical LDA atom.

    Parameters
    ----------
    atomic_number : int
        Nuclear charge :math:`Z`.
    points, r_max : int, float
        Uniform radial grid: ``points`` nodes out to ``r_max`` Bohr.  The default
        resolves the :math:`1s` shell of the first two rows; heavier atoms want
        more points.
    mixing : float
        Linear density-mixing fraction.  Small values are slower but stable.
    configuration : dict, optional
        ``{(n, l): occupancy}``.  Defaults to the aufbau ground state.

    Returns
    -------
    AtomicResult
        Converged orbitals, eigenvalues, density and potentials.
    """
    Z = int(atomic_number)
    occupations = dict(configuration if configuration is not None
                       else ground_state_config(Z))

    step = r_max / (points + 1)
    r = np.arange(1, points + 1) * step
    nuclear = -Z / r

    # Thomas-Fermi-like starting density: a screened exponential holding Z
    # electrons is close enough for the mixing to take over.
    scale = max(Z ** (1.0 / 3.0), 1.0)
    density = Z * (scale ** 3 / np.pi) * np.exp(-2.0 * scale * r)
    density *= Z / max(np.trapezoid(4.0 * np.pi * density * r * r, r), 1e-30)

    orbitals: dict = {}
    eigenvalues: dict = {}
    v_hartree = np.zeros_like(r)
    v_xc = np.zeros_like(r)
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        v_hartree = hartree_potential(r, density)
        _e_xc, v_xc = lda_xc(density)
        v_effective = nuclear + v_hartree + v_xc

        new_density = np.zeros_like(r)
        for (n, l), occupancy in occupations.items():
            if occupancy <= 0:
                continue
            u, eps = solve_radial(r, v_effective, l, n - l - 1)
            orbitals[(n, l)] = u
            eigenvalues[(n, l)] = eps
            new_density += occupancy * u * u / (4.0 * np.pi * r * r)

        change = float(np.max(np.abs(new_density - density)))
        density = (1.0 - mixing) * density + mixing * new_density
        if change < tolerance:
            converged = True
            break

    # Final potentials consistent with the converged density.
    v_hartree = hartree_potential(r, density)
    e_xc, v_xc = lda_xc(density)
    v_effective = nuclear + v_hartree + v_xc

    # Total energy from the eigenvalue sum, correcting the double counting.
    band = sum(occupations[k] * eigenvalues[k] for k in eigenvalues)
    hartree_energy = 0.5 * np.trapezoid(v_hartree * density * 4.0 * np.pi * r * r, r)
    xc_energy = np.trapezoid(e_xc * density * 4.0 * np.pi * r * r, r)
    xc_potential_energy = np.trapezoid(v_xc * density * 4.0 * np.pi * r * r, r)
    total = band - hartree_energy + xc_energy - xc_potential_energy

    return AtomicResult(
        atomic_number=Z, r=r, orbitals=orbitals, eigenvalues=eigenvalues,
        occupations=occupations, density=density, v_effective=v_effective,
        v_hartree=v_hartree, v_xc=v_xc, total_energy=float(total),
        converged=converged, iterations=iteration,
        details={"points": points, "r_max": r_max, "mixing": mixing})
