# -*- coding: utf-8 -*-
# file: basis/atomic_solver.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Self-consistent all-electron radial atom: the Python interface of the native solver.

The atom is the reference calculation a pseudopotential or a numerical-orbital basis is
built *from*: the all-electron orbitals :math:`u_{nl} = rR_{nl}`, their eigenvalues, the
density and the screening potentials.  Every number is computed in Rust
(``mandacaru-core``, ``models/atom.rs``): the spherical Kohn-Sham problem on a uniform radial
grid (a tridiagonal eigensolve per subshell), Pulay mixing in the radial metric, LDA or PBE,
and the nonrelativistic, scalar-relativistic (Koelling-Harmon) or Dirac radial equation.
A relativistic atom finishes on the fourth-order Numerov shoot the pseudopotential
generators use, so the atom and the pseudization agree on where a relativistic s level is.

This module only validates and converts arguments, calls the extension and holds the
result (:class:`AtomicResult`); the configuration rule (aufbau unless an ``f`` valence shell
is unbound) is documented at :func:`relaxed_configuration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Default radial grid: points and outer radius (Bohr).
DEFAULT_POINTS = 4000
DEFAULT_R_MAX = 25.0


def _context(resources=None):
    from .. import _native

    return _native.Context(**dict(resources or {}))


def lda_exchange(rho):
    r"""Slater exchange: ``(energy per electron e_x, potential v_x)`` of density ``rho``."""
    terms = _context().lda_terms(np.ascontiguousarray(rho, dtype=float).ravel())
    shape = np.shape(rho)
    return terms[0].reshape(shape), terms[1].reshape(shape)


def lda_correlation(rho):
    r"""Perdew-Zunger (1981) correlation: ``(e_c, v_c)`` of density ``rho``."""
    terms = _context().lda_terms(np.ascontiguousarray(rho, dtype=float).ravel())
    shape = np.shape(rho)
    return terms[2].reshape(shape), terms[3].reshape(shape)


def lda_xc(rho):
    """Total LDA exchange-correlation ``(e_xc, v_xc)`` (Hartree)."""
    terms = _context().lda_terms(np.ascontiguousarray(rho, dtype=float).ravel())
    shape = np.shape(rho)
    return (terms[0] + terms[2]).reshape(shape), (terms[1] + terms[3]).reshape(shape)


def solve_radial(r, potential, l: int, n_nodes: int):
    r"""Bound state of ``-1/2 u'' + [l(l+1)/2r^2 + V] u = eps u`` on a uniform grid.

    ``n_nodes`` (``n - l - 1``) selects the eigenvalue.  Returns ``(u, eps)`` with
    :math:`u = rR` normalized to ``int u^2 dr = 1`` and positive near the origin.
    """
    return _context().solve_radial(
        np.ascontiguousarray(r, dtype=float),
        np.ascontiguousarray(potential, dtype=float), int(l), int(n_nodes))


def hartree_potential(r, rho):
    r"""Radial Hartree potential of a spherical density on the grid ``r``."""
    return _context().radial_hartree(np.ascontiguousarray(r, dtype=float),
                                     np.ascontiguousarray(rho, dtype=float))


def relaxed_configuration(atomic_number: int, *, xc: str = "lda",
                          relativity: str = "none", r_max: float = DEFAULT_R_MAX,
                          points: int | None = None):
    r"""The reference configuration to solve this atom in: ``{(n, l): q}``.

    **The aufbau filling is used unless it produces an unbound valence state.**  That is the
    whole rule, deliberately narrower than "whichever configuration has the lowest energy":
    the margins by which LDA prefers chromium's ``3d^5 4s^1`` shrink toward zero with the
    grid, so this function declines to reproduce the Cr and Cu anomalies.  It repairs a
    sharp defect: under aufbau the ``f`` shell of La, Ac, Th and Pa can be a box state
    (positive, or negative and tiny with its maximum against the wall of the box), and a
    norm-conserving channel built from one is meaningless.  Moving one electron to the ``d``
    shell binds it, and the lowest-energy bound one-electron rearrangement is returned.

    A configuration with no occupied ``f`` valence shell returns immediately with no solve.
    ``points`` defaults to a floor that scales with :math:`Z` (12000 or ``300 Z``): a
    bound-state test on a coarse grid reads discretization error.
    """
    raw = _context().relaxed_configuration(
        int(atomic_number), functional=xc, relativity=str(relativity),
        r_max=float(r_max), points=None if points is None else int(points))
    return {(n, l): q for n, l, q in raw}


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
    #: ``(n, l, kappa) -> u``, filled only by a ``relativity="dirac"`` run.
    #: The spin-orbit-resolved counterparts of ``orbitals``/``eigenvalues``/
    #: ``occupations``, whose ``(n, l)`` entries then hold the ``(2j+1)``
    #: -weighted average -- so every consumer that predates spin-orbit
    #: coupling keeps reading the quantity it always read.
    orbitals_j: dict = field(default_factory=dict)
    eigenvalues_j: dict = field(default_factory=dict)
    occupations_j: dict = field(default_factory=dict)

    def radial(self, n: int, l: int) -> np.ndarray:
        """``R_nl(r) = u_nl / r``."""
        return self.orbitals[(n, l)] / self.r

    @property
    def relativity(self) -> str:
        """Which radial equation produced this atom."""
        return str(self.details.get("relativity", "none"))

    @property
    def xc(self) -> str:
        """Which exchange-correlation functional produced it."""
        return str(self.details.get("xc", "lda"))

    def spin_orbit_splitting(self, n: int, l: int) -> float:
        r""":math:`\varepsilon_{l+1/2} - \varepsilon_{l-1/2}` (Hartree).

        Zero for an s shell, and zero for any atom not solved with
        ``relativity="dirac"`` -- in which case it is zero because the
        equation has no spin-orbit term, not because the atom has no
        splitting.
        """
        if int(l) == 0 or not self.eigenvalues_j:
            return 0.0
        return float(self.eigenvalues_j[(n, l, -(l + 1))]
                     - self.eigenvalues_j[(n, l, l)])

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
               tolerance: float = 1e-6, mixing: float = 0.3, configuration=None,
               confinement=None, xc: str = "lda", relativity: str = "none",
               polish: int = 12, resources=None) -> AtomicResult:
    r"""Self-consistent spherical atom.

    Parameters
    ----------
    atomic_number : int
        Nuclear charge :math:`Z`.
    points, r_max : int, float
        Uniform radial grid: ``points`` nodes out to ``r_max`` Bohr.  The default
        resolves the :math:`1s` shell of the first two rows; heavier atoms want more points.
    mixing : float
        Fraction of the linear step taken by the Pulay mixer (and the whole step whenever
        the Pulay extrapolation is not a density).
    configuration : dict, optional
        ``{(n, l): occupancy}``.  Defaults to :func:`relaxed_configuration`: the aufbau
        ground state unless it leaves the valence unbound.
    confinement : callable or ndarray, optional
        An extra external potential (Hartree) added **only when integrating the orbitals**
        -- a confining wall for a localized basis (see :mod:`mandacaru.basis.nao_ae`).  A
        callable is evaluated on the radial grid; an array must already be on it.  It is
        *not* part of the returned ``v_effective`` / ``v_hartree`` / ``v_xc``, which remain
        the genuine self-consistent potentials of the (confined) density.
    xc : str
        ``"lda"`` (Slater + Perdew-Zunger) or ``"pbe"``.
    relativity : str
        ``"none"`` (the default), ``"scalar"`` (Koelling-Harmon: mass-velocity and Darwin,
        no spin-orbit) or ``"dirac"`` (each ``j`` solved separately).  A ``"dirac"`` atom
        additionally fills ``orbitals_j`` / ``eigenvalues_j`` / ``occupations_j``; its
        ``(n, l)`` entries carry the ``(2j+1)``-weighted average, so a consumer that does not
        know about spin-orbit coupling sees the scalar-relativistic atom.  Both relativistic
        treatments also use the relativistic correction to LDA/PBE exchange.
    polish : int
        Extra self-consistency steps a **relativistic** atom takes with the Numerov shoot
        the pseudopotential generators use, after the tridiagonal field has converged (0
        disables it; ignored for ``relativity="none"``).  Only the valence is re-solved.
        Without it the atom and the generators discretize a relativistic s state two
        different ways and disagree about where the level is (9.5 mHa for oxygen's 2s).
    resources : dict, optional
        Keyword arguments of the native ``Context`` (threads, memory).

    Returns
    -------
    AtomicResult
        Converged orbitals, eigenvalues, density and potentials.
    """
    if callable(confinement):
        confinement = confinement(
            np.arange(1, points + 1, dtype=float) * r_max / (points + 1))
    context = _context(resources)
    spec = (None if configuration is None else
            [(int(n), int(l), float(q)) for (n, l), q in configuration.items() if q > 0])
    raw = context.solve_atom(
        int(atomic_number), points=points, r_max=r_max, max_iterations=max_iterations,
        tolerance=tolerance, mixing=mixing, configuration=spec,
        confinement=None if confinement is None else np.ascontiguousarray(confinement, dtype=float),
        functional=xc, relativity=str(relativity), polish=int(polish))
    name, functional = raw.pop("relativity"), raw.pop("functional")
    return AtomicResult(**raw, details={
        "engine": "rust", "points": points, "r_max": r_max, "mixing": mixing,
        "confined": confinement is not None, "xc": functional,
        "relativity": name, "relativistic_exchange": name != "none",
        "memory_peak_bytes": context.memory_peak_bytes})
