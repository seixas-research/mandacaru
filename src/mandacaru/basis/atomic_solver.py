# -*- coding: utf-8 -*-
# file: basis/atomic_solver.py

# This code is part of Mandacaru.
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
module free of any external data -- consistent with Mandacaru generating every
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

    def partition_density(self, valence):
        """``(core, valence)`` spherical densities for a valence selection.

        ``valence`` is an iterable of ``(n, l)`` subshells treated as valence;
        everything else in the configuration is core.  This is the split a
        nonlinear core correction is built from
        (:func:`mandacaru.pseudopotentials.oncv.partial_core_density`).
        """
        wanted = {(int(n), int(l)) for n, l in valence}
        core = np.zeros_like(self.r)
        outer = np.zeros_like(self.r)
        weight = 4.0 * np.pi * self.r * self.r
        if self.occupations_j:
            items = [((n, l), q, self.orbitals_j[(n, l, k)])
                     for (n, l, k), q in self.occupations_j.items() if q > 0]
        else:
            items = [((n, l), q, self.orbitals[(n, l)])
                     for (n, l), q in self.occupations.items() if q > 0]
        for key, occupancy, u in items:
            share = occupancy * u * u / weight
            if key in wanted:
                outer += share
            else:
                core += share
        return core, outer

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
               configuration=None, confinement=None,
               xc: str = "lda", relativity: str = "none",
               polish: int = 12,
               grid: str = "uniform") -> AtomicResult:
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
    confinement : callable or ndarray, optional
        An extra external potential (Hartree) added **only when integrating
        the orbitals** -- a confining wall for a localized basis (see
        :mod:`mandacaru.basis.nao_ae`).  A callable is evaluated on the radial
        grid; an array must already be on it.  It is *not* part of the
        returned ``v_effective`` / ``v_hartree`` / ``v_xc``, which remain the
        genuine self-consistent potentials of the (confined) density.
    xc : str
        ``"lda"`` (Slater + Perdew-Zunger, the default and the historical
        behavior) or ``"pbe"`` (:mod:`mandacaru.basis.xc`).
    relativity : str
        ``"none"`` (the default), ``"scalar"`` (Koelling-Harmon: mass-velocity
        and Darwin, no spin-orbit) or ``"dirac"`` (each ``j`` solved
        separately).  See :mod:`mandacaru.basis.relativity`.  A ``"dirac"``
        atom additionally fills ``orbitals_j`` / ``eigenvalues_j`` /
        ``occupations_j``; its ``(n, l)`` entries carry the ``(2j+1)``-weighted
        average, so a consumer that does not know about spin-orbit coupling
        sees the scalar-relativistic atom.
    polish : int
        Extra self-consistency steps a **relativistic** atom takes with the
        Numerov shoot the pseudopotential generators use, after the
        tridiagonal field has converged (0 disables it; ignored when
        ``relativity="none"``).  Only the valence is re-solved.  Without it
        the atom and the generators discretize a relativistic s state two
        different ways -- its :math:`r^\gamma` cusp is resolved well by
        neither -- and disagree about where the level is: 9.5 mHa for
        oxygen's 2s, which reaches the ionic potential as an 8e-4 Hartree
        error in its tail.
    grid : str
        ``"uniform"`` (the default) is the grid every consumer of this atom
        expects.  ``"log"`` solves each state on a logarithmic grid instead
        (:mod:`~mandacaru.basis.loggrid}`), which is the only way to converge a
        relativistic :math:`l=0` level, and is meant for **measuring** the
        uniform atom rather than for feeding a generator: the pseudopotential
        constructions need waves that satisfy the uniform-grid equation, which
        a splined log-grid wave does not.  ``polish`` is forced off with it --
        polishing re-solves the valence with the uniform Numerov shoot, so
        running both makes them fight over the valence levels and oxygen's
        relativistic total-energy shift comes out with the wrong sign.

    Returns
    -------
    AtomicResult
        Converged orbitals, eigenvalues, density and potentials.
    """
    # Imported here, not at module scope: mandacaru.basis.xc builds the LDA
    # branch of its dispatcher out of `lda_xc` above, so a top-level import
    # would close a cycle.
    from .relativity import (degeneracy, kappa_values,
                             solve_radial_relativistic)
    from .xc import xc_potential

    Z = int(atomic_number)
    occupations = dict(configuration if configuration is not None
                       else ground_state_config(Z))
    relativity = str(relativity).strip().lower()
    j_resolved = relativity in ("dirac", "full", "relativistic")
    if j_resolved:
        from ._config import split_configuration
        occupations_j = split_configuration(occupations)
    else:
        occupations_j = {}

    step = r_max / (points + 1)
    r = np.arange(1, points + 1) * step
    nuclear = -Z / r
    if confinement is None:
        wall = np.zeros_like(r)
    elif callable(confinement):
        wall = np.asarray(confinement(r), dtype=float)
    else:
        wall = np.asarray(confinement, dtype=float)
        if wall.shape != r.shape:
            raise ValueError("confinement array must match the radial grid")

    # Thomas-Fermi-like starting density: a screened exponential holding Z
    # electrons is close enough for the mixing to take over.
    scale = max(Z ** (1.0 / 3.0), 1.0)
    density = Z * (scale ** 3 / np.pi) * np.exp(-2.0 * scale * r)
    density *= Z / max(np.trapezoid(4.0 * np.pi * density * r * r, r), 1e-30)

    orbitals: dict = {}
    eigenvalues: dict = {}
    orbitals_j: dict = {}
    eigenvalues_j: dict = {}
    v_hartree = np.zeros_like(r)
    v_xc = np.zeros_like(r)
    converged = False
    iteration = 0
    shell = 4.0 * np.pi * r * r

    non_relativistic = relativity in ("none", "nonrelativistic",
                                      "non-relativistic", "nr")
    # V' and V'' do not depend on the energy, so they are built once per
    # potential and shared by every state and every M(eps) step.
    cache: dict = {"derivatives": None, "energy": {}}

    def solve_one(key, l, n_nodes, potential, kappa=None):
        """One radial state, at whichever level of theory was asked for."""
        if non_relativistic and grid == "uniform":
            return solve_radial(r, potential, l, n_nodes)
        u, eps = solve_radial_relativistic(
            r, potential, l, n_nodes, kappa=kappa,
            treatment=("dirac" if j_resolved
                       else ("none" if non_relativistic else relativity)),
            atomic_number=Z, energy_guess=cache["energy"].get(key),
            derivatives=cache["derivatives"], grid=grid)
        cache["energy"][key] = eps
        return u, eps

    for iteration in range(1, max_iterations + 1):
        v_hartree = hartree_potential(r, density)
        _e_xc, v_xc = xc_potential(r, density, xc)
        v_effective = nuclear + v_hartree + v_xc
        if not non_relativistic:
            from .relativity import potential_derivatives
            cache["derivatives"] = potential_derivatives(
                r, v_effective + wall, Z)

        new_density = np.zeros_like(r)
        if j_resolved:
            # Each j is its own radial equation; the (n, l) entries below are
            # the (2j+1) average, which is what the scalar-relativistic atom
            # would have produced.
            for (n, l), occupancy in occupations.items():
                if occupancy <= 0:
                    continue
                for k in kappa_values(l):
                    share = occupations_j[(n, l, k)]
                    u, eps = solve_one((n, l, k), l, n - l - 1,
                                       v_effective + wall, k)
                    orbitals_j[(n, l, k)] = u
                    eigenvalues_j[(n, l, k)] = eps
                    if share > 0:
                        new_density += share * u * u / shell
        else:
            for (n, l), occupancy in occupations.items():
                if occupancy <= 0:
                    continue
                u, eps = solve_one((n, l), l, n - l - 1,
                                   v_effective + wall)
                orbitals[(n, l)] = u
                eigenvalues[(n, l)] = eps
                new_density += occupancy * u * u / shell

        change = float(np.max(np.abs(new_density - density)))
        density = (1.0 - mixing) * density + mixing * new_density
        if change < tolerance:
            converged = True
            break

    if not non_relativistic and polish and grid == "uniform":
        # Finish on Numerov orbitals.  The tridiagonal solve above is second
        # order; the pseudopotential generators refine their partial waves
        # with a fourth-order Numerov shoot, and a relativistic s state --
        # whose r^gamma cusp neither discretization resolves well -- lands
        # 9.5 mHa apart in the two.  A pseudization whose waves are not
        # eigenstates of the potential it was given is not a small error: it
        # left the PAW-LCAO ionic potential 8e-4 Hartree off -Z_ion/r out to 11
        # Bohr.  Re-converging here with the same solver the generators use
        # makes the atom and the pseudization agree by construction.
        from ..pseudopotentials.oncv import bound_state

        from ._config import valence_subshells

        # Only the valence is re-solved.  The generators pseudize valence
        # states and nothing else; a core state contributes a *density*, its
        # tridiagonal one is already converged, and it is both the most
        # expensive to shoot (deep, many nodes) and the least likely to
        # bracket.  Re-solving the whole atom instead made the heavy end of
        # the library 39 hours of work rather than 5.
        wanted = {(int(n), int(l)) for n, l in valence_subshells(Z)}

        def is_valence(key):
            return (int(key[0]), int(key[1])) in wanted

        for _ in range(polish):
            v_hartree = hartree_potential(r, density)
            _e_xc, v_xc = xc_potential(r, density, xc)
            v_effective = nuclear + v_hartree + v_xc
            new_density = np.zeros_like(r)
            for key, occupancy in (occupations_j if j_resolved
                                   else occupations).items():
                if occupancy <= 0:
                    continue
                n, l = key[0], key[1]
                k = key[2] if j_resolved else None
                table = eigenvalues_j if j_resolved else eigenvalues
                store = orbitals_j if j_resolved else orbitals
                if is_valence(key):
                    try:
                        u, eps = bound_state(r, v_effective + wall, l,
                                             table[key], Z,
                                             treatment=relativity, kappa=k)
                        store[key], table[key] = u, eps
                    except RuntimeError:
                        # The shoot could not bracket it; its tridiagonal
                        # solution stands.
                        pass
                new_density += occupancy * store[key] ** 2 / shell
            change = float(np.max(np.abs(new_density - density)))
            density = (1.0 - mixing) * density + mixing * new_density
            if change < tolerance:
                break
        # One last solve in the potential that will actually be *reported*.
        # The loop above updates the density after its final solve, so
        # without this the returned eigenvalues belong to a potential one
        # step behind the returned one -- and a generator that re-shoots in
        # the reported potential lands somewhere else (3.4e-2 Hartree for
        # oxygen's relativistic 2s, which is the whole problem this polish
        # exists to remove).
        v_hartree = hartree_potential(r, density)
        _e_xc, v_xc = xc_potential(r, density, xc)
        v_effective = nuclear + v_hartree + v_xc
        for key, occupancy in (occupations_j if j_resolved
                               else occupations).items():
            if occupancy <= 0:
                continue
            if not is_valence(key):
                continue
            n, l = key[0], key[1]
            k = key[2] if j_resolved else None
            table = eigenvalues_j if j_resolved else eigenvalues
            store = orbitals_j if j_resolved else orbitals
            try:
                store[key], table[key] = bound_state(
                    r, v_effective + wall, l, table[key], Z,
                    treatment=relativity, kappa=k)
            except RuntimeError:
                pass

    if j_resolved:
        from .relativity import j_average
        for (n, l) in occupations:
            eigenvalues[(n, l)] = j_average(
                {k: eigenvalues_j[(n, l, k)] for k in kappa_values(l)}, l)
            weights = {k: float(degeneracy(k)) for k in kappa_values(l)}
            total_weight = sum(weights.values())
            averaged = sum(weights[k] * orbitals_j[(n, l, k)]
                           for k in kappa_values(l)) / total_weight
            orbitals[(n, l)] = averaged / np.sqrt(
                np.trapezoid(averaged * averaged, r))

    # Final potentials consistent with the converged density.
    v_hartree = hartree_potential(r, density)
    e_xc, v_xc = xc_potential(r, density, xc)
    v_effective = nuclear + v_hartree + v_xc

    # Total energy from the eigenvalue sum, correcting the double counting.
    if j_resolved:
        band = sum(occupations_j[k] * eigenvalues_j[k]
                   for k in eigenvalues_j if occupations_j.get(k, 0) > 0)
    else:
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
        orbitals_j=orbitals_j, eigenvalues_j=eigenvalues_j,
        occupations_j=occupations_j,
        details={"points": points, "r_max": r_max, "mixing": mixing,
                 "confined": confinement is not None,
                 "xc": xc, "relativity": relativity})
