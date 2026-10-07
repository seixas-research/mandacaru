# -*- coding: utf-8 -*-
# file: algorithms/dft.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Kohn-Sham density functional theory behind ``Mandacaru(method="dft")``.

The Kohn-Sham problem is solved in the same localized basis, on the same grid
and with the same external potential as every other method here, so a DFT
energy and an ADAPT-VQE energy differ only in how the electrons are treated:

.. math::

    E = \operatorname{tr}(D h) + \tfrac12 \operatorname{tr}(D J[D])
        + E_{xc}[\rho + \tilde\rho_c] + E_{\text{const}},
    \qquad
    F = h + J[D] + V_{xc}.

``h`` is the core Hamiltonian of the integrals (kinetic, local and nonlocal
pseudopotential), ``J`` the Hartree matrix of the same Coulomb operator the
quantum Hamiltonian's two-body tensor holds -- by default one Poisson solve of
the density per iteration (``hartree="poisson"``, :class:`PoissonHartree`),
so a semilocal energy needs no :math:`M^4` tensor (the forces, a hybrid's
exchange and the export to a quantum method still build theirs, when asked);
``hartree="tensor"`` contracts the tensor instead, equal to round-off -- and :math:`\rho(\mathbf r) =
\sum_{pq} D_{pq}\,\phi_p(\mathbf r)\phi_q^*(\mathbf r)` the density of the
Loewdin-orthonormal orbitals on the grid.  The exchange-correlation energy and
potential come from :mod:`mandacaru.integrals.exchange_correlation`, with a
dataset's partial core density :math:`\tilde\rho_c` added when it carries one.
:math:`E_{\text{const}}` is the ion-ion repulsion, any frozen one-center
constant of the integrals and, with a core correction, the per-atom offset
that keeps the valence-only energy zero
(:func:`~mandacaru.integrals.exchange_correlation.core_correction_offset`).

Functionals: LDA, PBE, the r\ :sup:`2`\ SCAN meta-GGA, whose
kinetic-energy density :math:`\tau = \tfrac12\sum_{pq}D_{pq}\nabla\phi_p
\cdot\nabla\phi_q^*` is built from spectral gradients of the orbitals, and
the screened hybrid HSE06 (molecules and crystals):

.. math::

    E_{xc} = E_{xc}^{\rm sl}[\rho]
        - \tfrac a4 \sum_{pqrs} D_{sr} D_{qp}\,\langle pr|sq\rangle^{\rm SR},
    \qquad
    F \mathrel{+}= -\tfrac a2 K^{\rm SR}[D],

with :math:`a = 1/4`, the semilocal part from
:mod:`mandacaru.basis.hse` and the short-range tensor
:math:`\langle pq|\operatorname{erfc}(\omega r_{12})/r_{12}|rs\rangle` the
full one minus the long-range one over the same pair densities
(:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.short_range_two_body`).
On PAW-LCAO those are the augmented pair densities, and each augmentation
sphere adds its one-center exact exchange and removes its share of the
frozen one-center semilocal exchange
(:mod:`mandacaru.pseudopotentials.onecenter`).  Like the meta-GGA it is
solved in the generalized Kohn-Sham sense (a nonlocal operator).  The
D4 dispersion correction (``dispersion="d4"``) is added from the external
``dftd4`` library, parameterized for the functional that ran.

PAW-LCAO
--------
The PAW-LCAO datasets here linearize the one-center Hartree and
exchange-correlation energies around their reference atom: a fixed coupling
:math:`D^{ion}` in the nonlocal term and a per-species constant in the
integrals.  The augmentation spheres therefore need no exchange-correlation
evaluation of their own; the Kohn-Sham energy is the smooth one, with the
compensation charges in the Hartree term (their low-rank multipole terms,
the same ones the two-body tensor carries) and the dataset's smooth core in the functional, exactly how the dataset was
unscreened.  The datasets are LDA: another functional on them is a mismatch
between molecule and dataset, and is warned about.

Restricted for a closed shell, unrestricted when ``n_alpha != n_beta``
(:class:`UnrestrictedKohnSham`).  Levels fill by aufbau, except a degenerate
shell that straddles the Fermi level -- spin-restricted O2's pi*, the OH
radical's beta pi -- whose electrons are shared equally by its members
(:func:`shell_occupations`): the zero-temperature ensemble, which the energy,
the density and the forces then describe.  A **periodic** geometry (any ``atoms.pbc``)
takes the crystal path instead -- Bloch states on a Monkhorst-Pack mesh with
smearing, spin-polarized when the atoms carry initial moments, PAW-LCAO only
(:mod:`~mandacaru.algorithms.periodic_dft`); D4 there is the lattice sum,
in the energy, the forces and the stress.

The converged Kohn-Sham orbitals are exported like the Hartree-Fock ones: the
result carries the many-body Hamiltonian written in them, so
``Mandacaru(method="adapt-vqe", **result.as_quantum_problem())`` continues
from a Kohn-Sham reference.
"""

from __future__ import annotations

import inspect
import warnings
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.hamiltonian import spin_block_integrals
from ..core.mapping import Fermion
from ..integrals import exchange_correlation as xc_grid
from ..units import to_bohr
from .hartree_fock import (DIIS, LEVEL_SHIFT, RHFResult, UHFResult,
                           _level_shifted, natural_orbitals,
                           transform_integrals)
from .periodic_dft import scf_record
from .pseudo_forces import AlgebraicEnergy
from .mean_field import MeanFieldResult, _MeanFieldDriver

#: Default exchange-correlation functional of ``method="dft"``.
DEFAULT_XC = "lda"

#: An energy rise (Hartree) below this is grid and round-off noise, not the
#: oscillation the level shift exists to damp.  A shift switched on by noise
#: near convergence freezes the density change just above ``tol``: stretched
#: H2 with PBE stalled at 1e-7 for 200 iterations after a 1.7e-10 Ha rise.
LEVEL_SHIFT_TRIGGER = 1e-6

#: Iterations between restarts of the DIIS history.  Near convergence an
#: r2SCAN SCF can stall with the extrapolation pinned by stale vectors, the
#: energy oscillating at 1e-8 Ha for all 200 iterations (H2O, PAW-LCAO DZP,
#: oxygen displaced by 0.004 Angstrom).  Clearing the history every 10
#: iterations finished it in 90 (every 15: 111; every 20: 177); LDA and PBE
#: converge before the first restart matters.
DIIS_RESTART = 10

#: How a molecular Kohn-Sham run builds its Hartree matrix (``hartree=``):
#: ``"poisson"`` solves the Poisson equation of the density once per
#: iteration, the compensation multipoles of a PAW-LCAO basis added as their
#: low-rank matrices (:class:`PoissonHartree`); ``"tensor"`` contracts the
#: two-body tensor.  Both integrate the same operator -- the spectral
#: isolated kernel -- and agree to round-off.
HARTREE_METHODS = ("poisson", "tensor")

#: The default of ``hartree=`` (HISTORY, "K4").
DEFAULT_HARTREE = "poisson"


def _stalled(history) -> bool:
    """Whether the SCF made no progress over the last :data:`DIIS_RESTART`
    iterations -- the density change not even halved -- the only time a DIIS
    restart is taken.  An unconditional one, every ten iterations, threw
    triplet O2 (UKS, PAW-LCAO SZ) from 1e-4 to 0.18 Ha off its energy at
    iteration 40 and then kept the density change at 1e-7..1e-5, above the
    tolerance, while the energy had settled to 1e-13 Ha."""
    if len(history) < DIIS_RESTART:
        return False
    return (history[-1]["residual"]
            > 0.5 * history[-DIIS_RESTART]["residual"])


#: Smallest eigenvalue of the normalized overlap of the DIIS errors below
#: which they count as linearly dependent (:class:`_IndependentDIIS`).
#: Water (PAW-LCAO DZP) never goes below 1e-7; OH in single zeta reaches
#: 1e-16 by the seventh vector.
DIIS_DEPENDENCE = 1e-12


class _IndependentDIIS(DIIS):
    """:class:`~.hartree_fock.DIIS` that drops its oldest vectors while the
    errors, the new one included, are linearly dependent.

    A small basis has a small error space: OH in PAW-LCAO single zeta has
    five orbitals, and an alpha channel of four occupied and one empty
    orbital has commutator errors in four real dimensions.  Eight vectors
    there make the DIIS system singular, its solve returns coefficients of
    1e5 and more, and the extrapolated operator throws the density back to
    the core guess's: OH did not converge in 200 iterations from three of
    nine nearby geometries, triplet O2 took 70 to 186.  Pruned, they take
    24 to 45 and 12.  Errors that stay independent leave the arithmetic --
    and so every well-conditioned run -- exactly as before.
    """

    def extrapolate(self, F, D):
        error = self.error(F, D)
        while self._errors:
            errors = (self._errors + [error])[-self.depth:]
            if len(errors) < 2:
                break
            overlap = np.array([[np.vdot(a, b) for b in errors]
                                for a in errors])
            norms = np.sqrt(np.real(np.diag(overlap)))
            if norms.min() == 0.0 or np.min(np.linalg.eigvalsh(
                    overlap / np.outer(norms, norms))) > DIIS_DEPENDENCE:
                break
            self._errors.pop(0)
            self._focks.pop(0)
        return super().extrapolate(F, D)


#: Kohn-Sham levels closer than this (Hartree) form one degenerate shell when
#: it straddles the Fermi level, and share its electrons equally
#: (:func:`shell_occupations`).  Wide enough for a pair a slightly bent
#: geometry or the grid splits (OH with its hydrogen 0.08 Angstrom off the
#: axis: 2.2e-4 Ha between the beta pi levels), narrow enough that a small
#: real gap stays integer (H2 at 4 Angstrom: 5.5e-3 Ha between sigma_g and
#: sigma_u).
DEGENERACY_WINDOW = 1e-3


def shell_occupations(levels, n: int, window: float = DEGENERACY_WINDOW):
    r"""Occupations (0 to 1) of ``levels`` (ascending) holding ``n``
    electrons of one spin, or ``None`` for the aufbau determinant.

    Aufbau fills the lowest ``n`` levels.  When the ``n``-th and the next
    one are closer than ``window``, the degenerate shell they belong to
    (consecutive levels closer than ``window``) is partly filled, and its
    electrons are spread equally over it: the zero-temperature ensemble
    (Mermin's functional at :math:`T \to 0`) of a symmetric shell.  Filling
    one member instead breaks the symmetry, and the occupied member is then
    pushed above the empty one (OH's beta pi: no integer aufbau solution
    exists) or the SCF wanders between members (closed-shell O2's pi*).
    ``None`` -- the common case -- keeps the determinant's own arithmetic.
    """
    levels = np.real(np.asarray(levels))
    size = len(levels)
    if n <= 0 or n >= size or levels[n] - levels[n - 1] >= window:
        return None
    low = n - 1
    while low > 0 and levels[low] - levels[low - 1] < window:
        low -= 1
    high = n
    while high + 1 < size and levels[high + 1] - levels[high] < window:
        high += 1
    occupations = np.zeros(size)
    occupations[:low] = 1.0
    occupations[low:high + 1] = (n - low) / (high + 1 - low)
    return occupations

#: Dispersion corrections ``dispersion=`` accepts.
DISPERSION_CORRECTIONS = ("d4",)

#: The functionals D4 is parameterized for, by the ``dftd4`` method name.
D4_METHODS = {"pbe": "pbe", "r2scan": "r2scan", "hse06": "hse06"}


class MOTensor:
    """The two-body tensor in a set of orbitals, transformed when first read.

    A Kohn-Sham run on the Poisson route never builds the two-body tensor:
    its Hartree term is a Poisson solve.  Only the handoff to a quantum
    method reads the tensor in the Kohn-Sham orbitals, so the result carries
    this recipe in its ``eri_mo`` and builds it then
    (:class:`_DeferredTensor`), from the integrals' own (cached) tensor.
    ``orbitals`` are columns over the integrals' Loewdin basis, the basis of
    :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.two_body`.
    """

    def __init__(self, integrals, orbitals):
        self.integrals = integrals
        self.orbitals = np.asarray(orbitals)

    def build(self) -> np.ndarray:
        C = self.orbitals
        g = self.integrals.two_body()
        return np.real_if_close(np.einsum(
            "ap,bq,cr,ds,abcd->pqrs", C.conj(), C.conj(), C, C, g,
            optimize=True))


class _DeferredTensor:
    """Builds an ``eri_mo`` held as a :class:`MOTensor` the first time it is
    read, and keeps the array."""

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name == "eri_mo" and isinstance(value, MOTensor):
            value = value.build()
            object.__setattr__(self, name, value)
        return value


def determinant_energy(h_mo, g_occ, n_alpha: int, n_beta: int) -> float:
    r"""Hartree-Fock energy of the determinant that fills the lowest
    ``n_alpha`` and ``n_beta`` orbitals of one spatial set (Hartree, no
    constant): :math:`\sum_i n_i h_{ii} + \tfrac12\sum_{ij} n_i n_j
    \langle ij|ij\rangle - \tfrac12\sum_\sigma\sum_{ij\in\sigma}
    \langle ij|ji\rangle`.

    ``g_occ`` holds :math:`\langle ij|kl\rangle` over (at least) the first
    ``max(n_alpha, n_beta)`` orbitals -- all the determinant needs, which the
    Poisson route computes without the whole tensor."""
    n = max(int(n_alpha), int(n_beta))
    counts = ((np.arange(n) < n_alpha).astype(float)
              + (np.arange(n) < n_beta).astype(float))
    h = np.real(np.diagonal(np.asarray(h_mo))[:n])
    g = np.real(np.asarray(g_occ)[:n, :n, :n, :n])
    coulomb = np.einsum("ijij->ij", g)
    exchange = np.einsum("ijji->ij", g)
    value = counts @ h + 0.5 * counts @ coulomb @ counts
    for m in (int(n_alpha), int(n_beta)):
        value -= 0.5 * np.sum(exchange[:m, :m])
    return float(value)


@dataclass
class KohnShamResult(_DeferredTensor, RHFResult):
    """A converged closed-shell Kohn-Sham determinant.

    The fields of :class:`~mandacaru.algorithms.hartree_fock.RHFResult` (MO
    energies are the Kohn-Sham eigenvalues; ``h_mo`` and ``eri_mo`` are the
    integrals in the Kohn-Sham orbitals) plus the energy decomposition, all in
    Hartree.  ``electronic_energy`` excludes every constant; the dispersion
    and core-correction constants are kept apart so each can be reported.
    On the Poisson route (``hartree="poisson"``) ``eri_mo`` is transformed
    the first time it is read (:class:`MOTensor`).

    ``occupations`` is ``None`` for the aufbau determinant (the lowest
    ``n_occupied`` orbitals doubly occupied).  When a degenerate shell at the
    Fermi level is partly filled (:func:`shell_occupations`), the Kohn-Sham
    state is an ensemble: ``occupations`` then holds each canonical orbital's
    occupation, both spins (0 to 2), and the energy, the density and the
    forces are the ensemble's.
    """

    functional: str = DEFAULT_XC
    hartree_energy: float = 0.0
    xc_energy: float = 0.0
    #: A hybrid's exact-exchange part of ``xc_energy`` (zero otherwise).
    exact_exchange_energy: float = 0.0
    core_correction_energy: float = 0.0
    dispersion_energy: float = 0.0
    #: One record per iteration (:func:`~.periodic_dft.scf_record`), for the
    #: run log's ``[SCF ITERATIONS]`` table.
    history: list = field(default_factory=list)
    #: Fractional occupations of the canonical orbitals, or ``None``.
    occupations: np.ndarray | None = None
    #: :attr:`determinant_energy`, computed by the SCF.
    reference_energy: float = 0.0

    @property
    def determinant_energy(self) -> float:
        r"""Energy of the Kohn-Sham determinant in the many-body Hamiltonian.

        :math:`\sum_i 2h_{ii} + \sum_{ij}(2\langle ij|ij\rangle -
        \langle ij|ji\rangle)` over the occupied Kohn-Sham orbitals (Hartree,
        no constant; :func:`determinant_energy`): the Hartree-Fock functional
        of these orbitals, which is what the exported problem's reference
        state has -- not the Kohn-Sham energy.  With fractional
        ``occupations`` it is the determinant of the lowest ``n_occupied``
        orbitals, one member of the ensemble.
        """
        return float(self.reference_energy)

    def __repr__(self) -> str:
        return (f"KohnShamResult({self.functional}, "
                f"E_elec={self.electronic_energy:.6f}, "
                f"n_occ={self.n_occupied}, converged={self.converged})")


class LazyHamiltonian:
    """The many-body Hamiltonian in the Kohn-Sham orbitals, built on demand.

    Spin-blocking the integrals allocates :math:`(2M)^4` entries -- 1.6 GB at
    50 orbitals -- for an operator only the handoff to a quantum method
    (:meth:`~mandacaru.algorithms.mean_field.MeanFieldResult.as_quantum_problem`)
    reads.  :class:`~mandacaru.algorithms.mean_field.MeanFieldResult` builds it
    the first time the attribute is read, from the Kohn-Sham result ``scf``
    (whose ``eri_mo`` may itself be deferred).
    """

    def __init__(self, scf, constant):
        self.scf = scf
        self.constant = complex(constant)

    def build(self) -> Fermion:
        h_so, g_so = spin_block_integrals(self.scf.h_mo, self.scf.eri_mo)
        n_modes = h_so.shape[0]
        return (Fermion.from_integrals(h_so, g_so)
                + Fermion({(): self.constant}, n_modes=n_modes))


class PoissonHartree:
    r"""The Hartree terms of the Loewdin basis without the two-body tensor.

    :math:`J[D]` is the potential of one density:
    :math:`J = X^\dagger J_{AO}[X D X^\dagger]\,X`, with :math:`J_{AO}`
    from :meth:`~mandacaru.integrals.direct.DirectCoulomb.coulomb` -- one
    solve with the spectral isolated kernel the tensor is built with, and
    the PAW-LCAO compensation multipoles (every :math:`L` the dataset
    carries) as their low-rank matrices, so it equals the tensor's
    contraction to round-off.  :meth:`orbital_integrals` gives
    :math:`\langle ij|kl\rangle` over a few orbitals (the occupied ones of
    a determinant energy) with :math:`n^2/2` solves.
    """

    def __init__(self, integrals):
        self.integrals = integrals
        self.direct = integrals.direct_coulomb()
        self.X = integrals._lowdin_x()

    def coulomb(self, D) -> np.ndarray:
        X = self.X
        return X.conj().T @ self.direct.coulomb(X @ D @ X.conj().T) @ X

    def orbital_integrals(self, C) -> np.ndarray:
        """``<ij|kl>`` over the Loewdin-basis orbital columns ``C``."""
        return self.direct.orbital_integrals(self.X @ np.asarray(C))


class KohnSham:
    r"""Closed-shell (restricted) Kohn-Sham SCF in an orthonormal basis.

    Parameters
    ----------
    h : (M, M) array
        Core Hamiltonian in the orthonormal basis (Hartree).
    eri : (M, M, M, M) array or PoissonHartree
        ``<pq|rs>`` in physicists' notation, same basis -- or the
        :class:`PoissonHartree` of the integrals, which builds the Hartree
        matrix by a Poisson solve and never forms the tensor.
    orbitals : (M, ngrid) array
        The orthonormal basis functions sampled on ``grid``.
    grid : Grid
        The integration grid.
    n_electrons : int
        Even electron count.
    functional : str
        ``"lda"``, ``"pbe"``, ``"r2scan"`` or ``"hse06"``.  HSE06's hole
        model bends its reduced gradient continuously from s = 1
        (:data:`~mandacaru.basis.hse.S_BEND_OFFSET`), so its forces are the
        energy's derivative.
    core_density : (ngrid,) array, optional
        Partial core density added to the valence density inside the
        functional.
    core_tau : (ngrid,) array, optional
        The cores' kinetic-energy density, added to a meta-GGA's
        (:func:`~mandacaru.integrals.exchange_correlation.core_tau_on_grid`).
    relativistic : bool
        Apply the relativistic exchange factor (LDA and PBE), as a
        relativistic dataset was unscreened with.
    short_range_eri : (M, M, M, M) array, optional
        A hybrid's exchange tensor, ``<pq|erfc(omega r12)/r12|rs>`` in the
        same basis and layout as ``eri``; required by a hybrid
        (:data:`~mandacaru.integrals.exchange_correlation.HYBRIDS`).
    screening : (float, float), optional
        A hybrid's ``(omega, fraction)``; the functional's own by default.
        ``omega`` must be the one ``short_range_eri`` was built with.
    one_center : (ndarray, object), optional
        A hybrid's PAW-LCAO one-center terms: the projections ``C`` (``(M,
        P)``, this basis) and the
        :class:`~mandacaru.pseudopotentials.onecenter.OneCenterHybrid` built
        with the same ``screening``.  Its energy at :math:`C^\dagger D C` is
        part of the exchange-correlation energy and :math:`C O C^\dagger` of
        the operator.

    A hybrid is solved in the generalized Kohn-Sham sense, as the meta-GGA
    is: the operator is the derivative of the energy with respect to the
    density matrix, here with the nonlocal
    :math:`-\tfrac a2 K^{\rm SR}[D]` alongside the semilocal potential.
    """

    def __init__(self, h, eri, orbitals, grid, n_electrons: int,
                 functional: str = DEFAULT_XC, core_density=None,
                 relativistic: bool = False, core_tau=None,
                 short_range_eri=None, screening=None, one_center=None):
        self.h = np.asarray(h, dtype=complex)
        #: The Poisson route's Hartree builder, or ``None`` with a tensor.
        self.poisson = eri if isinstance(eri, PoissonHartree) else None
        self.eri = (None if self.poisson is not None
                    else np.asarray(eri, dtype=complex))
        self.orbitals = np.asarray(orbitals)
        self.grid = grid
        self.M = self.h.shape[0]
        self._set_electrons(n_electrons)
        self.functional = xc_grid.resolve_functional(functional)
        self.relativistic = bool(relativistic)
        self.screening = None
        self.short_range_eri = None
        if xc_grid.is_hybrid(self.functional):
            if short_range_eri is None:
                raise ValueError(f"xc={self.functional!r} is a hybrid: it "
                                 "needs its short-range exchange tensor")
            self.screening = tuple(float(v) for v in (
                screening if screening is not None
                else xc_grid.HYBRIDS[self.functional]))
            self.short_range_eri = np.asarray(short_range_eri, dtype=complex)
        self.one_center = None
        if one_center is not None and self.screening is not None:
            projections, terms = one_center
            self.one_center = (np.asarray(projections, dtype=complex), terms)
        #: The exact-exchange part of the last exchange-correlation energy.
        self.exact_exchange_energy = 0.0
        self.core_density = (None if core_density is None
                             else np.asarray(core_density, dtype=float))
        self.meta = xc_grid.is_meta_gga(self.functional)
        self._orbital_gradients = None
        self._core_tau = None
        if self.meta:
            self._orbital_gradients = xc_grid.gradient(grid, self.orbitals)
            if core_tau is not None:
                self._core_tau = np.asarray(core_tau, dtype=float)

    def _set_electrons(self, n_electrons) -> None:
        if n_electrons % 2 != 0:
            raise ValueError("closed-shell Kohn-Sham needs an even number of "
                             "electrons")
        self.n_electrons = int(n_electrons)
        self.n_occ = self.n_electrons // 2
        if self.n_occ > self.M:
            raise ValueError(
                f"{n_electrons} electrons need > {self.M} spatial orbitals")

    # -- the density and its matrices ------------------------------------- #

    def _density_matrix(self, C, occupations=None) -> np.ndarray:
        """``D_pq = 2 sum_i^occ C_pi C*_qi``, or ``2 sum_i f_i C_pi C*_qi``
        with the per-spin ``occupations`` ``f`` of a partly filled shell."""
        if occupations is not None:
            return 2.0 * ((C * occupations) @ C.conj().T)
        occupied = C[:, :self.n_occ]
        return 2.0 * (occupied @ occupied.conj().T)

    def density(self, D) -> np.ndarray:
        r"""Valence density :math:`\sum_{pq} D_{pq}\phi_p\phi_q^*` on the grid."""
        phi = self.orbitals
        return np.real(np.sum(phi * (D @ phi.conj()), axis=0))

    def kinetic_energy_density(self, D) -> np.ndarray:
        r""":math:`\tau = \tfrac12\sum_{pq}D_{pq}\nabla\phi_p\cdot\nabla\phi_q^*`."""
        tau = np.zeros(self.orbitals.shape[1])
        for dphi in self._orbital_gradients:
            tau += np.real(np.sum(dphi * (D @ dphi.conj()), axis=0))
        return 0.5 * tau

    def _hartree(self, D) -> np.ndarray:
        r""":math:`J_{pq} = \sum_{rs} D_{sr}\langle pr|qs\rangle`: one
        Poisson solve on the Poisson route, else the same contraction as the
        Hartree-Fock Coulomb matrix (density entering as ``D_sr``, see
        ``RHF._fock``)."""
        if self.poisson is not None:
            return self.poisson.coulomb(D)
        return np.einsum("sr,prqs->pq", D, self.eri, optimize=True)

    def _mo_integrals(self, C, n_occupied: int):
        """``(h_mo, eri_mo, g_occ)`` in the orbitals ``C``: ``eri_mo`` the
        whole tensor, or on the Poisson route its deferred
        :class:`MOTensor`; ``g_occ`` the block over the first
        ``n_occupied`` orbitals (all a determinant energy needs)."""
        if self.poisson is None:
            h_mo, eri_mo = transform_integrals(self.h, self.eri, C)
            occ = slice(0, n_occupied)
            return h_mo, eri_mo, eri_mo[occ, occ, occ, occ]
        h_mo = C.conj().T @ self.h @ C
        eri_mo = MOTensor(self.poisson.integrals, C)
        return h_mo, eri_mo, self.poisson.orbital_integrals(
            C[:, :n_occupied])

    def _exchange_matrix(self, R) -> np.ndarray:
        r""":math:`K_{pq} = \sum_{rs} R_{sr}\langle pr|sq\rangle^{\rm SR}`,
        the short-range exchange matrix of a hybrid."""
        return np.einsum("sr,prsq->pq", R, self.short_range_eri,
                         optimize=True)

    def _xc(self, D):
        """``(E_xc, V_xc)``: the functional's energy and its matrix.

        A hybrid's short-range exact exchange is included in both:
        :math:`-\\tfrac a4\\,\\mathrm{tr}(D K[D])` and
        :math:`-\\tfrac a2 K[D]` (``D`` holds both spins); the semilocal
        short-range exchange it replaces is the valence density's
        (:func:`~mandacaru.integrals.exchange_correlation.evaluate`).
        """
        valence = rho = self.density(D)
        if self.core_density is not None:
            rho = rho + self.core_density
        tau = None
        if self.meta:
            tau = self.kinetic_energy_density(D)
            if self._core_tau is not None:
                tau = tau + self._core_tau
        terms = xc_grid.evaluate(self.grid, rho, self.functional,
                                 relativistic=self.relativistic, tau=tau,
                                 screening=self.screening,
                                 exchange_density=valence)
        phi = self.orbitals
        dV = self.grid.dV
        V = (phi.conj() * terms.potential) @ phi.T * dV
        if terms.tau_potential is not None:
            # dE/dD_qp through tau: (1/2) int df/dtau grad phi_p* . grad phi_q.
            weight = 0.5 * terms.tau_potential
            for dphi in self._orbital_gradients:
                V = V + (dphi.conj() * weight) @ dphi.T * dV
        energy = terms.energy
        if self.screening is not None:
            fraction = self.screening[1]
            K = self._exchange_matrix(D)
            self.exact_exchange_energy = -0.25 * fraction * float(
                np.real(np.sum(D * K.T)))
            energy = energy + self.exact_exchange_energy
            V = V - 0.5 * fraction * K
            if self.one_center is not None:
                C, hybrid = self.one_center
                spheres = hybrid.evaluate(C.conj().T @ D @ C)
                self.exact_exchange_energy += spheres.exact_exchange
                energy = energy + spheres.energy
                V = V + C @ spheres.operators[0] @ C.conj().T
        return energy, 0.5 * (V + V.conj().T)

    def _fock(self, D):
        """``(F, E_elec, E_H, E_xc)`` at the density matrix ``D``."""
        J = self._hartree(D)
        e_xc, V_xc = self._xc(D)
        e_one = float(np.real(np.sum(D * self.h.T)))
        e_hartree = 0.5 * float(np.real(np.sum(D * J.T)))
        F = self.h + J + V_xc
        return 0.5 * (F + F.conj().T), e_one + e_hartree + e_xc, e_hartree, e_xc

    # -- the SCF ---------------------------------------------------------- #

    def run(self, max_iter: int = 200, tol: float = 1e-8,
            diis: bool = True) -> KohnShamResult:
        """Iterate to self-consistency from the core guess.

        DIIS-extrapolated, its history cleared every :data:`DIIS_RESTART`
        iterations, with the Hartree-Fock solver's level shift
        switched on when an iteration raises the energy by more than
        :data:`LEVEL_SHIFT_TRIGGER` and off again once the energy has settled
        below it.  Converged
        when both the energy change and the largest density-matrix change are
        below ``tol``.

        Each iteration fills the levels by aufbau, except a degenerate shell
        straddling the Fermi level, which :func:`shell_occupations` fills
        equally (the result's ``occupations``).  Closed-shell O2 is the case:
        its pi* pair holds two of the four electrons it can take, and
        filling one member let the SCF land, by the path, either on that
        symmetry-broken determinant (-895.42 eV, PAW-LCAO SZ, LDA) or on a
        self-consistent state 24 eV higher; the ensemble is -895.95 eV.
        """
        _eps, C = np.linalg.eigh(self.h)
        occupations = shell_occupations(_eps, self.n_occ)
        D = self._density_matrix(C, occupations)
        energy = np.inf
        converged = False
        mixer = _IndependentDIIS() if diis else None
        shift = 0.0
        it = 0
        history, start = [], time.perf_counter()
        for it in range(1, max_iter + 1):
            F, current, _e_h, _e_xc = self._fock(D)
            if mixer is not None and it % DIIS_RESTART == 0 \
                    and _stalled(history):
                mixer = _IndependentDIIS()
            if current > energy + LEVEL_SHIFT_TRIGGER and shift == 0.0:
                shift = LEVEL_SHIFT
            elif shift and abs(current - energy) < LEVEL_SHIFT_TRIGGER:
                # The oscillation is over; the shift would now only slow the
                # last digits of the density down.
                shift = 0.0
            F_iter = mixer.extrapolate(F, 0.5 * D) if mixer else F
            if shift:
                F_iter = _level_shifted(F_iter, 0.5 * D, shift)
            _eps, C = np.linalg.eigh(F_iter)
            occupations = shell_occupations(_eps, self.n_occ)
            D_new = self._density_matrix(C, occupations)
            change = float(np.max(np.abs(D_new - D)))
            history.append(scf_record(it, start, current, energy, change))
            if abs(current - energy) < tol and change < tol:
                D, energy = D_new, current
                converged = True
                break
            D, energy = D_new, current

        F, energy, e_hartree, e_xc = self._fock(D)
        eps, C = np.linalg.eigh(F)                  # canonical KS orbitals
        h_mo, eri_mo, g_occ = self._mo_integrals(C, self.n_occ)
        # The last iteration's occupations: at convergence its levels are
        # these to within ``tol``, and inside a shell they are equal anyway.
        return KohnShamResult(
            electronic_energy=energy, mo_energies=np.real(eps),
            mo_coefficients=C, n_occupied=self.n_occ, converged=converged,
            h_mo=np.real_if_close(h_mo),
            eri_mo=(eri_mo if isinstance(eri_mo, MOTensor)
                    else np.real_if_close(eri_mo)),
            reference_energy=determinant_energy(h_mo, g_occ, self.n_occ,
                                                self.n_occ),
            n_iterations=it, functional=self.functional,
            hartree_energy=e_hartree, xc_energy=e_xc, history=history,
            exact_exchange_energy=self.exact_exchange_energy,
            occupations=(None if occupations is None
                         else 2.0 * occupations))


@dataclass
class KohnShamUResult(_DeferredTensor, UHFResult):
    """A converged spin-unrestricted Kohn-Sham determinant.

    The fields of :class:`~mandacaru.algorithms.hartree_fock.UHFResult`
    (MO energies are the Kohn-Sham eigenvalues of each spin; the many-body
    integrals are in the natural orbitals of the total density, and
    ``reference_energy`` is the Hartree-Fock energy of their determinant) plus
    the Kohn-Sham energy decomposition, all in Hartree.  On the Poisson route
    ``eri_mo`` is transformed the first time it is read (:class:`MOTensor`).

    ``occupations_alpha`` / ``occupations_beta`` are ``None`` for a channel
    filled by aufbau, else the occupations (0 to 1) of its canonical orbitals
    when a degenerate shell at its Fermi level is partly filled
    (:func:`shell_occupations`) -- the OH radical's beta pi, for one.
    """

    functional: str = DEFAULT_XC
    hartree_energy: float = 0.0
    xc_energy: float = 0.0
    #: A hybrid's exact-exchange part of ``xc_energy`` (zero otherwise).
    exact_exchange_energy: float = 0.0
    core_correction_energy: float = 0.0
    dispersion_energy: float = 0.0
    #: One record per iteration (:func:`~.periodic_dft.scf_record`).
    history: list = field(default_factory=list)
    #: Fractional occupations of each channel's canonical orbitals, or None.
    occupations_alpha: np.ndarray | None = None
    occupations_beta: np.ndarray | None = None

    @property
    def determinant_energy(self) -> float:
        """The natural-orbital determinant's energy in the exported many-body
        Hamiltonian (no constant), as for the restricted result."""
        return float(self.reference_energy)


class UnrestrictedKohnSham(KohnSham):
    r"""Spin-unrestricted Kohn-Sham: one determinant per spin.

    :math:`F_\sigma = h + J[D_\uparrow + D_\downarrow] + V_{xc}^\sigma`, the
    exchange-correlation potentials from the spin-polarized functional
    (:func:`~mandacaru.integrals.exchange_correlation.evaluate_spin`); the
    spin-unpolarized partial core is split evenly between the channels.  The
    loop is :class:`KohnSham`'s -- DIIS per spin, restarted every
    :data:`DIIS_RESTART` iterations, and the level shift.
    """

    def __init__(self, h, eri, orbitals, grid, n_alpha: int, n_beta: int,
                 **options):
        self._spins = (int(n_alpha), int(n_beta))
        super().__init__(h, eri, orbitals, grid, sum(self._spins), **options)

    def _set_electrons(self, n_electrons) -> None:
        self.n_alpha, self.n_beta = self._spins
        self.n_electrons = int(n_electrons)
        if max(self._spins) > self.M:
            raise ValueError("more electrons of one spin than spatial "
                             "orbitals")

    @staticmethod
    def _spin_density(C, n: int, occupations=None) -> np.ndarray:
        """One channel's density matrix: its lowest ``n`` orbitals, or the
        fractional ``occupations`` of a partly filled shell."""
        if occupations is not None:
            return (C * occupations) @ C.conj().T
        occupied = C[:, :n]
        return occupied @ occupied.conj().T

    def _xc_spin(self, Da, Db):
        """``(E_xc, V_xc^alpha, V_xc^beta)``."""
        half_core = None if self.core_density is None else \
            0.5 * self.core_density
        rho = []
        tau = []
        valence = []
        for D in (Da, Db):
            valence.append(self.density(D))
            rho.append(valence[-1] + (0.0 if half_core is None
                                      else half_core))
            if self.meta:
                t = self.kinetic_energy_density(D)
                tau.append(t + (0.0 if self._core_tau is None
                                else 0.5 * self._core_tau))
        terms = xc_grid.evaluate_spin(
            self.grid, rho[0], rho[1], self.functional,
            relativistic=self.relativistic,
            tau_up=tau[0] if self.meta else None,
            tau_dn=tau[1] if self.meta else None, screening=self.screening,
            exchange_densities=valence)
        phi, dV = self.orbitals, self.grid.dV
        energy = terms.energy
        exact = 0.0
        matrices = []
        spheres = None
        if self.one_center is not None:
            # The one-center terms of both channels at once: the frozen
            # semilocal part depends on their sum.
            C, hybrid = self.one_center
            spheres = hybrid.evaluate(C.conj().T @ Da @ C,
                                      C.conj().T @ Db @ C)
            exact += spheres.exact_exchange
            energy += spheres.semilocal
        for s, (D, v, v_tau) in enumerate((
                (Da, terms.potential_up, terms.tau_potential_up),
                (Db, terms.potential_dn, terms.tau_potential_dn))):
            V = (phi.conj() * v) @ phi.T * dV
            if v_tau is not None:
                for dphi in self._orbital_gradients:
                    V = V + (dphi.conj() * (0.5 * v_tau)) @ dphi.T * dV
            if self.screening is not None:
                # -(a/2) tr(D_s K[D_s]) per spin, and its derivative -a K.
                fraction = self.screening[1]
                K = self._exchange_matrix(D)
                exact -= 0.5 * fraction * float(np.real(np.sum(D * K.T)))
                V = V - fraction * K
            if spheres is not None:
                V = V + C @ spheres.operators[s] @ C.conj().T
            matrices.append(0.5 * (V + V.conj().T))
        self.exact_exchange_energy = exact
        return energy + exact, matrices[0], matrices[1]

    def _fock_pair(self, Da, Db):
        """``(F_alpha, F_beta, E_elec, E_H, E_xc)``."""
        D = Da + Db
        J = self._hartree(D)
        e_xc, Va, Vb = self._xc_spin(Da, Db)
        e_one = float(np.real(np.sum(D * self.h.T)))
        e_hartree = 0.5 * float(np.real(np.sum(D * J.T)))
        Fa, Fb = self.h + J + Va, self.h + J + Vb
        return (0.5 * (Fa + Fa.conj().T), 0.5 * (Fb + Fb.conj().T),
                e_one + e_hartree + e_xc, e_hartree, e_xc)

    def run(self, max_iter: int = 200, tol: float = 1e-8,
            diis: bool = True) -> KohnShamUResult:
        """Iterate to self-consistency from the core guess (both spins),
        each channel filled as :meth:`KohnSham.run` fills its levels."""
        na, nb = self.n_alpha, self.n_beta
        _eps, C = np.linalg.eigh(self.h)
        fa, fb = shell_occupations(_eps, na), shell_occupations(_eps, nb)
        Da, Db = self._spin_density(C, na, fa), self._spin_density(C, nb, fb)
        energy = np.inf
        converged = False
        mixers = (_IndependentDIIS(), _IndependentDIIS()) if diis else None
        shift = 0.0
        it = 0

        history, start = [], time.perf_counter()
        for it in range(1, max_iter + 1):
            Fa, Fb, current, _e_h, _e_xc = self._fock_pair(Da, Db)
            if mixers is not None and it % DIIS_RESTART == 0 \
                    and _stalled(history):
                mixers = (_IndependentDIIS(), _IndependentDIIS())
            if current > energy + LEVEL_SHIFT_TRIGGER and shift == 0.0:
                shift = LEVEL_SHIFT
            elif shift and abs(current - energy) < LEVEL_SHIFT_TRIGGER:
                shift = 0.0
            if mixers is not None:
                Fa, Fb = (mixers[0].extrapolate(Fa, Da),
                          mixers[1].extrapolate(Fb, Db))
            if shift:
                Fa, Fb = (_level_shifted(Fa, Da, shift),
                          _level_shifted(Fb, Db, shift))
            _ea, Ca = np.linalg.eigh(Fa)
            _eb, Cb = np.linalg.eigh(Fb)
            fa, fb = shell_occupations(_ea, na), shell_occupations(_eb, nb)
            Da_new, Db_new = (self._spin_density(Ca, na, fa),
                              self._spin_density(Cb, nb, fb))
            change = max(np.max(np.abs(Da_new - Da)),
                         np.max(np.abs(Db_new - Db)))
            history.append(scf_record(it, start, current, energy, change))
            if abs(current - energy) < tol and change < tol:
                Da, Db, energy = Da_new, Db_new, current
                converged = True
                break
            Da, Db, energy = Da_new, Db_new, current

        Fa, Fb, energy, e_hartree, e_xc = self._fock_pair(Da, Db)
        epsa, Ca = np.linalg.eigh(Fa)               # canonical KS orbitals
        epsb, Cb = np.linalg.eigh(Fb)
        occupations, C_no = natural_orbitals(Da + Db)
        # The exported reference fills the lowest natural orbitals: the
        # Hartree-Fock energy of that determinant is what the many-body
        # Hamiltonian assigns it.
        h_mo, eri_mo, g_occ = self._mo_integrals(C_no, max(na, nb))
        reference = determinant_energy(h_mo, g_occ, na, nb)
        return KohnShamUResult(
            electronic_energy=energy, n_alpha=na, n_beta=nb,
            mo_energies_alpha=np.real(epsa), mo_energies_beta=np.real(epsb),
            mo_coefficients_alpha=Ca, mo_coefficients_beta=Cb,
            natural_occupations=occupations, natural_orbitals=C_no,
            h_mo=np.real_if_close(h_mo),
            eri_mo=(eri_mo if isinstance(eri_mo, MOTensor)
                    else np.real_if_close(eri_mo)),
            converged=converged, n_iterations=it, reference_energy=reference,
            functional=self.functional, hartree_energy=e_hartree,
            xc_energy=e_xc, history=history,
            exact_exchange_energy=self.exact_exchange_energy,
            occupations_alpha=fa, occupations_beta=fb)


def _datasets_relativistic(datasets) -> bool:
    """The datasets' common relativistic-exchange flag (they must agree)."""
    flags = {bool(getattr(d, "relativistic_exchange", False))
             for d in datasets or ()}
    if len(flags) > 1:
        raise ValueError("the datasets mix relativistic and non-relativistic "
                         "exchange; the functional cannot match both")
    return flags.pop() if flags else False


def _warn_functional_mismatch(datasets, functional: str) -> None:
    """Warn when the datasets were generated with another functional, and
    name the library folder of datasets generated with ``functional`` when
    the PAW-LCAO library ships one (``directory="pbe-sr"`` for PBE)."""
    generated = sorted({str(getattr(d, "xc", "lda") or "lda").lower()
                        for d in datasets or ()})
    if generated and generated != [functional]:
        from ..pseudopotentials.environment import LIBRARY_FOLDERS

        families = {str(getattr(d, "family", "")).lower()
                    for d in datasets or ()}
        relativity = {str(getattr(d, "relativity", "scalar")).lower()
                      for d in datasets or ()}
        folder = (LIBRARY_FOLDERS.get(("paw-lcao", functional,
                                       relativity.pop()))
                  if families == {"paw-lcao"} and len(relativity) == 1
                  else None)
        hint = (f"; Mandacaru(directory={folder!r}) reads the PAW-LCAO "
                f"datasets generated with {functional.upper()}"
                if folder is not None else "")
        warnings.warn(
            f"xc={functional!r} on datasets generated with "
            f"{'/'.join(g.upper() for g in generated)}: the molecule and its "
            f"datasets use different functionals{hint}", RuntimeWarning,
            stacklevel=3)


def _hybrid_setup(functional: str) -> dict:
    """``[SCF SETUP]``'s ``exact_exchange`` entry of a hybrid, else empty."""
    if not xc_grid.is_hybrid(functional):
        return {}
    omega, fraction = xc_grid.HYBRIDS[xc_grid.resolve_functional(functional)]
    return {"exact_exchange": (
        f"{fraction:g} of the short-range exchange, erfc(omega r)/r with "
        f"omega = {omega:g} 1/Bohr (generalized Kohn-Sham)")}


def resolve_hartree(hartree) -> str:
    """Validate a ``hartree=`` option (:data:`HARTREE_METHODS`)."""
    key = DEFAULT_HARTREE if hartree is None else str(hartree).strip().lower()
    if key not in HARTREE_METHODS:
        raise ValueError(f"unknown hartree={hartree!r}; available: "
                         f"{', '.join(HARTREE_METHODS)}")
    return key


def kohn_sham_solver(integrals, n_electrons: int,
                     functional: str = DEFAULT_XC, spins=None,
                     screening=None, hartree: str = DEFAULT_HARTREE
                     ) -> KohnSham:
    """The :class:`KohnSham` problem of ``integrals``, not yet solved --
    :class:`UnrestrictedKohnSham` when ``spins = (n_alpha, n_beta)`` differ.

    ``integrals`` is a :class:`~mandacaru.core.hamiltonian.MolecularIntegrals`
    (or its PAW-LCAO subclass) in its orthonormalized basis; the orbitals are
    taken from the same grid samples its integrals were computed from.  A
    hybrid builds its short-range exchange tensor here
    (:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.short_range_two_body`);
    ``screening = (omega, fraction)``
    overrides the functional's own parameters.  ``hartree`` picks how the
    Hartree matrix is built (:data:`HARTREE_METHODS`): ``"poisson"`` never
    forms the two-body tensor (:class:`PoissonHartree`), ``"tensor"``
    contracts it.
    """
    hartree = resolve_hartree(hartree)
    if not getattr(integrals, "orthogonalize", False):
        raise ValueError("Kohn-Sham needs the Loewdin-orthonormalized basis")
    if getattr(integrals, "periodic", False):
        raise NotImplementedError(
            "method='dft' is molecular (Gamma point, isolated boundary "
            "conditions) for now")
    if getattr(integrals, "spin_orbit_coupling", None):
        raise NotImplementedError(
            "method='dft' has no spin-orbit term in its Kohn-Sham operator")
    functional = xc_grid.resolve_functional(functional)
    datasets = integrals.pseudopotentials or []
    _warn_functional_mismatch(datasets, functional)
    X = integrals._lowdin_x()
    orbitals = X.T @ integrals._engine._psi
    core = core_tau = None
    if datasets:
        core = xc_grid.core_density_on_grid(integrals.grid, datasets,
                                            _centers(integrals))
        if xc_grid.is_meta_gga(functional):
            core_tau = xc_grid.core_tau_on_grid(integrals.grid, datasets,
                                                _centers(integrals))
    options = dict(functional=functional, core_density=core,
                   core_tau=core_tau,
                   relativistic=(xc_grid.takes_relativistic_exchange(functional)
                                 and _datasets_relativistic(datasets)))
    if xc_grid.is_hybrid(functional):
        screening = tuple(float(v) for v in (
            screening if screening is not None
            else xc_grid.HYBRIDS[functional]))
        options.update(screening=screening,
                       short_range_eri=integrals.short_range_two_body(
                           screening[0]))
        spheres = integrals.one_center_hybrid(*screening)
        if spheres is not None:
            X = integrals._lowdin_x()
            options.update(one_center=(X.conj().T @ integrals.projections(),
                                       spheres))
    coulomb = (PoissonHartree(integrals) if hartree == "poisson"
               else integrals.two_body())
    if spins is not None and int(spins[0]) != int(spins[1]):
        return UnrestrictedKohnSham(integrals.one_body(), coulomb, orbitals,
                                    integrals.grid, int(spins[0]),
                                    int(spins[1]), **options)
    return KohnSham(integrals.one_body(), coulomb, orbitals,
                    integrals.grid, n_electrons, **options)


def _centers(integrals):
    """Atomic positions in Bohr, the frame the grid and potentials use."""
    return [np.asarray(c, dtype=float) for _Z, c in integrals._potentials.nuclei]


def kohn_sham(integrals, n_electrons: int, functional: str = DEFAULT_XC,
              spins=None, screening=None, hartree: str = DEFAULT_HARTREE,
              **run_options):
    """Solve the Kohn-Sham problem on ``integrals``: restricted, or
    unrestricted when ``spins = (n_alpha, n_beta)`` differ (``screening``
    and ``hartree`` as for :func:`kohn_sham_solver`)."""
    result = kohn_sham_solver(integrals, n_electrons, functional,
                              spins=spins, screening=screening,
                              hartree=hartree).run(**run_options)
    result.core_correction_energy = float(
        sum(xc_grid.core_correction_offset(d)
            for d in integrals.pseudopotentials or []))
    return result


# --------------------------------------------------------------------------- #
# Nuclear gradient.
# --------------------------------------------------------------------------- #

class KohnShamEnergy(AlgebraicEnergy):
    r"""The Kohn-Sham energy as a function of the AO matrices ``(S, h, g)``.

    The density matrix ``D`` is held fixed in the Loewdin basis, so the AO
    density follows the overlap, :math:`P = S^{-1/2} D S^{-1/2}`:

    .. math::

        E(S, h, g) = \operatorname{tr}\big(P\,(h + V_{xc})\big)
          + \tfrac12 \sum P_{ca} P_{db}\, g_{abcd} + c .

    The exchange-correlation energy enters **linearized** about the converged
    density, through the fixed AO matrix :math:`V_{xc}` of its potential and
    the constant :math:`c = E_{xc} - \operatorname{tr}(P_0 V_{xc})`: exact in
    value at the reference and in every first derivative, which is all a
    gradient needs.  Holding ``D`` fixed is exact because the Kohn-Sham
    energy is stationary with respect to orbital rotations at convergence,
    at the occupations the SCF converged with (an ensemble's fractional ones
    included: ``D`` is rebuilt with them).
    """

    def __init__(self, D, V_xc, constant: float):
        self.D = np.asarray(D, dtype=complex)
        self.V_xc = np.asarray(V_xc, dtype=complex)
        self.constant = float(constant)

    def density(self, S) -> np.ndarray:
        """The AO density matrix ``P`` at overlap ``S``."""
        X = self.lowdin(S)
        return X @ self.D @ X.conj().T

    def __call__(self, S, h, g) -> float:
        P = self.density(S)
        one = np.sum(P * (h + self.V_xc).T)
        hartree = 0.5 * np.einsum("ca,db,abcd->", P, P, g, optimize=True)
        return float(np.real(one + hartree)) + self.constant

    def orbital_gradient(self, S, h, g, step: float = 1e-5) -> None:
        """Not measured: a converged Kohn-Sham density is stationary."""
        return None


class KohnShamField:
    r"""The exchange-correlation terms the integral derivatives do not cover.

    ``pulay(dpsi)``: the change of the AO matrix
    :math:`V_{xc,\mu\nu} = \int\psi_\mu^* v_{xc}\psi_\nu
    (+ \tfrac12\int\partial_\tau f\,\nabla\psi_\mu^*\cdot\nabla\psi_\nu)`
    when basis functions move by ``dpsi`` under the fixed potential.
    ``hellmann_feynman(atom, k)``: :math:`\int v_{xc}\,\partial\tilde\rho_c
    /\partial R_{A,k}` (plus :math:`\int\partial_\tau f\,\partial\tau_c/
    \partial R_{A,k}` for a meta-GGA) -- the partial core density and its
    kinetic-energy density move with their atom.  ``core_potential``, when
    given, is the potential the moving core sees instead of ``potential``: a
    hybrid's semilocal short-range exchange is the valence density's alone,
    so its potential acts on the orbitals but not on the core.
    """

    def __init__(self, grid, psi, potential, tau_potential, core_functions,
                 core_tau_functions, centers, delta, core_potential=None):
        self.grid = grid
        self.psi = psi
        self.v = potential
        self.v_core = potential if core_potential is None else core_potential
        self.v_tau = tau_potential
        self.core_functions = core_functions
        self.core_tau_functions = core_tau_functions
        self.centers = centers
        self.delta = float(delta)
        self.grad_psi = (xc_grid.gradient(grid, psi)
                         if tau_potential is not None else None)

    def matrix(self) -> np.ndarray:
        """``V_xc`` over the AO basis (Hartree)."""
        psi, dV = self.psi, self.grid.dV
        V = ((psi.conj() * self.v) @ psi.T) * dV
        if self.v_tau is not None:
            for d in self.grad_psi:
                V = V + 0.5 * ((d.conj() * self.v_tau) @ d.T) * dV
        return 0.5 * (V + V.conj().T)

    def pulay(self, dpsi) -> np.ndarray:
        return self._pulay_with(self.v, self.v_tau, dpsi)

    def _matrix_of(self, v, v_tau) -> np.ndarray:
        psi, dV = self.psi, self.grid.dV
        V = ((psi.conj() * v) @ psi.T) * dV
        if v_tau is not None:
            for d in self.grad_psi:
                V = V + 0.5 * ((d.conj() * v_tau) @ d.T) * dV
        return 0.5 * (V + V.conj().T)

    def _pulay_with(self, v, v_tau, dpsi) -> np.ndarray:
        psi, dV = self.psi, self.grid.dV
        out = ((dpsi.conj() * v) @ psi.T + (psi.conj() * v) @ dpsi.T) * dV
        if v_tau is not None:
            w = 0.5 * v_tau
            for d, dd in zip(self.grad_psi, xc_grid.gradient(self.grid, dpsi)):
                out = out + ((dd.conj() * w) @ d.T + (d.conj() * w) @ dd.T) * dV
        return out

    def hellmann_feynman(self, atom: int, k: int) -> float:
        function = self.core_functions[atom]
        if function is None:
            return 0.0
        from .pseudo_forces import _moved_radial
        drho = _moved_radial(function, self.centers[atom], self.grid, k,
                             self.delta)
        value = np.sum(self.v_core * drho)
        tau_function = self.core_tau_functions[atom]
        if self.v_tau is not None and tau_function is not None:
            dtau = _moved_radial(tau_function, self.centers[atom], self.grid,
                                 k, self.delta)
            value = value + np.sum(self.v_tau * dtau)
        return float(value * self.grid.dV)


class UnrestrictedKohnShamField(KohnShamField):
    r""":class:`KohnShamField` for a spin-resolved potential.

    The linearized exchange-correlation energy
    :math:`\sum_\sigma\mathrm{tr}(P_\sigma V^\sigma) = \mathrm{tr}(P\bar V)
    + \mathrm{tr}(P_m\Delta V)`, with :math:`\bar V = (V^\uparrow +
    V^\downarrow)/2`, :math:`\Delta V = (V^\uparrow - V^\downarrow)/2` and the
    magnetization :math:`P_m = P_\uparrow - P_\downarrow`.  The base field
    carries :math:`\bar v` (the average also moves the partial core, split
    evenly between the spins); :meth:`spin_term` adds the derivative of the
    magnetization part, :math:`\mathrm{tr}(dP_m\,\Delta V) +
    \mathrm{tr}(P_m\,d\Delta V)`, with :math:`P_m = X D_m X^\dagger` held at
    fixed :math:`D_m` in the orthonormal basis as the energy holds :math:`D`.
    """

    def __init__(self, grid, psi, v_mean, v_tau_mean, v_half, v_tau_half,
                 magnetization, core_functions, core_tau_functions, centers,
                 delta, core_potential=None):
        super().__init__(grid, psi, v_mean, v_tau_mean, core_functions,
                         core_tau_functions, centers, delta,
                         core_potential=core_potential)
        if self.grad_psi is None and v_tau_half is not None:
            self.grad_psi = xc_grid.gradient(grid, psi)
        self.v_half = v_half
        self.v_tau_half = v_tau_half
        self.Dm = np.asarray(magnetization, dtype=complex)
        self.delta_matrix = self._matrix_of(v_half, v_tau_half)

    def spin_term(self, S0, dS, dpsi, step) -> float:
        from .pseudo_forces import AlgebraicEnergy

        def magnetization(S):
            X = AlgebraicEnergy.lowdin(S)
            return X @ self.Dm @ X.conj().T

        value = 0.0
        if dS is not None:
            dPm = (magnetization(S0 + step * dS)
                   - magnetization(S0 - step * dS)) / (2.0 * step)
            value += float(np.real(np.sum(dPm * self.delta_matrix.T)))
        if dpsi is not None:
            dDelta = self._pulay_with(self.v_half, self.v_tau_half, dpsi)
            value += float(np.real(np.sum(magnetization(S0) * dDelta.T)))
        return value


def _core_functions(datasets):
    """Per atom, ``r -> core density`` for the core correction, or ``None``."""
    return [xc_grid.core_density_function(dataset) for dataset in datasets]


def _core_tau_functions(datasets):
    """Per atom, ``r -> core kinetic-energy density``, or ``None``."""
    return [xc_grid.core_tau_function(dataset) for dataset in datasets]


def kohn_sham_gradient(integrals, scf: KohnShamResult, *, atom_of_orbital,
                       orbital_delta=None, include_pulay: bool = True):
    r"""Hellmann-Feynman + Pulay gradient of a converged Kohn-Sham energy.

    Every integral derivative -- overlap (augmented), kinetic, local,
    nonlocal, PAW-LCAO compensation charges and the two-electron tensor --
    is :func:`~mandacaru.algorithms.pseudo_forces.pseudo_nuclear_gradient`'s,
    contracted with :class:`KohnShamEnergy` instead of a many-body state.
    :class:`KohnShamField` adds the exchange-correlation potential acting on
    moving basis functions and the moving partial core.  Returns a
    :class:`~mandacaru.algorithms.forces.ForceResult` (eV/Angstrom) whose
    ``details["energy_hartree"]`` is the rebuilt total energy without the
    dispersion and core-correction constants.

    A hybrid's exact exchange is linearized like the semilocal potential:
    its derivative with respect to the density matrix, :math:`-\tfrac a2
    K^{\rm SR}` plus the PAW-LCAO one-center operator :math:`C O
    C^\dagger`, joins the fixed one-body matrix (which carries the overlap
    dependence of :math:`P = S^{-1/2}DS^{-1/2}`), and the derivatives of the
    short-range tensor and of the projections are
    :class:`~mandacaru.algorithms.pseudo_forces.ScreenedExchange`'s.  The
    partial core moves under the potential without the semilocal
    short-range exchange, which is the valence density's.
    """
    from .forces import DEFAULT_ORBITAL_DELTA
    from .pseudo_forces import pseudo_nuclear_gradient

    delta = DEFAULT_ORBITAL_DELTA if orbital_delta is None else float(
        orbital_delta)
    if isinstance(scf, KohnShamUResult):
        return _unrestricted_gradient(integrals, scf, atom_of_orbital, delta,
                                      include_pulay)
    n_electrons = 2 * scf.n_occupied
    solver = kohn_sham_solver(integrals, n_electrons, scf.functional)
    C = scf.mo_coefficients
    # The state the SCF converged to: an ensemble's fractional occupations
    # (each spin's half of the stored ones) or the aufbau determinant.
    D = solver._density_matrix(
        C, None if scf.occupations is None else 0.5 * scf.occupations)
    valence = rho = solver.density(D)
    if solver.core_density is not None:
        rho = rho + solver.core_density
    tau = None
    if solver.meta:
        tau = solver.kinetic_energy_density(D)
        if solver._core_tau is not None:
            tau = tau + solver._core_tau
    terms = xc_grid.evaluate(integrals.grid, rho, solver.functional,
                             relativistic=solver.relativistic, tau=tau,
                             screening=solver.screening,
                             exchange_density=valence)
    core_potential = None
    if solver.screening is not None and solver.core_density is not None:
        core_potential = xc_grid.evaluate(
            integrals.grid, rho, solver.functional,
            relativistic=solver.relativistic,
            screening=(solver.screening[0], 0.0)).potential
    datasets = integrals.pseudopotentials or []
    nothing = [None] * len(integrals.nuclei)
    field = KohnShamField(
        integrals.grid, np.ascontiguousarray(integrals._engine._psi),
        terms.potential, terms.tau_potential,
        _core_functions(datasets) if datasets else nothing,
        _core_tau_functions(datasets) if datasets else nothing,
        _centers(integrals), delta, core_potential=core_potential)
    V_xc = field.matrix()
    X = integrals._lowdin_x()
    P0 = X @ D @ X.conj().T
    e_xc = terms.energy
    exchange = None
    if solver.screening is not None:
        e_exact, V_exact, operators = _exact_exchange_terms(solver, [D])
        e_xc += e_exact
        root = np.linalg.inv(X)                       # S^(1/2)
        V_xc = V_xc + root @ V_exact[0] @ root
        exchange = _screened_exchange(solver, [(-0.25, P0)], operators)
    constant = e_xc - float(np.real(np.sum(P0 * V_xc.T)))
    energy = KohnShamEnergy(D, V_xc, constant)
    return pseudo_nuclear_gradient(
        integrals, None, None, atom_of_orbital=atom_of_orbital,
        orbital_delta=delta, include_pulay=include_pulay,
        orbital_gradient=False, energy=energy, field=field,
        exchange=exchange)


def _exact_exchange_terms(solver, densities):
    r"""``(E, [V_sigma], [O_sigma])`` of a hybrid's exact exchange in the
    solver's (Loewdin) basis.

    ``densities`` is ``[D]`` for a closed shell (both spins) or ``[D_a,
    D_b]``.  ``E`` is the grid and one-center exact exchange plus the
    one-center frozen semilocal term; ``V_sigma`` its derivative matrices
    (:math:`-\tfrac a2 K[D]` or :math:`-a K[D_\sigma]`, plus
    :math:`C O_\sigma C^\dagger`); ``O_sigma`` the one-center ``(P, P)``
    operators, or ``None`` without augmentation spheres.
    """
    fraction = solver.screening[1]
    closed = len(densities) == 1
    weight = 0.25 if closed else 0.5
    energy = 0.0
    matrices = []
    for D in densities:
        K = solver._exchange_matrix(D)
        energy -= weight * fraction * float(np.real(np.sum(D * K.T)))
        matrices.append(-2.0 * weight * fraction * K)
    operators = None
    if solver.one_center is not None:
        C, spheres = solver.one_center
        terms = spheres.evaluate(*[C.conj().T @ D @ C for D in densities])
        energy += terms.energy
        operators = list(terms.operators)
        matrices = [V + C @ O @ C.conj().T
                    for V, O in zip(matrices, operators)]
    return energy, matrices, operators


def _screened_exchange(solver, densities, operators):
    """The :class:`~.pseudo_forces.ScreenedExchange` of ``solver``'s hybrid
    over the AO ``densities = [(weight / fraction, P), ...]``."""
    from .pseudo_forces import ScreenedExchange

    omega, fraction = solver.screening
    return ScreenedExchange(
        omega=omega, densities=[(w * fraction, P) for w, P in densities],
        operators=operators)


def _unrestricted_gradient(integrals, scf, atom_of_orbital, delta,
                           include_pulay):
    """:func:`kohn_sham_gradient` of a :class:`KohnShamUResult`."""
    from .pseudo_forces import pseudo_nuclear_gradient

    solver = kohn_sham_solver(integrals, scf.n_alpha + scf.n_beta,
                              scf.functional, spins=(scf.n_alpha, scf.n_beta))
    Da = solver._spin_density(scf.mo_coefficients_alpha, scf.n_alpha,
                              scf.occupations_alpha)
    Db = solver._spin_density(scf.mo_coefficients_beta, scf.n_beta,
                              scf.occupations_beta)
    half_core = (0.0 if solver.core_density is None
                 else 0.5 * solver.core_density)
    tau = [None, None]
    if solver.meta:
        half_tau = (0.0 if solver._core_tau is None
                    else 0.5 * solver._core_tau)
        tau = [solver.kinetic_energy_density(D) + half_tau for D in (Da, Db)]
    valence = [solver.density(Da), solver.density(Db)]
    terms = xc_grid.evaluate_spin(
        integrals.grid, valence[0] + half_core, valence[1] + half_core,
        solver.functional, relativistic=solver.relativistic, tau_up=tau[0],
        tau_dn=tau[1], screening=solver.screening,
        exchange_densities=valence)
    meta = terms.tau_potential_up is not None
    core_potential = None
    if solver.screening is not None and solver.core_density is not None:
        bare = xc_grid.evaluate_spin(
            integrals.grid, valence[0] + half_core, valence[1] + half_core,
            solver.functional, relativistic=solver.relativistic,
            screening=(solver.screening[0], 0.0))
        core_potential = 0.5 * (bare.potential_up + bare.potential_dn)

    def mean(a, b):
        return None if a is None else 0.5 * (a + b)

    def half(a, b):
        return None if a is None else 0.5 * (a - b)

    datasets = integrals.pseudopotentials or []
    nothing = [None] * len(integrals.nuclei)
    field = UnrestrictedKohnShamField(
        integrals.grid, np.ascontiguousarray(integrals._engine._psi),
        mean(terms.potential_up, terms.potential_dn),
        mean(terms.tau_potential_up, terms.tau_potential_dn) if meta else None,
        half(terms.potential_up, terms.potential_dn),
        half(terms.tau_potential_up, terms.tau_potential_dn) if meta else None,
        Da - Db,
        _core_functions(datasets) if datasets else nothing,
        _core_tau_functions(datasets) if datasets else nothing,
        _centers(integrals), delta, core_potential=core_potential)
    V_mean = field.matrix()
    X = integrals._lowdin_x()
    D = Da + Db
    P0 = X @ D @ X.conj().T
    e_xc = terms.energy
    exchange = None
    if solver.screening is not None:
        e_exact, (Va, Vb), operators = _exact_exchange_terms(solver,
                                                             [Da, Db])
        e_xc += e_exact
        root = np.linalg.inv(X)                       # S^(1/2)
        V_mean = V_mean + root @ (0.5 * (Va + Vb)) @ root
        field.delta_matrix = (field.delta_matrix
                              + root @ (0.5 * (Va - Vb)) @ root)
        exchange = _screened_exchange(
            solver, [(-0.5, X @ Da @ X.conj().T), (-0.5, X @ Db @ X.conj().T)],
            operators)
    # The constant keeps the energy's value exact at the reference: the
    # magnetization part's value stays in it, its derivative comes from
    # `field.spin_term`.
    constant = e_xc - float(np.real(np.sum(P0 * V_mean.T)))
    energy = KohnShamEnergy(D, V_mean, constant)
    return pseudo_nuclear_gradient(
        integrals, None, None, atom_of_orbital=atom_of_orbital,
        orbital_delta=delta, include_pulay=include_pulay,
        orbital_gradient=False, energy=energy, field=field,
        exchange=exchange)


def d4_dispersion(atoms, functional: str, grad: bool = False) -> dict:
    """D4 of ``atoms`` for ``functional`` from ``dftd4``: ``energy``
    (Hartree) and, with ``grad``, ``gradient`` ``dE/dR`` (Hartree/Bohr,
    ``(n_atoms, 3)``) and ``virial`` (Hartree, ``(3, 3)``).

    Periodic along the directions ``atoms.pbc`` marks: the lattice sum of a
    crystal (its pair dispersion and the coordination numbers that set the
    C6 coefficients both run over the images), whose virial over the cell
    volume is the dispersion's contribution to the stress in ASE's sign."""
    try:
        from dftd4.interface import DampingParam, DispersionModel
    except ImportError as error:
        raise ImportError(
            "dispersion='d4' needs the dftd4 package: "
            "pip install 'mandacaru[dispersion]'") from error
    periodic = np.asarray(atoms.pbc, dtype=bool)
    lattice = (to_bohr(np.asarray(atoms.get_cell()), "angstrom")
               if periodic.any() else None)
    model = DispersionModel(np.asarray(atoms.get_atomic_numbers()),
                            to_bohr(np.asarray(atoms.get_positions()),
                                    "angstrom"),
                            lattice=lattice,
                            periodic=periodic if periodic.any() else None)
    result = model.get_dispersion(DampingParam(method=D4_METHODS[functional]),
                                  grad=grad)
    out = {"energy": float(result["energy"])}
    if grad:
        out["gradient"] = np.asarray(result["gradient"], dtype=float)
        out["virial"] = np.asarray(result["virial"], dtype=float)
    return out


def d4_dispersion_gradient(atoms, functional: str) -> np.ndarray:
    """D4 dispersion gradient ``dE/dR`` (Hartree/Bohr), ``(n_atoms, 3)``."""
    return d4_dispersion(atoms, functional, grad=True)["gradient"]


def d4_dispersion_energy(atoms, functional: str) -> float:
    """D4 dispersion energy (Hartree) of ``atoms`` for ``functional``,
    periodic along ``atoms.pbc``."""
    return d4_dispersion(atoms, functional)["energy"]


@dataclass(frozen=True)
class PeriodicDFTResult:
    r"""A periodic Kohn-Sham result, per cell, energies in ``energy_unit``.

    ``optimal_energy`` is the :math:`\sigma \to 0` estimate the calculator
    reports as the energy; ``free_energy`` is :math:`F = E - \sigma S`, the
    variational quantity.  ``scf`` keeps the band energies (Hartree), the
    occupations, the k-points and their weights.
    """

    method: str
    optimal_energy: float
    free_energy: float
    fermi_level: float
    scf: object
    energy_unit: str
    timings: dict | None = None

    @property
    def success(self) -> bool:
        return bool(self.scf.converged)

    @property
    def optimal_parameters(self) -> np.ndarray:
        return np.empty(0, dtype=float)

    @property
    def num_evaluations(self) -> int:
        return int(self.scf.n_iterations)

    def in_units(self, units: str = "eV") -> float:
        from ..units import convert_energy
        return float(convert_energy(self.optimal_energy, self.energy_unit,
                                    units))


class DFTDriver(_MeanFieldDriver):
    """Closed-shell Kohn-Sham DFT on the shared geometry and basis pipeline.

    ``device`` is accepted and ignored: the calculation is classical and runs
    where Python runs.
    """

    _kind = "dft"
    #: The Kohn-Sham energy is stationary in its density matrix, so its
    #: gradient needs no orbital response: the calculator asks
    #: :meth:`nuclear_gradient` instead of refusing forces.
    analytic_gradient = True

    def __init__(self, xc: str = DEFAULT_XC, dispersion: str | None = None,
                 smearing=None, hartree: str = DEFAULT_HARTREE,
                 **driver_kwargs: object) -> None:
        from .periodic_dft import resolve_smearing
        self.xc = xc_grid.resolve_functional(xc)
        self.dispersion = self._check_dispersion(dispersion, self.xc)
        self.smearing = smearing
        resolve_smearing(smearing)               # validated now, used later
        #: How a molecule's Hartree matrix is built (:data:`HARTREE_METHODS`);
        #: a crystal's is always a periodic Poisson solve.
        self.hartree = resolve_hartree(hartree)
        self._periodic = False
        driver_kwargs.pop("device", None)
        super().__init__(**driver_kwargs)

    # -- the periodic path -------------------------------------------------- #

    @staticmethod
    def _is_periodic(atoms) -> bool:
        return atoms is not None and bool(np.any(atoms.get_pbc()))

    def _build_hamiltonian(self, atoms):
        """A crystal is built by :func:`~mandacaru.pseudopotentials.periodic_paw.build_crystal`;
        a molecule by the shared builder."""
        self._periodic = self._is_periodic(atoms)
        if not self._periodic:
            return self._build_molecular_integrals(atoms)
        if self.hartree != "poisson":
            raise ValueError(
                f"hartree={self.hartree!r} is molecular: a crystal's Hartree "
                "potential is always the periodic Poisson solve; drop it for "
                "a periodic geometry")
        if self.kinetic == "fd":
            # It used to be ignored: the crystal's kinetic energy is always
            # spectral, and a crystal compared with a molecule run on the
            # stencil differed by Hartrees (HISTORY, "K20").
            raise ValueError(
                "a crystal's kinetic energy is spectral, exact on its Bloch "
                "sums; kinetic='fd' is the molecular finite-difference "
                "stencil -- drop it for a periodic geometry (a molecule "
                "compared with a crystal should use kinetic='spectral')")
        from ..pseudopotentials.periodic_paw import build_crystal
        from ._hamiltonian_from_atoms import pseudopotential_family, resolve_basis

        name, spec = resolve_basis(self.basis)
        family = pseudopotential_family(name)
        if family is None or family.name not in ("paw-lcao", "upaw-lcao"):
            raise NotImplementedError(
                "periodic method='dft' needs a PAW-LCAO or UPAW-LCAO basis; "
                f"got {self.basis!r}")
        options = {k: v for k, v in dict(spec or {}).items() if k != "name"}
        crystal, context = build_crystal(
            atoms, self.h, options,
            kpts={"size": tuple(self.kpts), "gamma": self.kpts_gamma},
            family=family.name, grid=self.grid, ghosts=self.ghosts,
            full_mesh=self.electric_field is not None)
        context["integrals"] = crystal
        context["n_electrons"] = float(context["n_electrons"]) - float(
            self.charge)
        self._gradient_context = context
        self._basis_symbols = list(atoms.get_chemical_symbols())
        n = int(round(context["n_electrons"]))
        return None, (n // 2, n - n // 2), crystal.M

    def _build_molecular_integrals(self, atoms):
        """The shared builder, stopped after the integrals.

        Kohn-Sham needs ``h``, the two-body tensor and the grid; the RHF
        orbitals and the second-quantized operator the builder would make
        next are never used (the Hamiltonian is exported in the Kohn-Sham
        orbitals instead, and only on demand), and on H2O DZP they were a
        third of the run.
        """
        from ._hamiltonian_from_atoms import build_basis_hamiltonian

        (_none, num_particles, n_orbitals, profile,
         context) = build_basis_hamiltonian(
            atoms, self.basis, self.grid, self.h, self.charge,
            self.n_electrons, spin=self.spin, kinetic=self.kinetic,
            commensurate=self._grid_commensurate(),
            active_space=self.active_space, hamiltonian=False,
            ghosts=self.ghosts, electric_field=self.electric_field)
        self._integration_profile = profile
        self._gradient_context = context
        self._basis_symbols = list(atoms.get_chemical_symbols())
        return None, num_particles, n_orbitals

    def _check_kpts(self) -> None:
        if not self._periodic:
            super()._check_kpts()

    @staticmethod
    def _check_dispersion(dispersion, functional: str) -> str | None:
        if dispersion in (None, False):
            return None
        key = str(dispersion).strip().lower().replace("-", "")
        if key not in DISPERSION_CORRECTIONS:
            raise ValueError(f"unknown dispersion correction {dispersion!r}; "
                             f"available: {', '.join(DISPERSION_CORRECTIONS)}")
        if xc_grid.has_nonlocal_correlation(functional):
            raise ValueError(
                f"xc={functional!r} already carries its dispersion (rVV10 "
                f"nonlocal correlation); adding D4 would count it twice")
        if functional not in D4_METHODS:
            raise ValueError(
                f"D4 has no parameters for xc={functional!r}; use one of "
                f"{', '.join(sorted(D4_METHODS))}")
        return key

    def _citation_config(self) -> dict[str, object]:
        """Cite Kohn-Sham, the functional and the dispersion that ran."""
        config = super()._citation_config()
        functional = {"lda": ("PerdewZunger1981",),
                      "pbe": ("PBE1996", "PerdewWang1992"),
                      "r2scan": ("Furness2020", "PerdewWang1992"),
                      "hse06": ("Heyd2003", "Krukau2006", "PBE1996",
                                "PerdewWang1992"),
                      "r2scan-rvv10": ("Furness2020", "PerdewWang1992",
                                       "Vydrov2010", "Sabatini2013",
                                       "RomanPerez2009", "Ning2022")}[self.xc]
        dispersion = ("Caldeweyher2019",) if self.dispersion else ()
        paw_exchange = ()
        if xc_grid.is_hybrid(self.xc):
            from ._hamiltonian_from_atoms import (pseudopotential_family,
                                                  resolve_basis)
            family = pseudopotential_family(resolve_basis(self.basis)[0])
            if family is not None and family.name in ("paw-lcao",
                                                      "upaw-lcao"):
                # The one-center exact exchange of the augmentation spheres.
                paw_exchange = ("Paier2005",)
        crystal = ()
        if self._periodic:
            from .periodic_dft import resolve_smearing
            method, _width = resolve_smearing(self.smearing)
            crystal = ("MonkhorstPack1976", "Pulay1980", "Kerker1981",
                       "MethfesselPaxton1989" if method == "methfessel-paxton"
                       else "Mermin1965")
            if self._gradient_context["integrals"].symmetry is not None:
                crystal += ("Togo2018",)          # the space group (spglib)
        spin = ()
        scf = getattr(self, "_scf", None)
        if self.xc == "lda" and (isinstance(scf, KohnShamUResult)
                                 or getattr(scf, "n_spins", 1) == 2):
            # The LDA's spin interpolation; PBE and r2SCAN polarize through
            # the Perdew-Wang uniform gas, already cited.
            spin = ("vonBarthHedin1972",)
        config["extras"] = (tuple(config.get("extras", ())) + functional
                            + paw_exchange + dispersion + crystal + spin)
        return config

    def _configure(self, hamiltonian: Fermion, num_particles: tuple[int, int],
                   n_orbitals: int) -> None:
        """Solve Kohn-Sham on the builder's integrals and export the
        Hamiltonian in the Kohn-Sham orbitals."""
        context = self._gradient_context or {}
        integrals = context.get("integrals")
        if integrals is None:
            raise ValueError(
                "method='dft' needs molecular integrals from a geometry and "
                "basis; a cached or custom qubit Hamiltonian does not contain "
                "the spatial integrals needed for SCF")
        if self._periodic:
            self._configure_periodic(context, num_particles, n_orbitals)
            return
        full_particles = tuple(int(n) for n in num_particles)
        # n_alpha != n_beta (initial magnetic moments, odd electron counts)
        # selects the unrestricted solver.
        self._scf = kohn_sham(integrals, sum(full_particles), self.xc,
                              spins=full_particles, hartree=self.hartree)
        # The exported Hamiltonian and every one-particle picture (density,
        # populations, cube files) are in the model orbitals: the Kohn-Sham
        # orbitals, or the natural orbitals of an unrestricted run.
        integrals.mo_coefficients = self._model_orbitals()
        if self.dispersion:
            # Over the real atoms: a ghost has no electrons to disperse.
            real = [i for i in range(len(self.atoms)) if i not in self.ghosts]
            self._scf.dispersion_energy = d4_dispersion_energy(
                self.atoms[real], self.xc)
        constant = complex(integrals.constant_energy
                           + integrals.nuclear_repulsion)
        M = integrals.n_orbitals
        self.fermion_hamiltonian = LazyHamiltonian(self._scf, constant)
        self.hamiltonian = self.fermion_hamiltonian
        self.num_particles = full_particles
        self.n_spatial_orbitals = int(M)
        self.n_qubits = 2 * M - (2 if self.mapping == "parity_reduced" else 0)
        self._configured = True

    def _configure_periodic(self, context, num_particles, n_orbitals) -> None:
        """Solve the crystal's Kohn-Sham problem."""
        from .periodic_dft import PeriodicKohnSham

        crystal = context["integrals"]
        datasets = crystal.datasets
        _warn_functional_mismatch(datasets, self.xc)
        constant = float(sum(d.one_center_energy for d in datasets)
                         + sum(xc_grid.core_correction_offset(d)
                               for d in datasets))
        solver = PeriodicKohnSham(
            crystal, context["n_electrons"], self.xc, smearing=self.smearing,
            relativistic=(xc_grid.takes_relativistic_exchange(self.xc)
                          and _datasets_relativistic(datasets)),
            constant=constant,
            magnetic_moments=np.delete(
                self.atoms.get_initial_magnetic_moments(), list(self.ghosts)),
            field=self.electric_field,
            field_size=(tuple(self.kpts) if self.electric_field is not None
                        else None))
        self._scf = solver.run()
        # The D4 lattice sum: a constant of the electronic problem, added to
        # the reported energies (`_result`) and its derivatives to the
        # forces and the stress.
        # Over the real atoms, in the same cell: a counterpoise fragment's
        # lattice sum, a ghost having no electrons to disperse.
        real = [i for i in range(len(self.atoms)) if i not in self.ghosts]
        self._dispersion_energy = (
            d4_dispersion_energy(self.atoms[real], self.xc)
            if self.dispersion else 0.0)
        # Kept for the non-self-consistent spectra: it holds the converged
        # potential (`PeriodicKohnSham.bands`).
        self._periodic_solver = solver
        self.fermion_hamiltonian = None
        self.hamiltonian = None
        self.num_particles = tuple(int(n) for n in num_particles)
        self.n_spatial_orbitals = int(n_orbitals)
        self.n_qubits = 0
        self._configured = True

    def nuclear_gradient(self, orbital_delta=None, include_pulay: bool = True):
        """Analytic forces of the converged Kohn-Sham energy (eV/Angstrom).

        :func:`kohn_sham_gradient`, plus the D4 gradient when it ran.
        ``details["energy_hartree"]`` is the rebuilt total energy, constants
        included, for the calculator's consistency check.
        """
        from .forces import HA_BOHR_TO_EV_ANGSTROM

        if self._periodic:
            return self._crystal_gradient(orbital_delta, include_pulay)
        context = self._gradient_context
        scf = self._scf
        result = kohn_sham_gradient(
            context["integrals"], scf,
            atom_of_orbital=context["atom_of_orbital"],
            orbital_delta=orbital_delta, include_pulay=include_pulay)
        if self.dispersion:
            dispersion = (d4_dispersion_gradient(self.atoms, self.xc)
                          * HA_BOHR_TO_EV_ANGSTROM)
            result.hellmann_feynman = result.hellmann_feynman + dispersion
            result.gradient = result.gradient + dispersion
            result.forces = -result.gradient
            result.details["dispersion_gradient"] = dispersion
        result.details["method"] = "kohn-sham"
        result.details["energy_hartree"] += (scf.core_correction_energy
                                             + scf.dispersion_energy)
        return result

    # -- spectra: eigenvalues, bands, densities of states ------------------- #

    def _require_scf(self):
        if getattr(self, "_scf", None) is None:
            raise ValueError(
                "the Kohn-Sham spectrum needs a converged run, so the energy "
                "has to have been computed first: atoms.calc = "
                "Mandacaru(method='dft', ...); atoms.get_potential_energy().")
        return self._scf

    def _require_crystal(self, what: str):
        self._require_scf()
        if not self._periodic:
            raise NotImplementedError(
                f"{what} needs a periodic geometry: a molecule has discrete "
                "levels and no k-dependence.  Use calc.dos() or calc.pdos() "
                "for its broadened spectrum.")
        return self._gradient_context["integrals"]

    def _orbital_shells(self) -> list:
        """``(atom, l)`` of every basis function, in the matrices' order."""
        context = self._gradient_context
        basis = list(context["integrals"].basis)
        owners = list(context["atom_of_orbital"])
        if (not self._periodic and (context.get("frozen")
                                    or context.get("active_space"))) \
                or len(basis) != len(owners):
            raise NotImplementedError(
                "the projected spectrum needs every basis function in the "
                "Kohn-Sham problem; this run froze or removed some")
        return [(int(a), int(f.l)) for a, f in zip(owners, basis)]

    def mean_field_rdm(self) -> np.ndarray:
        """The Kohn-Sham determinant's one-RDM (molecules), or the
        ensemble's when a degenerate shell is partly filled.

        A crystal's density is k-resolved and augmented on the cell grid: its
        populations go through :meth:`crystal_partition` instead, and its
        cube files and dipoles are not wired.
        """
        if self._periodic:
            raise NotImplementedError(
                "cube files, dipoles and natural orbitals of a periodic "
                "Kohn-Sham run are not wired yet; population analysis is "
                "(population=, get_charges), and calc.pdos() gives the "
                "per-atom projections of the states")
        gamma = super().mean_field_rdm()
        # A partly filled degenerate shell: the ensemble's density, not that
        # of the determinant filling one member.
        scf = self._scf
        if self._unrestricted():
            channels = [(scf.mo_coefficients_alpha, scf.occupations_alpha),
                        (scf.mo_coefficients_beta, scf.occupations_beta)]
        else:
            half = (None if scf.occupations is None
                    else 0.5 * scf.occupations)
            channels = [(scf.mo_coefficients, half)] * 2
        V = np.asarray(self.result.model_orbitals)
        M = V.shape[1]
        for block, (C, f) in zip((slice(0, M), slice(M, 2 * M)), channels):
            if f is not None:
                A = V.conj().T @ np.asarray(C)
                gamma[block, block] = (A * f) @ A.conj().T
        return gamma

    def crystal_partition(self, method: str = "hirshfeld", numbers=None):
        """Hirshfeld, Voronoi or Bader populations of the converged crystal.

        :func:`~mandacaru.algorithms.charges.partition_crystal` on the
        k-point density matrices of the SCF; the calculator's
        ``population=``, ``get_charges`` and ``atomic_partition`` reach it.
        """
        from .charges import partition_crystal

        from dataclasses import replace

        self._require_crystal("a crystal population analysis")
        partition = partition_crystal(
            self._gradient_context["integrals"],
            self._periodic_solver.density_matrices, method=method,
            numbers=numbers)
        # One owner for the total moment: the occupations (what
        # `get_total_magnetic_moment` reports); the grid integral equals it
        # up to quadrature.
        return replace(partition, total_magnetic_moment=float(
            getattr(self._scf, "magnetic_moment", 0.0)))

    def _unrestricted(self) -> bool:
        return isinstance(getattr(self, "_scf", None), KohnShamUResult)

    def _molecular_levels(self):
        """``[(eigenvalues, coefficients, n_occupied)]`` per spin (Hartree)."""
        scf = self._scf
        if self._unrestricted():
            return [(np.asarray(scf.mo_energies_alpha, float),
                     np.asarray(scf.mo_coefficients_alpha), scf.n_alpha),
                    (np.asarray(scf.mo_energies_beta, float),
                     np.asarray(scf.mo_coefficients_beta), scf.n_beta)]
        return [(np.asarray(scf.mo_energies, float),
                 np.asarray(scf.mo_coefficients), scf.n_occupied)]

    def get_number_of_spins(self) -> int:
        """2 for a spin-unrestricted molecule or spin-polarized crystal."""
        if self._periodic:
            return int(getattr(self._require_scf(), "n_spins", 1))
        return 2 if self._unrestricted() else 1

    def get_ibz_k_points(self) -> np.ndarray:
        """The SCF k-points (fractional), reduced; Gamma for a molecule."""
        self._require_scf()
        if not self._periodic:
            return np.zeros((1, 3))
        return np.asarray(self._gradient_context["kpoints_fractional"], float)

    def get_k_point_weights(self) -> np.ndarray:
        """Weights of :meth:`get_ibz_k_points`, summing to one."""
        scf = self._require_scf()
        return (np.asarray(scf.weights, float) if self._periodic
                else np.ones(1))

    def get_eigenvalues(self, kpt: int = 0, spin: int = 0) -> np.ndarray:
        """Kohn-Sham eigenvalues (eV) at the ``kpt``-th irreducible k-point.

        Crystals are on the plane-wave reference of
        :meth:`~mandacaru.algorithms.periodic_dft.PeriodicKohnSham.eigenvalue_reference`.
        """
        from ..units import HARTREE_TO_EV

        scf = self._require_scf()
        if spin not in range(self.get_number_of_spins()):
            raise ValueError(f"spin must be one of "
                             f"{list(range(self.get_number_of_spins()))}: the "
                             "run has that many spin channels")
        if self._periodic:
            values = scf.eigenvalues[kpt]
            if self.get_number_of_spins() == 2:
                values = values[spin]
        else:
            values = [self._molecular_levels()[spin][0]][kpt]
        return np.asarray(values, dtype=float) * HARTREE_TO_EV

    def get_fermi_level(self) -> float:
        """The Fermi level (eV); midway between HOMO and LUMO for a molecule."""
        from ..units import HARTREE_TO_EV

        scf = self._require_scf()
        if self._periodic:
            return float(scf.fermi_level) * HARTREE_TO_EV
        occupied, empty = [], []
        for levels, _C, n in self._molecular_levels():
            occupied.extend(levels[:n])
            empty.extend(levels[n:])
        homo = max(occupied)
        mu = homo if not empty else 0.5 * (homo + min(empty))
        return float(mu) * HARTREE_TO_EV

    def _band_path(self, path, npoints: int):
        """The path in the cell the potential was converged in: ``self.atoms``
        is the copy ASE's ``calculate`` stored, not the user's Atoms."""
        atoms = self.atoms
        return atoms.cell.bandpath(path, npoints=int(npoints),
                                   pbc=atoms.get_pbc())

    def band_structure(self, path=None, npoints: int = 100):
        """Non-self-consistent Kohn-Sham bands along a high-symmetry path.

        The converged potential is frozen and :math:`H(\\mathbf k)` is
        diagonalized at every point of ``path`` -- a string such as
        ``"GXWKGLUWLK"``, or ``None`` for the lattice's own path -- so the
        bands are continuous, not limited to the SCF mesh.  Returns ASE's
        :class:`~ase.spectrum.band_structure.BandStructure` (eV, one spin
        channel, ``reference`` the Fermi level); ``.plot()`` draws it.
        """
        from ase.spectrum.band_structure import BandStructure

        from ..units import HARTREE_TO_EV

        crystal = self._require_crystal("a band structure")
        bandpath = self._band_path(path, npoints)
        eigenvalues, _ = self._periodic_solver.bands(
            crystal.cartesian_kpoints(bandpath.kpts))
        self._cite("SetyawanCurtarolo2010")
        if eigenvalues.ndim == 2:                  # one channel: (1, nk, nb)
            eigenvalues = eigenvalues[None]
        return BandStructure(bandpath, eigenvalues * HARTREE_TO_EV,
                             reference=self.get_fermi_level())

    def fat_bands(self, path=None, npoints: int = 100):
        """:meth:`band_structure` plus each state's weight on each atomic shell.

        Returns ``(band_structure, weights)``; ``weights[(atom, l)]`` is a
        ``(n_kpoints, n_bands)`` array of Loewdin weights, and the weights of
        one state sum to one over every ``(atom, l)``.
        """
        from ase.spectrum.band_structure import BandStructure

        from ..units import HARTREE_TO_EV

        crystal = self._require_crystal("fat bands")
        bandpath = self._band_path(path, npoints)
        eigenvalues, weights = self._periodic_solver.bands(
            crystal.cartesian_kpoints(bandpath.kpts), projections=True)
        self._cite("SetyawanCurtarolo2010", "Loewdin1950")
        shells = self._orbital_shells()
        grouped = {}
        for mu, shell in enumerate(shells):
            grouped[shell] = grouped.get(shell, 0.0) + weights[..., mu, :]
        energies = eigenvalues[None] if eigenvalues.ndim == 2 else eigenvalues
        structure = BandStructure(bandpath, energies * HARTREE_TO_EV,
                                  reference=self.get_fermi_level())
        return structure, dict(sorted(grouped.items()))

    def _spectrum(self, kpts, projections: bool):
        """``(eigenvalues eV (nk, n), weights (nk,), state weights, symmetry)``.

        ``kpts=None`` is the SCF mesh; any other mesh is diagonalized
        non-self-consistently after the same symmetry reduction.
        """
        from ..units import HARTREE_TO_EV

        scf = self._require_scf()
        if not self._periodic:
            if kpts is not None:
                raise ValueError("kpts samples a crystal's Brillouin zone; "
                                 "a molecule has none")
            # One "k-point" per spin channel; weight 1/2 each, so the
            # two-electrons-per-state broadening counts one per state.
            channels = self._molecular_levels()
            weights = np.full(len(channels), 1.0 / len(channels))
            levels = np.array([e for e, _C, _n in channels])
            state = (np.array([np.abs(C) ** 2 for _e, C, _n in channels])
                     if projections else None)
            return levels * HARTREE_TO_EV, weights, state, None
        crystal = self._gradient_context["integrals"]
        spin = self.get_number_of_spins() == 2
        if kpts is None and not projections:
            levels = np.array(scf.eigenvalues)
            weights = np.asarray(scf.weights, float)
            if spin:
                # Each (k, spin) pair is an entry of weight w_k / 2: the
                # two-per-state broadening then counts one electron per state.
                levels = np.concatenate([levels[:, 0], levels[:, 1]])
                weights = np.concatenate([weights, weights]) / 2.0
            return levels * HARTREE_TO_EV, weights, None, crystal.symmetry
        from ..pseudopotentials.periodic_paw import reduce_mesh
        from ._hamiltonian_from_atoms import monkhorst_pack_kpts

        spec = (kpts if kpts is not None
                else {"size": tuple(self.kpts), "gamma": self.kpts_gamma})
        _size, _gamma, mesh = monkhorst_pack_kpts(spec)
        points, weights, symmetry = reduce_mesh(mesh, crystal.symmetry)
        eigenvalues, state = self._periodic_solver.bands(
            crystal.cartesian_kpoints(points), projections=projections)
        if spin:
            eigenvalues = eigenvalues.reshape(-1, eigenvalues.shape[-1])
            weights = np.concatenate([weights, weights]) / 2.0
            if state is not None:
                state = state.reshape(-1, *state.shape[-2:])
        return eigenvalues * HARTREE_TO_EV, weights, state, symmetry

    @staticmethod
    def _energy_axis(eigenvalues, width: float, npoints: int, energies):
        if energies is not None:
            return np.asarray(energies, dtype=float)
        return np.linspace(eigenvalues.min() - 5.0 * width,
                           eigenvalues.max() + 5.0 * width, int(npoints))

    def dos(self, width: float = 0.1, npoints: int = 2001, kpts=None,
            energies=None):
        """Gaussian-broadened density of states, ``(energies, dos)``.

        ``energies`` in eV (on the eigenvalues' reference -- subtract
        :meth:`get_fermi_level` to center it), ``dos`` in states/eV per cell
        (per molecule), both spins: it integrates to twice the number of
        bands.  ``width`` is the Gaussian's standard deviation (eV).  A
        crystal's ``kpts`` (a mesh, as for the calculator) is diagonalized
        non-self-consistently -- a denser mesh than the SCF one gives a
        smoother curve; ``None`` uses the SCF mesh.
        """
        from .periodic_dft import broadened_dos

        eigenvalues, weights, _state, _sym = self._spectrum(kpts, False)
        axis = self._energy_axis(eigenvalues, width, npoints, energies)
        return axis, broadened_dos(axis, eigenvalues, weights, width)

    def pdos(self, width: float = 0.1, npoints: int = 2001, kpts=None,
             energies=None):
        """Density of states projected on atomic shells, ``(energies, pdos)``.

        ``pdos[(atom, l)]`` is the share of :meth:`dos` carried by the
        Loewdin-orthogonalized orbitals of angular momentum ``l`` on atom
        ``atom`` (index into the Atoms); the shells sum to the total.  On a
        symmetry-reduced mesh each atom's share is averaged over the atoms
        the operations map it to, which makes it the full mesh's.
        """
        from .periodic_dft import broadened_dos

        eigenvalues, weights, state, symmetry = self._spectrum(kpts, True)
        axis = self._energy_axis(eigenvalues, width, npoints, energies)
        per_orbital = broadened_dos(axis, eigenvalues, weights, width,
                                    state_weights=state)
        self._cite("Loewdin1950")
        grouped = {}
        for mu, shell in enumerate(self._orbital_shells()):
            grouped[shell] = grouped.get(shell, 0.0) + per_orbital[:, mu]
        if symmetry is not None:
            maps = symmetry.atom_maps
            grouped = {(atom, l): sum(grouped.get((int(m[atom]), l), 0.0)
                                      for m in maps) / len(maps)
                       for (atom, l) in grouped}
        return axis, dict(sorted(grouped.items()))

    def fermi_surface(self, size) -> np.ndarray:
        """Band energies (eV) on the full Gamma-centered ``size`` mesh.

        ``size = (n1, n2, n3)``; the mesh is ``k = (i1/n1, i2/n2, i3/n3)``,
        the grid a ``.bxsf`` file spans.  Only the irreducible points are
        diagonalized (non-self-consistently, the potential frozen) and the
        rest are filled in by symmetry -- a band energy is invariant under
        the crystal's rotations and under time reversal.  Returns
        ``(n_bands, n1, n2, n3)``, or ``(2, n_bands, n1, n2, n3)`` for a
        spin-polarized crystal, on the eigenvalues' reference
        (:meth:`get_fermi_level` is on the same one).
        """
        from ..core.symmetry import irreducible_kpoints
        from ..units import HARTREE_TO_EV

        crystal = self._require_crystal("a Fermi surface")
        size = tuple(int(n) for n in size)
        if len(size) != 3 or min(size) < 1:
            raise ValueError("size must be three positive integers "
                             f"(n1, n2, n3); got {size!r}")
        index = np.indices(size).reshape(3, -1).T
        mesh = index / np.asarray(size, dtype=float)
        operations = (crystal.symmetry.keeping_mesh(mesh)
                      if crystal.symmetry is not None else None)
        if operations is not None:
            zone = irreducible_kpoints(mesh, operations.info,
                                       time_reversal=True)
            points, mapping = zone.points, zone.mapping
        else:
            # Time reversal alone: -k is the mesh point (-i mod n).
            flat = np.ravel_multi_index(index.T, size)
            partner = np.ravel_multi_index((-index % size).T, size)
            keep = np.flatnonzero(flat <= partner)
            slot = np.empty(len(mesh), dtype=int)
            slot[keep] = np.arange(len(keep))
            slot[partner[keep]] = slot[keep]
            points, mapping = mesh[keep], slot
        eigenvalues, _ = self._periodic_solver.bands(
            crystal.cartesian_kpoints(points))
        eigenvalues = np.asarray(eigenvalues) * HARTREE_TO_EV
        full = eigenvalues[..., mapping, :]             # (..., nk, M)
        return np.moveaxis(full, -1, -2).reshape(
            full.shape[:-2] + (full.shape[-1],) + size)

    def get_polarization(self, kpts=None) -> np.ndarray:
        """The Berry-phase polarization of an insulating crystal (C/m^2).

        Strings of k-points are cut from the ``kpts`` mesh size (default:
        the SCF's) and diagonalized at the converged potential
        (:func:`~mandacaru.algorithms.berry_phase.berry_polarization`).
        Returns the vector on the branch nearest zero; the whole
        :class:`~mandacaru.algorithms.berry_phase.Polarization` -- quanta,
        Berry phases, ionic and electronic parts -- is left on
        :attr:`polarization_result`.  A polarization is defined modulo its
        quanta: compare *changes* along a path (``raw``).
        """
        from .berry_phase import berry_polarization

        self._require_crystal("a Berry-phase polarization")
        if kpts is None:
            size = tuple(self.kpts)
        elif isinstance(kpts, dict):
            size = tuple(kpts.get("size", ()))
        else:
            size = tuple(kpts)
        self._cite("KingSmith1993", "Resta1994")
        solver = self._periodic_solver
        if solver.field is not None:
            self._cite("Souza2002", "Umari2002")
            # In a field the states are the SCF's own (a rediagonalization
            # at the frozen potential would drop the field term): the
            # polarization comes from their strings, on the SCF's mesh.
            from .berry_phase import occupied_bands, polarization_from_phases
            if kpts is not None and size != tuple(solver.field_size):
                raise ValueError("in a finite field the polarization is that "
                                 "of the SCF's own mesh; leave kpts unset")
            occupied_bands(self._scf)
            self.polarization_result = polarization_from_phases(
                solver.crystal, solver.field_state[1], 2.0,
                solver.field_size)
        else:
            self.polarization_result = berry_polarization(solver, self._scf,
                                                          size)
        return self.polarization_result.vector

    def _mesh_size(self, kpts) -> tuple:
        """The full mesh a ``kpts`` option names (default the SCF's)."""
        if kpts is None:
            return tuple(self.kpts)
        if isinstance(kpts, dict):
            return tuple(kpts.get("size", ()))
        return tuple(kpts)

    def projectabilities(self, guess, *, kpts=None, bands=None):
        """Each Bloch state's projectability onto target atomic orbitals
        (:class:`~mandacaru.algorithms.band_selection.Projectabilities`).

        ``guess`` names the targets as for :meth:`wannier` -- orbitals on
        atoms, ``{"Cu": "d"}`` or ``[("Si", "sp3")]`` -- and they are the
        atoms' own first-zeta basis orbitals, so the projectability is
        exact (PAW overlap included), lies in [0, 1] and sums over every
        band to the number of targets.  ``kpts`` is a full mesh (default the
        SCF's size) or fractional k-points as rows; ``bands`` default every
        band.  ``wannier(..., windows="auto")`` freezes the widest energy
        window whose states all have ``p >= 0.95`` and leaves out the
        states below 0.02.
        """
        from .band_selection import (Projectabilities,
                                     mesh_projectabilities,
                                     point_projectabilities)
        from .wannier import resolve_guess

        crystal = self._require_crystal("projectabilities")
        solver = self._periodic_solver
        trials = resolve_guess(guess, self.atoms, crystal)
        bands = tuple(range(crystal.M)) if bands is None else tuple(
            int(b) for b in bands)
        explicit = kpts is not None and np.ndim(kpts) == 2
        size = None if explicit else self._mesh_size(kpts)
        energies, values = [], []
        for spin in range(int(solver.n_spins)):
            if explicit:
                fractional = np.asarray(kpts, dtype=float)
                e, p = point_projectabilities(
                    solver, crystal.cartesian_kpoints(fractional), bands,
                    trials, spin)
            else:
                fractional, e, p = mesh_projectabilities(solver, size, bands,
                                                         trials, spin)
            energies.append(e)
            values.append(p)
        self._cite("Loewdin1950", "Sayfutyarova2017", "Qiao2023")
        squeeze = (lambda x: x[0]) if solver.n_spins == 1 else np.stack
        return Projectabilities(
            kpoints=fractional, energies=squeeze(energies),
            values=squeeze(values), bands=bands,
            labels=tuple(t.label for t in trials),
            fermi_level=self.get_fermi_level(), size=size)

    def natural_orbitals(self, method: str = "rpa", *, kpts=None,
                         bands=None, cutoff=None, threshold=None):
        """The Bloch natural orbitals of the crystal's correlated density
        matrix (:class:`~mandacaru.algorithms.rpa_density.
        CrystalNaturalOrbitals`), ranked by how far their occupations are
        from 2 or 0.

        ``method="rpa"``: the direct-RPA (ring coupled-cluster doubles)
        amplitudes on the full ``kpts`` mesh (default the SCF's size),
        from every band of the basis (or ``bands``), the Coulomb
        integrals on q + G within ``cutoff`` (Bohr^-1); ``threshold`` is
        the deviation from 2 or 0 that selects an orbital.  The result's
        ``n_functions`` and projectabilities drive ``wannier(...,
        windows=result)``.  Spin-restricted crystals.
        """
        from .rpa_density import DEFAULT_DEVIATION_THRESHOLD, crystal_rpa

        if method != "rpa":
            raise ValueError(f"unknown natural-orbital method {method!r}; "
                             "use 'rpa'")
        self._require_crystal("natural orbitals")
        size = self._mesh_size(kpts)
        self._cite("Scuseria2008", "Furche2008")
        self.natural_orbitals_result = crystal_rpa(
            self._periodic_solver, size, float(self._scf.fermi_level),
            bands=bands, cutoff=cutoff,
            threshold=(DEFAULT_DEVIATION_THRESHOLD if threshold is None
                       else float(threshold)))
        return self.natural_orbitals_result

    def wannier(self, n_functions=None, *, guess="bonds", windows=None,
                bands=None, kpts=None, trial_radial="gaussian", **options):
        """Maximally localized Wannier functions of a crystal's bands
        (:func:`~mandacaru.algorithms.wannier.wannier_functions`).

        Without ``windows`` the ``bands`` (indices; default the occupied
        bands of an insulator) are an isolated group, one function per
        band.  With ``windows`` -- ``{"outer": (lo, hi), "frozen": (lo,
        hi)}`` in eV, on the scale of :meth:`get_fermi_level` --
        ``n_functions`` are disentangled from the ``bands`` (default every
        band) inside the outer window, the frozen window's states kept
        exactly.  ``windows`` can instead choose the states by what they
        are (:mod:`~mandacaru.algorithms.band_selection`): ``"auto"`` by
        their projectability onto the guess' atomic orbitals (the frozen
        window the widest energy window whose states all reach 0.95, states
        below 0.02 left out; ``{"projectability": (outer, frozen)}`` sets
        the two), ``"scdm"`` by the erfc-weighted projections of the
        selected-columns scheme with no windows (``{"scdm": (mu, sigma)}``
        in eV), ``"rpa"`` by the projectability onto the most correlated
        RPA natural orbitals (:meth:`natural_orbitals`, or its result in
        place of the string), which also sets ``n_functions``.  ``guess`` gives the starting trial orbitals, one per
        function: ``"bonds"`` (s Gaussians on the nearest-neighbor bond
        midpoints, a covalent crystal's valence), ``"sp3"`` (hybrids on
        every atom toward its four neighbors), orbitals on atoms, or
        centers (Angstrom) for s Gaussians.  Orbitals on atoms
        (:func:`~mandacaru.algorithms.wannier.orbital_trials`) are a dict
        ``{site: orbitals}`` or a list of ``(site, orbitals)`` or ``(site,
        orbitals, frame)``: ``site`` a chemical symbol (every atom of it),
        an atom index or, in a list, a position (Angstrom); ``orbitals`` a
        name or a list of names -- a shell ``"s"``, ``"p"``, ``"d"``,
        ``"f"`` (every m, from -l to l), one real harmonic (``"px"``,
        ``"py"``, ``"pz"``, ``"dxy"``, ``"dyz"``, ``"dz2"``, ``"dxz"``,
        ``"dx2-y2"``), ``"t2g"`` or ``"eg"``, or hybrids ``"sp"``,
        ``"sp2"``, ``"sp3"``; ``frame`` the local x, y, z axes as rows
        (default Cartesian), in which the harmonics, the t2g/eg split and
        the hybrids' directions are defined.  For example ``{"V":
        "t2g"}``, ``[("Cu", ["s", "p", "d"])]``, ``[("V", "t2g", frame)]``.
        An explicit guess (orbitals or centers) sets ``n_functions`` by
        default.  ``trial_radial`` is ``"gaussian"`` or ``"basis"``, the
        radial functions of the atoms' own basis orbitals.  ``kpts`` is the full mesh (default: the SCF's
        size), diagonalized at the converged potential.  A spin-polarized
        crystal returns a
        :class:`~mandacaru.algorithms.wannier.SpinWannierResult`: each
        channel's functions from its own Bloch states, the trial orbitals
        shared, and ``bands``, ``n_functions`` and ``windows`` each either
        shared or a pair ``(up, down)``.
        """
        from .band_selection import resolve_selection, selection_citations
        from .berry_phase import occupied_bands
        from .wannier import per_spin, resolve_guess, wannier_functions

        self._require_crystal("Wannier functions")
        n_spins = int(self._periodic_solver.n_spins)
        if bands is None:
            if windows is None:
                try:
                    counts = occupied_bands(self._scf)
                except NotImplementedError as error:
                    raise ValueError(
                        "the occupied bands are not a separated group (a "
                        "metal): give the bands of the group (bands=[...]) "
                        "or energy windows to disentangle the functions "
                        "from") from error
                if min(counts) == 0:
                    raise ValueError(
                        f"a spin channel has no occupied bands ({counts}): "
                        "give the bands of the group explicitly")
                bands = ([list(range(c)) for c in counts] if n_spins == 2
                         else list(range(counts[0])))
            else:
                bands = list(range(self._periodic_solver.crystal.M))
        trials = resolve_guess(guess, self.atoms,
                               self._periodic_solver.crystal, trial_radial)
        explicit = not isinstance(guess, str)
        size = self._mesh_size(kpts)
        selection = resolve_selection(windows)
        if selection is not None:
            if selection.kind == "natural" and selection.natural is None:
                selection.natural = self.natural_orbitals(method="rpa",
                                                          kpts=size)
            if selection.kind == "natural" and n_functions is None:
                n_functions = selection.natural.n_functions
            windows = selection
        counts = []
        for group, count, window in zip(per_spin(bands, n_spins, "bands"),
                                        per_spin(n_functions, n_spins, "int"),
                                        per_spin(windows, n_spins,
                                                 "windows")):
            group = list(group)
            if count is None:
                count = (len(trials) if explicit or window is not None
                         else len(group))
            if len(trials) != count:
                raise ValueError(
                    f"the guess gives {len(trials)} trial orbitals for "
                    f"{count} functions: give one per function")
            if window is None and count != len(group):
                raise ValueError(
                    f"{count} functions from {len(group)} bands: without "
                    "windows the bands are an isolated group, one function "
                    f"per band; give {count} bands (bands=[...]) or energy "
                    "windows to disentangle them: windows={'outer': (lo, "
                    "hi), 'frozen': (lo, hi)} (eV)")
            counts.append(int(count))
        self._cite("Marzari1997", "Marzari2012")
        if windows is not None:
            self._cite(*selection_citations(selection))
        n_functions = counts[0] if n_spins == 1 else tuple(counts)
        self.wannier_result = wannier_functions(
            self._periodic_solver, size, bands, trials,
            n_functions=n_functions, windows=windows,
            fermi_level=float(self._scf.fermi_level), cite=self._cite,
            **options)
        return self.wannier_result

    def write_fermi_surface(self, path, size, *, spin=None,
                            bands=None) -> str:
        """Write :meth:`fermi_surface` as an XCrySDen ``.bxsf`` file.

        ``bands`` selects band indices (default: every band); a
        spin-polarized crystal needs ``spin`` (0 or 1), one channel per
        file.  The Fermi energy in the header is :meth:`get_fermi_level`.
        Returns the path written (:func:`~mandacaru.utils.bxsf.write_bxsf`).
        """
        from ..units import ANGSTROM_TO_BOHR
        from ..utils.bxsf import write_bxsf

        energies = self.fermi_surface(size)
        if self.get_number_of_spins() == 2:
            if spin not in (0, 1):
                raise ValueError("a spin-polarized crystal has one Fermi "
                                 "surface per channel: pass spin=0 or 1")
            energies = energies[spin]
        elif spin not in (None, 0):
            raise ValueError("spin selects a channel of a spin-polarized "
                             "crystal; this one is not")
        if bands is not None:
            energies = energies[np.asarray(bands, dtype=int)]
        crystal = self._gradient_context["integrals"]
        # Rows b_i in Bohr^-1 -> Angstrom^-1 (with the 2 pi).
        reciprocal = crystal.cartesian_kpoints(np.eye(3)) * ANGSTROM_TO_BOHR
        return write_bxsf(path, energies, reciprocal, self.get_fermi_level(),
                          comment=f"{self.atoms.get_chemical_formula()} "
                                  f"{self.xc} band energies (eV)")

    # -- crystal forces and stress ------------------------------------------ #

    def _kpts_spec(self) -> dict:
        return {"size": tuple(self.kpts), "gamma": self.kpts_gamma}

    def _crystal_gradient(self, orbital_delta, include_pulay):
        r""":func:`~mandacaru.algorithms.crystal_forces.crystal_gradient` of
        the converged crystal: :math:`dF/d\mathbf R`, F the free energy."""
        from .crystal_forces import DEFAULT_DELTA, crystal_gradient

        from .forces import HA_BOHR_TO_EV_ANGSTROM

        result = crystal_gradient(
            self._periodic_solver,
            delta=DEFAULT_DELTA if orbital_delta is None else float(
                orbital_delta),
            include_pulay=include_pulay)
        if self.dispersion:
            dispersion = (d4_dispersion_gradient(self.atoms, self.xc)
                          * HA_BOHR_TO_EV_ANGSTROM)
            result.hellmann_feynman = result.hellmann_feynman + dispersion
            result.gradient = result.gradient + dispersion
            result.forces = -result.gradient
            result.details["dispersion_gradient"] = dispersion
        # The calculator checks the energy the gradient belongs to against
        # the reported one; a crystal's gradient is of F at the reported
        # state, so both are carried.
        result.details["energy_hartree"] = float(
            self._scf.extrapolated_energy + self._dispersion_energy)
        result.details["free_energy_hartree"] = float(
            self._scf.free_energy + self._dispersion_energy)
        return result

    def crystal_stress(self, strain=None) -> np.ndarray:
        """``(3, 3)`` stress of the converged crystal, Hartree/Bohr^3, ASE's
        sign (:func:`~mandacaru.algorithms.crystal_forces.crystal_stress`)."""
        from ._hamiltonian_from_atoms import pseudopotential_family, resolve_basis
        from .crystal_forces import DEFAULT_STRAIN, crystal_stress

        self._require_crystal("a stress")
        name, spec = resolve_basis(self.basis)
        family = pseudopotential_family(name)
        options = {k: v for k, v in dict(spec or {}).items() if k != "name"}
        stress = crystal_stress(
            self._periodic_solver, self.atoms, options, family.name,
            self._kpts_spec(),
            strain=DEFAULT_STRAIN if strain is None else float(strain))
        if self.dispersion:
            # dftd4's virial over the volume is the stress in ASE's sign.
            volume = abs(np.linalg.det(to_bohr(np.asarray(self.atoms.get_cell()),
                                               "angstrom")))
            stress = stress + d4_dispersion(self.atoms, self.xc,
                                            grad=True)["virial"] / volume
        return stress

    def _model_orbitals(self) -> np.ndarray:
        """The orbitals the many-body problem is exported in."""
        scf = self._scf
        if isinstance(scf, KohnShamUResult):
            return np.asarray(scf.natural_orbitals)
        return np.asarray(scf.mo_coefficients)

    def _result(self, timings, run_t0) -> MeanFieldResult:
        """The :class:`MeanFieldResult` of the Kohn-Sham SCF.

        The total energy carries the dispersion and core-correction
        constants; the reference energy is the Kohn-Sham determinant's in the
        exported Hamiltonian, which has neither.
        """
        if self._periodic:
            scf = self._scf
            dispersion = self._dispersion_energy
            result = PeriodicDFTResult(
                method=self._kind,
                optimal_energy=self._to_energy_units(
                    scf.extrapolated_energy + dispersion),
                free_energy=self._to_energy_units(scf.free_energy + dispersion),
                fermi_level=self._to_energy_units(scf.fermi_level),
                scf=scf, energy_unit=self._energy_unit_label())
            self._finalize_timings(timings, run_t0)
            return replace(result, timings=timings.as_dict())
        integrals = self._gradient_context["integrals"]
        constant = float(integrals.constant_energy + integrals.nuclear_repulsion)
        scf = self._scf
        total = (scf.electronic_energy + scf.core_correction_energy
                 + scf.dispersion_energy + constant)
        # Built once, timings included: `dataclasses.replace` would read every
        # field and so build the lazy Hamiltonian it exists to defer.
        self._finalize_timings(timings, run_t0)
        return MeanFieldResult(
            method=self._kind, optimal_energy=self._to_energy_units(total),
            reference_energy=self._to_energy_units(
                scf.determinant_energy + constant), scf=scf,
            model_orbitals=self._model_orbitals().copy(),
            fermion_hamiltonian=self.fermion_hamiltonian,
            num_particles=self.num_particles,
            n_spatial_orbitals=self.n_spatial_orbitals,
            energy_unit=self._energy_unit_label(),
            timings=timings.as_dict())

    def _log_title(self) -> str:
        return f"DFT ({self.xc.upper()})"

    def _electron_fields(self) -> dict:
        """The molecular ``[ELECTRONS]`` block, with a crystal's kinetic
        operator: always spectral (the line said "finite difference")."""
        fields = super()._electron_fields()
        if self._periodic:
            fields["kinetic operator"] = "spectral"
        return fields

    def _scf_setup_fields(self) -> dict:
        """``[SCF SETUP]``: the functional and what the SCF stops on."""
        if self._periodic:
            from .periodic_dft import PeriodicKohnSham, resolve_smearing
            from ..units import HARTREE_TO_EV
            method, width = resolve_smearing(self.smearing)
            defaults = inspect.signature(PeriodicKohnSham.run).parameters
            crystal = self._gradient_context["integrals"]
            return {
                "scf_method": "periodic Kohn-Sham (Bloch states)",
                "xc_functional": self.xc.upper(),
                **_hybrid_setup(self.xc),
                "k_points_irreducible": (
                    f"{len(crystal.kpoints)} (time reversal)"
                    if crystal.symmetry is None else
                    f"{len(crystal.kpoints)} ({crystal.symmetry.n_operations} "
                    f"of {crystal.symmetry.n_space_group} space-group "
                    f"operations on the grid, and time reversal)"),
                "smearing": f"{method}, {width * HARTREE_TO_EV:g} eV",
                "spin": ("polarized (initial moments "
                         f"{[round(float(m), 3) for m in self.atoms.get_initial_magnetic_moments()]})"
                         if np.any(self.atoms.get_initial_magnetic_moments())
                         else "restricted"),
                "max_iterations": defaults["max_iter"].default,
                "convergence_Hartree": (
                    f"{defaults['tol'].default:g} (free energy change) and "
                    f"{defaults['density_tol'].default:g} electrons "
                    "(density residual)"),
                "mixing": ("Pulay, Kerker-preconditioned (the total "
                           "density; the magnetization undamped)"
                           if np.any(self.atoms.get_initial_magnetic_moments())
                           else "Pulay, Kerker-preconditioned"),
                "energy_unit": self._energy_unit_label(),
            }
        defaults = inspect.signature(KohnSham.run).parameters
        hybrid = _hybrid_setup(self.xc)
        return {
            "scf_method": ("unrestricted Kohn-Sham (open shell)"
                           if self._unrestricted() else
                           "restricted Kohn-Sham (closed shell)"),
            "xc_functional": self.xc.upper(),
            **hybrid,
            "dispersion": (self.dispersion or "none").upper(),
            "hartree": ("Poisson solve of the density per iteration (no "
                        "two-body tensor)" if self.hartree == "poisson"
                        else "contracted from the two-body tensor"),
            "max_iterations": defaults["max_iter"].default,
            "convergence_Hartree": (f"{defaults['tol'].default:g} (energy "
                                    f"change and largest density change)"),
            "acceleration": (f"DIIS; level shift {LEVEL_SHIFT:g} Hartree "
                             f"while the energy rises by more than "
                             f"{LEVEL_SHIFT_TRIGGER:g} Hartree"),
            "energy_unit": self._energy_unit_label(),
        }

    def _scf_summary_fields(self, result: MeanFieldResult) -> dict:
        """``[SCF SUMMARY]``: the Kohn-Sham energy and its decomposition."""
        if self._periodic:
            unit = result.energy_unit
            scf = self._scf
            gap = scf.band_gap
            fields = {"converged": str(bool(scf.converged)),
                      f"optimal_energy_{unit}": f"{result.optimal_energy:.10f}",
                      f"free_energy_{unit}": f"{result.free_energy:.10f}",
                      f"fermi_level_{unit}": f"{result.fermi_level:.6f}",
                      f"band_gap_{unit}": ("none (metallic)" if gap is None
                                           else f"{self._to_energy_units(gap):.6f}"),
                      "scf_iterations": result.num_evaluations}
            if getattr(scf, "n_spins", 1) == 2:
                fields["magnetic_moment_bohr_magneton"] = \
                    f"{scf.magnetic_moment:.6f}"
            if self.dispersion:
                fields[f"dispersion_energy_{unit}"] = \
                    f"{self._to_energy_units(self._dispersion_energy):.10f}"
            return fields
        fields = super()._scf_summary_fields(result)
        unit = result.energy_unit
        to_unit = self._to_energy_units
        scf = self._scf
        # The exported problem starts from the Kohn-Sham determinant, whose
        # energy in the many-body Hamiltonian is not the Kohn-Sham energy.
        fields[f"reference_energy_{unit}"] = \
            f"{result.reference_energy:.10f}"
        fields[f"hartree_energy_{unit}"] = \
            f"{to_unit(scf.hartree_energy):.10f}"
        fields[f"xc_energy_{unit}"] = f"{to_unit(scf.xc_energy):.10f}"
        if xc_grid.is_hybrid(self.xc):
            # Part of xc_energy, reported as its own fact: the share of the
            # exchange-correlation energy that the exact exchange carries.
            fields[f"exact_exchange_energy_{unit}"] = \
                f"{to_unit(scf.exact_exchange_energy):.10f}"
        if scf.core_correction_energy:
            fields[f"core_correction_energy_{unit}"] = \
                f"{to_unit(scf.core_correction_energy):.10f}"
        if self.dispersion:
            fields[f"dispersion_energy_{unit}"] = \
                f"{to_unit(scf.dispersion_energy):.10f}"
        return fields
