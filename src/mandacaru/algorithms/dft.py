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
pseudopotential), ``J`` the Hartree matrix contracted from the same two-body
tensor the quantum Hamiltonian uses, and :math:`\rho(\mathbf r) =
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
compensation charges in the Hartree term (they are in the two-body tensor)
and the dataset's smooth core in the functional, exactly how the dataset was
unscreened.  The datasets are LDA: another functional on them is a mismatch
between molecule and dataset, and is warned about.

Restricted for a closed shell, unrestricted when ``n_alpha != n_beta``
(:class:`UnrestrictedKohnSham`).  A **periodic** geometry (any ``atoms.pbc``)
takes the crystal path instead -- Bloch states on a Monkhorst-Pack mesh with
smearing, spin-polarized when the atoms carry initial moments, PAW-LCAO only
(:mod:`~mandacaru.algorithms.periodic_dft`); it has no dispersion yet.

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

#: Dispersion corrections ``dispersion=`` accepts.
DISPERSION_CORRECTIONS = ("d4",)

#: The functionals D4 is parameterized for, by the ``dftd4`` method name.
D4_METHODS = {"pbe": "pbe", "r2scan": "r2scan", "hse06": "hse06"}


@dataclass
class KohnShamResult(RHFResult):
    """A converged closed-shell Kohn-Sham determinant.

    The fields of :class:`~mandacaru.algorithms.hartree_fock.RHFResult` (MO
    energies are the Kohn-Sham eigenvalues; ``h_mo`` and ``eri_mo`` are the
    integrals in the Kohn-Sham orbitals) plus the energy decomposition, all in
    Hartree.  ``electronic_energy`` excludes every constant; the dispersion
    and core-correction constants are kept apart so each can be reported.
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

    @property
    def determinant_energy(self) -> float:
        r"""Energy of the Kohn-Sham determinant in the many-body Hamiltonian.

        :math:`\sum_i 2h_{ii} + \sum_{ij}(2\langle ij|ij\rangle -
        \langle ij|ji\rangle)` over the occupied Kohn-Sham orbitals (Hartree,
        no constant): the Hartree-Fock functional of these orbitals, which is
        what the exported problem's reference state has -- not the Kohn-Sham
        energy.
        """
        occ = slice(0, self.n_occupied)
        h = np.real(np.diagonal(self.h_mo)[occ])
        g = np.real(self.eri_mo[occ, occ, occ, occ])
        coulomb = np.einsum("ijij->ij", g)
        exchange = np.einsum("ijji->ij", g)
        return float(2.0 * h.sum() + np.sum(2.0 * coulomb - exchange))

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
    the first time the attribute is read.
    """

    def __init__(self, h_mo, eri_mo, constant):
        self.h_mo = h_mo
        self.eri_mo = eri_mo
        self.constant = complex(constant)

    def build(self) -> Fermion:
        h_so, g_so = spin_block_integrals(self.h_mo, self.eri_mo)
        n_modes = h_so.shape[0]
        return (Fermion.from_integrals(h_so, g_so)
                + Fermion({(): self.constant}, n_modes=n_modes))


class KohnSham:
    r"""Closed-shell (restricted) Kohn-Sham SCF in an orthonormal basis.

    Parameters
    ----------
    h : (M, M) array
        Core Hamiltonian in the orthonormal basis (Hartree).
    eri : (M, M, M, M) array
        ``<pq|rs>`` in physicists' notation, same basis.
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
        self.eri = np.asarray(eri, dtype=complex)
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

    def _density_matrix(self, C) -> np.ndarray:
        """``D_pq = 2 sum_i^occ C_pi C*_qi``."""
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
        # The same contraction as the Hartree-Fock Coulomb matrix, density
        # entering as D_sr (see RHF._fock).
        return np.einsum("sr,prqs->pq", D, self.eri, optimize=True)

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
        """
        _eps, C = np.linalg.eigh(self.h)
        D = self._density_matrix(C)
        energy = np.inf
        converged = False
        mixer = DIIS() if diis else None
        shift = 0.0
        it = 0
        history, start = [], time.perf_counter()
        for it in range(1, max_iter + 1):
            F, current, _e_h, _e_xc = self._fock(D)
            if mixer is not None and it % DIIS_RESTART == 0:
                mixer = DIIS()
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
            D_new = self._density_matrix(C)
            change = float(np.max(np.abs(D_new - D)))
            history.append(scf_record(it, start, current, energy, change))
            if abs(current - energy) < tol and change < tol:
                D, energy = D_new, current
                converged = True
                break
            D, energy = D_new, current

        F, energy, e_hartree, e_xc = self._fock(D)
        eps, C = np.linalg.eigh(F)                  # canonical KS orbitals
        h_mo, eri_mo = transform_integrals(self.h, self.eri, C)
        return KohnShamResult(
            electronic_energy=energy, mo_energies=np.real(eps),
            mo_coefficients=C, n_occupied=self.n_occ, converged=converged,
            h_mo=np.real_if_close(h_mo), eri_mo=np.real_if_close(eri_mo),
            n_iterations=it, functional=self.functional,
            hartree_energy=e_hartree, xc_energy=e_xc, history=history,
            exact_exchange_energy=self.exact_exchange_energy)


@dataclass
class KohnShamUResult(UHFResult):
    """A converged spin-unrestricted Kohn-Sham determinant.

    The fields of :class:`~mandacaru.algorithms.hartree_fock.UHFResult`
    (MO energies are the Kohn-Sham eigenvalues of each spin; the many-body
    integrals are in the natural orbitals of the total density, and
    ``reference_energy`` is the Hartree-Fock energy of their determinant) plus
    the Kohn-Sham energy decomposition, all in Hartree.
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
    def _spin_density(C, n: int) -> np.ndarray:
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

    def _determinant_energy(self, Ra, Rb) -> float:
        """Hartree-Fock energy of the determinant with densities ``Ra``,
        ``Rb`` -- what the exported many-body Hamiltonian assigns it."""
        D = Ra + Rb
        J = self._hartree(D)

        def exchange(R):
            return np.einsum("sr,prsq->pq", R, self.eri, optimize=True)
        value = (np.sum(D * self.h.T) + 0.5 * np.sum(D * J.T)
                 - 0.5 * np.sum(Ra * exchange(Ra).T)
                 - 0.5 * np.sum(Rb * exchange(Rb).T))
        return float(np.real(value))

    def run(self, max_iter: int = 200, tol: float = 1e-8,
            diis: bool = True) -> KohnShamUResult:
        """Iterate to self-consistency from the core guess (both spins)."""
        na, nb = self.n_alpha, self.n_beta
        _eps, C = np.linalg.eigh(self.h)
        Da, Db = self._spin_density(C, na), self._spin_density(C, nb)
        energy = np.inf
        converged = False
        mixers = (DIIS(), DIIS()) if diis else None
        shift = 0.0
        it = 0
        history, start = [], time.perf_counter()
        for it in range(1, max_iter + 1):
            Fa, Fb, current, _e_h, _e_xc = self._fock_pair(Da, Db)
            if mixers is not None and it % DIIS_RESTART == 0:
                mixers = (DIIS(), DIIS())
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
            Da_new, Db_new = self._spin_density(Ca, na), self._spin_density(
                Cb, nb)
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
        h_mo, eri_mo = transform_integrals(self.h, self.eri, C_no)
        reference = self._determinant_energy(self._spin_density(C_no, na),
                                             self._spin_density(C_no, nb))
        return KohnShamUResult(
            electronic_energy=energy, n_alpha=na, n_beta=nb,
            mo_energies_alpha=np.real(epsa), mo_energies_beta=np.real(epsb),
            mo_coefficients_alpha=Ca, mo_coefficients_beta=Cb,
            natural_occupations=occupations, natural_orbitals=C_no,
            h_mo=np.real_if_close(h_mo), eri_mo=np.real_if_close(eri_mo),
            converged=converged, n_iterations=it, reference_energy=reference,
            functional=self.functional, hartree_energy=e_hartree,
            xc_energy=e_xc, history=history,
            exact_exchange_energy=self.exact_exchange_energy)


def _datasets_relativistic(datasets) -> bool:
    """The datasets' common relativistic-exchange flag (they must agree)."""
    flags = {bool(getattr(d, "relativistic_exchange", False))
             for d in datasets or ()}
    if len(flags) > 1:
        raise ValueError("the datasets mix relativistic and non-relativistic "
                         "exchange; the functional cannot match both")
    return flags.pop() if flags else False


def _warn_functional_mismatch(datasets, functional: str) -> None:
    """Warn when the datasets were generated with another functional."""
    generated = sorted({str(getattr(d, "xc", "lda") or "lda").lower()
                        for d in datasets or ()})
    if generated and generated != [functional]:
        warnings.warn(
            f"xc={functional!r} on datasets generated with "
            f"{'/'.join(g.upper() for g in generated)}: the molecule and its "
            "datasets use different functionals", RuntimeWarning, stacklevel=3)


def _hybrid_setup(functional: str) -> dict:
    """``[SCF SETUP]``'s ``exact_exchange`` entry of a hybrid, else empty."""
    if not xc_grid.is_hybrid(functional):
        return {}
    omega, fraction = xc_grid.HYBRIDS[xc_grid.resolve_functional(functional)]
    return {"exact_exchange": (
        f"{fraction:g} of the short-range exchange, erfc(omega r)/r with "
        f"omega = {omega:g} 1/Bohr (generalized Kohn-Sham)")}


def kohn_sham_solver(integrals, n_electrons: int,
                     functional: str = DEFAULT_XC, spins=None,
                     screening=None) -> KohnSham:
    """The :class:`KohnSham` problem of ``integrals``, not yet solved --
    :class:`UnrestrictedKohnSham` when ``spins = (n_alpha, n_beta)`` differ.

    ``integrals`` is a :class:`~mandacaru.core.hamiltonian.MolecularIntegrals`
    (or its PAW-LCAO subclass) in its orthonormalized basis; the orbitals are
    taken from the same grid samples its integrals were computed from.  A
    hybrid builds its short-range exchange tensor here
    (:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.short_range_two_body`);
    ``screening = (omega, fraction)``
    overrides the functional's own parameters.
    """
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
    if spins is not None and int(spins[0]) != int(spins[1]):
        return UnrestrictedKohnSham(integrals.one_body(),
                                    integrals.two_body(), orbitals,
                                    integrals.grid, int(spins[0]),
                                    int(spins[1]), **options)
    return KohnSham(integrals.one_body(), integrals.two_body(), orbitals,
                    integrals.grid, n_electrons, **options)


def _centers(integrals):
    """Atomic positions in Bohr, the frame the grid and potentials use."""
    return [np.asarray(c, dtype=float) for _Z, c in integrals._potentials.nuclei]


def kohn_sham(integrals, n_electrons: int, functional: str = DEFAULT_XC,
              spins=None, screening=None, **run_options):
    """Solve the Kohn-Sham problem on ``integrals``: restricted, or
    unrestricted when ``spins = (n_alpha, n_beta)`` differ (``screening``
    as for :func:`kohn_sham_solver`)."""
    result = kohn_sham_solver(integrals, n_electrons, functional,
                              spins=spins, screening=screening
                              ).run(**run_options)
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
    energy is stationary with respect to orbital rotations at convergence.
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
    D = solver._density_matrix(C)
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
    Da = solver._spin_density(scf.mo_coefficients_alpha, scf.n_alpha)
    Db = solver._spin_density(scf.mo_coefficients_beta, scf.n_beta)
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


def d4_dispersion_gradient(atoms, functional: str) -> np.ndarray:
    """D4 dispersion gradient ``dE/dR`` (Hartree/Bohr), ``(n_atoms, 3)``."""
    from dftd4.interface import DampingParam, DispersionModel

    model = DispersionModel(np.asarray(atoms.get_atomic_numbers()),
                            to_bohr(np.asarray(atoms.get_positions()),
                                    "angstrom"))
    result = model.get_dispersion(DampingParam(method=D4_METHODS[functional]),
                                  grad=True)
    return np.asarray(result["gradient"], dtype=float)


def d4_dispersion_energy(atoms, functional: str) -> float:
    """D4 dispersion energy (Hartree) of ``atoms`` for ``functional``."""
    try:
        from dftd4.interface import DampingParam, DispersionModel
    except ImportError as error:
        raise ImportError(
            "dispersion='d4' needs the dftd4 package: "
            "pip install 'mandacaru[dispersion]'") from error
    model = DispersionModel(np.asarray(atoms.get_atomic_numbers()),
                            to_bohr(np.asarray(atoms.get_positions()),
                                    "angstrom"))
    result = model.get_dispersion(DampingParam(method=D4_METHODS[functional]),
                                  grad=False)
    return float(result["energy"])


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
                 smearing=None, **driver_kwargs: object) -> None:
        from .periodic_dft import resolve_smearing
        self.xc = xc_grid.resolve_functional(xc)
        self.dispersion = self._check_dispersion(dispersion, self.xc)
        self.smearing = smearing
        resolve_smearing(smearing)               # validated now, used later
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
            family=family.name)
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
            active_space=self.active_space, hamiltonian=False)
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
                                "PerdewWang1992")}[self.xc]
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
                              spins=full_particles)
        # The exported Hamiltonian and every one-particle picture (density,
        # populations, cube files) are in the model orbitals: the Kohn-Sham
        # orbitals, or the natural orbitals of an unrestricted run.
        integrals.mo_coefficients = self._model_orbitals()
        if self.dispersion:
            self._scf.dispersion_energy = d4_dispersion_energy(self.atoms,
                                                               self.xc)
        constant = complex(integrals.constant_energy
                           + integrals.nuclear_repulsion)
        M = integrals.n_orbitals
        self.fermion_hamiltonian = LazyHamiltonian(self._scf.h_mo,
                                                   self._scf.eri_mo, constant)
        self.hamiltonian = self.fermion_hamiltonian
        self.num_particles = full_particles
        self.n_spatial_orbitals = int(M)
        self.n_qubits = 2 * M - (2 if self.mapping == "parity_reduced" else 0)
        self._configured = True

    def _configure_periodic(self, context, num_particles, n_orbitals) -> None:
        """Solve the crystal's Kohn-Sham problem."""
        from .periodic_dft import PeriodicKohnSham

        if self.dispersion:
            raise NotImplementedError(
                "dispersion='d4' is molecular for now; the periodic D4 "
                "lattice sum is not wired")
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
            magnetic_moments=self.atoms.get_initial_magnetic_moments())
        self._scf = solver.run()
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
        """The Kohn-Sham determinant's one-RDM (molecules).

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
        return super().mean_field_rdm()

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

    # -- crystal forces and stress ------------------------------------------ #

    def _kpts_spec(self) -> dict:
        return {"size": tuple(self.kpts), "gamma": self.kpts_gamma}

    def _crystal_gradient(self, orbital_delta, include_pulay):
        r""":func:`~mandacaru.algorithms.crystal_forces.crystal_gradient` of
        the converged crystal: :math:`dF/d\mathbf R`, F the free energy."""
        from .crystal_forces import DEFAULT_DELTA, crystal_gradient

        result = crystal_gradient(
            self._periodic_solver,
            delta=DEFAULT_DELTA if orbital_delta is None else float(
                orbital_delta),
            include_pulay=include_pulay)
        # The calculator checks the energy the gradient belongs to against
        # the reported one; a crystal's gradient is of F at the reported
        # state, so both are carried.
        result.details["energy_hartree"] = float(
            self._scf.extrapolated_energy)
        result.details["free_energy_hartree"] = float(self._scf.free_energy)
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
        return crystal_stress(
            self._periodic_solver, self.atoms, options, family.name,
            self._kpts_spec(),
            strain=DEFAULT_STRAIN if strain is None else float(strain))

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
            result = PeriodicDFTResult(
                method=self._kind,
                optimal_energy=self._to_energy_units(scf.extrapolated_energy),
                free_energy=self._to_energy_units(scf.free_energy),
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
