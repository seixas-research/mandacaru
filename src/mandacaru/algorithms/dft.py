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

Functionals: LDA, PBE and the r\ :sup:`2`\ SCAN meta-GGA, whose
kinetic-energy density :math:`\tau = \tfrac12\sum_{pq}D_{pq}\nabla\phi_p
\cdot\nabla\phi_q^*` is built from spectral gradients of the orbitals.  The
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

Scope: closed shells.  A **periodic** geometry (any ``atoms.pbc``) takes the
crystal path instead -- Bloch states on a Monkhorst-Pack mesh with smearing,
PAW-LCAO only (:mod:`~mandacaru.algorithms.periodic_dft`); it has no forces
and no dispersion yet.

The converged Kohn-Sham orbitals are exported like the Hartree-Fock ones: the
result carries the many-body Hamiltonian written in them, so
``Mandacaru(method="adapt-vqe", **result.as_quantum_problem())`` continues
from a Kohn-Sham reference.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, replace

import numpy as np

from ..core.hamiltonian import spin_block_integrals
from ..core.mapping import Fermion
from ..integrals import exchange_correlation as xc_grid
from ..units import to_bohr
from .hartree_fock import (DIIS, LEVEL_SHIFT, RHFResult, _level_shifted,
                           transform_integrals)
from .pseudo_forces import AlgebraicEnergy
from .mean_field import MeanFieldResult, _MeanFieldDriver

#: Default exchange-correlation functional of ``method="dft"``.
DEFAULT_XC = "lda"

#: An energy rise (Hartree) below this is grid and round-off noise, not the
#: oscillation the level shift exists to damp.  A shift switched on by noise
#: near convergence freezes the density change just above ``tol``: stretched
#: H2 with PBE stalled at 1e-7 for 200 iterations after a 1.7e-10 Ha rise.
LEVEL_SHIFT_TRIGGER = 1e-6

#: Dispersion corrections ``dispersion=`` accepts.
DISPERSION_CORRECTIONS = ("d4",)

#: The functionals D4 is parameterized for, by the ``dftd4`` method name.
D4_METHODS = {"pbe": "pbe", "r2scan": "r2scan"}


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
    core_correction_energy: float = 0.0
    dispersion_energy: float = 0.0

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
        ``"lda"``, ``"pbe"`` or ``"r2scan"``.
    core_density : (ngrid,) array, optional
        Partial core density added to the valence density inside the
        functional.
    relativistic : bool
        Apply the relativistic exchange factor (LDA and PBE), as a
        relativistic dataset was unscreened with.
    """

    def __init__(self, h, eri, orbitals, grid, n_electrons: int,
                 functional: str = DEFAULT_XC, core_density=None,
                 relativistic: bool = False):
        self.h = np.asarray(h, dtype=complex)
        self.eri = np.asarray(eri, dtype=complex)
        self.orbitals = np.asarray(orbitals)
        self.grid = grid
        self.M = self.h.shape[0]
        if n_electrons % 2 != 0:
            raise ValueError("closed-shell Kohn-Sham needs an even number of "
                             "electrons")
        self.n_electrons = int(n_electrons)
        self.n_occ = self.n_electrons // 2
        if self.n_occ > self.M:
            raise ValueError(
                f"{n_electrons} electrons need > {self.M} spatial orbitals")
        self.functional = xc_grid.resolve_functional(functional)
        self.relativistic = bool(relativistic)
        self.core_density = (None if core_density is None
                             else np.asarray(core_density, dtype=float))
        self.meta = xc_grid.is_meta_gga(self.functional)
        self._orbital_gradients = None
        self._core_tau = None
        if self.meta:
            self._orbital_gradients = xc_grid.gradient(grid, self.orbitals)
            if self.core_density is not None:
                self._core_tau = xc_grid.weizsaecker_tau(grid,
                                                         self.core_density)

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

    def _xc(self, D):
        """``(E_xc, V_xc)``: the functional's energy and its matrix."""
        rho = self.density(D)
        if self.core_density is not None:
            rho = rho + self.core_density
        tau = None
        if self.meta:
            tau = self.kinetic_energy_density(D)
            if self._core_tau is not None:
                tau = tau + self._core_tau
        terms = xc_grid.evaluate(self.grid, rho, self.functional,
                                 relativistic=self.relativistic, tau=tau)
        phi = self.orbitals
        dV = self.grid.dV
        V = (phi.conj() * terms.potential) @ phi.T * dV
        if terms.tau_potential is not None:
            # dE/dD_qp through tau: (1/2) int df/dtau grad phi_p* . grad phi_q.
            weight = 0.5 * terms.tau_potential
            for dphi in self._orbital_gradients:
                V = V + (dphi.conj() * weight) @ dphi.T * dV
        return terms.energy, 0.5 * (V + V.conj().T)

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

        DIIS-extrapolated, with the Hartree-Fock solver's level shift
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
        for it in range(1, max_iter + 1):
            F, current, _e_h, _e_xc = self._fock(D)
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
            if abs(current - energy) < tol and np.max(np.abs(D_new - D)) < tol:
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
            hartree_energy=e_hartree, xc_energy=e_xc)


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


def kohn_sham_solver(integrals, n_electrons: int,
                     functional: str = DEFAULT_XC) -> KohnSham:
    """The :class:`KohnSham` problem of ``integrals``, not yet solved.

    ``integrals`` is a :class:`~mandacaru.core.hamiltonian.MolecularIntegrals`
    (or its PAW-LCAO subclass) in its orthonormalized basis; the orbitals are
    taken from the same grid samples its integrals were computed from.
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
    core = None
    if datasets:
        core = xc_grid.core_density_on_grid(integrals.grid, datasets,
                                            _centers(integrals))
    return KohnSham(integrals.one_body(), integrals.two_body(), orbitals,
                    integrals.grid, n_electrons, functional=functional,
                    core_density=core,
                    relativistic=(functional != "r2scan"
                                  and _datasets_relativistic(datasets)))


def _centers(integrals):
    """Atomic positions in Bohr, the frame the grid and potentials use."""
    return [np.asarray(c, dtype=float) for _Z, c in integrals._potentials.nuclei]


def kohn_sham(integrals, n_electrons: int, functional: str = DEFAULT_XC,
              **run_options) -> KohnShamResult:
    """Solve the closed-shell Kohn-Sham problem on ``integrals``."""
    result = kohn_sham_solver(integrals, n_electrons, functional).run(
        **run_options)
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
    /\partial R_{A,k}` (plus the core's von Weizsaecker kinetic-energy density
    for a meta-GGA) -- the partial core density moves with its atom.
    """

    def __init__(self, grid, psi, potential, tau_potential, core_density,
                 core_functions, centers, delta):
        self.grid = grid
        self.psi = psi
        self.v = potential
        self.v_tau = tau_potential
        self.core = core_density
        self.core_functions = core_functions
        self.centers = centers
        self.delta = float(delta)
        self.grad_psi = (xc_grid.gradient(grid, psi)
                         if tau_potential is not None else None)
        self.grad_core = (xc_grid.gradient(grid, core_density)
                          if tau_potential is not None
                          and core_density is not None else None)

    def matrix(self) -> np.ndarray:
        """``V_xc`` over the AO basis (Hartree)."""
        psi, dV = self.psi, self.grid.dV
        V = ((psi.conj() * self.v) @ psi.T) * dV
        if self.v_tau is not None:
            for d in self.grad_psi:
                V = V + 0.5 * ((d.conj() * self.v_tau) @ d.T) * dV
        return 0.5 * (V + V.conj().T)

    def pulay(self, dpsi) -> np.ndarray:
        psi, v, dV = self.psi, self.v, self.grid.dV
        out = ((dpsi.conj() * v) @ psi.T + (psi.conj() * v) @ dpsi.T) * dV
        if self.v_tau is not None:
            w = 0.5 * self.v_tau
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
        value = np.sum(self.v * drho)
        if self.v_tau is not None:
            # d(|grad rho|^2 / 8 rho) for the total core density.
            rho = self.core
            dgrad = xc_grid.gradient(self.grid, drho)
            g2 = np.sum(self.grad_core * self.grad_core, axis=0)
            cross = np.sum(self.grad_core * dgrad, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                dtau = np.where(rho > xc_grid.GRADIENT_DENSITY_FLOOR,
                                cross / (4.0 * rho) - g2 * drho
                                / (8.0 * rho * rho), 0.0)
            value = value + np.sum(self.v_tau * dtau)
        return float(value * self.grid.dV)


def _core_functions(datasets):
    """Per atom, ``r -> core density`` for the core correction, or ``None``."""
    return [xc_grid.core_density_function(dataset) for dataset in datasets]


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
    """
    from .forces import DEFAULT_ORBITAL_DELTA
    from .pseudo_forces import pseudo_nuclear_gradient

    delta = DEFAULT_ORBITAL_DELTA if orbital_delta is None else float(
        orbital_delta)
    n_electrons = 2 * scf.n_occupied
    solver = kohn_sham_solver(integrals, n_electrons, scf.functional)
    C = scf.mo_coefficients
    D = solver._density_matrix(C)
    rho = solver.density(D)
    if solver.core_density is not None:
        rho = rho + solver.core_density
    tau = None
    if solver.meta:
        tau = solver.kinetic_energy_density(D)
        if solver._core_tau is not None:
            tau = tau + solver._core_tau
    terms = xc_grid.evaluate(integrals.grid, rho, solver.functional,
                             relativistic=solver.relativistic, tau=tau)
    datasets = integrals.pseudopotentials or []
    field = KohnShamField(
        integrals.grid, np.ascontiguousarray(integrals._engine._psi),
        terms.potential, terms.tau_potential, solver.core_density,
        _core_functions(datasets) if datasets else
        [None] * len(integrals.nuclei), _centers(integrals), delta)
    V_xc = field.matrix()
    X = integrals._lowdin_x()
    P0 = X @ D @ X.conj().T
    constant = terms.energy - float(np.real(np.sum(P0 * V_xc.T)))
    energy = KohnShamEnergy(D, V_xc, constant)
    return pseudo_nuclear_gradient(
        integrals, None, None, atom_of_orbital=atom_of_orbital,
        orbital_delta=delta, include_pulay=include_pulay,
        orbital_gradient=False, energy=energy, field=field)


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
            return super()._build_hamiltonian(atoms)
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
                      "r2scan": ("Furness2020", "PerdewWang1992")}[self.xc]
        dispersion = ("Caldeweyher2019",) if self.dispersion else ()
        crystal = ()
        if self._periodic:
            from .periodic_dft import resolve_smearing
            method, _width = resolve_smearing(self.smearing)
            crystal = ("MonkhorstPack1976", "Pulay1980", "Kerker1981",
                       "MethfesselPaxton1989" if method == "methfessel-paxton"
                       else "Mermin1965")
        config["extras"] = (tuple(config.get("extras", ())) + functional
                            + dispersion + crystal)
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
        if full_particles[0] != full_particles[1]:
            raise NotImplementedError(
                "method='dft' is closed-shell (n_alpha = n_beta) for now")
        self._scf = kohn_sham(integrals, sum(full_particles), self.xc)
        if self.dispersion:
            self._scf.dispersion_energy = d4_dispersion_energy(self.atoms,
                                                               self.xc)
        constant = complex(integrals.constant_energy
                           + integrals.nuclear_repulsion)
        M = integrals.n_orbitals
        h_so, g_so = spin_block_integrals(self._scf.h_mo, self._scf.eri_mo)
        hamiltonian = Fermion.from_integrals(h_so, g_so)
        self.fermion_hamiltonian = hamiltonian + Fermion(
            {(): constant}, n_modes=2 * M)
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
            relativistic=(self.xc != "r2scan"
                          and _datasets_relativistic(datasets)),
            constant=constant)
        self._scf = solver.run()
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
            raise NotImplementedError(
                "periodic Kohn-Sham forces are not implemented yet")
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
        result = MeanFieldResult(
            method=self._kind, optimal_energy=self._to_energy_units(total),
            reference_energy=self._to_energy_units(
                scf.determinant_energy + constant), scf=scf,
            model_orbitals=np.asarray(scf.mo_coefficients).copy(),
            fermion_hamiltonian=self.fermion_hamiltonian,
            num_particles=self.num_particles,
            n_spatial_orbitals=self.n_spatial_orbitals,
            energy_unit=self._energy_unit_label())
        self._finalize_timings(timings, run_t0)
        return replace(result, timings=timings.as_dict())

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
                "k_points_irreducible": (f"{len(crystal.kpoints)} "
                                         "(time reversal)"),
                "smearing": f"{method}, {width * HARTREE_TO_EV:g} eV",
                "max_iterations": defaults["max_iter"].default,
                "convergence_Hartree": (
                    f"{defaults['tol'].default:g} (free energy change) and "
                    f"{defaults['density_tol'].default:g} electrons "
                    "(density residual)"),
                "mixing": "Pulay, Kerker-preconditioned",
                "energy_unit": self._energy_unit_label(),
            }
        defaults = inspect.signature(KohnSham.run).parameters
        return {
            "scf_method": "restricted Kohn-Sham (closed shell)",
            "xc_functional": self.xc.upper(),
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
        if scf.core_correction_energy:
            fields[f"core_correction_energy_{unit}"] = \
                f"{to_unit(scf.core_correction_energy):.10f}"
        if self.dispersion:
            fields[f"dispersion_energy_{unit}"] = \
                f"{to_unit(scf.dispersion_energy):.10f}"
        return fields
