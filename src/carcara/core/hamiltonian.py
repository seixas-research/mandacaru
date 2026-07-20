# -*- coding: utf-8 -*-
# file: core/hamiltonian.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Electronic-structure integrals and the molecular Hamiltonian.

:class:`MolecularIntegrals` computes the one- and two-body integrals over a
localized Full Atomic Orbitals (FAO) basis by driving the real-space
:class:`~carcara.integrals.IntegralEngine`, and assembles the second-quantized
molecular Hamiltonian as a :class:`~carcara.core.mapping.Fermion`.

Conventions (atomic units, Hartree):

* one-body ``h_pq = <p| -1/2 nabla^2 - sum_I Z_I/|r-R_I| |q>``;
* two-body in **physicists' notation**
  ``<pq|rs> = int int p*(1) q*(2) r(1) s(2)/r12`` -- electron 1 carries the
  orbital pair ``(p, r)`` and electron 2 the pair ``(q, s)`` (this is exactly
  what :meth:`IntegralEngine.two_body` returns);
* the Hamiltonian uses the ordering of :meth:`Fermion.from_integrals`,

  .. math::

      H = \sum_{PQ} h_{PQ}\, a^\dagger_P a_Q
        + \tfrac12 \sum_{PQRS} \langle PQ|RS\rangle\, a^\dagger_P a^\dagger_Q a_S a_R,

  over spin-orbitals, with the spin-blocked expansion of the spatial integrals
  (the standard physicists'-notation second-quantized electronic Hamiltonian).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..basis import FullAtomicOrbital
from ..integrals import Grid, IntegralEngine, Potentials
from ..units import to_bohr
from .mapping import Fermion


class MolecularIntegrals:
    """One- and two-body integrals over a localized basis for a molecule.

    Parameters
    ----------
    nuclei : sequence of ``(Z, position)``
        Nuclear charges and Cartesian positions (in ``units``) defining the
        electron-nuclear potential.
    basis : sequence of BasisFunction
        Spatial orbitals spanning the active space (e.g. ``FullAtomicOrbital``).
    grid : Grid
        Real-space integration grid.
    units : {"angstrom", "bohr"}
        Unit of the nuclear positions (default ``"angstrom"``).
    softening : float
        Coulomb softening length in **Bohr** (default ``1e-12``, effectively a bare
        ``-Z/r``): the electron-nuclear potential is ``-Z/max(r, softening)``, which
        bounds the singularity a nucleus samples when it sits on (or very near) a
        grid node.  The ASE-calculator path sets it to a fraction of the grid step
        so heavier-atom cores stay numerically finite on a coarse grid.
    """

    def __init__(self, nuclei: Sequence[tuple[float, np.ndarray]],
                 basis, grid: Grid, units: str = "angstrom",
                 orthogonalize: bool = True, softening: float = 1e-12,
                 pseudopotentials=None, kb_projectors=None):
        self.nuclei = [(float(Z), np.asarray(R, dtype=float)) for Z, R in nuclei]
        self.basis = list(basis)
        self.grid = grid
        self.units = units
        self.orthogonalize = orthogonalize
        self._engine = IntegralEngine(self.basis, grid)
        # With pseudopotentials the "nuclei" carry the *ionic* charges Z_ion, so
        # the nuclear repulsion below is already the ion-ion term.
        self.pseudopotentials = (list(pseudopotentials)
                                 if pseudopotentials is not None else None)
        self.kb_projectors = list(kb_projectors) if kb_projectors else []
        self._potentials = Potentials(self.nuclei, softening=softening,
                                      units=units,
                                      pseudopotentials=self.pseudopotentials)
        self._S: np.ndarray | None = None
        self._h1: np.ndarray | None = None
        self._eri: np.ndarray | None = None

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals."""
        return len(self.basis)

    def integration_profile(self) -> dict:
        """Timing / cores / peak-memory profile of the integral engine.

        Populated once the integrals have run (via :meth:`molecular_hamiltonian`,
        :meth:`one_body`, ...).  Keys: ``stages_s`` (per-stage wall times),
        ``total_s``, ``peak_memory_mb``, ``n_cores`` (OpenMP threads, ``None`` for
        the NumPy fallback) and ``backend``.
        """
        return self._engine.integration_profile()

    # -- overlap and orthogonalization ------------------------------------ #

    def overlap(self) -> np.ndarray:
        r"""Overlap matrix ``S_pq = <p|q>`` of the (generally non-orthogonal) basis."""
        if self._S is None:
            psi = np.stack([b.evaluate(self.grid.X, self.grid.Y, self.grid.Z).ravel()
                            for b in self.basis])
            S = (np.conj(psi) @ psi.T) * self.grid.dV
            self._S = 0.5 * (S + S.conj().T)
        return self._S

    def _lowdin_x(self) -> np.ndarray:
        """Symmetric orthogonalization matrix ``X = S^{-1/2}``."""
        S = self.overlap()
        w, U = np.linalg.eigh(S)
        return (U * (1.0 / np.sqrt(w))) @ U.conj().T

    @property
    def uses_pseudopotentials(self) -> bool:
        """True when the external potential is a sum of pseudopotentials."""
        return bool(self.pseudopotentials)

    def external_potential(self):
        """The callable the engine samples: pseudopotential or bare ``-Z/r``."""
        return (self._potentials.pseudopotential if self.uses_pseudopotentials
                else self._potentials.nuclear_potential)

    def kb_nonlocal(self) -> np.ndarray:
        r"""Kleinman-Bylander nonlocal matrix in the basis.

        .. math::

            H^{NL}_{\mu\nu} = \sum_p \langle\phi_\mu|\chi_p\rangle\,
                E^{KB}_p\, \langle\chi_p|\phi_\nu\rangle ,

        a sum of rank-one terms.  Only the ``(M, P)`` overlap matrix touches the
        grid (via :func:`carcara.integrals._backend.kb_projections`, C-accelerated);
        the rest is a small outer product.  Returns zeros when there are no
        projectors.
        """
        M = len(self.basis)
        if not self.kb_projectors:
            return np.zeros((M, M), dtype=complex)
        from ..integrals import _backend

        chi = np.stack([p.evaluate(self.grid.X, self.grid.Y,
                                   self.grid.Z).ravel()
                        for p in self.kb_projectors])
        overlaps = _backend.kb_projections(self._engine._psi, chi, self.grid.dV)
        energies = np.array([p.kb_energy for p in self.kb_projectors])
        return (overlaps * energies) @ overlaps.conj().T

    def _compute(self):
        T, V = self._engine.one_body(self.external_potential(),
                                     energy_units="Ha")
        one = T + V + self.kb_nonlocal()
        h = 0.5 * (one + one.conj().T)           # symmetrize away grid noise
        eri = self._engine.two_body(method="fft", energy_units="Ha")
        if self.orthogonalize:
            # Lowdin-orthonormalize the basis; the second-quantized Hamiltonian
            # requires an orthonormal orbital set.
            X = self._lowdin_x()
            h = X.conj().T @ h @ X
            eri = np.einsum("ap,bq,cr,ds,abcd->pqrs",
                            X.conj(), X, X.conj(), X, eri, optimize=True)
        self._h1, self._eri = h, eri

    # -- spatial integrals (Hartree) -------------------------------------- #

    def one_body(self) -> np.ndarray:
        r"""Spatial one-body core Hamiltonian ``h_pq = T_pq + V_pq`` (Hartree).

        In the orthonormalized orbital basis when ``orthogonalize=True``.
        """
        if self._h1 is None:
            self._compute()
        return self._h1

    def two_body(self) -> np.ndarray:
        r"""Spatial two-body tensor ``<pq|rs>`` in physicists' notation (Hartree)."""
        if self._eri is None:
            self._compute()
        return self._eri

    @property
    def nuclear_repulsion(self) -> float:
        r"""Nuclear repulsion energy ``sum_{I<J} Z_I Z_J/|R_I-R_J|`` (Hartree)."""
        e = 0.0
        pos = [to_bohr(R, self.units) for _Z, R in self.nuclei]
        Zs = [Z for Z, _R in self.nuclei]
        for i in range(len(self.nuclei)):
            for j in range(i + 1, len(self.nuclei)):
                e += Zs[i] * Zs[j] / np.linalg.norm(pos[i] - pos[j])
        return float(e)

    # -- spin-orbital integrals ------------------------------------------- #

    def spin_orbital_integrals(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Spin-orbital ``(h_so, g_so)`` for the Hamiltonian (spin-blocked).

        Spin-orbital ``P = p + sigma * M`` (``M`` spatial orbitals; ``sigma = 0``
        alpha for the first block, ``1`` beta for the second).  ``g_so`` is the
        two-electron integral in physicists' notation ``<PQ|RS>``, non-zero only
        when ``spin(P) == spin(R)`` (electron 1) and ``spin(Q) == spin(S)``
        (electron 2).  The returned tensors feed :meth:`Fermion.from_integrals`
        directly and yield a Hermitian, spin- and particle-number-conserving
        Hamiltonian.
        """
        return spin_block_integrals(self.one_body(), self.two_body())

    # -- Hartree-Fock ----------------------------------------------------- #

    def hartree_fock(self, n_electrons: int):
        """Run restricted Hartree-Fock in the (orthonormal) spatial basis.

        Returns an :class:`~carcara.algorithms.hartree_fock.RHFResult` with the MO
        coefficients, orbital energies and MO-basis integrals.  Requires
        ``orthogonalize=True`` (the default) so the basis overlap is the identity.
        """
        from ..algorithms.hartree_fock import RHF
        return RHF(self.one_body(), self.two_body(), n_electrons).run()

    # -- molecular Hamiltonian -------------------------------------------- #

    def molecular_hamiltonian(self, include_nuclear_repulsion: bool = True,
                              mo_basis: bool = False,
                              n_electrons: int | None = None,
                              frozen_orbitals=None) -> Fermion:
        """Assemble the second-quantized :class:`Fermion` Hamiltonian.

        Spin-orbitals are ordered alpha-block then beta-block, so the parity
        mapping's two-qubit reduction (which taper the alpha- and total-parity
        qubits) applies directly.

        With ``mo_basis=True`` the spatial integrals are first transformed to the
        restricted Hartree-Fock molecular-orbital basis (``n_electrons`` required),
        so the reference determinant is the HF ground state -- the basis expected
        by ADAPT-VQE and by variational algorithms in general.

        ``frozen_orbitals`` applies the **frozen-core approximation**: the given
        (doubly occupied) spatial MO indices are removed from the active space and
        replaced by their mean-field contribution -- a constant core energy plus an
        effective one-body potential on the remaining orbitals (see
        :func:`freeze_core_integrals`).  It requires ``mo_basis=True``; the returned
        Hamiltonian acts only on the active spin-orbitals.
        """
        frozen = sorted({int(i) for i in frozen_orbitals}) if frozen_orbitals \
            else []
        core_energy = 0.0
        if mo_basis:
            if n_electrons is None:
                raise ValueError("mo_basis=True requires n_electrons")
            rhf = self.hartree_fock(n_electrons)
            h_mo, eri_mo = rhf.h_mo, rhf.eri_mo
            if frozen:
                active = [p for p in range(self.n_orbitals) if p not in frozen]
                h_mo, eri_mo, core_energy = freeze_core_integrals(
                    h_mo, eri_mo, frozen, active)
            h_so, g_so = spin_block_integrals(h_mo, eri_mo)
        else:
            if frozen:
                raise ValueError(
                    "frozen_orbitals requires mo_basis=True (the frozen-core "
                    "approximation freezes canonical molecular orbitals)")
            h_so, g_so = self.spin_orbital_integrals()
        H = Fermion.from_integrals(h_so, g_so)
        const = core_energy + (self.nuclear_repulsion
                               if include_nuclear_repulsion else 0.0)
        if abs(const) > 1e-14:
            H = H + Fermion({(): complex(const)}, n_modes=h_so.shape[0])
        return H

    def hartree_fock_hamiltonian(self, n_electrons: int,
                                 include_nuclear_repulsion: bool = True) -> Fermion:
        r"""Reduce the molecular Hamiltonian to its Hartree-Fock (mean-field) form.

        Returns the diagonal one-body :class:`Fermion`

        .. math::

            H_{\mathrm{HF}} = \sum_P \varepsilon_P\, a^\dagger_P a_P + c ,

        where :math:`\varepsilon_P` are the (spin-orbital) RHF orbital energies and
        the constant :math:`c` is fixed so that, in the ``n_electrons`` sector, the
        aufbau (Hartree-Fock) determinant is the ground state with exactly the RHF
        total energy.  This is the mean-field Hamiltonian whose ground state is the
        HF determinant -- e.g. a cheap reference for the variational drivers.
        """
        rhf = self.hartree_fock(n_electrons)
        eps = np.real(np.asarray(rhf.mo_energies, dtype=float))
        M = len(eps)
        eps_so = np.concatenate([eps, eps])          # alpha block, then beta block
        n_so = 2 * M
        n_occ = n_electrons // 2
        occupied = list(range(n_occ)) + list(range(M, M + n_occ))

        terms = {((P, True), (P, False)): complex(eps_so[P]) for P in range(n_so)}
        H = Fermion(terms, n_modes=n_so)

        e_reference = float(sum(eps_so[P] for P in occupied))
        e_hf = rhf.electronic_energy + (self.nuclear_repulsion
                                        if include_nuclear_repulsion else 0.0)
        const = complex(e_hf - e_reference)
        if abs(const) > 1e-14:
            H = H + Fermion({(): const}, n_modes=n_so)
        return H


def spin_block_integrals(h: np.ndarray,
                         eri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Expand spatial integrals ``(h, <pq|rs>)`` to spin-orbitals (spin-blocked).

    Spin-orbital ``P = p + sigma * M`` (``sigma = 0`` alpha for the first ``M``,
    ``1`` beta for the second).  The one-body block is diagonal in spin; the
    physicists'-notation two-body tensor is non-zero only when
    ``spin(P) == spin(R)`` (electron 1) and ``spin(Q) == spin(S)`` (electron 2).
    """
    h = np.asarray(h)
    eri = np.asarray(eri)
    M = h.shape[0]
    n_so = 2 * M

    def spin(P):
        return P // M

    def orb(P):
        return P % M

    h_so = np.zeros((n_so, n_so), dtype=complex)
    for P in range(n_so):
        for Q in range(n_so):
            if spin(P) == spin(Q):
                h_so[P, Q] = h[orb(P), orb(Q)]

    g_so = np.zeros((n_so,) * 4, dtype=complex)
    for P in range(n_so):
        for Q in range(n_so):
            for R in range(n_so):
                for S in range(n_so):
                    if spin(P) == spin(R) and spin(Q) == spin(S):
                        g_so[P, Q, R, S] = eri[orb(P), orb(Q), orb(R), orb(S)]
    return h_so, g_so


def freeze_core_integrals(h_mo: np.ndarray, eri_mo: np.ndarray,
                          frozen: Sequence[int], active: Sequence[int]
                          ) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Frozen-core reduction of the MO-basis spatial integrals.

    Given the molecular-orbital one-body ``h_mo`` and physicists'-notation
    two-body ``<pq|rs>`` (``eri_mo``), a set of doubly occupied ``frozen`` (core)
    spatial orbitals and the complementary ``active`` orbitals, returns
    ``(h_active, eri_active, core_energy)`` for the reduced active-space
    Hamiltonian:

    .. math::

        E_{\text{core}} &= 2\sum_{i}h_{ii}
            + \sum_{ij}\bigl(2\langle ij|ij\rangle-\langle ij|ji\rangle\bigr), \\
        h^{\text{eff}}_{pq} &= h_{pq}
            + \sum_{i}\bigl(2\langle pi|qi\rangle-\langle pi|iq\rangle\bigr),

    with ``i, j`` over ``frozen`` and ``p, q`` over ``active``; ``eri_active`` is
    the ``active`` sub-block of ``eri_mo``.  Adding ``core_energy`` as a constant
    and using ``(h_active, eri_active)`` in the active space reproduces the full
    energy exactly for a determinant that keeps every frozen orbital doubly
    occupied (the frozen-core approximation).
    """
    h_mo = np.asarray(h_mo)
    eri_mo = np.asarray(eri_mo)
    frozen = list(frozen)
    active = list(active)

    core_energy = 0.0
    for i in frozen:
        core_energy += 2.0 * h_mo[i, i]
    for i in frozen:
        for j in frozen:
            core_energy += 2.0 * eri_mo[i, j, i, j] - eri_mo[i, j, j, i]

    n_act = len(active)
    h_eff = np.zeros((n_act, n_act), dtype=complex)
    for a, p in enumerate(active):
        for b, q in enumerate(active):
            val = h_mo[p, q]
            for i in frozen:
                val += 2.0 * eri_mo[p, i, q, i] - eri_mo[p, i, i, q]
            h_eff[a, b] = val

    eri_active = eri_mo[np.ix_(active, active, active, active)]
    return h_eff, eri_active, float(np.real(core_energy))


def minimal_fao_basis(nuclei, grid_units: str = "angstrom"):
    """Build one Slater-screened FAO (Full Atomic Orbital) 1s orbital per atom (minimal basis).

    ``nuclei`` is a sequence of ``(Z, position)``; returns a list of
    :class:`~carcara.basis.FullAtomicOrbital`, one 1s per center with the Slater
    effective charge for that atom's 1s.
    """
    basis = []
    for Z, R in nuclei:
        z_eff = FullAtomicOrbital.slater_effective_charge(int(round(Z)), 1, 0)
        basis.append(FullAtomicOrbital(1, 0, 0, Z=z_eff, center=R,
                                       units=grid_units))
    return basis

