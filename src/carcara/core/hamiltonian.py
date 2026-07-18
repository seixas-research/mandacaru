# -*- coding: utf-8 -*-
# file: core/hamiltonian.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Electronic-structure integrals and the molecular Hamiltonian.

:class:`MolecularIntegrals` computes the one- and two-body integrals over a
localized (hydrogenic) basis by driving the real-space
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
    """

    def __init__(self, nuclei: Sequence[tuple[float, np.ndarray]],
                 basis, grid: Grid, units: str = "angstrom",
                 orthogonalize: bool = True):
        self.nuclei = [(float(Z), np.asarray(R, dtype=float)) for Z, R in nuclei]
        self.basis = list(basis)
        self.grid = grid
        self.units = units
        self.orthogonalize = orthogonalize
        self._engine = IntegralEngine(self.basis, grid)
        self._potentials = Potentials(self.nuclei, units=units)
        self._S: np.ndarray | None = None
        self._h1: np.ndarray | None = None
        self._eri: np.ndarray | None = None

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals."""
        return len(self.basis)

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

    def _compute(self):
        T, V = self._engine.one_body(self._potentials.nuclear_potential,
                                     energy_units="Ha")
        h = 0.5 * ((T + V) + (T + V).conj().T)   # symmetrize away grid noise
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
                              n_electrons: int | None = None) -> Fermion:
        """Assemble the second-quantized :class:`Fermion` Hamiltonian.

        Spin-orbitals are ordered alpha-block then beta-block, so the parity
        mapping's two-qubit reduction (which taper the alpha- and total-parity
        qubits) applies directly.

        With ``mo_basis=True`` the spatial integrals are first transformed to the
        restricted Hartree-Fock molecular-orbital basis (``n_electrons`` required),
        so the reference determinant is the HF ground state -- the basis expected
        by ADAPT-VQE and by variational algorithms in general.
        """
        if mo_basis:
            if n_electrons is None:
                raise ValueError("mo_basis=True requires n_electrons")
            rhf = self.hartree_fock(n_electrons)
            h_so, g_so = spin_block_integrals(rhf.h_mo, rhf.eri_mo)
        else:
            h_so, g_so = self.spin_orbital_integrals()
        H = Fermion.from_integrals(h_so, g_so)
        if include_nuclear_repulsion:
            H = H + Fermion({(): complex(self.nuclear_repulsion)},
                            n_modes=2 * self.n_orbitals)
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


def minimal_fao_basis(nuclei, grid_units: str = "angstrom"):
    """Build one Slater-screened hydrogenic 1s orbital per atom (minimal basis).

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


# Backward-compatible aliases.  ``MolecularIntegrals`` is basis-agnostic (it
# drives the integral engine over any localized basis); it was previously named
# ``HydrogenicIntegrals``.  ``minimal_fao_basis`` was ``minimal_hydrogenic_basis``.
HydrogenicIntegrals = MolecularIntegrals
minimal_hydrogenic_basis = minimal_fao_basis
