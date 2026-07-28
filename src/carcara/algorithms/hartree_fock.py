# -*- coding: utf-8 -*-
# file: algorithms/hartree_fock.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Restricted Hartree-Fock (RHF) and the molecular-orbital basis.

Variational quantum algorithms are almost always run in the **molecular-orbital
(MO) basis**: the Slater determinant filling the lowest MOs is then the
Hartree-Fock ground state, a *stationary* point of the energy.  By Brillouin's
theorem single-excitation gradients vanish there, so ADAPT-VQE
(:mod:`carcara.algorithms.adapt_vqe`) selects the physically relevant double
excitations first and converges to the FCI ground state -- behavior that does
*not* hold from an arbitrary (e.g. raw orthogonalized-AO) reference determinant.

:class:`RHF` is a small closed-shell self-consistent-field solver operating on an
**already orthonormal** spatial basis (as produced by
:class:`~carcara.core.hamiltonian.MolecularIntegrals` with ``orthogonalize=True``,
i.e. overlap :math:`S = I`).  It returns the MO coefficients and the one- and
two-body integrals rotated into the MO basis, ready for
:meth:`~carcara.core.mapping.Fermion.from_integrals`.

Conventions: ``h`` is the ``(M, M)`` spatial core Hamiltonian and ``eri`` the
``(M, M, M, M)`` two-electron integral in **physicists' notation**
:math:`\langle pq|rs\rangle` (matching Carcará throughout); the chemists'-notation
integral used inside the Fock build is :math:`(pq|rs) = \langle pr|qs\rangle`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RHFResult:
    """Outcome of a restricted Hartree-Fock calculation."""

    electronic_energy: float          # <HF| H_elec |HF> (no nuclear repulsion)
    mo_energies: np.ndarray           # orbital energies (ascending)
    mo_coefficients: np.ndarray       # C: columns are MOs in the input basis
    n_occupied: int                   # doubly occupied spatial orbitals
    converged: bool
    h_mo: np.ndarray                  # one-body core Hamiltonian in the MO basis
    eri_mo: np.ndarray                # <pq|rs> (physicists') in the MO basis
    n_iterations: int = 0

    def __repr__(self) -> str:
        return (f"RHFResult(E_elec={self.electronic_energy:.6f}, "
                f"n_occ={self.n_occupied}, converged={self.converged})")


def transform_integrals(h: np.ndarray, eri: np.ndarray,
                        C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Rotate ``h`` and physicists'-notation ``eri`` into the basis ``C``.

    ``h_new = C^\dagger h C`` and
    ``eri_new[p,q,r,s] = sum C*_ap C*_bq C_cr C_ds <ab|cd>``.
    """
    h_new = C.conj().T @ h @ C
    eri_new = np.einsum("ap,bq,cr,ds,abcd->pqrs",
                        C.conj(), C.conj(), C, C, eri, optimize=True)
    return h_new, eri_new


class RHF:
    r"""Closed-shell restricted Hartree-Fock on an orthonormal spatial basis.

    Parameters
    ----------
    h : (M, M) array
        Spatial one-body core Hamiltonian ``T + V`` in an orthonormal basis.
    eri : (M, M, M, M) array
        Two-electron integrals ``<pq|rs>`` in physicists' notation.
    n_electrons : int
        Total electron count (must be even for closed-shell RHF).
    """

    def __init__(self, h: np.ndarray, eri: np.ndarray, n_electrons: int):
        self.h = np.asarray(h, dtype=complex)
        self.eri = np.asarray(eri, dtype=complex)
        self.M = self.h.shape[0]
        if n_electrons % 2 != 0:
            raise ValueError("RHF requires an even number of electrons")
        self.n_electrons = int(n_electrons)
        self.n_occ = self.n_electrons // 2
        if self.n_occ > self.M:
            raise ValueError(
                f"{n_electrons} electrons need > {self.M} spatial orbitals")

    def _density(self, C: np.ndarray) -> np.ndarray:
        """Closed-shell density ``D_pq = 2 sum_i^occ C_pi C*_qi``."""
        Cocc = C[:, :self.n_occ]
        return 2.0 * (Cocc @ Cocc.conj().T)

    def _fock(self, D: np.ndarray) -> np.ndarray:
        r"""Fock matrix ``F = h + J - K/2`` from the density ``D``.

        In chemists' notation ``(pq|rs) = <pr|qs> = eri[p,r,q,s]``:
        ``J_pq = sum_rs D_rs (pq|rs)``, ``K_pq = sum_rs D_rs (pr|qs)``.
        """
        # J_pq = sum_rs D_rs eri[p,r,q,s]; K_pq = sum_rs D_rs eri[p,r,s,q].
        J = np.einsum("rs,prqs->pq", D, self.eri, optimize=True)
        K = np.einsum("rs,prsq->pq", D, self.eri, optimize=True)
        return self.h + J - 0.5 * K

    def _electronic_energy(self, D: np.ndarray, F: np.ndarray) -> float:
        return float(np.real(0.5 * np.sum(D * (self.h + F).T)))

    def run(self, max_iter: int = 200, tol: float = 1e-9) -> RHFResult:
        """Run the SCF loop and return the converged :class:`RHFResult`."""
        # Core-Hamiltonian initial guess.
        eps, C = np.linalg.eigh(self.h)
        D = self._density(C)
        energy = np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            F = self._fock(D)
            F = 0.5 * (F + F.conj().T)
            eps, C = np.linalg.eigh(F)
            D_new = self._density(C)
            new_energy = self._electronic_energy(D_new, F)
            if abs(new_energy - energy) < tol and \
                    np.max(np.abs(D_new - D)) < tol:
                D, energy = D_new, new_energy
                converged = True
                break
            D, energy = D_new, new_energy

        F = 0.5 * (self._fock(D) + self._fock(D).conj().T)
        energy = self._electronic_energy(D, F)
        h_mo, eri_mo = transform_integrals(self.h, self.eri, C)
        return RHFResult(
            electronic_energy=energy, mo_energies=np.real(eps),
            mo_coefficients=C, n_occupied=self.n_occ, converged=converged,
            h_mo=np.real_if_close(h_mo), eri_mo=np.real_if_close(eri_mo),
            n_iterations=it)


class UHF:
    r"""Unrestricted (open-shell) Hartree-Fock on an orthonormal spatial basis.

    Handles an arbitrary number of :math:`\alpha` and :math:`\beta` electrons, so
    it covers open-shell atoms (e.g. the hydrogen doublet or lithium
    :math:`1s^2 2s^1`) that closed-shell :class:`RHF` cannot -- used to compute the
    isolated-atom energies that reference molecular dissociation curves.

    Parameters
    ----------
    h, eri : arrays
        Spatial core Hamiltonian and physicists'-notation two-electron integrals
        (as for :class:`RHF`).
    n_alpha, n_beta : int
        Numbers of spin-up and spin-down electrons.
    """

    def __init__(self, h: np.ndarray, eri: np.ndarray,
                 n_alpha: int, n_beta: int):
        self.h = np.asarray(h, dtype=complex)
        self.eri = np.asarray(eri, dtype=complex)
        self.M = self.h.shape[0]
        self.na, self.nb = int(n_alpha), int(n_beta)
        if max(self.na, self.nb) > self.M:
            raise ValueError("more electrons of one spin than spatial orbitals")

    def _density(self, C: np.ndarray, n: int) -> np.ndarray:
        Cocc = C[:, :n]
        return Cocc @ Cocc.conj().T if n else np.zeros((self.M, self.M), complex)

    def _coulomb(self, D: np.ndarray) -> np.ndarray:
        return np.einsum("rs,prqs->pq", D, self.eri, optimize=True)

    def _exchange(self, D: np.ndarray) -> np.ndarray:
        return np.einsum("rs,prsq->pq", D, self.eri, optimize=True)

    def run(self, max_iter: int = 300, tol: float = 1e-9) -> float:
        """Run the SCF loop; return the electronic energy ``<H_elec>`` (Hartree)."""
        eps, C = np.linalg.eigh(self.h)
        Da, Db = self._density(C, self.na), self._density(C, self.nb)
        energy = np.inf
        for _ in range(max_iter):
            Jt = self._coulomb(Da + Db)
            Fa = self.h + Jt - self._exchange(Da)
            Fb = self.h + Jt - self._exchange(Db)
            Fa = 0.5 * (Fa + Fa.conj().T)
            Fb = 0.5 * (Fb + Fb.conj().T)
            _, Ca = np.linalg.eigh(Fa)
            _, Cb = np.linalg.eigh(Fb)
            Da_new, Db_new = self._density(Ca, self.na), self._density(Cb, self.nb)
            new_energy = float(np.real(0.5 * (
                np.sum((Da + Db) * self.h.T)
                + np.sum(Da * Fa.T) + np.sum(Db * Fb.T))))
            if (abs(new_energy - energy) < tol
                    and np.max(np.abs(Da_new - Da)) < tol):
                Da, Db = Da_new, Db_new
                energy = new_energy
                break
            Da, Db = Da_new, Db_new
            energy = new_energy
        return energy
