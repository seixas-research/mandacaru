# -*- coding: utf-8 -*-
# file: algorithms/hartree_fock.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Restricted / unrestricted Hartree-Fock and the molecular-orbital basis.

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

**Open shells.**  An odd electron count (or any :math:`n_\alpha \ne n_\beta`
state) has no closed-shell RHF solution.  :class:`UHF` solves the unrestricted
problem instead, and :meth:`UHF.solve` turns its two sets of spin orbitals into
**one** common spatial basis -- the *natural orbitals* of the total density
:math:`D_\alpha + D_\beta`, ordered by occupation.  Filling the first
:math:`n_\alpha` of them with spin-up and the first :math:`n_\beta` with
spin-down electrons is the natural reference determinant, and because the basis
is shared by both spins the spin-orbital Hamiltonian keeps exactly the
alpha-block / beta-block form the closed-shell path produces, so every ansatz,
pool and mapping downstream is unchanged.  That reference is not a stationary
point of the energy (it is not the UHF determinant itself, which lives in two
different spatial bases), so single excitations carry a non-zero gradient and
ADAPT-VQE will pick some up -- harmless, and expected.

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


@dataclass
class UHFResult:
    """Outcome of an unrestricted Hartree-Fock calculation.

    Besides the two sets of spin orbitals it carries the **natural orbitals**
    of the total density -- one spatial basis shared by both spins -- and the
    one- and two-body integrals rotated into it, which is what an open-shell
    molecular Hamiltonian is built from (see the module docstring).
    """

    electronic_energy: float          # <UHF| H_elec |UHF> (no nuclear repulsion)
    n_alpha: int
    n_beta: int
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    mo_coefficients_alpha: np.ndarray
    mo_coefficients_beta: np.ndarray
    natural_occupations: np.ndarray   # eigenvalues of D_alpha + D_beta, descending
    natural_orbitals: np.ndarray      # columns: natural orbitals in the input basis
    h_mo: np.ndarray                  # one-body Hamiltonian in the natural-orbital basis
    eri_mo: np.ndarray                # <pq|rs> (physicists') in that basis
    converged: bool
    n_iterations: int = 0
    reference_energy: float = 0.0     # <ref| H_elec |ref> of the NO reference determinant

    @property
    def spin_contamination(self) -> float:
        """``<S^2> - S(S+1)`` of the UHF determinant (0 for a pure spin state)."""
        Ca = self.mo_coefficients_alpha[:, :self.n_alpha]
        Cb = self.mo_coefficients_beta[:, :self.n_beta]
        overlap = Ca.conj().T @ Cb
        s = 0.5 * (self.n_alpha - self.n_beta)
        return float(self.n_beta - np.sum(np.abs(overlap) ** 2)) if self.n_beta \
            else 0.0 * s

    def __repr__(self) -> str:
        return (f"UHFResult(E_elec={self.electronic_energy:.6f}, "
                f"n_alpha={self.n_alpha}, n_beta={self.n_beta}, "
                f"converged={self.converged})")


def natural_orbitals(density: np.ndarray):
    """Eigen-decompose a total density ``D_alpha + D_beta``.

    Returns ``(occupations, orbitals)`` with the occupations (in ``[0, 2]``)
    sorted descending and the orbitals as columns in the same order -- a single
    orthonormal spatial basis whose first orbitals hold the most charge.
    """
    D = np.asarray(density)
    D = 0.5 * (D + D.conj().T)
    occ, C = np.linalg.eigh(D)
    order = np.argsort(occ)[::-1]
    return np.real(occ[order]), C[:, order]


class UHF:
    r"""Unrestricted (open-shell) Hartree-Fock on an orthonormal spatial basis.

    Handles an arbitrary number of :math:`\alpha` and :math:`\beta` electrons, so
    it covers open-shell atoms and radicals (the hydrogen doublet, lithium
    :math:`1s^2 2s^1`, any odd-electron molecule) that closed-shell :class:`RHF`
    cannot.  :meth:`run` returns the energy (the isolated-atom references of
    the dissociation curves); :meth:`solve` returns the full
    :class:`UHFResult`, including the natural-orbital basis the open-shell
    molecular Hamiltonian is written in.

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

    def _fock_pair(self, Da, Db):
        Jt = self._coulomb(Da + Db)
        Fa = self.h + Jt - self._exchange(Da)
        Fb = self.h + Jt - self._exchange(Db)
        return 0.5 * (Fa + Fa.conj().T), 0.5 * (Fb + Fb.conj().T)

    def _energy(self, Da, Db, Fa, Fb) -> float:
        return float(np.real(0.5 * (
            np.sum((Da + Db) * self.h.T)
            + np.sum(Da * Fa.T) + np.sum(Db * Fb.T))))

    def run(self, max_iter: int = 300, tol: float = 1e-9) -> float:
        """Run the SCF loop; return the electronic energy ``<H_elec>`` (Hartree)."""
        return self.solve(max_iter=max_iter, tol=tol).electronic_energy

    def solve(self, max_iter: int = 300, tol: float = 1e-9) -> UHFResult:
        """Run the SCF loop and return the full :class:`UHFResult`.

        The initial guess breaks the alpha/beta symmetry slightly for
        :math:`n_\alpha \ne n_\beta` (they start from the same core-Hamiltonian
        orbitals but different occupations, which is enough); for
        :math:`n_\alpha = n_\beta` the solution is the RHF one.
        """
        eps, C = np.linalg.eigh(self.h)
        Ca, Cb = C, C
        epsa = epsb = np.real(eps)
        Da, Db = self._density(C, self.na), self._density(C, self.nb)
        energy = np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            Fa, Fb = self._fock_pair(Da, Db)
            epsa, Ca = np.linalg.eigh(Fa)
            epsb, Cb = np.linalg.eigh(Fb)
            Da_new, Db_new = self._density(Ca, self.na), self._density(Cb, self.nb)
            new_energy = self._energy(Da, Db, Fa, Fb)
            if (abs(new_energy - energy) < tol
                    and np.max(np.abs(Da_new - Da)) < tol
                    and np.max(np.abs(Db_new - Db)) < tol):
                Da, Db = Da_new, Db_new
                energy = new_energy
                converged = True
                break
            Da, Db = Da_new, Db_new
            energy = new_energy

        Fa, Fb = self._fock_pair(Da, Db)
        energy = self._energy(Da, Db, Fa, Fb)
        occ, C_no = natural_orbitals(Da + Db)
        h_mo, eri_mo = transform_integrals(self.h, self.eri, C_no)

        # Energy of the reference determinant in the natural-orbital basis:
        # the first n_alpha (n_beta) NOs spin-up (spin-down).
        Ra, Rb = self._density(C_no, self.na), self._density(C_no, self.nb)
        FRa, FRb = self._fock_pair(Ra, Rb)
        reference = self._energy(Ra, Rb, FRa, FRb)

        return UHFResult(
            electronic_energy=energy, n_alpha=self.na, n_beta=self.nb,
            mo_energies_alpha=np.real(epsa), mo_energies_beta=np.real(epsb),
            mo_coefficients_alpha=Ca, mo_coefficients_beta=Cb,
            natural_occupations=occ, natural_orbitals=C_no,
            h_mo=np.real_if_close(h_mo), eri_mo=np.real_if_close(eri_mo),
            converged=converged, n_iterations=it, reference_energy=reference)
