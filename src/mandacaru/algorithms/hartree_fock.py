# -*- coding: utf-8 -*-
# file: algorithms/hartree_fock.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Restricted / unrestricted Hartree-Fock and the molecular-orbital basis.

Variational quantum algorithms are almost always run in the **molecular-orbital
(MO) basis**: the Slater determinant filling the lowest MOs is then the
Hartree-Fock ground state, a *stationary* point of the energy.  By Brillouin's
theorem single-excitation gradients vanish there, so ADAPT-VQE
(:mod:`mandacaru.algorithms.adapt_vqe`) selects the physically relevant double
excitations first and converges to the FCI ground state -- behavior that does
*not* hold from an arbitrary (e.g. raw orthogonalized-AO) reference determinant.

:class:`RHF` is a small closed-shell self-consistent-field solver operating on an
**already orthonormal** spatial basis (as produced by
:class:`~mandacaru.core.hamiltonian.MolecularIntegrals` with ``orthogonalize=True``,
i.e. overlap :math:`S = I`).  It returns the MO coefficients and the one- and
two-body integrals rotated into the MO basis, ready for
:meth:`~mandacaru.core.mapping.Fermion.from_integrals`.

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
:math:`\langle pq|rs\rangle` (matching Mandacaru throughout); the chemists'-notation
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


class DIIS:
    """Pulay's direct inversion in the iterative subspace, for an orthonormal basis.

    Keeps the last ``depth`` Fock matrices and their commutator errors
    ``e = FD - DF`` (the overlap is the identity here) and returns the linear
    combination of Fock matrices that minimizes the norm of the extrapolated
    error.  Falls back to the latest Fock matrix while fewer than two vectors
    are stored, or when the small DIIS system is singular.
    """

    def __init__(self, depth: int = 8):
        self.depth = int(depth)
        self._focks: list[np.ndarray] = []
        self._errors: list[np.ndarray] = []

    @staticmethod
    def error(F: np.ndarray, D: np.ndarray) -> np.ndarray:
        return F @ D - D @ F

    def extrapolate(self, F: np.ndarray, D: np.ndarray) -> np.ndarray:
        e = self.error(F, D)
        self._focks.append(F)
        self._errors.append(e)
        if len(self._focks) > self.depth:
            self._focks.pop(0)
            self._errors.pop(0)
        n = len(self._focks)
        if n < 2:
            return F
        B = np.zeros((n + 1, n + 1), dtype=complex)
        for i in range(n):
            for j in range(n):
                B[i, j] = np.vdot(self._errors[i], self._errors[j])
        B[n, :n] = B[:n, n] = -1.0
        rhs = np.zeros(n + 1, dtype=complex)
        rhs[n] = -1.0
        try:
            coeff = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return F
        if not np.all(np.isfinite(coeff)):
            return F
        return sum(c * f for c, f in zip(coeff, self._focks))


#: Level shift (Hartree) applied to the virtual space once an SCF is seen to
#: raise its energy -- the classic cure for oscillation between two
#: configurations, which plain Roothaan iteration is prone to in a basis with
#: near-degenerate compact functions.
LEVEL_SHIFT = 0.5


def _level_shifted(F: np.ndarray, D_half: np.ndarray, shift: float) -> np.ndarray:
    """``F + shift * (1 - P)`` with ``P`` the occupied projector (``D_half``)."""
    n = F.shape[0]
    return F + shift * (np.eye(n) - D_half)


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
        # With <pr|qs> = int p*(1) q(1) r*(2) s(2) the electron-2 pair r*(2) s(2)
        # is weighted by sum_j C_sj C*_rj = D_sr / 2, i.e. the density enters
        # as D_sr, NOT D_rs -- the two differ by a complex conjugate, which is
        # invisible for real orbitals and wrong for the complex spherical
        # harmonics of any p or d shell:
        #   J_pq = sum_rs D_sr eri[p,r,q,s],   K_pq = sum_rs D_sr eri[p,r,s,q].
        J = np.einsum("sr,prqs->pq", D, self.eri, optimize=True)
        K = np.einsum("sr,prsq->pq", D, self.eri, optimize=True)
        return self.h + J - 0.5 * K

    def _electronic_energy(self, D: np.ndarray, F: np.ndarray) -> float:
        return float(np.real(0.5 * np.sum(D * (self.h + F).T)))

    def run(self, max_iter: int = 200, tol: float = 1e-9,
            diis: bool = True, guesses: int = 2) -> RHFResult:
        """Run the SCF and return the lowest converged :class:`RHFResult`.

        The Fock matrix is DIIS-extrapolated (Pulay), and a level shift is
        switched on for the rest of the run the first time an iteration
        *raises* the energy -- the signature of oscillation between two
        configurations.  Both keep the iteration on the variational path a
        plain Roothaan loop can leave in a basis with compact, near-degenerate
        functions.  ``diis=False`` restores the bare iteration.

        A converged SCF is a stationary point, not necessarily the minimum:
        with ``guesses=2`` (default) the loop is run from the bare-core
        guess and from a "screened core" guess (the core Hamiltonian plus the
        Coulomb field of the core-guess density, damped) and the lower
        converged energy is returned -- the second start reaches the ground
        configuration where the first sometimes lands on an excited one.
        """
        eps, C = np.linalg.eigh(self.h)
        D_core = self._density(C)
        starts = [D_core]
        if guesses >= 2:
            F1 = self.h + 0.5 * self._fock_two_electron(D_core)
            starts.append(self._density(np.linalg.eigh(0.5 * (F1 + F1.conj().T))[1]))
        best = None
        for D0 in starts[:max(int(guesses), 1)]:
            result = self._scf(D0, max_iter, tol, diis)
            if best is None or (result.converged and not best.converged) or (
                    result.converged == best.converged
                    and result.electronic_energy < best.electronic_energy - 1e-12):
                best = result
        return best

    def _fock_two_electron(self, D: np.ndarray) -> np.ndarray:
        return self._fock(D) - self.h

    def _scf(self, D: np.ndarray, max_iter: int, tol: float,
             diis: bool) -> RHFResult:
        """One SCF run from the density ``D``."""
        energy = np.inf
        converged = False
        it = 0
        mixer = DIIS() if diis else None
        shift = 0.0
        for it in range(1, max_iter + 1):
            F = self._fock(D)
            F = 0.5 * (F + F.conj().T)
            current = self._electronic_energy(D, F)
            if current > energy + 1e-10 and shift == 0.0:
                shift = LEVEL_SHIFT                # energy went up: damp
            F_iter = mixer.extrapolate(F, 0.5 * D) if mixer else F
            if shift:
                F_iter = _level_shifted(F_iter, 0.5 * D, shift)
            eps, C = np.linalg.eigh(F_iter)
            D_new = self._density(C)
            if abs(current - energy) < tol and np.max(np.abs(D_new - D)) < tol:
                D, energy = D_new, current
                converged = True
                break
            D, energy = D_new, current

        F = 0.5 * (self._fock(D) + self._fock(D).conj().T)
        eps, _C_final = np.linalg.eigh(F)          # canonical orbital energies
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
        # D enters as D_sr (see RHF._fock): complex orbitals need it.
        return np.einsum("sr,prqs->pq", D, self.eri, optimize=True)

    def _exchange(self, D: np.ndarray) -> np.ndarray:
        return np.einsum("sr,prsq->pq", D, self.eri, optimize=True)

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

    def solve(self, max_iter: int = 300, tol: float = 1e-9,
              guesses: int = 2) -> UHFResult:
        r"""Run the SCF and return the lowest converged :class:`UHFResult`.

        The initial guess breaks the alpha/beta symmetry slightly for
        :math:`n_\alpha \ne n_\beta` (they start from the same core-Hamiltonian
        orbitals but different occupations, which is enough); for
        :math:`n_\alpha = n_\beta` the solution is the RHF one.  As for
        :meth:`RHF.run`, a second, screened-core start is tried and the lower
        converged solution kept.
        """
        eps, C = np.linalg.eigh(self.h)
        starts = [C]
        if guesses >= 2:
            D0 = self._density(C, self.na) + self._density(C, self.nb)
            F1 = self.h + 0.5 * self._coulomb(D0)
            starts.append(np.linalg.eigh(0.5 * (F1 + F1.conj().T))[1])
        best = None
        for C0 in starts[:max(int(guesses), 1)]:
            result = self._scf(C0, max_iter, tol)
            if best is None or (result.converged and not best.converged) or (
                    result.converged == best.converged
                    and result.electronic_energy < best.electronic_energy - 1e-12):
                best = result
        return best

    def _scf(self, C: np.ndarray, max_iter: int, tol: float) -> UHFResult:
        """One SCF run from the orbitals ``C``."""
        eps = np.real(np.diag(C.conj().T @ self.h @ C))
        Ca, Cb = C, C
        epsa = epsb = np.real(eps)
        Da, Db = self._density(C, self.na), self._density(C, self.nb)
        energy = np.inf
        converged = False
        it = 0
        mixer_a, mixer_b = DIIS(), DIIS()
        shift = 0.0
        for it in range(1, max_iter + 1):
            Fa, Fb = self._fock_pair(Da, Db)
            new_energy = self._energy(Da, Db, Fa, Fb)
            if new_energy > energy + 1e-10 and shift == 0.0:
                shift = LEVEL_SHIFT
            Fa_it, Fb_it = mixer_a.extrapolate(Fa, Da), mixer_b.extrapolate(Fb, Db)
            if shift:
                Fa_it = _level_shifted(Fa_it, Da, shift)
                Fb_it = _level_shifted(Fb_it, Db, shift)
            epsa, Ca = np.linalg.eigh(Fa_it)
            epsb, Cb = np.linalg.eigh(Fb_it)
            Da_new, Db_new = self._density(Ca, self.na), self._density(Cb, self.nb)
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
        epsa, Ca = np.linalg.eigh(Fa)             # canonical (unshifted) orbitals
        epsb, Cb = np.linalg.eigh(Fb)
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
