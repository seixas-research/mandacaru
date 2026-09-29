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

    def _frontier_energies(self) -> tuple[float, float]:
        """Validated occupied/unoccupied frontier energies in Hartree."""
        if not self.converged:
            raise ValueError("HOMO-LUMO energies require a converged RHF reference")
        energies = np.asarray(self.mo_energies)
        if (energies.ndim != 1 or not np.isrealobj(energies)
                or not np.all(np.isfinite(energies)) or np.any(np.diff(energies) < 0)):
            raise ValueError("mo_energies must be finite real energies in ascending order")
        occupied = self.n_occupied
        if (isinstance(occupied, (bool, np.bool_))
                or not isinstance(occupied, (int, np.integer))
                or not 0 < occupied < energies.size):
            raise ValueError("HOMO-LUMO gap requires occupied and unoccupied orbitals")
        return float(energies[occupied - 1]), float(energies[occupied])

    @property
    def homo_energy(self) -> float:
        """Highest occupied canonical RHF orbital energy, in Hartree."""
        return self._frontier_energies()[0]

    @property
    def lumo_energy(self) -> float:
        """Lowest unoccupied canonical RHF orbital energy, in Hartree."""
        return self._frontier_energies()[1]

    @property
    def homo_lumo_gap(self) -> float:
        """``epsilon_LUMO - epsilon_HOMO`` in Hartree for the RHF reference.

        This mean-field orbital gap is distinct from a correlated neutral
        excitation energy or a quasiparticle gap. A converged reference with
        at least one occupied and one unoccupied spatial orbital is required.
        """
        homo, lumo = self._frontier_energies()
        return lumo - homo

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

    :math:`h_{\text{new}} = C^\dagger h C` and
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
        _eps, C = np.linalg.eigh(self.h)
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
              guesses: int = 3) -> UHFResult:
        r"""Run the SCF and return the lowest converged :class:`UHFResult`.

        The core and screened-core starts cover ordinary closed and open shells.
        Equal alpha/beta populations need a third, deliberately spin-broken
        occupied/virtual rotation: otherwise their identical initial densities
        keep UHF on the RHF branch even when a lower broken-symmetry solution
        exists (for example, a stretched bond).  The lowest converged result
        is retained.
        """
        _eps, C = np.linalg.eigh(self.h)
        starts = [(C, C)]
        if guesses >= 2:
            D0 = self._density(C, self.na) + self._density(C, self.nb)
            F1 = self.h + 0.5 * self._coulomb(D0)
            screened = np.linalg.eigh(0.5 * (F1 + F1.conj().T))[1]
            starts.append((screened, screened))
        if guesses >= 3 and self.na == self.nb and 0 < self.na < self.M:
            occupied, virtual = self.na - 1, self.na
            angle = 0.3
            cosine, sine = np.cos(angle), np.sin(angle)
            alpha, beta = C.copy(), C.copy()
            alpha[:, occupied] = (cosine * C[:, occupied]
                                  + sine * C[:, virtual])
            alpha[:, virtual] = (-sine * C[:, occupied]
                                 + cosine * C[:, virtual])
            beta[:, occupied] = (cosine * C[:, occupied]
                                 - sine * C[:, virtual])
            beta[:, virtual] = (sine * C[:, occupied]
                                + cosine * C[:, virtual])
            starts.append((alpha, beta))
        best = None
        for Ca0, Cb0 in starts[:max(int(guesses), 1)]:
            result = self._scf(Ca0, Cb0, max_iter, tol)
            if best is None or (result.converged and not best.converged) or (
                    result.converged == best.converged
                    and result.electronic_energy < best.electronic_energy - 1e-12):
                best = result
        return best

    def _scf(self, Ca: np.ndarray, Cb: np.ndarray,
             max_iter: int, tol: float) -> UHFResult:
        """One SCF run from independent alpha and beta orbital guesses."""
        epsa = np.real(np.diag(Ca.conj().T @ self.h @ Ca))
        epsb = np.real(np.diag(Cb.conj().T @ self.h @ Cb))
        Da, Db = self._density(Ca, self.na), self._density(Cb, self.nb)
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


@dataclass
class GHFResult:
    """Outcome of a generalized (spinor) Hartree-Fock calculation.

    The orbitals are two-component spinors: column ``k`` of
    ``mo_coefficients`` is ``(alpha part; beta part)`` over the ``M`` spatial
    functions, so the array is ``(2M, 2M)``.  ``h_mo`` / ``eri_mo`` are the
    spin-orbital integrals in the spinor basis, in the **mode order** of the
    exported Hamiltonian (:meth:`GHF.mode_order`), not the energy order.
    """

    electronic_energy: float          # <GHF| H_elec |GHF> (no nuclear repulsion)
    n_electrons: int
    mo_energies: np.ndarray           # spinor energies, ascending
    mo_coefficients: np.ndarray       # (2M, 2M): spinors in the input basis
    h_mo: np.ndarray                  # (2M, 2M) in the spinor basis, mode order
    eri_mo: np.ndarray                # (2M)^4 <PQ|RS> in the spinor basis, mode order
    converged: bool
    n_iterations: int = 0
    #: The same spinors in mode order (:meth:`GHF.mode_order`): column ``P``
    #: is the spinor of mode ``P`` of the exported Hamiltonian.
    mode_coefficients: np.ndarray | None = None
    #: Largest splitting within the energy-ordered pairs (0-1, 2-3, ...) of
    #: occupied spinors: zero for a Kramers-paired (time-reversal
    #: symmetric) closed shell.  Measured, not imposed.
    kramers_pairing: float = 0.0

    def __repr__(self) -> str:
        return (f"GHFResult(E_elec={self.electronic_energy:.6f}, "
                f"n_electrons={self.n_electrons}, converged={self.converged})")


def spinor_integrals(h: np.ndarray, eri: np.ndarray, C: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    r"""``(h', g')`` in the spinor basis ``C`` (``(2M, K)``).

    ``h`` is the ``(2M, 2M)`` spin-orbital one-body matrix and ``eri`` the
    spin-free ``(M, M, M, M)`` physicists' tensor, so
    :math:`\langle PQ|RS\rangle' = \sum_{\sigma\tau}\sum_{pqrs}
    C^*_{p\sigma,P}C^*_{q\tau,Q}C_{r\sigma,R}C_{s\tau,S}\langle pq|rs\rangle`:
    electron 1 keeps its spin between ``P`` and ``R``, electron 2 between
    ``Q`` and ``S``.  The result has ``K^4`` complex entries.
    """
    M = eri.shape[0]
    h_new = C.conj().T @ h @ C
    parts = (C[:M], C[M:])
    g = 0
    for Cs in parts:
        for Ct in parts:
            g = g + np.einsum("ap,bq,cr,ds,abcd->pqrs", Cs.conj(), Ct.conj(),
                              Cs, Ct, eri, optimize=True)
    return h_new, g


class GHF:
    r"""Generalized Hartree-Fock: one determinant of complex two-component
    spinors, on an orthonormal spatial basis.

    The orbitals are not assumed to have a definite :math:`S_z`, so a
    one-body term that couples the spins -- spin-orbit coupling
    (:mod:`mandacaru.core.spin_orbit`) -- enters the self-consistent Fock
    operator itself.  With a spin-diagonal ``h`` the GHF solutions include the
    RHF and UHF ones.

    The Fock matrix of the spinor density
    :math:`D_{PQ} = \sum_i^{occ} C_{Pi}C^*_{Qi}` is
    :math:`F = h + J - K`: the Coulomb term is the spatial one of
    :math:`D_{\alpha\alpha} + D_{\beta\beta}` on both diagonal blocks, and the
    exchange of each spin block :math:`(\sigma, \tau)` is the spatial exchange
    of :math:`D_{\sigma\tau}` -- so the off-diagonal blocks carry exchange
    only.  Both use the ``D_sr`` contraction :class:`RHF` explains.

    Parameters
    ----------
    h : (2M, 2M) array
        Spin-orbital one-body Hamiltonian, alpha block first (the layout of
        :func:`~mandacaru.core.hamiltonian.spin_block_integrals`).
    eri : (M, M, M, M) array
        Spatial two-electron integrals ``<pq|rs>`` in physicists' notation.
    n_electrons : int
        Total electron count.

    """

    def __init__(self, h: np.ndarray, eri: np.ndarray, n_electrons: int):
        self.h = np.asarray(h, dtype=complex)
        self.eri = np.asarray(eri, dtype=complex)
        self.M = self.eri.shape[0]
        if self.h.shape != (2 * self.M, 2 * self.M):
            raise ValueError(f"h must be ({2 * self.M}, {2 * self.M}) for "
                             f"{self.M} spatial orbitals, got {self.h.shape}")
        self.n = int(n_electrons)
        if not 0 < self.n <= 2 * self.M:
            raise ValueError(f"{n_electrons} electrons do not fit "
                             f"{2 * self.M} spin-orbitals")

    def _density(self, C: np.ndarray) -> np.ndarray:
        Cocc = C[:, :self.n]
        return Cocc @ Cocc.conj().T

    def _fock(self, D: np.ndarray) -> np.ndarray:
        M = self.M
        blocks = [[D[s * M:(s + 1) * M, t * M:(t + 1) * M] for t in (0, 1)]
                  for s in (0, 1)]
        J = np.einsum("sr,prqs->pq", blocks[0][0] + blocks[1][1], self.eri,
                      optimize=True)
        F = self.h.copy()
        for s in (0, 1):
            for t in (0, 1):
                K = np.einsum("sr,prsq->pq", blocks[s][t], self.eri,
                              optimize=True)
                F[s * M:(s + 1) * M, t * M:(t + 1) * M] -= K
                if s == t:
                    F[s * M:(s + 1) * M, t * M:(t + 1) * M] += J
        return 0.5 * (F + F.conj().T)

    def _energy(self, D: np.ndarray, F: np.ndarray) -> float:
        return float(np.real(0.5 * np.sum(D * (self.h + F).T)))

    def energy_of(self, C: np.ndarray) -> float:
        """The energy of the determinant of the first ``n_electrons`` columns
        of ``C`` -- e.g. an RHF or UHF determinant written as spinors, in a
        Hamiltonian with spin-orbit coupling."""
        D = self._density(np.asarray(C, dtype=complex))
        return self._energy(D, self._fock(D))

    def solve(self, max_iter: int = 300, tol: float = 1e-9,
              guesses=()) -> GHFResult:
        r"""Run the SCF and return the lowest converged :class:`GHFResult`.

        Starts from the core guess, the screened core guess, and every
        ``(2M, 2M)`` spinor matrix in ``guesses`` (e.g. the RHF or UHF
        orbitals, :meth:`collinear_spinors`) -- so a GHF started from a
        converged RHF/UHF determinant can only end at or below its energy in
        this Hamiltonian.
        """
        _eps, C = np.linalg.eigh(self.h)
        starts = [C]
        D0 = self._density(C)
        F1 = self.h + 0.5 * (self._fock(D0) - self.h)
        starts.append(np.linalg.eigh(0.5 * (F1 + F1.conj().T))[1])
        starts.extend(np.asarray(g, dtype=complex) for g in guesses)
        best = None
        for C0 in starts:
            result = self._scf(C0, max_iter, tol)
            if best is None or (result.converged and not best.converged) or (
                    result.converged == best.converged
                    and result.electronic_energy < best.electronic_energy - 1e-12):
                best = result
        return best

    @staticmethod
    def collinear_spinors(Ca: np.ndarray, Cb: np.ndarray, n_alpha: int,
                          n_beta: int) -> np.ndarray:
        """Spatial alpha and beta orbitals as a ``(2M, 2M)`` spinor matrix
        whose first ``n_alpha + n_beta`` columns are the occupied ones."""
        M = Ca.shape[0]
        alpha = np.vstack([Ca, np.zeros_like(Ca)])
        beta = np.vstack([np.zeros_like(Cb), Cb])
        occupied = [alpha[:, :n_alpha], beta[:, :n_beta]]
        virtual = [alpha[:, n_alpha:], beta[:, n_beta:]]
        out = np.hstack(occupied + virtual)
        if out.shape != (2 * M, 2 * M):
            raise ValueError("alpha and beta orbitals must be square")
        return out

    def mode_order(self) -> np.ndarray:
        """Spinor index of each exported mode: spinor ``2i`` goes to mode
        ``i`` (the first half) and ``2i + 1`` to mode ``M + i``, so the
        reference that fills the first ``ceil(N/2)`` modes of the first half
        and ``floor(N/2)`` of the second is the GHF determinant."""
        M = self.M
        order = np.empty(2 * M, dtype=int)
        order[:M] = np.arange(0, 2 * M, 2)
        order[M:] = np.arange(1, 2 * M, 2)
        return order

    def _scf(self, C: np.ndarray, max_iter: int, tol: float) -> GHFResult:
        D = self._density(C)
        energy = np.inf
        converged = False
        it = 0
        mixer = DIIS()
        shift = 0.0
        for it in range(1, max_iter + 1):
            F = self._fock(D)
            current = self._energy(D, F)
            if current > energy + 1e-10 and shift == 0.0:
                shift = LEVEL_SHIFT
            F_iter = mixer.extrapolate(F, D)
            if shift:
                F_iter = _level_shifted(F_iter, D, shift)
            _eps, C = np.linalg.eigh(F_iter)
            D_new = self._density(C)
            if abs(current - energy) < tol and np.max(np.abs(D_new - D)) < tol:
                D, energy = D_new, current
                converged = True
                break
            D, energy = D_new, current

        F = self._fock(D)
        eps, C = np.linalg.eigh(F)                 # canonical spinors
        energy = self._energy(D, F)
        occupied = eps[:self.n]
        pairs = occupied[:self.n - self.n % 2].reshape(-1, 2)
        pairing = float(np.max(pairs[:, 1] - pairs[:, 0])) if len(pairs) else 0.0
        ordered = C[:, self.mode_order()]
        h_mo, eri_mo = spinor_integrals(self.h, self.eri, ordered)
        return GHFResult(
            electronic_energy=energy, n_electrons=self.n,
            mo_energies=np.real(eps), mo_coefficients=C, h_mo=h_mo,
            eri_mo=eri_mo, mode_coefficients=ordered, converged=converged,
            n_iterations=it,
            kramers_pairing=pairing)
