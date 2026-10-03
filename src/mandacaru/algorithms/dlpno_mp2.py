# -*- coding: utf-8 -*-
# file: algorithms/dlpno_mp2.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Domain-based local pair natural orbital MP2 (DLPNO-MP2), closed shell.

Canonical MP2 couples every occupied pair to every virtual pair.  DLPNO-MP2
(Pinski, Riplinger, Valeev and Neese, J. Chem. Phys. 143, 034108 (2015))
makes the same second-order energy and density local:

1. the active occupied orbitals are **localized** (Foster-Boys,
   :mod:`~mandacaru.algorithms.local_correlation`);
2. the virtual space is spanned by **projected atomic orbitals** (PAOs), and
   each localized orbital gets a **domain** of atoms, hence of PAOs;
3. every pair :math:`(i, j)` gets an initial semicanonical MP2 amplitude in
   its pair domain, whose pair density is diagonalized: the eigenvectors are
   its **pair natural orbitals** (PNOs), and those with occupation below
   ``pno_cutoff`` are dropped;
4. the local MP2 amplitude equations are solved in each pair's PNO space,
   coupled through the occupied Fock matrix (localized orbitals do not
   diagonalize it) and the PNO overlaps between pairs.

With a zero cutoff and domains covering the molecule the PNO space is the
whole virtual space, and the energy and density are canonical MP2 exactly
(the test of this module).  Every integral comes from an orbital-integral
provider (:mod:`~mandacaru.algorithms.orbital_integrals`), so the same code
runs on the tensor or integral-direct, never forming the :math:`M^4` tensor
in the second case.

What it returns is what the active-space selector consumes from canonical
MP2: the correlation energy, the virtual natural orbitals of the MP2 density
(back-transformed from the PNO spaces into the canonical virtual orbitals)
and the occupied block of the density in the canonical occupied orbitals.

Distant pairs are **prescreened** by the semicanonical dipole estimate
(:func:`_screen_pairs`, ``pair_cutoff``) and only their estimate enters the
energy; the amplitude equations couple an orbital only to those whose
occupied Fock element exceeds :data:`FOCK_COUPLING`; the exchange blocks are
solved on local boxes when the provider's integrals are configured for it
(:attr:`~mandacaru.integrals.direct.DirectCoulomb.box_threshold`).

Not here: the PNO truncation correction, and a sparse (atom-blocked)
representation of the PAOs and PNOs -- every vector is stored over all
:math:`M` orbitals, so the per-pair work carries a factor :math:`M` and the
method is not yet linear even where the pair count is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .local_correlation import (DOMAIN_POPULATION, boys_localize,
                                orbital_domains)

__all__ = ["DLPNOResult", "dlpno_mp2", "DEFAULT_PNO_CUTOFF"]

#: PNO occupation below which a pair natural orbital is dropped
#: (ORCA's ``TCutPNO`` default for "NormalPNO").
DEFAULT_PNO_CUTOFF = 1e-8

#: Differential overlap above which a basis function's atom joins an
#: orbital's domain (Pinski et al.'s ``TCutDO``).
DEFAULT_DOI_CUTOFF = 1e-2

#: Estimated pair energy (Hartree, magnitude) below which a pair whose
#: domains do not overlap is dropped and its estimate added to the energy
#: (Pinski et al.'s ``TCutPre``).
DEFAULT_PAIR_CUTOFF = 1e-6

#: Occupied Fock coupling (Hartree) below which two localized orbitals are
#: treated as uncoupled in the amplitude equations.
FOCK_COUPLING = 1e-6

#: Overlap eigenvalue below which a direction of a pair domain's PAOs is
#: linearly dependent and removed.
PAO_DEPENDENCE = 1e-7

#: Largest residual norm of the amplitude equations at convergence.
RESIDUAL_TOLERANCE = 1e-8

#: Largest number of amplitude iterations.
MAX_ITERATIONS = 100


@dataclass
class DLPNOResult:
    """The MP2 quantities the active-space selector reads.

    Attributes
    ----------
    correlation_energy : float
        Local MP2 correlation energy (Hartree) of the correlated occupied
        orbitals.
    occupied_occupations : ndarray
        Diagonal of the occupied block of the density, canonical orbitals.
    virtual_occupations : ndarray
        Virtual natural occupations, descending.
    rotation : ndarray
        ``(M, M)``: identity on the occupied block, the virtual natural
        orbitals (descending occupation) on the virtual block.
    occupied_density : ndarray
        ``2 + D_oo`` over every occupied orbital, canonical basis; frozen
        (uncorrelated) orbitals have exactly 2 on the diagonal.
    pno_counts : dict
        Number of PNOs kept per pair, for the run log.
    iterations : int
    """

    correlation_energy: float
    occupied_occupations: np.ndarray
    virtual_occupations: np.ndarray
    rotation: np.ndarray
    occupied_density: np.ndarray
    pno_counts: dict = field(default_factory=dict)
    iterations: int = 0


def _orthonormal_domain(paos, overlap_floor: float = PAO_DEPENDENCE):
    """Orthonormal combinations of a domain's PAOs (canonical
    orthonormalization)."""
    S = paos.conj().T @ paos
    S = 0.5 * (S + S.conj().T)
    w, V = np.linalg.eigh(S)
    keep = w > overlap_floor * max(float(w.max()), 1.0)
    return paos @ (V[:, keep] / np.sqrt(w[keep]))


def _screen_pairs(lmo, paos, owners, lmo_atoms, F, position, F_occ,
                  cutoff):
    r"""``(pairs, weak_energy)``: the pairs to treat, and the estimated
    energy of the ones dropped.

    A pair whose atom domains overlap is always kept.  Any other is estimated
    by the semicanonical dipole approximation (Pinski et al. 2015): with
    :math:`\mu_{ia} = \langle i|\mathbf r|\tilde a\rangle` over the
    semicanonical PAOs :math:`\tilde a` of orbital :math:`i`'s own domain and
    :math:`\mathbf R` the vector between the orbitals' centroids,

    .. math::

        \varepsilon_{ij} \approx -\frac{4}{R^6}\sum_{ab}
        \frac{[\mu_{ia}\cdot\mu_{jb}
                - 3(\mu_{ia}\cdot\hat R)(\mu_{jb}\cdot\hat R)]^2}
             {\epsilon_a + \epsilon_b - F_{ii} - F_{jj}} ,

    the dispersion-like leading term of a distant pair.  Kept when
    :math:`|\varepsilon_{ij}|` reaches ``cutoff``; ``cutoff=None`` keeps
    every pair.
    """
    n = lmo.shape[1]
    every = [(i, j) for i in range(n) for j in range(i, n)]
    if cutoff is None:
        return every, 0.0
    centroids = np.array([[float(lmo[:, i] @ r @ lmo[:, i]) for r in position]
                          for i in range(n)])
    dipoles, energies = [], []
    for i in range(n):
        dom = [mu for mu, A in enumerate(owners) if A in set(lmo_atoms[i])]
        Q = _orthonormal_domain(paos[:, dom])
        Fd = np.real(Q.conj().T @ F @ Q)
        e, V = np.linalg.eigh(0.5 * (Fd + Fd.T))
        Qs = np.real(Q @ V)
        dipoles.append(np.array([lmo[:, i] @ r @ Qs for r in position]))
        energies.append(e)
    pairs, weak = [], 0.0
    for i, j in every:
        if i == j or set(lmo_atoms[i]) & set(lmo_atoms[j]):
            pairs.append((i, j))
            continue
        R = centroids[j] - centroids[i]
        distance = float(np.linalg.norm(R))
        unit = R / distance
        mi, mj = dipoles[i], dipoles[j]                    # (3, a), (3, b)
        coupling = mi.T @ mj - 3.0 * np.outer(unit @ mi, unit @ mj)
        gap = (energies[i][:, None] + energies[j][None, :]
               - F_occ[i, i] - F_occ[j, j])
        estimate = -4.0 / distance ** 6 * float(np.sum(coupling ** 2 / gap))
        if abs(estimate) >= cutoff:
            pairs.append((i, j))
        else:
            weak += estimate
    return pairs, weak


def dlpno_mp2(integrals, n_doubly: int, *, frozen=(), domains: str = "local",
              pno_cutoff: float = DEFAULT_PNO_CUTOFF,
              doi_cutoff: float = DEFAULT_DOI_CUTOFF,
              pair_cutoff: float = DEFAULT_PAIR_CUTOFF,
              domain_population: float = DOMAIN_POPULATION) -> DLPNOResult:
    r"""Closed-shell DLPNO-MP2 in the molecular orbitals of ``integrals``.

    Parameters
    ----------
    integrals : orbital-integral provider
        Its molecular orbitals are the canonical RHF ones; its first
        ``n_doubly`` orbitals are occupied.
    n_doubly : int
        Number of (doubly) occupied orbitals.
    frozen : sequence of int
        Occupied orbitals left uncorrelated (the frozen core).
    domains : {"local", "full"}
        ``"local"``: each localized orbital's atoms by Loewdin population;
        ``"full"``: every atom (canonical MP2 when ``pno_cutoff`` is 0).
    pno_cutoff : float
        Pair natural orbitals with occupation below this are dropped.
    """
    M = integrals.n_orbitals
    o = int(n_doubly)
    frozen = sorted({int(i) for i in frozen})
    correlated = [i for i in range(o) if i not in set(frozen)]
    weights = [2.0] * o + [0.0] * (M - o)
    F = np.real(integrals.fock(weights))                  # MO basis, (M, M)

    # Localized orbitals of the correlated space, as MO-basis vectors.
    canonical = np.eye(M)[:, correlated]
    position = np.real(integrals.position_mo())
    lmo, U = boys_localize(position, canonical)
    lmo = np.real(lmo)
    n = lmo.shape[1]
    F_occ = lmo.T @ F @ lmo                                # (n, n)

    # PAOs: the basis functions with every occupied orbital projected out,
    # in the molecular orbitals.  The molecular orbitals are real functions;
    # a complex basis function (a complex spherical harmonic) is replaced by
    # its real and imaginary parts, two real functions on the same atom, so
    # everything below is real.  The set is redundant, which the per-pair
    # orthonormalization removes.
    basis = integrals.basis_in_mo()                        # (M, M)
    atom_of = np.asarray(integrals.atom_of())
    parts = np.concatenate([np.real(basis), np.imag(basis)], axis=1)
    owners = np.concatenate([atom_of, atom_of])
    doi = integrals.differential_overlap(lmo)              # (n, 2M)
    alive = np.linalg.norm(parts, axis=0) > 1e-10
    parts, owners, doi = parts[:, alive], owners[alive], doi[:, alive]
    virtual_projector = np.diag([0.0] * o + [1.0] * (M - o))
    paos = virtual_projector @ parts
    norms = np.linalg.norm(paos, axis=0)
    alive = norms > 1e-8
    paos, owners = paos[:, alive] / norms[alive], owners[alive]
    doi = doi[:, alive]
    if domains == "full":
        lmo_atoms = [tuple(sorted(set(atom_of.tolist())))] * n
    else:
        # An orbital's domain: every atom carrying a function whose
        # differential overlap with it reaches `doi_cutoff` (Pinski et al.'s
        # TCutDO), plus its own atoms by Loewdin population.
        populated = orbital_domains(basis.conj().T @ lmo, atom_of,
                                    threshold=domain_population)
        lmo_atoms = [tuple(sorted(set(populated[i])
                                  | set(owners[doi[i] >= doi_cutoff].tolist())))
                     for i in range(n)]

    def pao_domain(i, j):
        atoms = set(lmo_atoms[i]) | set(lmo_atoms[j])
        return [mu for mu, A in enumerate(owners) if A in atoms]

    pairs, weak_energy = _screen_pairs(
        lmo, paos, owners, lmo_atoms, F, position, F_occ,
        pair_cutoff if domains != "full" else None)
    domain_of = {pair: pao_domain(*pair) for pair in pairs}
    K_pao = integrals.exchange_blocks(lmo, paos, domain_of)

    # Per pair: orthonormal domain, semicanonical guess, PNOs.
    pno = {}
    eps = {}
    counts = {}
    for (i, j) in pairs:
        dom = domain_of[(i, j)]
        Q = _orthonormal_domain(paos[:, dom])              # MO-basis columns
        R = np.linalg.lstsq(paos[:, dom], Q, rcond=None)[0]  # PAO -> Q
        Fd = np.real(Q.conj().T @ F @ Q)
        e, V = np.linalg.eigh(0.5 * (Fd + Fd.T))
        Qs = Q @ V                                         # semicanonical
        to_s = R @ V                                       # PAO coeffs
        K = np.real(to_s.T @ K_pao[(i, j)] @ to_s)
        T = -K / (e[:, None] + e[None, :] - F_occ[i, i] - F_occ[j, j])
        Tt = 2.0 * T - T.T
        D = (Tt.T @ T + Tt @ T.T) * (2.0 / (1.0 + (i == j)))
        occ, d = np.linalg.eigh(0.5 * (D + D.T))
        keep = occ >= pno_cutoff if pno_cutoff > 0 else np.ones_like(occ,
                                                                    bool)
        if not np.any(keep):
            keep[np.argmax(occ)] = True
        C_pno = Qs @ d[:, keep]
        Fp = np.real(C_pno.conj().T @ F @ C_pno)
        ep, W = np.linalg.eigh(0.5 * (Fp + Fp.T))
        pno[(i, j)] = C_pno @ W                            # semicanonical PNOs
        eps[(i, j)] = ep
        counts[(i, j)] = int(ep.size)

    # Exchange integrals in each pair's PNO space.
    K = {}
    for (i, j) in pairs:
        dom = domain_of[(i, j)]
        to_p = np.linalg.lstsq(paos[:, dom], pno[(i, j)], rcond=None)[0]
        K[(i, j)] = np.real(to_p.T @ K_pao[(i, j)] @ to_p)

    def amplitude(T, k, l):
        """T^{kl} in pair (min, max)'s PNO space (T^{lk} = T^{kl}^T)."""
        return T[(k, l)] if k <= l else T[(l, k)].T

    def overlap(a, b):
        return np.real(pno[a].conj().T @ pno[b])

    def key(k, l):
        return (k, l) if k <= l else (l, k)

    # The occupied Fock matrix of localized orbitals decays with distance:
    # only the couplings above FOCK_COUPLING enter the residual.
    coupled = [[k for k in range(n) if abs(F_occ[i, k]) > FOCK_COUPLING]
               for i in range(n)]
    T = {p: -K[p] / (eps[p][:, None] + eps[p][None, :]
                     - F_occ[p[0], p[0]] - F_occ[p[1], p[1]]) for p in pairs}
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        largest = 0.0
        new = {}
        for (i, j) in pairs:
            e = eps[(i, j)]
            Tij = T[(i, j)]
            R = K[(i, j)] + e[:, None] * Tij + Tij * e[None, :]
            for k in coupled[i]:
                if key(k, j) in T:
                    S = overlap((i, j), key(k, j))
                    R -= F_occ[i, k] * (S @ amplitude(T, k, j) @ S.T)
            for k in coupled[j]:
                if key(i, k) in T:
                    S = overlap((i, j), key(i, k))
                    R -= F_occ[k, j] * (S @ amplitude(T, i, k) @ S.T)
            largest = max(largest, float(np.max(np.abs(R))) if R.size else 0.)
            new[(i, j)] = Tij - R / (e[:, None] + e[None, :]
                                     - F_occ[i, i] - F_occ[j, j])
        T = new
        if largest < RESIDUAL_TOLERANCE:
            break

    energy = weak_energy
    for (i, j) in pairs:
        Tij = T[(i, j)]
        factor = 1.0 if i == j else 2.0
        energy += factor * float(np.sum(K[(i, j)] * (2.0 * Tij - Tij.T)))

    # Unrelaxed MP2 density.  Virtual block, summed over ordered pairs:
    # D_vv = 2 sum_ij C_ij (Tt^ij T^ij^T) C_ij^H.
    D_vv = np.zeros((M, M), dtype=complex)
    for (i, j) in pairs:
        C = pno[(i, j)]
        Tij = T[(i, j)]
        orders = [Tij] if i == j else [Tij, Tij.T]
        for t in orders:
            tt = 2.0 * t - t.T
            D_vv += 2.0 * C @ (tt @ t.T) @ C.conj().T
    # Occupied block in the localized orbitals:
    # D_ij = -2 sum_k sum_ab Tt^ik_ab T^jk_ab  (common basis through PNOs).
    D_loc = np.zeros((n, n))
    partners = [[k for k in range(n) if key(i, k) in T] for i in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in set(partners[i]) & set(partners[j]):
                Cik, Cjk = pno[key(i, k)], pno[key(j, k)]
                tik = amplitude(T, i, k)
                tjk = amplitude(T, j, k)
                S = np.real(Cik.conj().T @ Cjk)
                total += float(np.sum((2.0 * tik - tik.T)
                                      * (S @ tjk @ S.T)))
            D_loc[i, j] = -2.0 * total
    D_loc = 0.5 * (D_loc + D_loc.T)

    # Back to canonical orbitals: virtual natural orbitals over the
    # canonical virtuals, occupied block over every occupied orbital.
    D_virtual = np.real(D_vv[o:, o:])
    D_virtual = 0.5 * (D_virtual + D_virtual.T)
    values, vectors = np.linalg.eigh(D_virtual)
    values, vectors = values[::-1], vectors[:, ::-1]
    rotation = np.eye(M)
    rotation[o:, o:] = vectors
    occupied_density = 2.0 * np.eye(o)
    block = U @ D_loc @ U.T                     # localized -> canonical
    occupied_density[np.ix_(correlated, correlated)] += block
    return DLPNOResult(
        correlation_energy=energy,
        occupied_occupations=np.diag(occupied_density).copy(),
        virtual_occupations=values, rotation=rotation,
        occupied_density=occupied_density, pno_counts=counts,
        iterations=iterations)
