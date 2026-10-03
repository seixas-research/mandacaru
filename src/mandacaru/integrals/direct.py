# -*- coding: utf-8 -*-
# file: integrals/direct.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Integral-direct Coulomb and exchange: two-electron work without the tensor.

The FFT two-body tensor (:meth:`IntegralEngine.two_body`) holds every
:math:`\langle ab|cd\rangle` of the basis: :math:`M^4` numbers, about
:math:`48 M^4` bytes at its peak, built from :math:`M^2/2` Poisson solves and
an :math:`M^4 G` contraction.  On a 17 GB machine that stops near
:math:`M = 135`; benzene in PAW-LCAO-DZP (:math:`M = 108`) did not finish in
25 minutes.  Most consumers never need the whole tensor:

* a **Fock matrix** needs :math:`J[D]` and :math:`K[D]`.  :math:`J` is the
  potential of one density, one Poisson solve.  :math:`K` is, for every
  occupied orbital :math:`\phi_i`, the potentials of the :math:`M` pair
  densities :math:`\phi_i^*\chi_q`: :math:`oM` solves and an :math:`oM^2G`
  contraction.  Benzene DZP: 9.9 s per build, against a tensor that never
  finished;
* an **active-space Hamiltonian** needs the integrals of :math:`n` orbitals,
  :math:`n^2/2` solves and an :math:`n^4` result.

Both are here, exact to round-off against the tensor (H2O DZP:
:math:`|\Delta J|, |\Delta K| < 10^{-14}` Ha), with the same Poisson kernel
the tensor uses and the same family augmentation.

The augmentation
----------------

A PAW-LCAO basis adds its compensation charges to the tensor as a low-rank
sum (:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.two_body_augmentation`):

.. math::

    \Delta\langle pq|rs\rangle = \sum_c \bigl(Q^c_{pr} W^c_{qs}
        + W^c_{pr} Q^c_{qs}\bigr) + \sum_{cd} U_{cd}\, Q^c_{pr} Q^d_{qs} ,

with :math:`M \times M` matrices :math:`Q^c, W^c` per multipole channel
:math:`c` and a small channel coupling :math:`U`.  Its contribution to
:math:`J`, :math:`K` or to the integrals of any orbital set is a few matrix
products, so it is applied exactly and never as an :math:`M^4` array.

Conventions follow :class:`~mandacaru.algorithms.hartree_fock.RHF`:
physicists' :math:`\langle pq|rs\rangle` (electron 1 carries :math:`p, r`),
:math:`J_{pq} = \sum_{rs} D_{sr}\langle pr|qs\rangle` and
:math:`K_{pq} = \sum_{rs} D_{sr}\langle pr|sq\rangle`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DirectCoulomb"]

#: Pair densities solved per Poisson batch.  Each batch holds two
#: ``(block, G)`` complex stacks; 64 keeps benzene DZP under 100 MB.
POISSON_BLOCK = 64

#: Density-matrix eigenvalues below this (relative to the largest) carry no
#: electrons and are skipped when :math:`K` is built orbital by orbital.
OCCUPATION_FLOOR = 1e-12


#: Default relative threshold of a local exchange term: grid points where
#: :math:`|\phi_k|^2` is below this fraction of its peak are left out of the
#: box around :math:`\phi_k`, and basis functions whose product norm with it
#: is below this fraction of the largest are skipped.  Measured on C16H34
#: PAW-LCAO-SZ: 1e-8 gives 1.8x over the global build at an exchange-energy
#: error of 1e-7 Ha; 1e-6 gives 3.0x at 3e-5 Ha, too coarse.
LOCAL_BOX_THRESHOLD = 1e-8

#: Box edges are rounded up to a multiple of this, so boxes of similar size
#: share one Poisson solver (its kernel transform is the expensive part).
BOX_QUANTUM = 4


def pivoted_cholesky(A, tolerance: float = 1e-12) -> np.ndarray:
    r"""Columns :math:`L` with :math:`A \approx L L^\dagger` for a Hermitian
    positive semi-definite ``A``, pivoting on the largest remaining diagonal.

    Applied to a density matrix in the atomic-orbital basis it returns
    occupied orbitals (as many as its rank) that are **localized** without
    any localization step -- each pivot is the most occupied basis function
    left, and the Schur complement removes only what overlaps it (the
    "Cholesky molecular orbitals" of Aquilante, Pedersen and Koch).  Cost
    :math:`O(M r^2)`.
    """
    A = np.asarray(A)
    A = 0.5 * (A + A.conj().T)
    n = A.shape[0]
    d = np.real(np.diag(A)).copy()
    floor = tolerance * max(float(d.max()), 1.0) if n else 0.0
    columns = []
    for _ in range(n):
        p = int(np.argmax(d))
        if d[p] <= floor:
            break
        column = A[:, p].copy()
        for l in columns:
            column = column - l * np.conj(l[p])
        column = column / np.sqrt(d[p])
        columns.append(column)
        d = d - np.abs(column) ** 2
        d[p] = 0.0
    return (np.array(columns).T if columns
            else np.zeros((n, 0), dtype=A.dtype))


class DirectCoulomb:
    """Coulomb and exchange of a basis without its two-body tensor.

    Parameters
    ----------
    integrals : MolecularIntegrals
        Supplies the sampled basis (``_engine._psi``), the grid and, for a
        PAW-LCAO basis, the compensation channels.  Only molecular
        (non-periodic) integrals with the FFT kernel are supported.
    box_threshold : float, optional
        Solve the exchange on local boxes at this density threshold
        (:meth:`exchange_local`); ``None`` keeps every solve on the whole
        grid, exact to round-off.
    """

    def __init__(self, integrals, box_threshold: float | None = None):
        if getattr(integrals, "periodic", False):
            raise NotImplementedError(
                "integral-direct Coulomb is molecular only: a periodic "
                "kernel needs the periodic Poisson solver and is not "
                "implemented here")
        from .poisson import PoissonFFTSolver

        self.integrals = integrals
        self.psi = integrals._engine._psi                     # (M, G)
        self.grid = integrals.grid
        self.dV = float(self.grid.dV)
        self.solver = PoissonFFTSolver(self.grid.shape, step=self.grid.step)
        self._augmentation = None
        self._box_solvers: dict = {}
        self._supports = None
        #: ``None``: every exchange solve on the whole grid (exact);
        #: a number: on local boxes (:meth:`exchange_local`) at that
        #: threshold.
        self.box_threshold = box_threshold

    @property
    def n_basis(self) -> int:
        return int(self.psi.shape[0])

    # -- the family augmentation ------------------------------------------- #

    def augmentation(self):
        """``(Q, W, U)`` per multipole channel (AO basis), or ``None``."""
        if self._augmentation is None:
            moments = getattr(self.integrals, "compensation_moments", None)
            channels = moments() if moments is not None else {}
            if not channels:
                self._augmentation = False
            else:
                Ws = self.integrals.compensation_potentials()
                U = np.asarray(self.integrals.compensation_coulomb())
                order = self.integrals.multipole_channels()
                self._augmentation = ([np.asarray(channels[c]) for c in order],
                                      [np.asarray(Ws[c]) for c in order], U)
        return self._augmentation or None

    # -- Coulomb and exchange ---------------------------------------------- #

    def potential(self, densities) -> np.ndarray:
        """Hartree potentials of a ``(k, G)`` stack of densities."""
        return self.solver.solve_stack(np.ascontiguousarray(densities))

    def coulomb(self, D) -> np.ndarray:
        r""":math:`J_{pq} = \sum_{rs} D_{sr}\langle pr|qs\rangle` (AO basis)."""
        D = np.asarray(D)
        psi = self.psi
        density = np.sum((D.T @ psi) * np.conj(psi), axis=0)
        V = self.potential(density[None, :])[0]
        J = (np.conj(psi) * V) @ psi.T * self.dV
        aug = self.augmentation()
        if aug is not None:
            Qs, Ws, U = aug
            tQ = np.array([np.sum(Q * D.T) for Q in Qs])
            tW = np.array([np.sum(W * D.T) for W in Ws])
            coupled = U @ tQ
            for c, (Q, W) in enumerate(zip(Qs, Ws)):
                J = J + Q * (tW[c] + coupled[c]) + W * tQ[c]
        return J

    def exchange(self, D) -> np.ndarray:
        r""":math:`K_{pq} = \sum_{rs} D_{sr}\langle pr|sq\rangle` (AO basis).

        ``D`` is factored as :math:`\sum_k w_k c_k c_k^\dagger` (its
        eigen-decomposition), and for each orbital :math:`\phi_k = \sum_s
        c_{sk}\chi_s` the pair densities :math:`\phi_k^*\chi_q` are solved in
        batches: :math:`K_{pq} \mathrel{+}= w_k\sum_g \chi_p^*\phi_k\,
        \Phi_{kq}`.
        """
        D = np.asarray(D)
        D = 0.5 * (D + D.conj().T)
        weights, vectors = np.linalg.eigh(D)
        floor = OCCUPATION_FLOOR * max(float(np.max(np.abs(weights))), 1.0)
        keep = np.abs(weights) > floor
        psi = self.psi
        K = np.zeros((self.n_basis, self.n_basis), dtype=complex)
        for w, c in zip(weights[keep], vectors[:, keep].T):
            phi = c @ psi
            left = np.conj(psi) * phi
            for q0 in range(0, self.n_basis, POISSON_BLOCK):
                block = self.potential(np.conj(phi) * psi[q0:q0 + POISSON_BLOCK])
                K[:, q0:q0 + POISSON_BLOCK] += w * (left @ block.T) * self.dV
        aug = self.augmentation()
        if aug is not None:
            Qs, Ws, U = aug
            QD = [Q @ D for Q in Qs]
            for c, (Q, W) in enumerate(zip(Qs, Ws)):
                K = K + QD[c] @ W + W @ D @ Q
                for d, Qd in enumerate(Qs):
                    if U[c, d] != 0:
                        K = K + U[c, d] * (QD[c] @ Qd)
        return K

    # -- local boxes ------------------------------------------------------- #

    def _box_solver(self, shape):
        from .poisson import PoissonFFTSolver

        if shape not in self._box_solvers:
            self._box_solvers[shape] = PoissonFFTSolver(
                shape, step=self.grid.step)
        return self._box_solvers[shape]

    def _bounding_box(self, mask3d):
        """Slices of the smallest box (edges rounded up to
        :data:`BOX_QUANTUM`, clipped to the grid) holding every true point."""
        boxes = []
        for axis, n in enumerate(mask3d.shape):
            other = tuple(a for a in range(3) if a != axis)
            hit = np.flatnonzero(mask3d.any(axis=other))
            lo, hi = int(hit[0]), int(hit[-1]) + 1
            width = min(-(-(hi - lo) // BOX_QUANTUM) * BOX_QUANTUM, n)
            lo = max(0, min(lo, n - width))
            boxes.append(slice(lo, lo + width))
        return tuple(boxes)

    def function_supports(self, threshold: float = LOCAL_BOX_THRESHOLD):
        r"""Bounding box of every basis function (where :math:`|\chi|^2`
        exceeds ``threshold`` of its peak), as index ranges ``(3, 2)``."""
        if self._supports is None:
            shape = self.grid.shape
            out = []
            for chi in self.psi:
                weight = np.abs(chi.reshape(shape)) ** 2
                box = self._bounding_box(weight > threshold * weight.max())
                out.append([(b.start, b.stop) for b in box])
            self._supports = np.array(out)                     # (M, 3, 2)
        return self._supports

    def _touching(self, box):
        """Basis functions whose support meets ``box``."""
        sup = self.function_supports()
        lo = np.array([b.start for b in box])
        hi = np.array([b.stop for b in box])
        inside = np.all((sup[:, :, 0] < hi) & (sup[:, :, 1] > lo), axis=1)
        return np.flatnonzero(inside)

    def local_factors(self, D) -> np.ndarray:
        r"""AO columns :math:`l_k` with :math:`D = \sum_k l_k l_k^\dagger`,
        as localized as possible.

        An idempotent closed-shell density (every occupation equal, as every
        SCF iterate is) is factored into its **Foster-Boys** orbitals: they
        decay fastest, so their boxes are smallest (C16H34 PAW-LCAO-SZ: 61% of
        the grid at a 1e-8 threshold against 75% for pivoted Cholesky
        factors).  Anything else falls back to :func:`pivoted_cholesky`.
        """
        from ..algorithms.local_correlation import (boys_localize,
                                                    position_matrices)

        ints = self.integrals
        if not getattr(ints, "orthogonalize", True):
            return pivoted_cholesky(D)
        X = ints._lowdin_x()
        half = ints.overlap() @ X                      # S^{1/2}
        D_orth = half @ D @ half.conj().T
        D_orth = 0.5 * (D_orth + D_orth.conj().T)
        weights, vectors = np.linalg.eigh(D_orth)
        floor = OCCUPATION_FLOOR * max(float(np.max(np.abs(weights))), 1.0)
        occupied = weights > floor
        w = weights[occupied]
        if not w.size or np.ptp(w) > 1e-8 * w.max():
            return pivoted_cholesky(D)
        if getattr(self, "_position", None) is None:
            self._position = position_matrices(ints)
        # Boys rotates with real angles, which keeps the orbitals real
        # functions only if they are real to begin with: the eigenvectors of
        # D in a basis of complex harmonics are arbitrary complex mixtures,
        # so they are made conjugation-real first (localization failed
        # without this: boxes covered 99% of the grid on C16H34).
        occupied_orbitals = ints.real_orbitals(vectors[:, occupied], ())
        lmo, _U = boys_localize(self._position, occupied_orbitals)
        return X @ lmo * np.sqrt(w.mean())

    def exchange_local(self, D, threshold: float = LOCAL_BOX_THRESHOLD):
        r""":meth:`exchange` with every Poisson solve on a local box.

        ``D`` is factored into localized orbitals :math:`\phi_k`
        (:meth:`local_factors`).  Both densities of term ``k``,
        :math:`\phi_k^*\chi_q` and :math:`\chi_p^*\phi_k`, carry
        :math:`\phi_k`, so they are taken on the box where
        :math:`|\phi_k|^2` exceeds ``threshold`` of its peak, only for the
        basis functions whose product with :math:`\phi_k` is not negligible
        (``threshold`` of the largest :math:`\|\phi_k\chi_q\|^2`) and that
        reach that box, and solved there by the same
        zero-padded FFT the global solve uses -- exact when the box covers
        the density.  The work per orbital is then independent of the size
        of the molecule.
        """
        D = np.asarray(D)
        D = 0.5 * (D + D.conj().T)
        factors = self.local_factors(D)
        shape = self.grid.shape
        M = self.n_basis
        K = np.zeros((M, M), dtype=complex)
        psi3 = self.psi.reshape((M,) + shape)
        phis = factors.T @ self.psi                            # (o, G)
        # Product screening: the pair density phi_k^* chi_q matters only
        # where both are large.  ||phi_k chi_q||^2 for every (k, q) is one
        # matrix product; a function is kept for orbital k when its product
        # norm reaches `threshold` of the largest one.
        products = (np.abs(phis) ** 2) @ (np.abs(self.psi) ** 2).T
        supports = self.function_supports()
        for k, c in enumerate(factors.T):
            phi = phis[k].reshape(shape)
            kept = np.flatnonzero(products[k] >= threshold * products[k].max())
            weight = np.abs(phi) ** 2
            mask = weight > threshold * weight.max()
            # Restrict the box to where a kept function lives.
            lo = supports[kept, :, 0].min(axis=0)
            hi = supports[kept, :, 1].max(axis=0)
            region = np.zeros(shape, dtype=bool)
            region[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
            box = self._bounding_box(mask & region if (mask & region).any()
                                     else mask)
            near = np.intersect1d(kept, self._touching(box))
            phi_box = phi[box].reshape(-1)
            chi_box = psi3[(near,) + box].reshape(len(near), -1)
            solver = self._box_solver(tuple(b.stop - b.start for b in box))
            left = np.conj(chi_box) * phi_box
            block = np.zeros((len(near), len(near)), dtype=complex)
            for q0 in range(0, len(near), POISSON_BLOCK):
                potentials = solver.solve_stack(
                    np.conj(phi_box) * chi_box[q0:q0 + POISSON_BLOCK])
                block[:, q0:q0 + POISSON_BLOCK] = left @ potentials.T * self.dV
            K[np.ix_(near, near)] += block
        aug = self.augmentation()
        if aug is not None:
            Qs, Ws, U = aug
            QD = [Q @ D for Q in Qs]
            for c, (Q, W) in enumerate(zip(Qs, Ws)):
                K = K + QD[c] @ W + W @ D @ Q
                for d, Qd in enumerate(Qs):
                    if U[c, d] != 0:
                        K = K + U[c, d] * (QD[c] @ Qd)
        return K

    def fock_two_electron(self, D, exact: bool = False) -> np.ndarray:
        r""":math:`J[D] - \tfrac12 K[D]` (AO basis), the closed-shell RHF
        two-electron Fock term; the exchange on local boxes when
        :attr:`box_threshold` is set, unless ``exact``.  Only the bulk of an
        SCF takes the local one: every quantity built from the converged
        orbitals (the frozen core, the Fock matrix a selector reads) asks
        for the exact one."""
        K = (self.exchange(D) if self.box_threshold is None or exact
             else self.exchange_local(D, self.box_threshold))
        return self.coulomb(D) - 0.5 * K

    # -- integrals of an orbital set --------------------------------------- #

    def _transformed_augmentation(self, left, right):
        """``(Q, W, U)`` with every channel matrix as ``left^H M right``."""
        aug = self.augmentation()
        if aug is None:
            return None
        Qs, Ws, U = aug
        Ld = np.asarray(left).conj().T
        right = np.asarray(right)
        return ([Ld @ Q @ right for Q in Qs], [Ld @ W @ right for W in Ws], U)

    def pair_exchange(self, occupied, virtual) -> np.ndarray:
        r""":math:`\langle ii|aa\rangle` for orbitals ``occupied`` (AO
        columns, index ``i``) and ``virtual`` (index ``a``): the coupling of
        the pair excitation :math:`i^2 \to a^2`, the exchange-type integral of
        the density :math:`\phi_i^*\phi_a` with itself.  One Poisson solve
        per ``(i, a)``."""
        occupied, virtual = np.asarray(occupied), np.asarray(virtual)
        phi_i = occupied.T @ self.psi
        phi_a = virtual.T @ self.psi
        out = np.zeros((phi_i.shape[0], phi_a.shape[0]), dtype=complex)
        for i, left in enumerate(phi_i):
            for a0 in range(0, phi_a.shape[0], POISSON_BLOCK):
                rho = np.conj(left) * phi_a[a0:a0 + POISSON_BLOCK]
                potential = self.potential(rho)
                out[i, a0:a0 + POISSON_BLOCK] = np.sum(
                    rho * potential, axis=1) * self.dV
        aug = self._transformed_augmentation(occupied, virtual)
        if aug is not None:
            Qs, Ws, U = aug
            for c, (Q, W) in enumerate(zip(Qs, Ws)):
                out += 2.0 * Q * W
                for d, Qd in enumerate(Qs):
                    if U[c, d] != 0:
                        out += U[c, d] * Q * Qd
        return out

    def orbital_integrals(self, C) -> np.ndarray:
        r""":math:`\langle pq|rs\rangle` over the orbitals ``C`` (AO columns).

        :math:`n^2/2` pair densities are solved (the Hermitian half) and the
        tensor is assembled from their Gram matrices, as the full engine does;
        the augmentation is transformed into the orbitals first.  Cost
        :math:`n^2` solves and :math:`n^4 G`, with :math:`n` the number of
        orbitals rather than of basis functions.
        """
        C = np.asarray(C)
        phi = C.T @ self.psi                                  # (n, G)
        n = phi.shape[0]
        iu, ju = np.triu_indices(n)
        index = np.empty((n, n), dtype=np.int64)
        index[iu, ju] = np.arange(iu.size)
        index[ju, iu] = np.arange(iu.size)
        densities = np.conj(phi[iu]) * phi[ju]                # rho_{ij}, i <= j
        potentials = np.concatenate([
            self.potential(densities[k:k + POISSON_BLOCK])
            for k in range(0, iu.size, POISSON_BLOCK)]) if iu.size else \
            np.zeros((0, phi.shape[1]), dtype=complex)
        # Gram blocks: <a b|c d> = sum_g rho_ac Phi_bd dV.  rho_ca = conj(rho_ac)
        # and Phi_db = conj(Phi_bd), so two Gram matrices cover every case.
        G1 = densities @ potentials.T * self.dV               # rho_u . Phi_v
        G2 = np.conj(densities) @ potentials.T * self.dV      # conj(rho_u) . Phi_v
        u = index.reshape(-1)                                 # over (a, c)
        lower = (np.arange(n)[:, None] > np.arange(n)[None, :]).reshape(-1)
        left, right = lower[:, None], lower[None, :]          # a > c, b > d
        g1, g2 = G1[np.ix_(u, u)], G2[np.ix_(u, u)]
        R = np.where(left, np.where(right, np.conj(g1), g2),
                     np.where(right, np.conj(g2), g1))        # R[(a,c),(b,d)]
        eri = R.reshape(n, n, n, n).transpose(0, 2, 1, 3).copy()
        aug = self.augmentation()
        if aug is not None:
            Qs, Ws, U = aug
            Cd = C.conj().T
            Qt = [Cd @ Q @ C for Q in Qs]
            Wt = [Cd @ W @ C for W in Ws]
            for c, (Q, W) in enumerate(zip(Qt, Wt)):
                eri += np.einsum("pr,qs->pqrs", Q, W)
                eri += np.einsum("pr,qs->pqrs", W, Q)
                coupled = sum(U[c, d] * Qd for d, Qd in enumerate(Qt)
                              if U[c, d] != 0)
                if not np.isscalar(coupled):
                    eri += np.einsum("pr,qs->pqrs", Q, coupled)
        return eri

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        r""":math:`K^{ij}_{ab} = \langle ij|ab\rangle` for local pairs.

        ``occupied`` and ``functions`` are AO columns (orbitals
        :math:`\phi_i` and virtual functions :math:`\tilde\chi_a`, e.g.
        PAOs); ``pairs`` maps ``(i, j)`` to the function indices of the pair's
        domain.  Grouped by ``j``: the potentials of
        :math:`\phi_j^*\tilde\chi_b` over the union of ``j``'s domains are
        solved once, and every pair contracts its own densities against them,
        so the work is the sum of the domain sizes, not :math:`o M`.
        """
        occupied, functions = np.asarray(occupied), np.asarray(functions)
        phi = occupied.T @ self.psi                            # (o, G)
        chi = functions.T @ self.psi                           # (m, G)
        by_j: dict[int, list] = {}
        for (i, j), domain in pairs.items():
            by_j.setdefault(j, []).append((i, list(domain)))
        aug = self._transformed_augmentation(occupied, functions)
        shape = self.grid.shape
        out = {}
        for j, members in by_j.items():
            union = sorted({b for _i, dom in members for b in dom})
            where = {b: k for k, b in enumerate(union)}
            box, solve = None, self.potential
            if self.box_threshold is not None:
                # rho_jb carries phi_j and rho_ia carries phi_i: the potential
                # is needed only where some partner phi_i lives, so the box
                # covers phi_j and its partners (local by pair screening).
                mask = np.zeros(shape, dtype=bool)
                for k in {j} | {i for i, _dom in members}:
                    weight = np.abs(phi[k].reshape(shape)) ** 2
                    mask |= weight > self.box_threshold * weight.max()
                box = self._bounding_box(mask)
                flat = np.zeros(shape, dtype=bool)
                flat[box] = True
                keep = np.flatnonzero(flat.reshape(-1))
                solve = self._box_solver(
                    tuple(b.stop - b.start for b in box)).solve_stack
            take = (lambda array: array) if box is None else (
                lambda array: array[..., keep])
            potentials = np.concatenate([
                solve(take(np.conj(phi[j]) * chi[union[k:k + POISSON_BLOCK]]))
                for k in range(0, len(union), POISSON_BLOCK)])
            for i, domain in members:
                rho = take(np.conj(phi[i]) * chi[domain])      # rho_ia
                cols = [where[b] for b in domain]
                K = rho @ potentials[cols].T * self.dV
                if aug is not None:
                    Qs, Ws, U = aug
                    for c, (Q, W) in enumerate(zip(Qs, Ws)):
                        K = K + np.outer(Q[i, domain], W[j, domain]) \
                            + np.outer(W[i, domain], Q[j, domain])
                        for d, Qd in enumerate(Qs):
                            if U[c, d] != 0:
                                K = K + U[c, d] * np.outer(Q[i, domain],
                                                           Qd[j, domain])
                out[(i, j)] = K
        return out
