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

__all__ = ["DirectCoulomb", "augmentation_coulomb", "augmentation_exchange",
           "augmentation_tensor"]

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


def augmentation_coulomb(augmentation, D) -> np.ndarray:
    r"""The family augmentation's :math:`J[D]` (AO basis): with
    :math:`t^Q_c = \mathrm{tr}(Q^c D)` and :math:`t^W_c` alike,
    :math:`\sum_c Q^c (t^W_c + (U t^Q)_c) + W^c t^Q_c`."""
    Qs, Ws, U = augmentation
    D = np.asarray(D)
    tQ = np.array([np.sum(Q * D.T) for Q in Qs])
    tW = np.array([np.sum(W * D.T) for W in Ws])
    coupled = U @ tQ
    J = 0.0
    for c, (Q, W) in enumerate(zip(Qs, Ws)):
        J = J + Q * (tW[c] + coupled[c]) + W * tQ[c]
    return J


def augmentation_exchange(augmentation, D) -> np.ndarray:
    r"""The family augmentation's :math:`K[D]` (AO basis),
    :math:`\sum_c Q^c D W^c + W^c D Q^c + \sum_{cd} U_{cd} Q^c D Q^d`."""
    Qs, Ws, U = augmentation
    D = np.asarray(D)
    QD = [Q @ D for Q in Qs]
    K = 0.0
    for c, (Q, W) in enumerate(zip(Qs, Ws)):
        if np.any(W):
            K = K + QD[c] @ W + W @ D @ Q
        for d, Qd in enumerate(Qs):
            if U[c, d] != 0:
                K = K + U[c, d] * (QD[c] @ Qd)
    return K


def augmentation_tensor(augmentation, C) -> np.ndarray:
    r"""The family augmentation's :math:`\Delta\langle pq|rs\rangle` over
    the orbitals ``C`` (AO columns): each channel matrix is transformed,
    :math:`C^\dagger Q^c C`, and the low-rank sum of the module docstring
    is formed in those orbitals only."""
    Qs, Ws, U = augmentation
    C = np.asarray(C)
    n = C.shape[1]
    Cd = C.conj().T
    Qt = [Cd @ Q @ C for Q in Qs]
    Wt = [Cd @ W @ C for W in Ws]
    eri = np.zeros((n, n, n, n), dtype=np.result_type(*Qt, *Wt, U, float))
    for c, (Q, W) in enumerate(zip(Qt, Wt)):
        if np.any(W):
            eri += np.einsum("pr,qs->pqrs", Q, W)
            eri += np.einsum("pr,qs->pqrs", W, Q)
        coupled = sum(U[c, d] * Qd for d, Qd in enumerate(Qt)
                      if U[c, d] != 0)
        if not np.isscalar(coupled):
            eri += np.einsum("pr,qs->pqrs", Q, coupled)
    return eri


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
        r"""``(Q, W, U)`` per channel (AO basis), or ``None``.

        The compensation multipoles, then -- for a PAW-LCAO Hamiltonian with
        exact spheres -- the one-center Coulomb factors
        (:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.one_center_channels`)
        as channels of their own with :math:`W = 0` and a diagonal block of
        ``U``: :math:`\sum_k \lambda_k G^k\otimes G^k` is the ``U`` term of
        the module docstring's sum."""
        if self._augmentation is None:
            moments = getattr(self.integrals, "compensation_moments", None)
            channels = moments() if moments is not None else {}
            spheres = getattr(self.integrals, "one_center_channels", None)
            G, values = spheres() if spheres is not None else ((), ())
            if not channels and not len(values):
                self._augmentation = False
            else:
                Qs, Ws = [], []
                U = np.zeros((0, 0))
                if channels:
                    Wd = self.integrals.compensation_potentials()
                    U = np.asarray(self.integrals.compensation_coulomb())
                    order = self.integrals.multipole_channels()
                    Qs = [np.asarray(channels[c]) for c in order]
                    Ws = [np.asarray(Wd[c]) for c in order]
                if len(values):
                    n, k = len(Qs), len(values)
                    zero = np.zeros_like(np.asarray(G[0]))
                    Qs = Qs + [np.asarray(g) for g in G]
                    Ws = Ws + [zero] * k
                    extended = np.zeros((n + k, n + k),
                                        dtype=np.result_type(U, float))
                    extended[:n, :n] = U
                    extended[n:, n:] = np.diag(np.asarray(values, dtype=float))
                    U = extended
                self._augmentation = (Qs, Ws, U)
        return self._augmentation or None

    # -- Coulomb and exchange ---------------------------------------------- #

    def _potential_dtype(self, density_dtype):
        """The dtype the solver returns for densities of ``density_dtype``:
        real for real densities on an even (real) kernel."""
        real = (not np.issubdtype(density_dtype, np.complexfloating)
                and bool(getattr(self.solver, "even", False)))
        return np.float64 if real else np.complex128

    def potential(self, densities, out=None) -> np.ndarray:
        """Hartree potentials of a ``(k, G)`` stack of densities, written to
        ``out`` when given (see
        :meth:`~mandacaru.integrals.poisson.PoissonFFTSolver.solve_stack`)."""
        return self.solver.solve_stack(np.ascontiguousarray(densities),
                                       out=out)

    def coulomb(self, D) -> np.ndarray:
        r""":math:`J_{pq} = \sum_{rs} D_{sr}\langle pr|qs\rangle` (AO basis).

        The density of a Hermitian ``D`` is real, and is solved through the
        real transforms."""
        D = np.asarray(D)
        psi = self.psi
        density = np.sum((D.T @ psi) * np.conj(psi), axis=0)
        if np.array_equal(D, D.conj().T):
            density = np.ascontiguousarray(density.real)
        V = self.potential(density[None, :])[0]
        J = (np.conj(psi) * V) @ psi.T * self.dV
        aug = self.augmentation()
        if aug is not None:
            J = J + augmentation_coulomb(aug, D)
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
        dtype = np.result_type(psi, vectors, float)
        K = np.zeros((self.n_basis, self.n_basis), dtype=dtype)
        rho = np.empty((min(POISSON_BLOCK, self.n_basis), psi.shape[1]),
                       dtype=dtype)
        potentials = np.empty_like(rho, dtype=self._potential_dtype(dtype))
        for w, c in zip(weights[keep], vectors[:, keep].T):
            phi = c @ psi
            left = np.conj(psi) * phi
            np.conjugate(phi, out=phi)
            for q0 in range(0, self.n_basis, POISSON_BLOCK):
                n = min(POISSON_BLOCK, self.n_basis - q0)
                np.multiply(phi, psi[q0:q0 + n], out=rho[:n])
                block = self.potential(rho[:n], out=potentials[:n])
                K[:, q0:q0 + n] += w * (left @ block.T) * self.dV
        aug = self.augmentation()
        if aug is not None:
            K = K + augmentation_exchange(aug, D)
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
        K = np.zeros((M, M), dtype=np.result_type(self.psi, factors, float))
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
            block = np.zeros((len(near), len(near)), dtype=K.dtype)
            for q0 in range(0, len(near), POISSON_BLOCK):
                potentials = solver.solve_stack(
                    np.conj(phi_box) * chi_box[q0:q0 + POISSON_BLOCK])
                block[:, q0:q0 + POISSON_BLOCK] = left @ potentials.T * self.dV
            K[np.ix_(near, near)] += block
        aug = self.augmentation()
        if aug is not None:
            K = K + augmentation_exchange(aug, D)
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
        out = np.zeros((phi_i.shape[0], phi_a.shape[0]),
                       dtype=np.result_type(phi_i, phi_a, float))
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

    def spinor_exchange(self, alpha, beta) -> np.ndarray:
        r""":math:`\langle ij|ji\rangle` over two-component spinors.

        Spinor :math:`i` has the AO-coefficient columns ``alpha[:, i]`` and
        ``beta[:, i]`` for its two spin components, and its pair density with
        :math:`j` sums over the spin it keeps,
        :math:`P_{ij} = \sum_\sigma \psi_{i\sigma}^*\psi_{j\sigma}`, so
        :math:`\langle ij|ji\rangle = \int P_{ij}\,\overline{V[P_{ij}]}` --
        real and non-negative.  One Poisson solve per pair :math:`i \le j`
        (the exchange energy of a spinor determinant needs nothing else), the
        augmentation transformed into the spinors' pair matrices
        :math:`\sum_\sigma c_\sigma^\dagger Q c_\sigma`.
        """
        alpha, beta = np.asarray(alpha), np.asarray(beta)
        phi = (alpha.T @ self.psi, beta.T @ self.psi)
        n = alpha.shape[1]
        out = np.zeros((n, n))
        for i in range(n):
            for j0 in range(i, n, POISSON_BLOCK):
                j = slice(j0, min(j0 + POISSON_BLOCK, n))
                rho = (np.conj(phi[0][i]) * phi[0][j]
                       + np.conj(phi[1][i]) * phi[1][j])
                potential = self.potential(rho)
                values = np.real(np.sum(rho * np.conj(potential), axis=1)
                                 ) * self.dV
                out[i, j] = values
                out[j, i] = values
        aug = self.augmentation()
        if aug is not None:
            Qs, Ws, U = aug

            def pair(Q):
                return (alpha.conj().T @ Q @ alpha
                        + beta.conj().T @ Q @ beta)

            Qt, Wt = [pair(Q) for Q in Qs], [pair(W) for W in Ws]
            extra = np.zeros((n, n), dtype=complex)
            for c, (Q, W) in enumerate(zip(Qt, Wt)):
                extra += Q * W.T + W * Q.T
                for d, Qd in enumerate(Qt):
                    if U[c, d] != 0:
                        extra += U[c, d] * Q * Qd.T
            out = out + np.real(extra)
        return out

    def orbital_integrals(self, C) -> np.ndarray:
        r""":math:`\langle pq|rs\rangle` over the orbitals ``C`` (AO columns).

        :math:`n^2/2` pair densities are solved (the Hermitian half), a block
        at a time, and each potential's slab of the tensor is one product, as
        the full engine does (:func:`~mandacaru.integrals.engine.contract_pair_potentials`);
        the augmentation is transformed into the orbitals first.  Cost
        :math:`n^2` solves and :math:`n^4 G`, with :math:`n` the number of
        orbitals rather than of basis functions, and grid memory of one
        block.
        """
        from .engine import contract_pair_potentials

        C = np.asarray(C)
        phi = C.T @ self.psi                                  # (n, G)
        n = phi.shape[0]
        iu, ju = np.triu_indices(n)
        # The engine's contraction, block by block: the potential of rho_bd
        # gives the whole slab eri[:, b, :, d] = (conj(phi) Phi_bd) phi^T dV,
        # and its flip eri[:, d, :, b] the conjugate transpose.
        dtype = np.result_type(phi, float)
        rho = np.empty((min(POISSON_BLOCK, max(iu.size, 1)), phi.shape[1]),
                       dtype=dtype)
        potentials = np.empty_like(rho, dtype=self._potential_dtype(dtype))
        eri = np.empty((n, n, n, n), dtype=potentials.dtype)
        real = not np.iscomplexobj(eri)
        conj_phi = np.conj(phi)
        for v0 in range(0, iu.size, POISSON_BLOCK):
            b, d = iu[v0:v0 + POISSON_BLOCK], ju[v0:v0 + POISSON_BLOCK]
            k = b.size
            np.multiply(conj_phi[b], phi[d], out=rho[:k])
            self.potential(rho[:k], out=potentials[:k])
            X = contract_pair_potentials(conj_phi, phi, potentials[:k],
                                         self.solver.workers)
            X *= self.dV
            flipped = X.transpose(0, 2, 1)
            eri[:, d, :, b] = flipped if real else np.conj(flipped)
            eri[:, b, :, d] = X
        aug = self.augmentation()
        if aug is not None:
            eri += augmentation_tensor(aug, C)
        return eri

    def ovov(self, occupied, virtual, occupied2=None, virtual2=None):
        r""":math:`(ia|jb) = \langle ij|ab\rangle` as an ``(o, v, o', v')``
        array: electron 1 on the pair densities :math:`\phi_i^*\phi_a` of
        ``occupied`` and ``virtual`` (AO columns), electron 2 on those of
        ``occupied2`` and ``virtual2`` (default: the same two sets) -- the
        blocks MP2 reads, an open shell's with one spin's orbitals on each
        electron.  :math:`o'v'` Poisson solves, a block at a time, each
        contracted against every electron-1 pair at once
        (:func:`~mandacaru.integrals.engine.contract_pair_potentials`), so
        no :math:`M^4` array and the grid memory of one block."""
        from .engine import contract_pair_potentials

        Co, Cv = np.asarray(occupied), np.asarray(virtual)
        Co2 = Co if occupied2 is None else np.asarray(occupied2)
        Cv2 = Cv if virtual2 is None else np.asarray(virtual2)
        conj_i = np.conj(Co.T @ self.psi)                     # (o, G)
        phi_a = Cv.T @ self.psi                               # (v, G)
        conj_j = np.conj(Co2.T @ self.psi)                    # (o', G)
        phi_b = Cv2.T @ self.psi                              # (v', G)
        o2, v2 = conj_j.shape[0], phi_b.shape[0]
        dtype = np.result_type(conj_i, phi_a, conj_j, phi_b, float)
        out = np.empty((conj_i.shape[0], phi_a.shape[0], o2, v2),
                       dtype=dtype)
        j_index, b_index = np.divmod(np.arange(o2 * v2), v2)
        rho = np.empty((min(POISSON_BLOCK, max(o2 * v2, 1)), self.psi.shape[1]),
                       dtype=np.result_type(conj_j, phi_b))
        potentials = np.empty_like(rho, dtype=self._potential_dtype(rho.dtype))
        for v0 in range(0, o2 * v2, POISSON_BLOCK):
            j, b = j_index[v0:v0 + POISSON_BLOCK], b_index[v0:v0 + POISSON_BLOCK]
            k = j.size
            np.multiply(conj_j[j], phi_b[b], out=rho[:k])
            self.potential(rho[:k], out=potentials[:k])
            X = contract_pair_potentials(conj_i, phi_a, potentials[:k],
                                         self.solver.workers)    # (k, o, v)
            X *= self.dV
            out[:, :, j, b] = X.transpose(1, 2, 0)
        aug = self.augmentation()
        if aug is not None:
            Qs, Ws, U = aug
            Ld, Ld2 = Co.conj().T, Co2.conj().T
            Q1 = [Ld @ Q @ Cv for Q in Qs]
            W1 = [Ld @ W @ Cv for W in Ws]
            Q2 = [Ld2 @ Q @ Cv2 for Q in Qs]
            W2 = [Ld2 @ W @ Cv2 for W in Ws]
            for c in range(len(Qs)):
                out += np.einsum("ia,jb->iajb", Q1[c], W2[c])
                out += np.einsum("ia,jb->iajb", W1[c], Q2[c])
                for d in range(len(Qs)):
                    if U[c, d] != 0:
                        out += U[c, d] * np.einsum("ia,jb->iajb", Q1[c], Q2[d])
        return out

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        r""":math:`K^{ij}_{ab} = \langle ij|ab\rangle` for local pairs.

        ``occupied`` and ``functions`` are AO columns (orbitals
        :math:`\phi_i` and virtual functions :math:`\tilde\chi_a`, e.g.
        PAOs); ``pairs`` maps ``(i, j)`` to the function indices of the pair's
        domain.  Grouped by ``j``: the potentials of
        :math:`\phi_j^*\tilde\chi_b` over the union of ``j``'s domains are
        solved once, a block at a time, and every pair contracts its own
        densities against each block, so the work is the sum of the domain
        sizes, not :math:`o M`, and the grid memory one block's.
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
            # The union's potentials a block at a time (never the whole
            # union resident): each block is contracted into every member
            # whose domain reaches it.
            blocks = {i: np.zeros((len(domain), len(domain)),
                                  dtype=np.result_type(phi, chi, float))
                      for i, domain in members}
            for k0 in range(0, len(union), POISSON_BLOCK):
                chunk = union[k0:k0 + POISSON_BLOCK]
                potentials = solve(take(np.conj(phi[j]) * chi[chunk]))
                for i, domain in members:
                    inside = [n for n, b in enumerate(domain)
                              if k0 <= where[b] < k0 + len(chunk)]
                    if not inside:
                        continue
                    rho = take(np.conj(phi[i]) * chi[domain])  # rho_ia
                    rows = [where[domain[n]] - k0 for n in inside]
                    blocks[i][:, inside] = rho @ potentials[rows].T * self.dV
            for i, domain in members:
                K = blocks[i]
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
