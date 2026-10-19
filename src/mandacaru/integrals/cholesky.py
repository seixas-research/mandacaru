# -*- coding: utf-8 -*-
# file: integrals/cholesky.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Cholesky-factored two-electron integrals, built on the grid without the
tensor.

The two-electron integrals are the Coulomb inner products of pair densities,
:math:`(ac|bd) = \iint \rho_{ac}(\mathbf r_1)\,\rho_{bd}(\mathbf r_2)/r_{12}`
with :math:`\rho_{ac} = \psi_a^*\psi_c`: a positive semidefinite matrix over
pairs, of numerical rank far below :math:`M^2`.  A pivoted, truncated Cholesky
decomposition (Beebe and Linderberg 1977; Koch, Sanchez de Meras and Pedersen
2003) writes it as

.. math::

    (ac|bd) \approx \sum_{r=1}^{R} B^r_{ac} B^r_{bd},
    \qquad B^r = (B^r)^T ,

so :math:`4 M^2 R` bytes (the upper triangles) stand for the :math:`8 M^4`
of the real tensor.  Every
residual Coulomb self-energy of a pair density left out is below the
``tolerance`` (Hartree), which bounds every integral's error by the
Cauchy-Schwarz inequality.  It does **not** bound an energy error by the same
number -- a density contraction sums many integrals -- so the tolerance is
chosen by the energies it keeps (the guide's table).

The factorization is generated matrix-free: the pair matrix is never formed.

* **Real pair densities.**  A molecule's basis functions use real spherical
  harmonics, so the pair densities :math:`\phi_i\phi_j` (:math:`i \le j`)
  are real, their potentials are real-transform Poisson solves, and the
  factors are real symmetric matrices :math:`B^r`.
* **The exact diagonal by Parseval.**  A pair density vanishes outside the
  box of the zero-padded mesh, so its Coulomb self-energy is
  :math:`dV^2 H^{-1}\sum_k |\hat\rho(k)|^2 \hat K(k)`: one pruned forward
  transform per pair and no inverse
  (:meth:`~mandacaru.integrals.poisson.PoissonFFTSolver.self_energies`).
* **Screening by support.**  The sampled functions are exactly zero beyond
  their cutoff radius, so two functions whose sampled supports do not meet
  have a pair density that is identically zero: never a pivot, never
  computed.
* **Batched two-step pivots** (Folkestad, Kjonstad and Koch 2019).  Each
  round takes the largest residual diagonals above ``span`` times the
  current maximum, solves their potentials as one batch, computes their
  exact columns, and runs a pivoted Cholesky inside the batch, accepting
  pivots while they stay above that floor.
* **A column is one product.**  The column of pivot :math:`p` over every pair
  is :math:`\mathrm{vec}((\phi \odot \Phi_p)\,\phi^T)\,dV`
  (:func:`~mandacaru.integrals.engine.contract_pair_potentials`), no pair
  densities to stream.
* **Factors grown in place**, in a buffer whose capacity doubles.
* **Threads that change nothing.**  The diagonal's batches of pair
  densities are shared out over the threads, each on an FFT workspace of
  its own; the columns and the updates by the factors run in fixed blocks
  of rows, one core each (:func:`~mandacaru.integrals.fft.parallel_blocks`).
  The work is cut by the problem, not by the thread count, so the factors
  and the pivots are the same bits on any number of cores.

The PAW-LCAO augmentation stays outside the factorization in its structured
form :math:`(Q, W, U)` and is added exactly by every consumer: its pair
matrix is not guaranteed positive semidefinite.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .direct import (augmentation_coulomb, augmentation_exchange,
                     augmentation_tensor)
from .engine import contract_pair_potentials, eri_memory_budget_mb
from .fft import parallel_blocks, parallel_matmul, thread_count

__all__ = ["CholeskyERI", "TwoBodySpec", "cholesky_two_body",
           "resolve_two_body", "TWO_BODY_METHODS",
           "DEFAULT_TOLERANCE", "CHOLESKY_SPAN", "CHOLESKY_BATCH"]

#: The representations ``Mandacaru(two_body=...)`` accepts: the dense
#: tensor (the default) and its Cholesky factors.
TWO_BODY_METHODS = ("dense", "cholesky")

#: Default truncation (Hartree) of the residual Coulomb self-energy of every
#: pair density: the loosest one that kept every measured energy within
#: 1e-6 Ha of the dense tensor -- RHF, MP2 and active-space FCI of water
#: (DZP), N2 (SZP) and LiH (DZP), and the RHF of benzene (DZP, M = 108),
#: whose error at 3e-7 Ha was 1.1e-6 Ha and at 1e-7 Ha 2.3e-7 Ha.
DEFAULT_TOLERANCE = 1e-7

#: Span factor of the batched pivots: a round accepts pivots whose residual
#: is above this fraction of the round's largest one.
CHOLESKY_SPAN = 1e-2

#: Pivot candidates whose columns are computed together in one round.
CHOLESKY_BATCH = 32

#: Pair densities per batch of the diagonal (Parseval) transforms; the
#: batches are shared out over the threads.
DIAGONAL_BLOCK = 64


#: Largest negative residual diagonal (relative to the initial largest)
#: tolerated as round-off before the factorization is declared broken.
NEGATIVE_RESIDUAL = 1e-9


@dataclass(frozen=True)
class TwoBodySpec:
    """A resolved ``two_body`` option: the method and, for ``"cholesky"``,
    its tolerance (Hartree).

    ``pivots`` is internal, never read from the user's option: the ordered
    ``(i, j)`` basis-function pairs of a reference factorization
    (:attr:`CholeskyERI.pivot_pairs`), which a displaced geometry reuses so
    that its factors are a smooth function of the nuclear positions
    (:func:`cholesky_two_body`).
    """

    method: str = "dense"
    tolerance: float | None = None
    pivots: tuple[tuple[int, int], ...] | None = None

    @property
    def cholesky(self) -> bool:
        return self.method == "cholesky"


def resolve_two_body(spec) -> TwoBodySpec:
    """Validate a ``two_body`` option.

    ``None`` or ``"dense"`` is the dense tensor; ``"cholesky"`` the factors
    at :data:`DEFAULT_TOLERANCE`; a dict ``{"method": ..., "tolerance": ...}``
    sets the tolerance of the factors (Hartree).  Anything else is refused
    with the accepted spellings.
    """
    if isinstance(spec, TwoBodySpec):
        return spec
    if spec is None:
        return TwoBodySpec()
    if isinstance(spec, str):
        options = {"method": spec}
    elif isinstance(spec, dict):
        options = dict(spec)
    else:
        raise TypeError(
            f"two_body must be 'dense', 'cholesky' or a dict "
            f"{{'method': ..., 'tolerance': ...}}, got {spec!r}")
    unknown = set(options) - {"method", "tolerance"}
    if unknown:
        raise ValueError(f"unknown two_body option(s) {sorted(unknown)}; "
                         "the keys are 'method' and 'tolerance'")
    method = str(options.get("method", "dense")).strip().lower()
    if method not in TWO_BODY_METHODS:
        raise ValueError(f"unknown two_body method {method!r}; use one of "
                         f"{TWO_BODY_METHODS}")
    tolerance = options.get("tolerance")
    if method == "dense":
        if tolerance is not None:
            raise ValueError("the dense two-body tensor is exact and takes no "
                             "tolerance; use {'method': 'cholesky', "
                             "'tolerance': ...} for the factors")
        return TwoBodySpec()
    tolerance = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    if not (np.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError(f"the Cholesky tolerance must be a positive number "
                         f"of Hartree, got {options.get('tolerance')!r}")
    return TwoBodySpec("cholesky", tolerance)


class CholeskyERI:
    r"""Two-electron integrals as Cholesky factors:
    :math:`\langle ab|cd\rangle = (ac|bd) = \sum_r B^r_{ac}B^r_{bd} +
    \Delta_{abcd}`.

    The factors are stored as real vectors over the pair functions
    :math:`\phi_i\phi_j` (:math:`i \le j`, the upper triangle) and unpacked
    into the real symmetric :math:`M \times M` matrices :math:`B^r`
    (:meth:`matrices`) a block at a time by every consumer; :math:`\Delta`
    is the family augmentation in its structured form.  Build one with
    :func:`cholesky_two_body`.

    Attributes
    ----------
    vectors : (P, R) float array
        The factor vectors over the ``P`` pair functions (Hartree^1/2).
    rows, cols : (P,) int arrays
        Pair function ``u`` is :math:`\phi_i\phi_j` with ``i = rows[u] <=
        j = cols[u]``.
    tolerance : float
        The truncation (Hartree).
    pivots : (R,) int array
        The pair function of each factor.
    max_residual : float
        The largest residual diagonal left (Hartree): every integral's error
        is below it.  ``nan`` for factors built on given pivots, which skip
        the diagonal.
    augmentation : tuple or None
        ``(Q, W, U)`` of the family's compensation charges (AO basis).
    profile : dict
        Seconds and counts of the generation stages.
    """

    def __init__(self, vectors, rows, cols, n_basis, tolerance, pivots,
                 max_residual, augmentation=None, profile=None):
        self.vectors = vectors
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.n_basis = int(n_basis)
        self.tolerance = float(tolerance)
        self.pivots = np.asarray(pivots, dtype=np.int64)
        self.max_residual = float(max_residual)
        self.augmentation = augmentation
        self.profile = dict(profile or {})

    # -- size --------------------------------------------------------------- #

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def pivot_pairs(self) -> np.ndarray:
        """``(R, 2)``: the ``(i, j)`` basis functions of each factor's pivot,
        in factor order -- what :func:`cholesky_two_body` takes as
        ``pivots``."""
        return np.column_stack([self.rows[self.pivots],
                                self.cols[self.pivots]])

    @property
    def nbytes(self) -> int:
        """Bytes retained: the factor vectors, the pair index and the
        structured augmentation."""
        total = self.vectors.nbytes + self.rows.nbytes + self.cols.nbytes
        if self.augmentation is not None:
            Qs, Ws, U = self.augmentation
            total += sum(np.asarray(a).nbytes for a in (*Qs, *Ws, U))
        return int(total)

    def summary(self) -> str:
        """The factors in one line, as the run log's ``[ELECTRONS]`` block
        writes them: tolerance, rank (and per basis function) and the largest
        residual diagonal -- or, for factors built on given pivots, that the
        diagonal was not computed."""
        residual = ("not computed (the reference's pivots)"
                    if not np.isfinite(self.max_residual)
                    else f"{self.max_residual:.3e} Ha")
        per_function = self.rank / max(self.n_basis, 1)
        return (f"Cholesky factors, tolerance {self.tolerance:.3e} Ha, rank "
                f"{self.rank} ({per_function:.2f} per basis function), "
                f"largest residual diagonal {residual}")

    #: Factors unpacked per block by the consumers.
    BLOCK = 64

    def _blocks(self):
        for start in range(0, self.rank, self.BLOCK):
            yield start, min(start + self.BLOCK, self.rank)

    # -- the factor matrices ------------------------------------------------ #

    def matrices(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        r"""``(k, M, M)``: the real symmetric factors :math:`B^r`, ``r`` in
        ``[start, stop)``."""
        stop = self.rank if stop is None else int(stop)
        v = self.vectors[:, int(start):stop].T                # (k, P)
        B = np.zeros((v.shape[0], self.n_basis, self.n_basis))
        B[:, self.rows, self.cols] = v
        B[:, self.cols, self.rows] = v
        return B

    def to_mo(self, C) -> np.ndarray:
        r"""``(R, n, n)``: :math:`C^\dagger B^r C` for orbitals ``C`` (AO
        columns), the factors of the molecular-orbital integrals.  The
        augmentation is not in them (:meth:`active_tensor` adds it)."""
        C = np.asarray(C)
        n = C.shape[1]
        out = np.empty((self.rank, n, n), dtype=np.result_type(C, float))
        Cd = C.conj().T
        for start, stop in self._blocks():
            out[start:stop] = Cd @ self.matrices(start, stop) @ C
        return out

    def pair_factors(self, C):
        r"""``(F, K)``: the integrals over the orbitals ``C`` (AO columns)
        as pair factors, :math:`(pr|qs) = \sum_{ij} K_{ij} (F_i)_{pr}
        (F_j)_{qs}` -- the factors :math:`C^\dagger B^r C` with :math:`K =
        1`, then the augmentation's channels :math:`C^\dagger Q C` and
        :math:`C^\dagger W C` with their coupling :math:`\begin{pmatrix} U
        & 1\\ 1 & 0\end{pmatrix}`.  What a double factorization starts from
        (:func:`~mandacaru.backends.factorization.spatial_double_factorization`)."""
        F = [self.to_mo(C)]
        blocks = [np.eye(self.rank)]
        if self.augmentation is not None:
            Qs, Ws, U = self._augmentation_blocks(C, C)
            n = len(Qs)
            F.append(np.asarray(Qs + Ws).reshape(2 * n, *F[0].shape[1:]))
            coupling = np.zeros((2 * n, 2 * n))
            coupling[:n, :n] = np.asarray(U, dtype=float)
            coupling[:n, n:] = coupling[n:, :n] = np.eye(n)
            blocks.append(coupling)
        from scipy.linalg import block_diag

        return np.concatenate(F), block_diag(*blocks)

    # -- consumers ---------------------------------------------------------- #

    def active_tensor(self, C) -> np.ndarray:
        r""":math:`\langle pq|rs\rangle` over the orbitals ``C`` (AO columns,
        e.g. the active ones), augmentation included: an :math:`n^4` array,
        never the :math:`M^4` one."""
        C = np.asarray(C)
        n = C.shape[1]
        L = self.to_mo(C).reshape(self.rank, n * n)
        eri = (L.T @ L).reshape(n, n, n, n).transpose(0, 2, 1, 3)
        eri = np.ascontiguousarray(eri)
        if self.augmentation is not None:
            eri = eri + augmentation_tensor(self.augmentation, C)
        return eri

    def _augmentation_blocks(self, left, right):
        """The channel matrices as ``left^H M right``."""
        Qs, Ws, U = self.augmentation
        Ld = np.asarray(left).conj().T
        right = np.asarray(right)
        return [Ld @ Q @ right for Q in Qs], [Ld @ W @ right for W in Ws], U

    def ovov(self, occupied, virtual, occupied2=None,
             virtual2=None) -> np.ndarray:
        r""":math:`(ia|jb) = \langle ij|ab\rangle` as an ``(o, v, o', v')``
        array, for orbitals ``occupied`` and ``virtual`` (AO columns) on
        electron 1 and ``occupied2``, ``virtual2`` (default: the same) on
        electron 2: the blocks MP2 needs, :math:`O(o^2v^2)` and not
        :math:`M^4`."""
        Co, Cv = np.asarray(occupied), np.asarray(virtual)
        same = occupied2 is None and virtual2 is None
        Co2 = Co if occupied2 is None else np.asarray(occupied2)
        Cv2 = Cv if virtual2 is None else np.asarray(virtual2)
        o, v = Co.shape[1], Cv.shape[1]
        o2, v2 = Co2.shape[1], Cv2.shape[1]
        out = np.zeros((o * v, o2 * v2),
                       dtype=np.result_type(Co, Cv, Co2, Cv2, float))
        Cod, Cod2 = Co.conj().T, Co2.conj().T
        for start, stop in self._blocks():
            B = self.matrices(start, stop)
            L = (Cod @ B @ Cv).reshape(-1, o * v)
            L2 = L if same else (Cod2 @ B @ Cv2).reshape(-1, o2 * v2)
            out += L.T @ L2
        out = out.reshape(o, v, o2, v2)
        if self.augmentation is not None:
            Qt, Wt, U = self._augmentation_blocks(Co, Cv)
            Qt2, Wt2, _U = (self._augmentation_blocks(Co2, Cv2)
                            if not same else (Qt, Wt, U))
            for c, (Q, W) in enumerate(zip(Qt, Wt)):
                out = out + np.einsum("ia,jb->iajb", Q, Wt2[c])
                out = out + np.einsum("ia,jb->iajb", W, Qt2[c])
                for d, Qd in enumerate(Qt2):
                    if U[c, d] != 0:
                        out = out + U[c, d] * np.einsum("ia,jb->iajb", Q, Qd)
        return out

    def coulomb(self, D) -> np.ndarray:
        r""":math:`J_{pq} = \sum_{rs} D_{sr}\langle pr|qs\rangle =
        \sum_r B^r_{pq}\,\mathrm{tr}(B^r D)` (AO basis)."""
        D = np.asarray(D)
        J = np.zeros((self.n_basis, self.n_basis),
                     dtype=np.result_type(D, float))
        for start, stop in self._blocks():
            B = self.matrices(start, stop)
            weights = np.einsum("krs,sr->k", B, D)
            J += np.einsum("k,kpq->pq", weights, B)
        if self.augmentation is not None:
            J = J + augmentation_coulomb(self.augmentation, D)
        return J

    def exchange(self, D) -> np.ndarray:
        r""":math:`K_{pq} = \sum_{rs} D_{sr}\langle pr|sq\rangle =
        \sum_r (B^r D B^r)_{pq}` (AO basis), through the eigenvectors of the
        Hermitian ``D``: :math:`\sum_i w_i\sum_r (B^r v_i)(B^r v_i)^\dagger`."""
        D = np.asarray(D)
        D = 0.5 * (D + D.conj().T)
        w, V = np.linalg.eigh(D)
        keep = np.abs(w) > 1e-12 * max(float(np.abs(w).max(initial=0.0)), 1.0)
        w, V = w[keep], V[:, keep]
        M = self.n_basis
        K = np.zeros((M, M), dtype=np.result_type(D, float))
        for start, stop in self._blocks():
            Y = self.matrices(start, stop) @ V                  # (k, M, o)
            Y = Y.transpose(1, 0, 2).reshape(M, -1)             # (M, k o)
            K += (Y * np.tile(w, stop - start)) @ Y.conj().T
        if self.augmentation is not None:
            K = K + augmentation_exchange(self.augmentation, D)
        return K

    def jk(self, D):
        """``(J, K)`` of :meth:`coulomb` and :meth:`exchange`."""
        return self.coulomb(D), self.exchange(D)

    #: No local exchange boxes: every contraction here is exact for the
    #: factors (the :class:`~mandacaru.integrals.direct.DirectCoulomb`
    #: interface the integral-direct SCF reads).
    box_threshold = None

    def fock_two_electron(self, D, exact: bool = True) -> np.ndarray:
        r""":math:`J[D] - \tfrac12 K[D]`, the closed-shell term (AO basis).
        ``exact`` is accepted for the interface of
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.fock_two_electron`;
        the factors have one contraction only."""
        return self.coulomb(D) - 0.5 * self.exchange(D)

    def orbital_integrals(self, C) -> np.ndarray:
        r""":math:`\langle pq|rs\rangle` over the orbitals ``C`` (AO
        columns), as
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.orbital_integrals`:
        :meth:`active_tensor`."""
        return self.active_tensor(C)

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        r""":math:`K^{ij}_{ab} = \langle ij|ab\rangle = (ia|jb)` for the
        pairs ``{(i, j): domain}`` (AO columns ``occupied`` and
        ``functions``), as
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.exchange_blocks`."""
        Co, Cf = np.asarray(occupied), np.asarray(functions)
        o, m = Co.shape[1], Cf.shape[1]
        L = np.empty((self.rank, o, m), dtype=np.result_type(Co, Cf, float))
        Cod = Co.conj().T
        for start, stop in self._blocks():
            L[start:stop] = Cod @ self.matrices(start, stop) @ Cf
        aug = (None if self.augmentation is None
               else self._augmentation_blocks(Co, Cf))
        out = {}
        for (i, j), domain in pairs.items():
            domain = list(domain)
            K = L[:, i, domain].T @ L[:, j, domain]
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

    def pair_exchange(self, occupied, virtual) -> np.ndarray:
        r""":math:`\langle ii|aa\rangle = (ia|ia)` over ``occupied`` x
        ``virtual`` (AO columns), as
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.pair_exchange`."""
        Co, Cv = np.asarray(occupied), np.asarray(virtual)
        out = np.zeros((Co.shape[1], Cv.shape[1]),
                       dtype=np.result_type(Co, Cv, float))
        Cod = Co.conj().T
        for start, stop in self._blocks():
            L = Cod @ self.matrices(start, stop) @ Cv
            out += np.sum(L * L, axis=0)
        if self.augmentation is not None:
            Qt, Wt, U = self._augmentation_blocks(Co, Cv)
            for c, (Q, W) in enumerate(zip(Qt, Wt)):
                out = out + 2.0 * Q * W
                for d, Qd in enumerate(Qt):
                    if U[c, d] != 0:
                        out = out + U[c, d] * Q * Qd
        return out

    def dense(self) -> np.ndarray:
        r"""The whole :math:`\langle ab|cd\rangle` tensor (AO basis,
        augmentation included) -- :math:`M^4`, for tests only."""
        return self.active_tensor(np.eye(self.n_basis))


# --------------------------------------------------------------------------- #
# Generation.
# --------------------------------------------------------------------------- #

class _PairFunctions:
    r"""The real pair densities :math:`\phi_i\phi_j` (:math:`i \le j`) the
    factorization runs over, evaluated on demand; pairs whose sampled supports
    do not meet are left out."""

    def __init__(self, psi, dV, workers=-1):
        self.dV = float(dV)
        self.workers = workers
        chi = np.ascontiguousarray(psi, dtype=np.float64)
        self.chi = chi
        M = chi.shape[0]
        support = (chi != 0).astype(np.float32)
        meets = (support @ support.T) > 0
        iu, ju = np.triu_indices(M)
        keep = meets[iu, ju]
        self.rows, self.cols = iu[keep], ju[keep]
        self.n_screened = int((M * (M + 1)) // 2 - keep.sum())
        #: Every pair function on the grid, ``(P, G)``, when it fits the
        #: budget: then a column is one matrix product against it.
        self.cache = None

    @property
    def size(self) -> int:
        return int(self.rows.size)

    def keep_all(self, budget_bytes: float) -> bool:
        """Form and keep every pair function if ``8 P G`` bytes fit."""
        P, G = self.size, self.chi.shape[1]
        if 8.0 * P * G > budget_bytes:
            return False
        cache = np.empty((P, G))

        def fill(lo, hi, _state):
            self.evaluate(np.arange(lo, hi), cache[lo:hi])

        parallel_blocks(P, DIAGONAL_BLOCK, fill, thread_count(self.workers))
        self.cache = cache
        return True

    def evaluate(self, index, out) -> np.ndarray:
        """``out[k]`` = pair function ``index[k]`` on the grid."""
        if self.cache is not None:
            np.take(self.cache, index, axis=0, out=out)
            return out
        np.multiply(self.chi[self.rows[index]], self.chi[self.cols[index]],
                    out=out)
        return out

    def columns(self, potentials) -> np.ndarray:
        """``(P, k)``: the Coulomb matrix columns of the pair functions whose
        potentials are ``potentials`` ``(k, G)``.  From the kept pair
        functions, blocks of rows on one core each
        (:func:`~mandacaru.integrals.fft.parallel_blocks`); else streamed."""
        if self.cache is not None:
            values = parallel_matmul(self.cache, potentials.T, self.workers)
            values *= self.dV
            return values
        X = contract_pair_potentials(self.chi, self.chi, potentials,
                                     self.workers)
        X *= self.dV
        return X[:, self.rows, self.cols].T


def _diagonal(pairs, solver) -> np.ndarray:
    """The Coulomb self-energy of every pair function (Parseval), blocks of
    :data:`DIAGONAL_BLOCK` pairs shared out over the solver's threads, each
    thread on a workspace of its own with one-thread transforms
    (:func:`~mandacaru.integrals.fft.parallel_blocks`): the pair
    formation, the transforms and the weighted sums all run concurrently,
    and each value is the same whatever the thread count."""
    from .fft import FFTWorkspace, WORKSPACE_BYTES

    P, G = pairs.size, pairs.chi.shape[1]
    diagonal = np.empty(P)
    if not P:
        return diagonal
    shared = solver.workspace
    threads = min(thread_count(getattr(solver, "workers", -1)),
                  -(-P // DIAGONAL_BLOCK))

    def buffers():
        # One thread's share of a workspace's batch, transformed on that
        # thread alone.
        space = (shared if threads <= 1 else
                 FFTWorkspace(shared.shape, shared.padded, 1,
                              batch_bytes=max(WORKSPACE_BYTES // threads,
                                              8 * shared.padded_size)))
        return space, np.empty((DIAGONAL_BLOCK, G))

    def block(lo, hi, state):
        space, work = state
        index = np.arange(lo, hi)
        values = pairs.evaluate(index, work[:hi - lo])
        diagonal[lo:hi] = solver.self_energies(values, workspace=space)

    parallel_blocks(P, DIAGONAL_BLOCK, block, threads, buffers)
    return diagonal


def _subtract_products(columns, left, right, workers) -> None:
    """``columns -= left @ right.T``, the update of the batch's columns by
    the factors so far, in fixed blocks of rows on one core each
    (:func:`~mandacaru.integrals.fft.parallel_matmul`): the same bits
    whatever the thread count."""
    columns -= parallel_matmul(left, np.ascontiguousarray(right.T), workers)


def cholesky_two_body(psi, solver, tolerance: float = DEFAULT_TOLERANCE, *,
                      augmentation=None, batch: int = CHOLESKY_BATCH,
                      span: float = CHOLESKY_SPAN,
                      max_memory_mb: float | None = None,
                      pivots=None) -> CholeskyERI:
    r"""Cholesky factors of the grid two-electron integrals of ``psi``.

    Without ``pivots`` the pivots are chosen as the factorization runs, which
    makes them -- and the truncated integrals -- jump as the geometry moves.
    With ``pivots`` (``(R, 2)`` basis-function pairs in factor order, a
    reference's :attr:`CholeskyERI.pivot_pairs`) the factors are those of
    the given pivots: with :math:`C` the Coulomb columns of the pivot pair
    densities and :math:`A` their pivot rows,
    :math:`(ac|bd) \approx [C A^{-1} C^T]_{ac,bd}`, the Coulomb-metric
    density fitting whose auxiliary functions are the pivot pair densities
    (Aquilante, Lindh and Pedersen 2008).  On the reference's own functions
    that is the reference factorization to round-off; on displaced
    functions it is a smooth function of the nuclear positions, which is
    what a central-difference force needs.  The diagonal is not computed,
    so :attr:`CholeskyERI.max_residual` is ``nan``.

    Parameters
    ----------
    psi : (M, G) float array
        The sampled basis functions: real, as a molecule's real spherical
        harmonics make them.
    solver : PoissonFFTSolver
        The isolated Coulomb solver of the grid; its kernel must be even
        (an orthogonal grid), which makes the pair matrix of real densities
        real and symmetric.
    tolerance : float
        Largest residual Coulomb self-energy (Hartree) of a pair density
        left out (:data:`DEFAULT_TOLERANCE`).
    augmentation : tuple, optional
        The family's ``(Q, W, U)``, kept beside the factors.
    batch, span : int, float
        Candidates per round and the span factor of the batched pivots.
    max_memory_mb : float, optional
        Working budget (the two-body one,
        :func:`~mandacaru.integrals.engine.eri_memory_budget_mb`, by
        default): when every pair function fits in half of it they are kept,
        and each column is one matrix product against them; otherwise the
        columns are streamed (:func:`~mandacaru.integrals.engine.contract_pair_potentials`).
    """
    if not getattr(solver, "even", False) or not hasattr(solver,
                                                         "self_energies"):
        raise ValueError(
            "the Cholesky integrals need the isolated solver of an "
            "orthogonal grid, whose kernel is even: a skewed grid's kernel "
            "is not symmetric, so its pair matrix has no Cholesky factors")
    if np.iscomplexobj(psi):
        raise ValueError(
            "the Cholesky integrals factor the real pair densities of a "
            "molecule's real basis functions; these samples are complex (a "
            "crystal's harmonics or Bloch functions)")
    tolerance = float(tolerance)
    if not tolerance > 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    t_start = perf_counter()
    pairs = _PairFunctions(np.asarray(psi), solver.dV,
                           getattr(solver, "workers", -1))
    P, G = pairs.size, pairs.chi.shape[1]
    budget = (eri_memory_budget_mb() if max_memory_mb is None
              else float(max_memory_mb)) * 2 ** 20
    profile = {"pair functions": P, "screened pairs": pairs.n_screened,
               "kept pair functions": pairs.keep_all(0.5 * budget)}
    if pivots is not None:
        return _fixed_pivot_factors(pairs, solver, tolerance, pivots, batch,
                                    augmentation, profile, t_start)

    # The exact diagonal, by Parseval: forward transforms only.
    t0 = perf_counter()
    diagonal = _diagonal(pairs, solver)
    profile["diagonal seconds"] = perf_counter() - t0
    initial = float(diagonal.max(initial=0.0))

    capacity = max(64, min(P, 8 * pairs.chi.shape[0]))
    L = np.empty((P, min(capacity, max(P, 1))), order="F")
    pivots: list[int] = []
    R = 0
    densities = np.empty((batch, G))
    potentials = np.empty((batch, G))
    seconds = {"columns": 0.0, "updates": 0.0}
    computed = rounds = 0
    while P:
        dmax = float(diagonal.max())
        if dmax <= tolerance:
            break
        floor = max(tolerance, span * dmax)
        order = np.argsort(diagonal)[::-1][:batch]
        candidates = order[diagonal[order] > floor]           # dmax is in
        k = candidates.size
        t0 = perf_counter()
        block = pairs.evaluate(candidates, densities[:k])
        solver.solve_stack(block, out=potentials[:k])
        columns = pairs.columns(potentials[:k])               # (P, k)
        t1 = perf_counter()
        if R:
            _subtract_products(columns, L[:, :R], L[candidates, :R],
                               pairs.workers)
        residual = columns[candidates, np.arange(k)].copy()
        accepted = 0
        for _ in range(k):
            j = int(np.argmax(residual))
            pivot = float(residual[j])
            if pivot <= floor:
                break
            if R == L.shape[1]:
                grown = np.empty((P, min(P, 2 * L.shape[1])), order="F")
                grown[:, :R] = L[:, :R]
                L = grown
            vector = columns[:, j] / np.sqrt(pivot)
            L[:, R] = vector
            R += 1
            accepted += 1
            pivots.append(int(candidates[j]))
            diagonal -= vector * vector
            diagonal[candidates[j]] = 0.0
            columns -= np.outer(vector, vector[candidates])
            residual = columns[candidates, np.arange(k)].copy()
            residual[j] = -np.inf
        if not accepted:
            # The accumulated diagonal over-estimated these residuals (round-
            # off): take the exact ones from their columns.
            diagonal[candidates] = np.maximum(
                columns[candidates, np.arange(k)], 0.0)
        lowest = float(diagonal.min())
        if lowest < -NEGATIVE_RESIDUAL * max(initial, 1.0):
            raise FloatingPointError(
                f"the Cholesky residual went negative ({lowest:.3e} Ha): the "
                "pair matrix is not positive semidefinite to round-off")
        np.maximum(diagonal, 0.0, out=diagonal)
        seconds["columns"] += t1 - t0
        seconds["updates"] += perf_counter() - t1
        computed += k
        rounds += 1
    profile.update({"columns seconds": seconds["columns"],
                    "updates seconds": seconds["updates"],
                    "columns computed": computed, "rounds": rounds,
                    "total seconds": perf_counter() - t_start})
    vectors = np.array(L[:, :R], order="F")
    return CholeskyERI(vectors, pairs.rows, pairs.cols,
                       pairs.chi.shape[0], tolerance,
                       pivots, float(diagonal.max(initial=0.0)),
                       augmentation=augmentation, profile=profile)


def _fixed_pivot_factors(pairs, solver, tolerance, pivots, batch,
                         augmentation, profile, t_start) -> CholeskyERI:
    """The factors of given pivots (:func:`cholesky_two_body`): their Coulomb
    columns ``C``, the Cholesky factor ``L_A`` of the pivot rows, and
    ``C L_A^{-T}``."""
    from scipy.linalg import solve_triangular

    pivots = np.asarray(pivots, dtype=np.int64).reshape(-1, 2)
    M = pairs.chi.shape[0]
    if pivots.size and (pivots.min() < 0 or pivots.max() >= M):
        raise ValueError(f"a pivot names a basis function outside the "
                         f"{M} of this basis")
    lookup = np.full((M, M), -1, dtype=np.int64)
    lookup[pairs.rows, pairs.cols] = np.arange(pairs.size)
    index = lookup[np.minimum(pivots[:, 0], pivots[:, 1]),
                   np.maximum(pivots[:, 0], pivots[:, 1])]
    if np.any(index < 0):
        i, j = pivots[int(np.argmax(index < 0))]
        raise ValueError(
            f"pivot pair ({i}, {j}) has no overlapping support here: the "
            "displaced basis no longer resembles the reference whose pivots "
            "were given")
    R, G = index.size, pairs.chi.shape[1]
    t0 = perf_counter()
    C = np.empty((pairs.size, R), order="F")
    densities = np.empty((min(batch, max(R, 1)), G))
    potentials = np.empty_like(densities)
    for start in range(0, R, batch):
        block = index[start:start + batch]
        k = block.size
        solver.solve_stack(pairs.evaluate(block, densities[:k]),
                           out=potentials[:k])
        C[:, start:start + k] = pairs.columns(potentials[:k])
    t1 = perf_counter()
    A = C[index]
    try:
        L_A = np.linalg.cholesky(0.5 * (A + A.T))
    except np.linalg.LinAlgError:
        raise FloatingPointError(
            "the Coulomb matrix of the given pivots is not positive definite "
            "here: the displaced basis is too far from the reference whose "
            "pivots were given") from None
    vectors = np.asfortranarray(
        solve_triangular(L_A, C.T, lower=True, check_finite=False).T)
    profile.update({"fixed pivots": True, "columns seconds": t1 - t0,
                    "updates seconds": perf_counter() - t1,
                    "columns computed": R, "rounds": 0,
                    "total seconds": perf_counter() - t_start})
    return CholeskyERI(vectors, pairs.rows, pairs.cols, M, tolerance, index,
                       float("nan"), augmentation=augmentation,
                       profile=profile)
