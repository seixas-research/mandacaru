# -*- coding: utf-8 -*-
# file: integrals/packed.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The real two-electron tensor stored once per symmetry-distinct entry.

Over real functions the integrals :math:`\langle ab|cd\rangle = (ac|bd)`
have the eightfold symmetry

.. math::

    (ac|bd) = (ca|bd) = (ac|db) = (ca|db) = (bd|ac) = (db|ac) = (bd|ca)
            = (db|ca),

so the tensor is a symmetric matrix :math:`W_{uv} = (ac|bd)` over the
:math:`P = M(M+1)/2` unordered pairs :math:`u = \{a, c\}`, :math:`v = \{b,
d\}`, and only its lower triangle -- :math:`P(P+1)/2 \approx M^4/8` numbers --
needs to be kept: eight bytes an entry make it :math:`M^4` bytes against the
:math:`8 M^4` of the full real array.

:class:`PackedERI` holds that triangle and gives every operation the
Hamiltonian pipeline needs without unpacking it:

* an orbital change (Loewdin, molecular orbitals, a semicanonical or natural
  rotation): :meth:`~PackedERI.transform`, two half-transforms through one
  ``P' x P`` intermediate;
* mixed-orbital blocks (MP2's :math:`\langle ij|ab\rangle`, an open shell's
  alpha-beta block): :meth:`~PackedERI.block`;
* Coulomb and exchange of a density (the SCF):
  :meth:`~PackedERI.coulomb`, :meth:`~PackedERI.exchange`;
* the slabs ``eri[p]`` the fermionic operator is written from
  (:meth:`~PackedERI.slab`), a subset of orbitals
  (:meth:`~PackedERI.subset`), a contraction with a two-body density matrix
  (:meth:`~PackedERI.contract`) and the structured augmentation added in
  place (:meth:`~PackedERI.add_augmentation`).

:meth:`~PackedERI.dense` writes the full array out; it is for the callers
that genuinely need it (a spinor or spin-orbital tensor, the tests), and
the array it returns is theirs to drop.

Pairs are numbered in the lower-triangle order of an ``(M, M)`` matrix,
:math:`u(a, c) = a(a+1)/2 + c` for :math:`c \le a`
(:func:`numpy.tril_indices`), and entry :math:`(u, v)`, :math:`v \le u`, is
stored at :math:`u(u+1)/2 + v` -- the upper-triangle column-major layout of
the BLAS packed routines.
"""

from __future__ import annotations

import numpy as np

from .. import _native

__all__ = ["PackedERI", "pair_index", "is_packed", "dense_tensor"]

#: Rows of the pair matrix gathered at a time by the blocked operations.
ROW_BLOCK = 256


def pair_index(n: int) -> np.ndarray:
    """``(n, n)``: the pair number :math:`u(a, c)` of every ordered pair,
    the same for ``(a, c)`` and ``(c, a)``."""
    return _native.packed_pair_indices(int(n))


def _triangle(n: int) -> int:
    return int(n) * (int(n) + 1) // 2


def is_packed(eri) -> bool:
    """Whether ``eri`` is a :class:`PackedERI`."""
    return isinstance(eri, PackedERI)


def dense_tensor(eri) -> np.ndarray:
    """``eri`` as a full ``(M, M, M, M)`` array: written out of a
    :class:`PackedERI` (a transient ``M^4`` array), unchanged otherwise."""
    return eri.dense() if is_packed(eri) else np.asarray(eri)


class PackedERI:
    r"""A real two-electron tensor :math:`\langle ab|cd\rangle` kept as the
    lower triangle of its pair matrix (see the module docstring).

    Parameters
    ----------
    data : (P(P+1)/2,) float array
        The triangle, row by row.
    n : int
        The number of orbitals.
    """

    def __init__(self, data, n: int):
        self.n = int(n)
        data = np.asarray(data, dtype=np.float64)
        expected = _triangle(self.n_pairs)
        if data.shape != (expected,):
            raise ValueError(f"a packed tensor of {self.n} orbitals holds "
                             f"{expected} numbers, got {data.shape}")
        self.data = data
        rows, cols = np.tril_indices(self.n)
        self._rows, self._cols = rows, cols
        self._pairs = pair_index(self.n)

    # -- size and layout ---------------------------------------------------- #

    @property
    def n_pairs(self) -> int:
        return _triangle(self.n)

    @property
    def shape(self) -> tuple:
        return (self.n,) * 4

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def real(self) -> "PackedERI":
        """The tensor itself: it is real (so :func:`numpy.real` of it is
        it, as for a real array)."""
        return self

    def __repr__(self) -> str:
        return f"PackedERI(n={self.n}, {self.nbytes / 2 ** 20:.1f} MiB)"

    def offsets(self) -> np.ndarray:
        """Where each row of the triangle starts."""
        return _native.packed_offsets(self.n_pairs)

    def pack_pairs(self, matrices) -> np.ndarray:
        """``(k, P)``: the lower triangles of ``(k, n, n)`` matrices."""
        return np.asarray(matrices)[..., self._rows, self._cols]

    def unpack_pairs(self, vectors, out=None) -> np.ndarray:
        """``(k, n, n)`` symmetric matrices from ``(k, P)`` pair vectors."""
        vectors = np.asarray(vectors)
        k = vectors.shape[0]
        if out is None:
            out = np.empty((k, self.n, self.n), dtype=vectors.dtype)
        out[:, self._rows, self._cols] = vectors
        out[:, self._cols, self._rows] = vectors
        return out

    def rows(self, index) -> np.ndarray:
        """``(k, P)``: the full rows ``index`` of the pair matrix."""
        idx = np.ascontiguousarray(index, dtype=np.int64)
        return _native.packed_rows(self.n_pairs, self.data, idx)

    def _set_row(self, u: int, values) -> None:
        """Write the triangle's row ``u`` (``values[:u + 1]``)."""
        start = u * (u + 1) // 2
        self.data[start:start + u + 1] = values[:u + 1]

    # -- construction ------------------------------------------------------- #

    @classmethod
    def zeros(cls, n: int) -> "PackedERI":
        return cls(np.zeros(_triangle(_triangle(n))), n)

    @classmethod
    def from_dense(cls, eri) -> "PackedERI":
        """The lower triangle of a full real tensor's pair matrix (its
        entries ``(ac|bd)`` with ``{a, c} >= {b, d}``; the other halves are
        assumed equal to them, as the integrals are)."""
        eri = np.asarray(eri)
        if np.iscomplexobj(eri):
            raise ValueError("the packed tensor is real: a complex tensor has "
                             "no eightfold symmetry")
        n = eri.shape[0]
        data = _native.packed_from_dense(np.ascontiguousarray(eri, dtype=np.float64))
        return cls(data, n)

    # -- writing it out ------------------------------------------------------ #

    def pair_matrix(self) -> np.ndarray:
        """``(P, P)``: the symmetric pair matrix :math:`W`."""
        return _native.packed_pair_matrix(self.n, self.data)

    def dense(self) -> np.ndarray:
        r"""The full ``(n, n, n, n)`` array :math:`\langle ab|cd\rangle`
        (a new ``M^4`` array; the store is unchanged)."""
        return _native.packed_dense(self.n, self.data)

    # -- in place ------------------------------------------------------------ #

    def scale(self, factor: float) -> "PackedERI":
        """Multiply in place; returns itself."""
        self.data *= float(factor)
        return self

    def add_augmentation(self, augmentation, scale: float = 1.0
                         ) -> "PackedERI":
        r"""Add ``scale`` times a
        :class:`~mandacaru.core.augmentation.PairAugmentation`
        (:math:`\sum_{ij} K_{ij} F^{(i)}_{pr} F^{(j)}_{qs}`) in place, a
        block of rows at a time; its factors must be symmetric, as a real
        basis makes them.  Returns itself."""
        F = np.asarray(augmentation.factors)
        if np.iscomplexobj(F) or F.shape[1:] != (self.n, self.n):
            raise ValueError("the augmentation must be real and over the "
                             f"tensor's {self.n} orbitals")
        if np.abs(F - F.transpose(0, 2, 1)).max(initial=0.0) > 1e-12 * max(
                np.abs(F).max(initial=0.0), 1.0):
            raise ValueError("a packed tensor takes symmetric augmentation "
                             "factors only")
        f_contig = np.ascontiguousarray(F, dtype=np.float64)
        coupling = np.ascontiguousarray(augmentation.coupling, dtype=np.float64)
        _native.packed_add_augmentation(self.n, self.data, f_contig, coupling, float(scale))
        return self

    # -- orbital changes ----------------------------------------------------- #

    def transform(self, C) -> "PackedERI":
        r"""The tensor over the orbitals :math:`\phi'_p = \sum_a C_{ap}
        \phi_a` (real ``C``, ``(n, m)``), still packed: the pair matrix
        :math:`T W T^T` by two half-transforms, through one ``P' x P``
        intermediate (about ``m^2 n^2 / 4`` numbers) and no ``M^4``
        array."""
        C = self._real(C)
        m = C.shape[1]
        data = _native.packed_transform(self.n, self.data, C)
        return PackedERI(data, m)

    def block(self, C1, C2, C3, C4) -> np.ndarray:
        r""":math:`\langle ij|kl\rangle` with :math:`i` over the columns of
        ``C1``, :math:`j` of ``C2``, :math:`k` of ``C3``, :math:`l` of
        ``C4`` (real), as an ``(n1, n2, n3, n4)`` array: electron 1 on
        ``(C1, C3)``, electron 2 on ``(C2, C4)``, two half-transforms."""
        C1, C2, C3, C4 = (self._real(C) for C in (C1, C2, C3, C4))
        return _native.packed_block(self.n, self.data, C1, C2, C3, C4)

    def subset(self, index) -> "PackedERI":
        """The tensor of the orbitals ``index`` alone (a gather, exact)."""
        idx = np.ascontiguousarray(index, dtype=np.int64)
        m = idx.size
        data = _native.packed_subset(self.n, self.data, idx)
        return PackedERI(data, m)

    @staticmethod
    def _real(C) -> np.ndarray:
        C = np.asarray(C)
        if np.iscomplexobj(C):
            if np.abs(C.imag).max(initial=0.0) > 0.0:
                raise ValueError("a packed (real) tensor transforms by real "
                                 "orbitals only")
            C = C.real
        return np.ascontiguousarray(C, dtype=np.float64)

    # -- contractions --------------------------------------------------------- #

    def coulomb(self, D) -> np.ndarray:
        r""":math:`J_{pq} = \sum_{rs} D_{sr}\langle pr|qs\rangle =
        \sum_{rs} (pq|rs) D_{sr}`: one packed symmetric matrix-vector
        product of the pair matrix (``dspmv``)."""
        D = np.asarray(D)
        if np.iscomplexobj(D):
            D = np.ascontiguousarray(D, dtype=np.complex128)
        else:
            D = np.ascontiguousarray(D, dtype=np.float64)
        return _native.packed_coulomb(self.n, self.data, D)

    def exchange(self, D) -> np.ndarray:
        r""":math:`K_{pq} = \sum_{rs} D_{sr}\langle pr|sq\rangle =
        \sum_{rs} (ps|rq) D_{sr}`: each row :math:`u = \{a, c\}` of the
        pair matrix, unpacked to :math:`R_{rq} = (ac|rq)`, adds
        :math:`D_{c\cdot} R` to row ``a`` of ``K`` and :math:`D_{a\cdot} R`
        to row ``c``."""
        D = np.asarray(D)
        if np.iscomplexobj(D):
            D = np.ascontiguousarray(D, dtype=np.complex128)
        else:
            D = np.ascontiguousarray(D, dtype=np.float64)
        return _native.packed_exchange(self.n, self.data, D)

    def slab(self, p: int) -> np.ndarray:
        r"""``eri[p]``: :math:`\langle pq|rs\rangle` over ``(q, r, s)``,
        ``(n, n, n)``."""
        return _native.packed_slab(self.n, self.data, int(p))

    def contract(self, G) -> complex | float:
        r""":math:`\sum_{pqrs} G_{pqrs}\langle pq|rs\rangle` for a full
        ``(n, n, n, n)`` array ``G`` (a two-body density matrix): ``G`` is
        summed over each pair's orderings and dotted with the triangle."""
        G = np.asarray(G)
        if np.iscomplexobj(G):
            G = np.ascontiguousarray(G, dtype=np.complex128)
        else:
            G = np.ascontiguousarray(G, dtype=np.float64)
        return _native.packed_contract(self.n, self.data, G)
