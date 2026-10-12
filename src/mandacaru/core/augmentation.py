# -*- coding: utf-8 -*-
# file: core/augmentation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""A two-body correction kept as a short sum of outer products.

The PAW-LCAO compensation charges add to the grid tensor

.. math::

    \Delta g_{pqrs} = \sum_{ij} K_{ij}\, F^{(i)}_{pr}\, F^{(j)}_{qs},

with a handful of ``(M, M)`` factor matrices :math:`F^{(i)}` -- the moments
:math:`Q^{A,LM}` and the compensation potentials :math:`W^{A,LM}` -- and a
small coupling matrix :math:`K` (:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.two_body_augmentation`).
Written out it is an ``M^4`` tensor; kept in this form it is ``k M^2``
numbers, and every operation the Hamiltonian needs works on the factors:

* an orbital change (Loewdin, molecular orbitals, an active block) acts on each
  factor alone, :math:`F \to B^\dagger F B` (:meth:`PairAugmentation.transformed`);
* adding it to a dense tensor is done in place, one ``M^3`` slab at a time
  (:meth:`PairAugmentation.add_to`), so no second ``M^4`` array exists.
"""

from __future__ import annotations

import numpy as np


class PairAugmentation:
    r""":math:`\Delta g_{pqrs} = \sum_{ij} K_{ij} F^{(i)}_{pr} F^{(j)}_{qs}`.

    Physicists' layout, as :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.two_body`:
    ``(p, r)`` belong to electron 1 and ``(q, s)`` to electron 2.

    Parameters
    ----------
    factors : array_like
        ``(k, n, n)``: the factor matrices :math:`F^{(i)}`.
    coupling : array_like
        ``(k, k)``: the coupling :math:`K`.
    """

    def __init__(self, factors, coupling):
        # Real for a molecule's real harmonics, complex otherwise.
        dtype = np.result_type(np.asarray(factors), np.asarray(coupling),
                               np.float64)
        factors = np.asarray(factors, dtype=dtype)
        coupling = np.asarray(coupling, dtype=dtype)
        if factors.ndim != 3 or factors.shape[1] != factors.shape[2]:
            raise ValueError(f"factors must be (k, n, n), got {factors.shape}")
        k = factors.shape[0]
        if coupling.shape != (k, k):
            raise ValueError(f"coupling must be ({k}, {k}), got "
                             f"{coupling.shape}")
        self.factors = factors
        self.coupling = coupling

    @property
    def n_orbitals(self) -> int:
        """Orbitals each factor spans."""
        return int(self.factors.shape[1])

    @property
    def rank(self) -> int:
        """Number of factor matrices."""
        return int(self.factors.shape[0])

    @property
    def nbytes(self) -> int:
        """Bytes held: ``k n^2`` factor entries and the ``k^2`` coupling."""
        return int(self.factors.nbytes + self.coupling.nbytes)

    def transformed(self, orbitals) -> PairAugmentation:
        r"""The correction over new orbitals :math:`\phi'_p = \sum_a B_{ap}\phi_a`.

        Bra indices take :math:`B^*` and ket indices :math:`B`, the pattern
        of every two-body transform here, so each factor becomes
        :math:`B^\dagger F B`.  ``orbitals`` may be rectangular ``(n, m)``:
        an active block.
        """
        B = np.asarray(orbitals)
        if B.ndim != 2 or B.shape[0] != self.n_orbitals:
            raise ValueError(f"orbitals must be ({self.n_orbitals}, m), got "
                             f"{B.shape}")
        factors = np.einsum("ap,iac,cr->ipr", B.conj(), self.factors, B,
                            optimize=True)
        return PairAugmentation(factors, self.coupling)

    def add_to(self, eri: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Add ``scale`` times the correction to ``eri`` in place; returns it.

        Blocked over the first index, so the only temporary is one
        ``(n, n, n)`` slab.
        """
        n = self.n_orbitals
        if eri.shape != (n, n, n, n):
            raise ValueError(f"eri must be ({n},) * 4, got {eri.shape}")
        k = self.rank
        # right[i, (q, s)] = sum_j K_ij F^(j)_qs
        right = (scale * self.coupling) @ self.factors.reshape(k, n * n)
        for p in range(n):
            # slab[r, q, s] = sum_i F^(i)_pr right[i, (q, s)]
            slab = (self.factors[:, p, :].T @ right).reshape(n, n, n)
            eri[p] += slab.transpose(1, 0, 2)
        return eri

    def dense(self) -> np.ndarray:
        """The ``(n, n, n, n)`` tensor (tests and the nuclear gradient)."""
        n = self.n_orbitals
        return self.add_to(np.zeros((n, n, n, n), dtype=self.factors.dtype))

    def __repr__(self) -> str:
        return (f"PairAugmentation(rank={self.rank}, "
                f"n_orbitals={self.n_orbitals})")
