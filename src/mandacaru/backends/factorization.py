# -*- coding: utf-8 -*-
# file: backends/factorization.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Double factorization: measuring :math:`\langle H\rangle` in ``O(M)`` bases.

Qubit-wise commuting grouping (:mod:`mandacaru.backends.measurement`) turns the
:math:`O(M^4)` Pauli strings of a molecular Hamiltonian into :math:`O(M^3)`
measurement circuits.  Measured on Mandacaru's own LiH/PAW-LCAO series that is
23 / 79 / 1,600 / 3,067 bases at 4 / 8 / 20 / 24 qubits -- roughly
:math:`0.2\,M^3` -- and at a few thousand shots per basis it is the *shot*
budget, not the Runtime's memory, that makes a hardware energy unaffordable.

**Double factorization changes the exponent.**  Following Motta *et al.*
(npj Quantum Inf. **7**, 83, 2021) and Huggins *et al.* (npj Quantum Inf.
**7**, 23, 2021), the two-body operator is written as a sum of squares of
one-body operators, each of which is *diagonal* in an orbital basis of its own:

.. math::

    H = E_0 + \sum_{pq} \tilde h_{pq}\, a^\dagger_p a_q
        + \tfrac12 \sum_{\ell} \lambda_\ell
          \Bigl( \sum_{k} g_{\ell k}\, \tilde n^{(\ell)}_k \Bigr)^2 ,

where :math:`\tilde n^{(\ell)}_k` is the occupation of orbital :math:`k` after
the rotation :math:`U_\ell`.  Every occupation in one leaf commutes with every
other, so **one** measurement circuit per leaf -- the basis change followed by a
computational-basis measurement of the whole register -- yields all
:math:`\langle \tilde n_j \tilde n_k\rangle` at once.  The one-body part takes
one further rotation, so the whole energy costs :math:`L + 1` bases with
:math:`L \le M(M+1)/2` exactly and :math:`L = O(M)` after truncation.

Derivation of the form, in the physicists' convention Mandacaru uses
(:meth:`~mandacaru.core.mapping.Fermion.from_integrals`,
:math:`H = \sum h_{pq} a^\dagger_p a_q
+ \tfrac12 \sum \langle pq|rs\rangle a^\dagger_p a^\dagger_q a_s a_r`).  With

.. math::

    a^\dagger_p a^\dagger_q a_s a_r
      = (a^\dagger_p a_r)(a^\dagger_q a_s) - \delta_{qr}\, a^\dagger_p a_s ,

the two-body term becomes :math:`\tfrac12 \sum g_{prqs} E_{pr} E_{qs}` with the
*chemists'* tensor :math:`g_{prqs} = \langle pq|rs\rangle` and
:math:`E_{pr} = a^\dagger_p a_r`, plus a one-body remainder
:math:`-\tfrac12 \sum_{pqs} \langle pq|qs\rangle a^\dagger_p a_s` which is
folded into :math:`\tilde h`.  Reshaping :math:`g` to the symmetric matrix
:math:`W_{(pr),(qs)}` and diagonalizing it gives the leaves
:math:`\lambda_\ell` and :math:`L^{(\ell)}_{pr}`; diagonalizing each
:math:`L^{(\ell)}` gives :math:`U_\ell` and :math:`g_{\ell k}`.

The rotations :math:`U_\ell` are realized on hardware by
:func:`givens_decomposition`: any real orthogonal :math:`M \times M` matrix is a
product of at most :math:`M(M-1)/2` Givens rotations on **adjacent** modes, and
under Jordan-Wigner each of those is a two-qubit number-conserving gate on
neighboring wires (Clements *et al.*, Optica **3**, 1460, 2016; Kivlichan
*et al.*, Phys. Rev. Lett. **120**, 110501, 2018).  After the rotation every
occupation is diagonal, so the leaf's whole contribution is a single
**diagonal** observable in ``I`` and ``Z`` -- one measurement basis per leaf.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Leaves whose weight is below this fraction of the largest are dropped.  The
#: first factorization's eigenvalues fall off fast -- the two-body tensor is
#: numerically low rank -- so this is where ``O(M^2)`` becomes ``O(M)``.
DEFAULT_LEAF_TOLERANCE = 1e-8

#: Orbitals whose coefficient in a leaf is below this are dropped from that
#: leaf's sum.  They cost nothing to measure (the rotation is measured whole)
#: but they inflate the 1-norm the shot estimate is built on.
DEFAULT_COEFFICIENT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class FactorizationLeaf:
    r"""One squared one-body operator: :math:`\tfrac12 \lambda (\sum_k g_k
    \tilde n_k)^2`.

    Attributes
    ----------
    weight : float
        :math:`\lambda_\ell`, the eigenvalue of the first factorization.  It
        carries the sign: the two-body matrix is symmetric but not positive
        definite.
    orbitals : (M, M) ndarray
        :math:`U_\ell`, the orbital rotation that diagonalizes this leaf.  Its
        columns are the rotated orbitals, so ``b = U.conj().T @ a``.
    coefficients : (M,) ndarray
        :math:`g_{\ell k}`, the eigenvalues of :math:`L^{(\ell)}` -- the weight
        of each rotated occupation in this leaf's sum.
    """

    weight: float
    orbitals: np.ndarray
    coefficients: np.ndarray

    @property
    def one_norm(self) -> float:
        r""":math:`\tfrac12 |\lambda| (\sum_k |g_k|)^2` -- this leaf's share."""
        return 0.5 * abs(self.weight) * float(np.abs(self.coefficients).sum()) ** 2


@dataclass(frozen=True)
class DoubleFactorization:
    r"""A Hamiltonian as a one-body term plus squares of one-body terms.

    Built by :func:`double_factorization`.  ``n_modes`` counts **spin
    orbitals**, matching the register the mapping produces.
    """

    n_modes: int
    constant: float
    one_body: np.ndarray
    one_body_orbitals: np.ndarray
    one_body_energies: np.ndarray
    leaves: list[FactorizationLeaf] = field(default_factory=list)
    #: Frobenius norm of the two-body tensor discarded by the leaf cutoff.
    truncation_error: float = 0.0

    @property
    def measurement_bases(self) -> int:
        """``L + 1``: one basis per leaf, plus one for the one-body term."""
        return len(self.leaves) + 1

    @property
    def one_norm(self) -> float:
        r"""1-norm of the factorized form, the shot budget's driver.

        The sum of the one-body eigenvalue magnitudes and each leaf's
        :math:`\tfrac12|\lambda|(\sum_k|g_k|)^2`.  It is the quantity a
        fault-tolerant cost estimate and a shot count both scale with, and it is
        *not* the same as the Pauli 1-norm of the same Hamiltonian -- which is
        the point of factorizing.
        """
        return float(np.abs(self.one_body_energies).sum()
                     + sum(leaf.one_norm for leaf in self.leaves))

    # -- exact reconstruction, for validation ----------------------------- #

    def energy(self, one_rdm: np.ndarray, two_rdm: np.ndarray) -> float:
        r"""``<H>`` from the RDMs, evaluated through the factorized form.

        Contracts the one-body part with :math:`\gamma` and each leaf with the
        two-body RDM rotated into that leaf's orbitals, which is exactly what
        the corresponding measurement circuit would return.  Used to check the
        factorization against the state vector before any shots are spent.

        ``one_rdm`` is :math:`\gamma_{pq} = \langle a^\dagger_p a_q\rangle` and
        ``two_rdm`` :math:`\Gamma_{pqrs} = \langle a^\dagger_p a^\dagger_q a_s
        a_r\rangle`, the convention of :mod:`mandacaru.algorithms.rdm`.
        """
        from .._native import factorized_energy_rdm as _native_energy

        return _native_energy(
            float(self.constant),
            np.ascontiguousarray(self.one_body, dtype=np.complex128),
            [float(leaf.weight) for leaf in self.leaves],
            [np.ascontiguousarray(leaf.orbitals, dtype=np.float64) for leaf in self.leaves],
            [np.ascontiguousarray(leaf.coefficients, dtype=np.float64) for leaf in self.leaves],
            np.ascontiguousarray(one_rdm, dtype=np.float64),
            np.ascontiguousarray(two_rdm, dtype=np.float64),
        )

    def to_fermion(self):
        """Rebuild the second-quantized operator this factorization represents.

        The definitive check: mapped to qubits it must equal the Hamiltonian it
        was built from.  ``O(L M^4)`` terms, so it is for small registers.
        """
        from ..core.mapping import Fermion

        operator = Fermion.from_integrals(self.one_body)
        if self.constant:
            operator = operator + Fermion({(): complex(self.constant)},
                                          n_modes=self.n_modes)
        for leaf in self.leaves:
            matrix = leaf.orbitals @ np.diag(leaf.coefficients) \
                @ leaf.orbitals.conj().T
            single = Fermion.from_integrals(matrix)
            operator = operator + (0.5 * leaf.weight) * (single * single)
        return operator


def double_factorization(h_pq, g_pqrs, tol: float = DEFAULT_LEAF_TOLERANCE,
                         coefficient_tol: float = DEFAULT_COEFFICIENT_TOLERANCE,
                         max_leaves: int | None = None,
                         constant: float = 0.0) -> DoubleFactorization:
    r"""Factorize ``H`` into a one-body term plus squares of one-body terms.

    Parameters
    ----------
    h_pq : (M, M) array_like
        One-body integrals over ``M`` **spin orbitals**.
    g_pqrs : (M, M, M, M) array_like
        Two-electron integrals in physicists' notation
        :math:`\langle pq|rs\rangle`, the same convention as
        :meth:`~mandacaru.core.mapping.Fermion.from_integrals`.
    tol : float
        Drop leaves whose ``|lambda|`` is below ``tol`` times the largest.
    coefficient_tol : float
        Zero the orbital coefficients below this within each leaf.
    max_leaves : int, optional
        Keep at most this many leaves, largest ``|lambda|`` first.  The honest
        knob for "``O(M)`` bases": the discarded weight is reported on
        :attr:`DoubleFactorization.truncation_error`.
    constant : float
        Scalar added to the energy (a frozen core, a nuclear repulsion).
    """
    h_arr = np.ascontiguousarray(h_pq, dtype=np.float64)
    g_arr = np.ascontiguousarray(g_pqrs, dtype=np.float64)
    m = int(h_arr.shape[0])
    if g_arr.shape != (m, m, m, m):
        raise ValueError(f"two-body tensor {g_arr.shape} does not match a "
                         f"{m}-mode one-body tensor")

    from .._native import double_factorization as _native_df

    n_modes, c, ob, ob_rot, ob_e, leaves_raw, trunc = _native_df(
        h_arr, g_arr, tol=float(tol), coefficient_tol=float(coefficient_tol),
        max_leaves=max_leaves, constant=float(constant)
    )
    leaves = [FactorizationLeaf(weight=float(w), orbitals=u, coefficients=val)
              for w, u, val in leaves_raw]
    return DoubleFactorization(
        n_modes=n_modes, constant=c, one_body=ob,
        one_body_orbitals=ob_rot, one_body_energies=ob_e,
        leaves=leaves, truncation_error=trunc)


def _pair_spectrum(eri=None, factors=None, coupling=None):
    r"""``(weights, vectors)`` of the spatial pair matrix
    :math:`W_{(pr),(qs)} = \langle pq|rs\rangle`, ``vectors[:, l]`` over
    the ``M^2`` pairs.

    From the tensor ``eri`` by an ``eigh`` of ``W``; from ``factors``
    ``(k, M, M)`` with :math:`W = \sum_{ij} K_{ij}\,\mathrm{vec}(F_i)
    \mathrm{vec}(F_j)^T` (``coupling`` :math:`K`, the identity by default:
    Cholesky factors) by the ``k x k`` problem :math:`R K R^T` of the QR
    decomposition :math:`F^T = QR`, so ``W`` is never formed when ``k`` is
    below ``M^2``.  Both give ``W``'s nonzero spectrum to round-off.
    """
    from .._native import pair_spectrum as _native_pair_spectrum

    if factors is not None:
        f_arr = np.ascontiguousarray(factors, dtype=np.float64)
        k_arr = (np.ascontiguousarray(coupling, dtype=np.float64)
                 if coupling is not None else None)
        return _native_pair_spectrum(factors=f_arr, coupling=k_arr)
    g_arr = np.ascontiguousarray(eri, dtype=np.float64)
    return _native_pair_spectrum(eri=g_arr)


def spatial_double_factorization(h, eri=None, *, factors=None, coupling=None,
                                 tol: float = DEFAULT_LEAF_TOLERANCE,
                                 coefficient_tol: float =
                                 DEFAULT_COEFFICIENT_TOLERANCE,
                                 max_leaves: int | None = None,
                                 constant: float = 0.0) -> DoubleFactorization:
    r""":func:`double_factorization` of a spin-free Hamiltonian, from its
    spatial integrals, without the spin-orbital ``(2M)^4`` tensor.

    ``h`` is the ``(M, M)`` spatial one-body matrix; the two-body part is
    either the spatial tensor ``eri`` (:math:`\langle pq|rs\rangle`,
    physicists' notation) or pair ``factors`` ``(k, M, M)``
    with the
    symmetric ``coupling`` ``(k, k)`` -- :math:`(pr|qs) = \sum_{ij}
    K_{ij}(F_i)_{pr}(F_j)_{qs}`, Cholesky factors with :math:`K = 1`, plus
    a structured augmentation's channels with theirs.

    The spin-orbital pair matrix of a spin-free tensor is the spatial one
    on every pair of spin-diagonal pairs, so its nonzero spectrum is the
    spatial one doubled: a spatial leaf :math:`(\lambda, v)` is the
    spin-orbital leaf of weight :math:`2\lambda` and matrix
    :math:`(v \oplus v)/\sqrt2`, rotated by :math:`U \oplus U`.  The
    result is that of :func:`double_factorization` on
    ``spin_block_integrals(h, eri)`` -- the same operator, leaves, weights,
    one-norm and measurement count -- except that every rotation is block
    diagonal in spin (the spin-orbital ``eigh`` may mix the two spins of a
    degenerate pair).  The leaves cost one ``M^2 x M^2`` eigenproblem (or,
    from ``k < M^2`` factors, a ``k x k`` one) instead of ``(2M)^2 x
    (2M)^2``.
    """
    from .._native import spatial_double_factorization as _native_sdf

    h_arr = np.ascontiguousarray(h, dtype=np.float64)
    m = int(h_arr.shape[0])
    if factors is not None:
        f_arr = np.ascontiguousarray(factors, dtype=np.float64)
        if f_arr.ndim != 3 or f_arr.shape[1:] != (m, m):
            raise ValueError(f"pair factors {f_arr.shape} do not match a {m}-orbital "
                             f"one-body matrix")
        k_arr = (np.ascontiguousarray(coupling, dtype=np.float64)
                 if coupling is not None else None)
        n_modes, c, ob, ob_rot, ob_e, leaves_raw, trunc = _native_sdf(
            h_arr, factors=f_arr, coupling=k_arr,
            tol=float(tol), coefficient_tol=float(coefficient_tol),
            max_leaves=max_leaves, constant=float(constant)
        )
    else:
        g_arr = np.ascontiguousarray(eri, dtype=np.float64)
        if g_arr.shape != (m, m, m, m):
            raise ValueError(f"two-body tensor {g_arr.shape} does not match a "
                             f"{m}-orbital one-body matrix")
        n_modes, c, ob, ob_rot, ob_e, leaves_raw, trunc = _native_sdf(
            h_arr, eri=g_arr,
            tol=float(tol), coefficient_tol=float(coefficient_tol),
            max_leaves=max_leaves, constant=float(constant)
        )
    leaves = [FactorizationLeaf(weight=float(w), orbitals=u, coefficients=val)
              for w, u, val in leaves_raw]
    return DoubleFactorization(
        n_modes=n_modes, constant=c, one_body=ob,
        one_body_orbitals=ob_rot, one_body_energies=ob_e,
        leaves=leaves, truncation_error=trunc)


def factorized_measurement_bases(factorization: DoubleFactorization) -> int:
    """How many measurement circuits the factorized energy needs.

    ``L + 1``.  Compare against
    :func:`~mandacaru.backends.measurement.qubit_wise_commuting_groups` on the
    same Hamiltonian.  A circuit run measures the energy in exactly these bases
    with ``Mandacaru(measurement_scheme="double-factorized")``: the Givens
    networks, the diagonal observables and the assembly of the energy are Rust
    (``quantum::factorized_measurement``), planned from the molecular model the
    calculation built, not from this object.
    """
    return factorization.measurement_bases


# --------------------------------------------------------------------------- #
# Realizing the rotations: Givens networks on adjacent modes.
# --------------------------------------------------------------------------- #

#: Rotation angles below this are dropped -- the gate would be the identity.
GIVENS_TOLERANCE = 1e-12


def givens_decomposition(unitary, tol: float = GIVENS_TOLERANCE):
    r"""Factor a real orthogonal matrix into **adjacent** Givens rotations.

    Returns ``(rotations, signs)``, where ``rotations`` is a list of
    ``(p, theta)`` -- a rotation mixing modes ``p`` and ``p + 1`` by ``theta``,
    in elimination order -- and ``signs`` the diagonal :math:`\pm 1` left over.
    Together they reproduce the input:

    .. math::

        U = G_{1}(-\theta_1)\, G_{2}(-\theta_2) \cdots G_{K}(-\theta_K)\, D ,

    with :math:`G_p(\theta)` the rotation
    :math:`\begin{pmatrix}\cos\theta & \sin\theta\\ -\sin\theta &
    \cos\theta\end{pmatrix}` on rows :math:`(p, p+1)`.

    The elimination is the ordinary Givens QR, restricted to **neighboring**
    rows so that every rotation is a nearest-neighbor gate under
    Jordan-Wigner: column by column, each sub-diagonal entry is zeroed against
    the row above it.  What is left is orthogonal and upper triangular, hence
    diagonal with entries :math:`\pm 1`.

    The signs are reported but are **not needed** to measure occupations:
    :math:`D` sends :math:`b_k \to \pm b_k`, which leaves
    :math:`\tilde n_k = b^\dagger_k b_k` unchanged.  They are returned so the
    decomposition can be checked against its input.
    """
    from .._native import givens_decomposition as _native_givens_decomposition

    return _native_givens_decomposition(
        np.asarray(unitary, dtype=np.float64),
        tol=float(tol),
    )
