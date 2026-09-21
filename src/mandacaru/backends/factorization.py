# -*- coding: utf-8 -*-
# file: backends/factorization.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Double factorization: measuring :math:`\langle H\rangle` in ``O(M)`` bases.

Qubit-wise commuting grouping (:mod:`mandacaru.backends.measurement`) turns the
:math:`O(M^4)` Pauli strings of a molecular Hamiltonian into :math:`O(M^3)`
measurement circuits.  Measured on Mandacaru's own LiH/PAW series that is
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
neighbouring wires (Clements *et al.*, Optica **3**, 1460, 2016; Kivlichan
*et al.*, Phys. Rev. Lett. **120**, 110501, 2018).  After the rotation every
occupation is diagonal, so the leaf's whole contribution is a single
**diagonal** observable in ``I`` and ``Z`` -- one PUB per leaf, one job for the
energy (:func:`factorized_energy`).
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

    def measurement_problems(self, n_qubits, occupied, generators, thetas,
                             mapping: str = "jordan_wigner") -> list[tuple]:
        """See :func:`measurement_problems`."""
        return measurement_problems(self, n_qubits, occupied, generators,
                                    thetas, mapping=mapping)

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
        gamma = np.asarray(one_rdm)
        gamma2 = np.asarray(two_rdm)
        energy = self.constant + float(np.real(np.einsum(
            "pq,pq->", self.one_body, gamma)))
        # <E_pr E_qs> = Gamma_pqrs + delta_rq gamma_ps, with
        # Gamma_pqrs = <a+_p a+_q a_s a_r>: the same reordering that produced
        # the chemists' tensor above, read backwards.
        pair = np.einsum("pqrs->prqs", gamma2) + np.einsum(
            "rq,ps->prqs", np.eye(self.n_modes), gamma)
        for leaf in self.leaves:
            matrix = leaf.orbitals @ np.diag(leaf.coefficients) \
                @ leaf.orbitals.conj().T
            energy += 0.5 * leaf.weight * float(np.real(np.einsum(
                "pr,qs,prqs->", matrix, matrix, pair)))
        return energy

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
    h_pq = np.asarray(h_pq)
    g = np.asarray(g_pqrs)
    m = int(h_pq.shape[0])
    if g.shape != (m, m, m, m):
        raise ValueError(f"two-body tensor {g.shape} does not match a "
                         f"{m}-mode one-body tensor")

    # <pq|rs> a+_p a+_q a_s a_r = <pq|rs> [ E_pr E_qs - delta_qr E_ps ].
    chemist = np.einsum("pqrs->prqs", g)
    one_body = np.asarray(h_pq, dtype=complex) \
        - 0.5 * np.einsum("pqqs->ps", g)

    # First factorization: the (pr),(qs) matrix is symmetric, so an eigenvalue
    # decomposition gives real leaves directly.  `eigh` needs it Hermitian;
    # symmetrizing costs nothing and absorbs the integrals' own round-off.
    w = chemist.reshape(m * m, m * m)
    w = 0.5 * (w + w.conj().T)
    weights, vectors = np.linalg.eigh(w)

    order = np.argsort(-np.abs(weights))
    weights, vectors = weights[order], vectors[:, order]
    largest = float(np.abs(weights[0])) if len(weights) else 0.0
    keep = np.abs(weights) > tol * max(largest, 1.0)
    if max_leaves is not None:
        cut = np.zeros_like(keep)
        cut[:int(max_leaves)] = True
        keep &= cut
    dropped = weights[~keep]
    truncation = float(np.sqrt(np.sum(dropped ** 2))) if dropped.size else 0.0

    leaves: list[FactorizationLeaf] = []
    for index in np.flatnonzero(keep):
        matrix = vectors[:, index].reshape(m, m)
        matrix = 0.5 * (matrix + matrix.conj().T)
        values, orbitals = np.linalg.eigh(matrix)
        values = np.where(np.abs(values) > coefficient_tol, values, 0.0)
        if not np.any(values):
            continue
        leaves.append(FactorizationLeaf(weight=float(weights[index]),
                                        orbitals=orbitals,
                                        coefficients=values))

    energies, rotation = np.linalg.eigh(0.5 * (one_body + one_body.conj().T))
    return DoubleFactorization(
        n_modes=m, constant=float(constant), one_body=one_body,
        one_body_orbitals=rotation, one_body_energies=energies,
        leaves=leaves, truncation_error=truncation)


def factorized_measurement_bases(factorization: DoubleFactorization) -> int:
    """How many measurement circuits the factorized energy needs.

    ``L + 1``.  Compare against
    :func:`~mandacaru.backends.measurement.qubit_wise_commuting_groups` on the
    same Hamiltonian: this is the number the pre-flight plan reports as the
    alternative, and the reason to implement the basis-rotation circuits.

    .. note::
       Mandacaru computes the factorization and can **evaluate** the energy
       through it exactly (:meth:`DoubleFactorization.energy`), but it does not
       yet emit the Givens-rotation circuits that realize ``U_l`` on a
       processor, so a shot-based run still goes through the qubit-wise
       commuting path.  What this function gives today is the honest count of
       what that path would cost.
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

    The elimination is the ordinary Givens QR, restricted to **neighbouring**
    rows so that every rotation is a nearest-neighbour gate under
    Jordan-Wigner: column by column, each sub-diagonal entry is zeroed against
    the row above it.  What is left is orthogonal and upper triangular, hence
    diagonal with entries :math:`\pm 1`.

    The signs are reported but are **not needed** to measure occupations:
    :math:`D` sends :math:`b_k \to \pm b_k`, which leaves
    :math:`\tilde n_k = b^\dagger_k b_k` unchanged.  They are returned so the
    decomposition can be checked against its input.
    """
    u = np.array(unitary, dtype=float)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"expected a square matrix, got {u.shape}")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), atol=1e-8):
        raise ValueError("the Givens decomposition needs a real orthogonal "
                         "matrix; this one is not")
    m = u.shape[0]
    rotations: list[tuple[int, float]] = []
    for column in range(m - 1):
        for row in range(m - 1, column, -1):
            below, above = u[row, column], u[row - 1, column]
            if abs(below) <= tol:
                continue
            theta = float(np.arctan2(below, above))
            cos, sin = np.cos(theta), np.sin(theta)
            u[row - 1], u[row] = (cos * u[row - 1] + sin * u[row],
                                  -sin * u[row - 1] + cos * u[row])
            rotations.append((row - 1, theta))
    return rotations, np.diag(u).copy()


def givens_generator(n_modes: int, p: int, mapping: str = "jordan_wigner"):
    r"""The anti-Hermitian generator of a Givens rotation on modes ``p``, ``p+1``.

    :math:`A = a^\dagger_p a_{p+1} - a^\dagger_{p+1} a_p`, so that
    :math:`e^{\theta A}` rotates the pair by :math:`\theta`.  Mapped to qubits
    it is two Pauli strings that **commute** (``XY`` and ``YX`` on the two
    wires, both squaring to ``ZZ``), which is what lets a circuit provider emit
    it exactly as a basis change, one CNOT ladder and one ``Rz``.
    """
    from ..core.mapping import Fermion

    operator = Fermion({((p, True), (p + 1, False)): 1.0 + 0j,
                        ((p + 1, True), (p, False)): -1.0 + 0j},
                       n_modes=int(n_modes))
    return operator.map_to_qubits(mapping)


def basis_rotation(n_modes: int, unitary, mapping: str = "jordan_wigner"):
    r"""``(generators, thetas)`` preparing the basis in which ``unitary`` is diagonal.

    The list is in **circuit order** -- the order a provider applies the gates
    in -- and the resulting unitary satisfies

    .. math::

        W^\dagger a_k W = \pm\, \bigl(U^\dagger a\bigr)_k ,

    so measuring the occupation of wire ``k`` on :math:`W|\psi\rangle` returns
    :math:`\langle \tilde n_k \rangle` in the rotated orbitals.  The sign is the
    leftover :math:`D` of :func:`givens_decomposition`; it cancels in every
    occupation, which is all this is used for.

    **Two reversals cancel here, and getting it wrong is silent.**  The
    verified identity is
    :math:`W = e^{A_{r_1}} \cdots e^{A_{r_K}}` over the *reversed* elimination
    order, while a circuit given ``[g_1, ..., g_K]`` applies ``g_1`` first and
    so realizes :math:`e^{g_K} \cdots e^{g_1}`.  The list is therefore the
    elimination order as it comes.  Written with one reversal too many the
    energy came out 0.15 Ha low with every individual piece looking sound, so
    a test extracts the single-particle matrix the circuit *actually* induces
    rather than trusting either convention.
    """
    rotations, _signs = givens_decomposition(unitary)
    generators, thetas = [], []
    for p, theta in rotations:
        generators.append(givens_generator(n_modes, p, mapping))
        thetas.append(float(theta))
    return generators, thetas


def _diagonal_observable(n_modes: int, coefficients, weight: float,
                         square: bool, mapping: str):
    """``weight * (sum_k c_k n_k)`` (or its square) as a qubit operator.

    Built through :class:`~mandacaru.core.mapping.Fermion` rather than by
    substituting ``n_k = (I - Z_k)/2`` by hand, so the convention is the one
    the rest of Mandacaru uses.  The result **must** be diagonal -- only ``I``
    and ``Z`` -- because that is the whole claim: after the rotation, one
    computational-basis measurement answers for the entire leaf.
    """
    from ..core.mapping import Fermion

    single = Fermion.from_integrals(np.diag(np.asarray(coefficients,
                                                       dtype=float)))
    operator = (single * single) if square else single
    qubit_operator = (weight * operator).map_to_qubits(mapping).simplify()
    offending = [label for label in qubit_operator.terms
                 if set(label) - {"I", "Z"}]
    if offending:
        raise RuntimeError(
            f"a rotated leaf should be diagonal, but {offending[:3]} are not; "
            f"the factorization or the mapping is wrong")
    return qubit_operator


def measurement_problems(factorization: "DoubleFactorization", n_qubits: int,
                         occupied, generators, thetas,
                         mapping: str = "jordan_wigner") -> list[tuple]:
    """One ``(n_qubits, occupied, generators, thetas, observable)`` per basis.

    Exactly the tuples
    :meth:`~mandacaru.backends.providers.QiskitProvider.energies` consumes, so
    the whole energy is **one job** of ``L + 1`` PUBs.  Each problem is the
    ansatz followed by that basis's Givens network, measured against a diagonal
    observable; summing the results and
    :attr:`DoubleFactorization.constant` gives the energy.

    ``generators`` / ``thetas`` describe the state to measure -- an ansatz's
    own, as ``driver.ansatz_problem()`` returns them.
    """
    if mapping != "jordan_wigner":
        raise NotImplementedError(
            f"the factorized measurement reads occupations off the "
            f"computational basis, which is the Jordan-Wigner picture; "
            f"mapping={mapping!r} does not have one.  Measure it through the "
            f"qubit-wise commuting path instead.")
    if int(n_qubits) != factorization.n_modes:
        raise ValueError(
            f"the factorization has {factorization.n_modes} modes but the "
            f"state has {n_qubits} qubits")
    base_generators, base_thetas = list(generators), list(thetas)
    problems = []

    def problem(rotation, observable):
        rotation_generators, rotation_thetas = rotation
        return (int(n_qubits), list(occupied),
                base_generators + rotation_generators,
                base_thetas + rotation_thetas, observable)

    # The one-body term, in the basis that diagonalizes it.
    problems.append(problem(
        basis_rotation(factorization.n_modes,
                       factorization.one_body_orbitals, mapping),
        _diagonal_observable(factorization.n_modes,
                             factorization.one_body_energies, 1.0, False,
                             mapping)))
    # One leaf, one rotation, one diagonal observable.
    for leaf in factorization.leaves:
        problems.append(problem(
            basis_rotation(factorization.n_modes, leaf.orbitals, mapping),
            _diagonal_observable(factorization.n_modes, leaf.coefficients,
                                 0.5 * leaf.weight, True, mapping)))
    return problems


def factorized_energy(provider, problem, factorization: "DoubleFactorization",
                      mapping: str = "jordan_wigner") -> float:
    """``<H>`` measured through the factorized form, in **one** job.

    ``problem`` is the ``(n_qubits, occupied, generators, thetas)`` tuple that
    describes the optimized state -- what
    ``driver.ansatz_problem()[:4]`` returns.

    The ``L + 1`` PUBs go to ``provider.energies`` together, so this costs one
    submission whatever the register size, against the ``O(M^3)`` measurement
    bases the qubit-wise commuting path needs.
    """
    problems = measurement_problems(factorization, *problem, mapping=mapping)
    values = provider.energies(problems)
    return float(factorization.constant + sum(float(v) for v in values))
