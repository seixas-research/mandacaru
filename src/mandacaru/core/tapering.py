# -*- coding: utf-8 -*-
# file: core/tapering.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Z\ :sub:`2` symmetry tapering: one qubit removed per conserved parity.

The parity mapping's two-qubit reduction
(:func:`~mandacaru.core.mapping.two_qubit_reduce`) removes exactly two qubits,
and it knows in advance which two: the alpha-sector parity and the total
parity, whose values the particle numbers fix.  That is a special case of a
general fact.  A molecular qubit Hamiltonian usually commutes with **more**
:math:`Z_2` operators than those two -- the spatial symmetries of the molecule
give further ones, so water in :math:`C_{2v}` or a homonuclear diatomic in
:math:`D_{\infty h}` has parities beyond particle number -- and each one is a
qubit that carries no information.

Nothing about the point group has to be supplied.  The symmetries are read off
the Hamiltonian itself, which is what makes this robust: a symmetry that the
orbitals happen to break is not found, and one that the molecule has but the
user did not think of is.

How it works
------------

Write each Pauli string as a pair of bit vectors, :math:`X`-support and
:math:`Z`-support.  Two strings commute exactly when their symplectic form
vanishes, :math:`x_1 \cdot z_2 + z_1 \cdot x_2 = 0 \pmod 2`.  So the Pauli
operators commuting with **every** term of :math:`H` are the kernel, over
:math:`\mathrm{GF}(2)`, of the matrix whose rows are the terms' symplectic
vectors with the halves swapped.  :func:`symmetry_generators` computes that
kernel and returns an independent set :math:`\tau_1 \ldots \tau_k` of
:math:`Z`-type generators.

For each :math:`\tau_j` pick a qubit :math:`q_j` where :math:`\tau_j` acts and
the others do not, and set :math:`\sigma_j = X_{q_j}`.  Then
:math:`U_j = (\tau_j + \sigma_j)/\sqrt2` is a Clifford, it is its own inverse,
and it maps :math:`\tau_j \mapsto \sigma_j`.  After conjugating by every
:math:`U_j` the Hamiltonian commutes with each :math:`X_{q_j}`, so each of those
qubits is a constant of motion: replace it by its eigenvalue :math:`\pm 1` and
delete it.  :func:`taper` does that, and :func:`sector_signs` reads the
eigenvalues off a reference determinant, which is the only way to land in the
sector that holds the ground state.

The trap this shares with the particle-number sector
----------------------------------------------------

Tapering the Hamiltonian is exact.  Tapering an **operator that does not commute
with the symmetries** is not: deleting a qubit replaces the operator by its
projection, and :math:`\exp(PAP) \neq P\exp(A)P`.  This is the same failure the
sector reduction documents (:mod:`mandacaru.core.sector`), and it has the same
consequence -- an ansatz built from projected generators explores the wrong
manifold and converges above the ground state, plausibly.  :func:`leaking_terms`
is therefore the check a caller must run over the operator pool before tapering
it, and :func:`taper` refuses an operator that fails it rather than returning a
projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import PauliSum

#: Largest coefficient still treated as zero when a tapered term cancels.
TAPER_ATOL = 1e-12


def symplectic_form(operator: PauliSum) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Z)`` support matrices of ``operator``'s terms, one row per term.

    ``X[t, q]`` is 1 when term ``t`` acts with ``X`` or ``Y`` on qubit ``q``, and
    ``Z[t, q]`` when it acts with ``Z`` or ``Y``.  Together they determine the
    term up to phase, and commutation is a bilinear form in them.
    """
    labels = list(operator.terms)
    n = operator.num_qubits
    x = np.zeros((len(labels), n), dtype=np.int8)
    z = np.zeros((len(labels), n), dtype=np.int8)
    for t, label in enumerate(labels):
        for q, letter in enumerate(label):
            if letter in ("X", "Y"):
                x[t, q] = 1
            if letter in ("Z", "Y"):
                z[t, q] = 1
    return x, z


def _gf2_kernel(matrix: np.ndarray) -> np.ndarray:
    """A basis of the null space of ``matrix`` over GF(2), as rows."""
    a = np.asarray(matrix, dtype=np.int8) % 2
    rows, cols = a.shape
    a = a.copy()
    pivots: list[int] = []
    row = 0
    for col in range(cols):
        candidates = np.flatnonzero(a[row:, col])
        if candidates.size == 0:
            continue
        pick = row + int(candidates[0])
        a[[row, pick]] = a[[pick, row]]
        for r in range(rows):
            if r != row and a[r, col]:
                a[r] ^= a[row]
        pivots.append(col)
        row += 1
        if row == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = np.zeros((len(free), cols), dtype=np.int8)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for r, p in enumerate(pivots):
            if a[r, f]:
                basis[k, p] = 1
    return basis


def symmetry_generators(operator: PauliSum) -> list[str]:
    r"""Independent :math:`Z`-type Pauli strings commuting with every term.

    Returns the labels of :math:`\tau_1 \ldots \tau_k`.  Empty when the operator
    has no :math:`Z_2` symmetry beyond the identity.

    Only :math:`Z`-type generators are searched for, which is not a restriction:
    any abelian group of commuting Pauli operators can be brought to
    :math:`Z`-type form by a Clifford, and for a molecular Hamiltonian in a real
    orbital basis the particle-number and spatial parities already are.  Keeping
    to :math:`Z` type is what makes the Clifford in :func:`taper` a product of
    single-qubit-anchored reflections rather than a general symplectic search.
    """
    x, z = symplectic_form(operator.simplify())
    if x.size == 0:
        return []
    # A Z-type candidate has no X support, so commuting with term t reduces to
    # x_t . z_cand = 0: the kernel of the X-support matrix.
    kernel = _gf2_kernel(x)
    labels = []
    for row in kernel:
        if not row.any():
            continue
        labels.append("".join("Z" if bit else "I" for bit in row))
    return labels


def leaking_terms(operator: PauliSum, generators) -> list[str]:
    """Labels of ``operator``'s terms that fail to commute with a ``tau``.

    Empty means the operator can be tapered exactly.  Anything else means
    tapering would replace it by its projection, which is not the same operator
    -- see this module's docstring.
    """
    if not generators:
        return []
    taus = [np.array([1 if letter == "Z" else 0 for letter in tau],
                     dtype=np.int8) for tau in generators]
    x, _z = symplectic_form(operator.simplify())
    labels = list(operator.simplify().terms)
    bad = []
    for t, label in enumerate(labels):
        # tau is Z-type, so the symplectic form is x_term . z_tau.
        if any(int(x[t] @ tau) % 2 for tau in taus):
            bad.append(label)
    return bad


def reduce_generators(generators) -> tuple[list[str], list[int]]:
    r"""``(reduced, anchors)``: a generating set in which each has its own qubit.

    Every :math:`\tau_j` needs an **anchor** qubit :math:`q_j` that it acts on
    and that no *other* generator acts on.  Two things depend on it: the Clifford
    :math:`U_j = (\tau_j + X_{q_j})/\sqrt2` is only unitary when
    :math:`\tau_j` anticommutes with :math:`X_{q_j}` (so :math:`\tau_j` must act
    there), and :math:`U_j` must commute with every other :math:`\tau_i` for the
    product of Cliffords to send each generator to its own :math:`X` (so no other
    generator may act there).

    Such a qubit generally does **not** exist for an arbitrary generating set, so
    the set is brought to reduced row echelon form over :math:`\mathrm{GF}(2)`
    first.  That is legitimate because a product of symmetries is a symmetry: the
    reduced operators generate the same group, and each has its pivot column to
    itself.  Getting this wrong is how the first version of this module failed --
    it eliminated to *choose* the anchors but then conjugated with the original,
    unreduced labels, so an anchor was a qubit the operator being used did not
    act on, and the rotated Hamiltonian kept a ``Z`` where an ``X`` was required.
    """
    labels = list(generators)
    if not labels:
        return [], []
    rows = np.array([[1 if letter == "Z" else 0 for letter in tau]
                     for tau in labels], dtype=np.int8)
    n_rows, n_cols = rows.shape
    pivots: list[int] = []
    row = 0
    for col in range(n_cols):
        candidates = np.flatnonzero(rows[row:, col])
        if candidates.size == 0:
            continue
        pick = row + int(candidates[0])
        rows[[row, pick]] = rows[[pick, row]]
        # Eliminate this column from *every* other row: that is what makes the
        # pivot exclusive, and an exclusive pivot is what an anchor is.
        for other in range(n_rows):
            if other != row and rows[other, col]:
                rows[other] ^= rows[row]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    reduced, anchors = [], []
    for r, pivot in enumerate(pivots):
        if not rows[r].any():
            continue
        reduced.append("".join("Z" if bit else "I" for bit in rows[r]))
        anchors.append(int(pivot))
    return reduced, anchors


def _clifford(tau: str, qubit: int) -> PauliSum:
    r""":math:`(\tau + X_q)/\sqrt2`, the Clifford mapping ``tau`` to ``X_q``."""
    n = len(tau)
    x_label = "".join("X" if q == qubit else "I" for q in range(n))
    root = 1.0 / np.sqrt(2.0)
    return PauliSum({tau: complex(root), x_label: complex(root)}, num_qubits=n)


def rotate_to_z_type(operator: PauliSum, generators, anchors=None) -> PauliSum:
    """Conjugate ``operator`` by the Cliffords that send each ``tau`` to ``X_q``.

    The result commutes with every ``X_q``, which is what lets those qubits be
    deleted.  Unitary, so the spectrum is untouched.
    """
    generators = list(generators)
    if anchors is None:
        generators, anchors = reduce_generators(generators)
    out = operator
    for tau, qubit in zip(generators, anchors):
        u = _clifford(tau, qubit)
        out = u.compose(out).compose(u)
    return out.simplify(TAPER_ATOL)


def sector_signs(reference_bits, generators) -> list[int]:
    r"""The eigenvalue :math:`\pm 1` of each ``tau`` on a reference determinant.

    ``reference_bits`` is the occupation bit string of the reference state in the
    same encoding the Hamiltonian is written in (``1`` = occupied).  A
    :math:`Z`-type ``tau`` is diagonal there, so its eigenvalue is just
    :math:`(-1)^{\text{overlap}}`.  Choosing the signs any other way puts the
    tapered Hamiltonian in a sector that need not contain the ground state --
    the tapered spectrum is then a *different* subset of the original one, and
    the lowest eigenvalue of the wrong subset is not the ground-state energy.
    """
    bits = np.asarray([int(b) for b in reference_bits], dtype=np.int8)
    signs = []
    for tau in generators:
        support = np.array([1 if letter == "Z" else 0 for letter in tau],
                           dtype=np.int8)
        if support.size != bits.size:
            raise ValueError(
                f"the reference has {bits.size} bits and the symmetry acts on "
                f"{support.size} qubits")
        signs.append(-1 if int(support @ bits) % 2 else +1)
    return signs


def taper(operator: PauliSum, generators, signs, anchors=None,
          rotate: bool = True) -> PauliSum:
    """Remove one qubit per symmetry generator, in the given sector.

    ``rotate=False`` skips the Clifford, for an operator already brought to
    ``X``-type form by a previous call -- which is how a pool is tapered
    consistently with its Hamiltonian.

    Raises
    ------
    ValueError
        If ``operator`` has a term that does not commute with a generator.
        Tapering it would return its projection instead, silently.
    """
    generators = list(generators)
    if not generators:
        return operator.simplify(TAPER_ATOL)
    if anchors is None:
        generators, anchors = reduce_generators(generators)
    if len(signs) != len(generators):
        raise ValueError(f"{len(signs)} signs for {len(generators)} generators")

    leaks = leaking_terms(operator, generators)
    if leaks:
        raise ValueError(
            f"{len(leaks)} term(s) of this operator do not commute with the "
            f"Z2 symmetries, the first being {leaks[0]!r}.  Deleting a qubit "
            f"would replace the operator by its projection, and "
            f"exp(PAP) != P exp(A) P -- an ansatz built from projected "
            f"generators explores the wrong manifold and converges above the "
            f"ground state without saying so.  Use a pool whose generators "
            f"conserve these symmetries, or do not taper.")

    rotated = (rotate_to_z_type(operator, generators, anchors) if rotate
               else operator.simplify(TAPER_ATOL))
    drop = set(int(q) for q in anchors)
    sign_of = {int(q): int(s) for q, s in zip(anchors, signs)}
    out: dict[str, complex] = {}
    for label, coefficient in rotated.terms.items():
        phase = 1.0
        for q in drop:
            letter = label[q]
            if letter == "I":
                continue
            if letter != "X":
                # Cannot happen once the rotation has run; a Z or Y on an anchor
                # would mean the operator did not commute with that X_q after
                # all, and folding it into a sign would be wrong.
                raise ValueError(
                    f"qubit {q} carries {letter!r} after the rotation, so the "
                    f"operator does not commute with X_{q} and cannot be "
                    f"tapered on it")
            phase *= sign_of[q]
        kept = "".join(c for q, c in enumerate(label) if q not in drop)
        out[kept] = out.get(kept, 0j) + coefficient * phase
    width = operator.num_qubits - len(drop)
    return PauliSum(out, num_qubits=width).simplify(TAPER_ATOL)


def tapered_reference_bits(reference_bits, anchors) -> list[int]:
    """The reference occupation with the tapered qubits removed."""
    drop = set(int(q) for q in anchors)
    return [int(b) for q, b in enumerate(reference_bits) if q not in drop]


@dataclass(frozen=True)
class TaperedRegister:
    """Everything a solver needs to run on a Z\\ :sub:`2`-tapered register.

    Built by :func:`taper_problem`, which is the single place that decides the
    symmetries, the anchors and the sector -- so the Hamiltonian, every pool
    generator and the reference determinant are guaranteed to have been reduced
    by the *same* Clifford.  Tapering them independently would silently put them
    on different registers.

    Attributes
    ----------
    hamiltonian : PauliSum
        The tapered Hamiltonian, on ``n_qubits``.
    generators : tuple of PauliSum
        The tapered pool generators that survived, in the order given.
    kept : tuple of int
        Indices into the *original* generator list that ``generators`` came from.
    dropped : tuple of int
        Indices of the generators that did **not** commute with the symmetries.
        They connect different symmetry sectors, so in the sector the reference
        fixes they cannot contribute -- an exact-ground-state amplitude on them
        is zero and ADAPT would screen them out at zero gradient.  Dropping them
        is therefore not an approximation *within this sector*; it does mean a
        ground state of different symmetry is unreachable, which is inherent to
        tapering and not to this choice.
    occupied : tuple of int
        Qubit indices set in the tapered reference determinant, so the ansatz can
        prepare it the way it prepares any Jordan-Wigner reference.
    symmetries, anchors, signs : tuple
        The group, the qubit removed per generator, and the sector.
    """

    hamiltonian: PauliSum
    generators: tuple
    kept: tuple[int, ...]
    dropped: tuple[int, ...]
    occupied: tuple[int, ...]
    symmetries: tuple[str, ...]
    anchors: tuple[int, ...]
    signs: tuple[int, ...]

    @property
    def n_qubits(self) -> int:
        return self.hamiltonian.num_qubits

    @property
    def removed(self) -> int:
        """How many qubits the taper took off."""
        return len(self.anchors)

    def taper_operator(self, operator: PauliSum) -> PauliSum:
        """``operator`` reduced by this register's own Clifford and sector.

        For an operator that was not among the ``generators`` handed to
        :func:`taper_problem` -- a coupled-exchange operator's member
        excitations, say -- but has to live on the same register.  Raises the
        ``ValueError`` of :func:`taper` when it does not commute with the
        symmetries.
        """
        return taper(operator, self.symmetries, self.signs, self.anchors)

    def summary(self) -> str:
        """One line for the run log."""
        line = (f"{self.removed} qubit(s) removed by Z2 symmetry "
                f"-> {self.n_qubits} on the register")
        if self.dropped:
            line += (f"; {len(self.dropped)} pool generator(s) dropped as "
                     f"symmetry-changing")
        return line


def taper_problem(hamiltonian: PauliSum, generators, reference_bits
                  ) -> TaperedRegister | None:
    """Taper a Hamiltonian, its pool and its reference together, or ``None``.

    ``None`` when the Hamiltonian has no :math:`Z_2` symmetry to exploit, which
    lets a caller fall through to the untapered path without a special case.

    ``generators`` are the pool's anti-Hermitian generators as
    :class:`~mandacaru.core.mapping.PauliSum` objects on the same register as
    ``hamiltonian``; ``reference_bits`` is the occupation bit string of the
    reference determinant in that same encoding.
    """
    reduced, anchors = reduce_generators(symmetry_generators(hamiltonian))
    if not reduced:
        return None
    signs = sector_signs(reference_bits, reduced)

    kept, dropped, tapered_generators = [], [], []
    for index, generator in enumerate(generators):
        if leaking_terms(generator, reduced):
            dropped.append(index)
            continue
        kept.append(index)
        tapered_generators.append(
            taper(generator, reduced, signs, anchors))

    bits = tapered_reference_bits(reference_bits, anchors)
    return TaperedRegister(
        hamiltonian=taper(hamiltonian, reduced, signs, anchors),
        generators=tuple(tapered_generators),
        kept=tuple(kept), dropped=tuple(dropped),
        occupied=tuple(q for q, bit in enumerate(bits) if bit),
        symmetries=tuple(reduced), anchors=tuple(anchors),
        signs=tuple(signs))
