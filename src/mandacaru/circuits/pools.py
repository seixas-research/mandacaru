# -*- coding: utf-8 -*-
# file: circuits/pools.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Operator pools for ADAPT-VQE.

ADAPT-VQE (Grimsley *et al.*, 2019) grows a variational ansatz one operator at a
time, drawing candidates from a fixed *operator pool*.  Each pool element is an
**anti-Hermitian generator** :math:`A` (so :math:`e^{\theta A}` is a
particle-number-conserving unitary for real :math:`\theta`), stored here as a
qubit :class:`~mandacaru.core.mapping.PauliSum` on the ansatz's ``N = 2M`` qubits.
The energy gradient of appending :math:`e^{\theta A_i}` at :math:`\theta_i = 0` is
:math:`\partial E/\partial\theta_i = \langle\psi|[H, A_i]|\psi\rangle`, which the
driver (:mod:`mandacaru.algorithms.adapt_vqe`) uses to select operators.

Four pools are provided, in increasing hardware-friendliness:

* **Fermionic** (:class:`FermionicPool`) -- the original ADAPT pool: spin-adapted
  single and double fermionic excitation generators
  :math:`a^\dagger_a a_i - a^\dagger_i a_a` and
  :math:`a^\dagger_a a^\dagger_b a_j a_i - \text{h.c.}`, mapped to qubits
  (Jordan-Wigner by default).  Most accurate, deepest circuits (the JW parity
  ``Z``-strings).
* **Qubit** (:class:`QubitPool`) -- qubit-ADAPT (Tang *et al.*, 2021): every
  individual Pauli string appearing in the mapped fermionic generators, taken
  as an independent generator :math:`i\,P`.  Largest pool, shallowest
  per-operator circuits; these strings break particle number by design.
* **QEB** (:class:`QEBPool`) -- Qubit-Excitation-Based (Yordanov *et al.*, 2021):
  the fermionic excitation generators with their JW ``Z``-strings removed, i.e.
  excitations acting only on the involved qubits.  Same excitation structure as
  the fermionic pool but distance-independent two-qubit cost.
* **CEO** (:class:`CEOPool`) -- Coupled-Exchange Operators (Ramôa *et al.*, 2024):
  QEB generators sharing the same qubit support are combined into a single
  generator (one variational parameter, one shared entangling structure -- the
  OVP-CEO variant).  With the present excitation enumeration each support
  carries one excitation, so this pool currently equals ``qeb`` (see
  :class:`CEOPool`).

All pools are built from the spin-blocked spin-orbital ordering used throughout
Mandacaru (first ``M`` :math:`\alpha`, next ``M`` :math:`\beta`) and only include
excitations that conserve the spin projection :math:`S_z`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mapping import Fermion, PauliSum, qubit_excitation
from .gates import double_excitation, single_excitation


# --------------------------------------------------------------------------- #
# Pool element.
# --------------------------------------------------------------------------- #

@dataclass
class PoolOperator:
    """One anti-Hermitian pool generator :math:`A` on ``n_qubits`` qubits.

    Parameters
    ----------
    label : str
        Human-readable identifier (e.g. ``"D(0,1->2,3)"``).
    generator : PauliSum
        The anti-Hermitian generator :math:`A` (a qubit operator); ``e^{theta A}``
        is unitary for real ``theta``.
    support : tuple of int
        Qubit indices the generator acts on non-trivially (used for circuit
        construction and CEO grouping).
    kind : str
        Pool-specific category (``"fermionic-single"``, ``"double"``, ``"pauli"``,
        ``"qeb-single"``, ``"ceo"`` ...).
    """

    label: str
    generator: PauliSum
    support: tuple[int, ...]
    kind: str
    _matrix: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def n_qubits(self) -> int:
        return self.generator.num_qubits

    def matrix(self) -> np.ndarray:
        """Dense matrix of the generator (cached)."""
        if self._matrix is None:
            self._matrix = self.generator.to_matrix()
        return self._matrix

    def __repr__(self) -> str:
        return f"PoolOperator({self.label!r}, kind={self.kind!r}, support={self.support})"


# --------------------------------------------------------------------------- #
# Shared helpers.
# --------------------------------------------------------------------------- #

def _support_of(op: PauliSum) -> tuple[int, ...]:
    """Qubit indices acted on by any non-identity Pauli in ``op``."""
    s = op.simplify()
    qubits: set[int] = set()
    for label in s.terms:
        for k, ch in enumerate(label):
            if ch != "I":
                qubits.add(k)
    return tuple(sorted(qubits))


def _spin_conserving_excitations(n_spatial_orbitals: int,
                                 num_particles: tuple[int, int]):
    """Occupied/virtual partition and the ``S_z``-conserving excitation index sets.

    Mirrors :class:`~mandacaru.circuits.ansatz.UCCSD`: the reference fills the lowest
    ``n_alpha`` alpha (block ``0..M-1``) and ``n_beta`` beta (block ``M..2M-1``)
    spin-orbitals; singles connect same-spin occ/virt, doubles conserve total
    ``S_z``.
    """
    M = int(n_spatial_orbitals)
    na, nb = int(num_particles[0]), int(num_particles[1])
    occ = list(range(na)) + list(range(M, M + nb))
    virt = ([a for a in range(M) if a not in occ]
            + [a for a in range(M, 2 * M) if a not in occ])

    def spin(p):
        return p // M

    singles = [(i, a) for i in occ for a in virt if spin(i) == spin(a)]

    doubles = []
    for x in range(len(occ)):
        for y in range(x + 1, len(occ)):
            i, j = occ[x], occ[y]
            for p in range(len(virt)):
                for q in range(p + 1, len(virt)):
                    a, b = virt[p], virt[q]
                    if spin(i) + spin(j) == spin(a) + spin(b):
                        doubles.append((i, j, a, b))
    return occ, virt, singles, doubles


# --------------------------------------------------------------------------- #
# Pool base class.
# --------------------------------------------------------------------------- #

class PoolBase:
    """Base class for ADAPT-VQE operator pools.

    A pool is built from ``n_spatial_orbitals`` spatial orbitals and a
    ``(n_alpha, n_beta)`` reference occupation, and exposes :meth:`operators` --
    the list of :class:`PoolOperator` candidates on ``n_qubits = 2 *
    n_spatial_orbitals`` qubits.
    """

    name = "base"

    #: Whether the pool's generators survive the parity two-qubit reduction
    #: (they must commute with both tapered symmetries).
    supports_two_qubit_reduction = False

    #: Whether every generator commutes with the number operators (N, Sz).
    conserves_particle_number = True

    def __init__(self, n_spatial_orbitals: int, num_particles: tuple[int, int],
                 mapping: str = "jordan_wigner",
                 two_qubit_reduction: bool = False):
        self.n_spatial_orbitals = int(n_spatial_orbitals)
        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.mapping = mapping
        self.two_qubit_reduction = bool(two_qubit_reduction)
        #: Spin-orbital (mode) count; the register is two smaller when tapered.
        self.n_modes = 2 * self.n_spatial_orbitals
        self.n_qubits = self.n_modes
        if self.two_qubit_reduction:
            if not self.supports_two_qubit_reduction:
                raise ValueError(
                    f"the {self.name!r} pool's generators do not commute with "
                    "the tapered parity symmetries, so the two-qubit "
                    "reduction would change them; use the 'fermionic', 'qeb' "
                    "or 'ceo' pool with mapping='parity'")
            self.n_qubits = self.n_modes - 2
        (self._occ, self._virt, self._singles,
         self._doubles) = _spin_conserving_excitations(
            self.n_spatial_orbitals, self.num_particles)
        self._operators: list[PoolOperator] | None = None

    @property
    def occupied_orbitals(self) -> tuple[int, ...]:
        """Occupied spin-orbital (= JW qubit) indices of the HF reference."""
        return tuple(self._occ)

    def _build(self) -> list[PoolOperator]:  # pragma: no cover - overridden
        raise NotImplementedError

    def operators(self) -> list[PoolOperator]:
        """The pool's candidate generators (built once and cached)."""
        if self._operators is None:
            self._operators = self._build()
        return self._operators

    def __len__(self) -> int:
        return len(self.operators())

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_qubits={self.n_qubits}, "
                f"size={len(self)})")

    # -- shared fermionic generators (used by several pools) --------------- #

    def _fermionic_generators(self) -> list[tuple[str, Fermion, tuple[int, ...]]]:
        """``(label, anti-Hermitian Fermion generator, spin-orbital support)``."""
        gens: list[tuple[str, Fermion, tuple[int, ...]]] = []
        for (i, a) in self._singles:
            gens.append((f"S({i}->{a})", single_excitation(i, a),
                         tuple(sorted((i, a)))))
        for (i, j, a, b) in self._doubles:
            gens.append((f"D({i},{j}->{a},{b})", double_excitation(i, j, a, b),
                         tuple(sorted((i, j, a, b)))))
        return gens


# --------------------------------------------------------------------------- #
# 1. Fermionic pool.
# --------------------------------------------------------------------------- #

class FermionicPool(PoolBase):
    """Spin-adapted single + double fermionic excitation generators.

    Mapped through the driver's own encoding, so the generators conserve the
    particle numbers whichever mapping the Hamiltonian uses.
    """

    name = "fermionic"
    supports_two_qubit_reduction = True

    def _build(self) -> list[PoolOperator]:
        ops: list[PoolOperator] = []
        for label, gen, support in self._fermionic_generators():
            pauli = gen.map_to_qubits(
                self.mapping, n_modes=self.n_modes,
                two_qubit_reduction=self.two_qubit_reduction,
                num_particles=self.num_particles).simplify()
            if not pauli.terms:
                continue
            kind = "fermionic-single" if label[0] == "S" else "fermionic-double"
            ops.append(PoolOperator(label, pauli, _support_of(pauli), kind))
        return ops


# --------------------------------------------------------------------------- #
# 2. Qubit pool (qubit-ADAPT).
# --------------------------------------------------------------------------- #

class QubitPool(PoolBase):
    r"""Individual Pauli strings as independent generators :math:`i\,P`.

    Every distinct Pauli string appearing in the image of the fermionic
    generators **under the requested encoding**, kept if it contains at least
    one ``X`` or ``Y`` (a pure ``Z`` string is diagonal and cannot lower the
    energy), becomes an anti-Hermitian generator :math:`A = i\,P`.
    """

    name = "qubit"

    #: Individual Pauli strings do **not** commute with N or Sz: qubit-ADAPT
    #: explores states outside the reference particle-number sector on purpose.
    #: That is true in every encoding, and it is why this pool cannot be
    #: tapered and is rejected by the particle-sector simulation.
    conserves_particle_number = False

    def _build(self) -> list[PoolOperator]:
        seen: set[str] = set()
        ops: list[PoolOperator] = []
        for _label, gen, _support in self._fermionic_generators():
            pauli = gen.map_to_qubits(self.mapping, n_modes=self.n_qubits)
            for string in pauli.simplify().terms:
                if string in seen:
                    continue
                if not any(ch in "XY" for ch in string):
                    continue
                seen.add(string)
                generator = PauliSum({string: 1j})   # i P is anti-Hermitian
                ops.append(PoolOperator(f"iP[{string}]", generator,
                                        _support_of(generator), "pauli"))
        return ops


# --------------------------------------------------------------------------- #
# 3. Qubit-Excitation-Based (QEB) pool.
# --------------------------------------------------------------------------- #

class QEBPool(PoolBase):
    r"""Qubit-excitation generators, built in the requested encoding.

    A qubit excitation moves occupation numbers *without* the fermionic parity
    sign: :math:`T - T^\dagger` with
    :math:`T = \prod q^\dagger_a \prod q_i` over the
    :func:`~mandacaru.core.mapping.qubit_excitation` ladder operators.  In
    Jordan-Wigner that is exactly "drop the ``Z`` strings outside the
    excitation's support" -- the single ``(i/2)(X_i Y_a - Y_i X_a)`` and the
    8-term double -- but the construction is defined in **every** encoding,
    because the update and flip sets are.

    That matters: simply reusing the JW strings under parity or Bravyi-Kitaev
    would give an operator that no longer commutes with the mapped number
    operator (Frobenius ``|[A, N]|`` of 4.0 and 4.24 on a 4-mode example), so
    the ansatz would leave the physical sector.  Built this way the generator
    is anti-Hermitian and conserves both particle numbers in any encoding, and
    the parity two-qubit reduction therefore applies to it.
    """

    name = "qeb"

    #: Qubit excitations commute with both tapered symmetries.
    supports_two_qubit_reduction = True

    def _qeb_operators(self) -> list[PoolOperator]:
        ops: list[PoolOperator] = []
        for (i, a) in self._singles:
            self._append_excitation(ops, f"QS({i}->{a})", (a,), (i,),
                                    "qeb-single")
        for (i, j, a, b) in self._doubles:
            # Mirrors a+_a a+_b a_j a_i of the fermionic generator.
            self._append_excitation(ops, f"QD({i},{j}->{a},{b})", (a, b),
                                    (j, i), "qeb-double")
        return ops

    def _append_excitation(self, ops, label, modes_out, modes_in, kind) -> None:
        generator = qubit_excitation(
            modes_out, modes_in, self.n_modes, self.mapping,
            two_qubit_reduction=self.two_qubit_reduction,
            num_particles=self.num_particles)
        if generator.terms:
            ops.append(PoolOperator(label, generator, _support_of(generator),
                                    kind))

    def _build(self) -> list[PoolOperator]:
        return self._qeb_operators()


# --------------------------------------------------------------------------- #
# 4. Coupled-Exchange-Operator (CEO) pool.
# --------------------------------------------------------------------------- #

class CEOPool(QEBPool):
    r"""Coupled-Exchange Operators: QEBs on a shared qubit support, combined.

    All QEB generators acting on the *same* set of qubits are summed into a single
    anti-Hermitian generator (the OVP-CEO variant: one variational parameter and
    one shared entangling structure per group), so several exchange terms ride a
    single CNOT ladder.

    **What this construction actually yields.** In Jordan-Wigner the excitation
    enumeration supplies exactly one spin-conserving excitation per qubit
    support, so every group is a *singleton* and the pool is
    generator-for-generator identical to ``qeb`` (verified in
    ``test/test_pool_encoding.py``).  Under parity / Bravyi-Kitaev the update
    and flip sets widen the supports, so distinct excitations can share one and
    are genuinely summed.  Either way the gate savings reported for CEO need
    several exchange directions per support *and* their specialized circuit
    synthesis, which is not implemented here.
    """

    name = "ceo"

    def _build(self) -> list[PoolOperator]:
        groups: dict[tuple[int, ...], PauliSum] = {}
        members: dict[tuple[int, ...], list[str]] = {}
        order: list[tuple[int, ...]] = []
        for op in self._qeb_operators():
            key = op.support
            if key not in groups:
                groups[key] = PauliSum()
                members[key] = []
                order.append(key)
            groups[key] = groups[key] + op.generator
            members[key].append(op.label)

        ops: list[PoolOperator] = []
        for key in order:
            generator = groups[key].simplify()
            if not generator.terms:
                continue
            # Label with the *full* qubit support (not just its endpoints, which
            # collide for non-contiguous supports) plus the coupled QEB
            # excitations, so every CEO operator is uniquely and descriptively
            # named -- e.g. "CEO[q0,q1,q4,q5]{QD(0,1->4,5)}".
            qubits = ",".join(f"q{q}" for q in key)
            excitations = "+".join(m[1:] for m in members[key])  # drop the "Q"
            label = f"CEO[{qubits}]{{{excitations}}}"
            ops.append(PoolOperator(label, generator, key, "ceo"))
        return ops


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #

_POOLS = {
    "fermionic": FermionicPool,
    "qubit": QubitPool,
    "qeb": QEBPool,
    "ceo": CEOPool,
}

# Friendly aliases.
_POOL_ALIASES = {
    "fermion": "fermionic",
    "uccsd": "fermionic",
    "qubit-adapt": "qubit",
    "pauli": "qubit",
    "qubit-excitation": "qeb",
    "ceo-ovp": "ceo",
    "coupled-exchange": "ceo",
}


def available_pools() -> list[str]:
    """Canonical pool names understood by :func:`build_pool`."""
    return list(_POOLS)


def build_pool(name: str, n_spatial_orbitals: int,
               num_particles: tuple[int, int],
               mapping: str = "jordan_wigner",
               two_qubit_reduction: bool = False) -> PoolBase:
    return _build_pool(name, n_spatial_orbitals, num_particles, mapping,
                       two_qubit_reduction)


def _build_pool(name: str, n_spatial_orbitals: int,
               num_particles: tuple[int, int],
               mapping: str = "jordan_wigner",
               two_qubit_reduction: bool = False) -> PoolBase:
    """Construct an :class:`PoolBase` by name.

    ``name`` is one of ``"fermionic"``, ``"qubit"``, ``"qeb"``, ``"ceo"`` (plus a
    few aliases).  ``n_spatial_orbitals`` and ``num_particles = (n_alpha,
    n_beta)`` define the qubit count and Hartree-Fock reference; ``mapping`` is
    the fermion-to-qubit mapping for the fermionic pool.
    """
    key = name.lower()
    key = _POOL_ALIASES.get(key, key)
    try:
        cls = _POOLS[key]
    except KeyError:
        raise ValueError(
            f"unknown pool {name!r}; choose from {sorted(_POOLS)} "
            f"(or aliases {sorted(_POOL_ALIASES)})") from None
    return cls(n_spatial_orbitals, num_particles, mapping=mapping,
               two_qubit_reduction=two_qubit_reduction)
