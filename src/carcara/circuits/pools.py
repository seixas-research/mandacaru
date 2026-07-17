# -*- coding: utf-8 -*-
# file: circuits/pools.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Operator pools for ADAPT-VQE.

ADAPT-VQE (Grimsley *et al.*, 2019) grows a variational ansatz one operator at a
time, drawing candidates from a fixed *operator pool*.  Each pool element is an
**anti-Hermitian generator** :math:`A` (so :math:`e^{\theta A}` is a
particle-number-conserving unitary for real :math:`\theta`), stored here as a
qubit :class:`~carcara.core.mapping.PauliSum` on the ansatz's ``N = 2M`` qubits.
The energy gradient of appending :math:`e^{\theta A_i}` at :math:`\theta_i = 0` is
:math:`\partial E/\partial\theta_i = \langle\psi|[H, A_i]|\psi\rangle`, which the
driver (:mod:`carcara.algorithms.adapt_vqe`) uses to select operators.

Four pools are provided, in increasing hardware-friendliness:

* **Fermionic** (:class:`FermionicPool`) -- the original ADAPT pool: spin-adapted
  single and double fermionic excitation generators
  :math:`a^\dagger_a a_i - a^\dagger_i a_a` and
  :math:`a^\dagger_a a^\dagger_b a_j a_i - \text{h.c.}`, mapped to qubits
  (Jordan-Wigner by default).  Most accurate, deepest circuits (the JW parity
  ``Z``-strings).
* **Qubit** (:class:`QubitPool`) -- qubit-ADAPT (Tang *et al.*, 2021): every
  individual Pauli string appearing in the JW-mapped fermionic generators, taken
  as an independent generator :math:`i\,P`.  Largest pool, shallowest
  per-operator circuits.
* **QEB** (:class:`QEBPool`) -- Qubit-Excitation-Based (Yordanov *et al.*, 2021):
  the fermionic excitation generators with their JW ``Z``-strings removed, i.e.
  excitations acting only on the involved qubits.  Same excitation structure as
  the fermionic pool but distance-independent two-qubit cost.
* **CEO** (:class:`CEOPool`) -- Coupled-Exchange Operators (Ramôa *et al.*, 2024):
  QEB generators sharing the same qubit support are combined into a single
  generator (one variational parameter, one shared entangling structure -- the
  OVP-CEO variant), giving the best accuracy-per-CNOT of the four.

All pools are built from the spin-blocked spin-orbital ordering used throughout
Carcará (first ``M`` :math:`\alpha`, next ``M`` :math:`\beta`) and only include
excitations that conserve the spin projection :math:`S_z`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from ..core.mapping import Fermion, PauliSum
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


def _strip_outside_support(op: PauliSum, support: Iterable[int]) -> PauliSum:
    """Force every qubit outside ``support`` to identity (drop JW ``Z``-strings)."""
    keep = set(support)
    out = PauliSum()
    for label, coeff in op.terms.items():
        chars = [ch if k in keep else "I" for k, ch in enumerate(label)]
        new = "".join(chars)
        out.terms[new] = out.terms.get(new, 0j) + coeff
    return out.simplify()


def _spin_conserving_excitations(n_spatial_orbitals: int,
                                 num_particles: tuple[int, int]):
    """Occupied/virtual partition and the ``S_z``-conserving excitation index sets.

    Mirrors :class:`~carcara.circuits.ansatz.UCCSD`: the reference fills the lowest
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

    def __init__(self, n_spatial_orbitals: int, num_particles: tuple[int, int],
                 mapping: str = "jordan_wigner"):
        self.n_spatial_orbitals = int(n_spatial_orbitals)
        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.mapping = mapping
        self.n_qubits = 2 * self.n_spatial_orbitals
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
    """Spin-adapted single + double fermionic excitation generators (JW-mapped)."""

    name = "fermionic"

    def _build(self) -> list[PoolOperator]:
        ops: list[PoolOperator] = []
        for label, gen, support in self._fermionic_generators():
            pauli = gen.map_to_qubits(self.mapping, n_modes=self.n_qubits).simplify()
            if not pauli.terms:
                continue
            kind = "fermionic-single" if label[0] == "S" else "fermionic-double"
            ops.append(PoolOperator(label, pauli, _support_of(pauli), kind))
        return ops


# --------------------------------------------------------------------------- #
# 2. Qubit pool (qubit-ADAPT).
# --------------------------------------------------------------------------- #

class QubitPool(PoolBase):
    r"""Individual JW Pauli strings as independent generators :math:`i\,P`.

    Every distinct Pauli string appearing in the Jordan-Wigner image of the
    fermionic generators, kept if it contains at least one ``X`` or ``Y`` (a pure
    ``Z`` string is diagonal and cannot lower the energy), becomes an
    anti-Hermitian generator :math:`A = i\,P`.
    """

    name = "qubit"

    def _build(self) -> list[PoolOperator]:
        seen: set[str] = set()
        ops: list[PoolOperator] = []
        for _label, gen, _support in self._fermionic_generators():
            pauli = gen.map_to_qubits("jordan_wigner", n_modes=self.n_qubits)
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
    r"""Qubit-excitation generators: fermionic excitations with ``Z``-strings dropped.

    Taking the JW image of each fermionic excitation generator and forcing every
    qubit *outside* the excitation's own support to the identity removes the
    Jordan-Wigner parity string, leaving a *qubit excitation* that acts only on
    the involved qubits -- e.g. the single ``(i/2)(X_i Y_a - Y_i X_a)`` and the
    8-term double on the four involved qubits.  The generator stays anti-Hermitian
    and particle-number conserving.
    """

    name = "qeb"

    def _qeb_operators(self) -> list[PoolOperator]:
        ops: list[PoolOperator] = []
        for label, gen, support in self._fermionic_generators():
            jw = gen.map_to_qubits("jordan_wigner", n_modes=self.n_qubits)
            qeb = _strip_outside_support(jw, support)
            if not qeb.terms:
                continue
            kind = "qeb-single" if label[0] == "S" else "qeb-double"
            ops.append(PoolOperator("Q" + label, qeb, tuple(support), kind))
        return ops

    def _build(self) -> list[PoolOperator]:
        return self._qeb_operators()


# --------------------------------------------------------------------------- #
# 4. Coupled-Exchange-Operator (CEO) pool.
# --------------------------------------------------------------------------- #

class CEOPool(QEBPool):
    r"""Coupled-Exchange Operators: QEBs on a shared qubit support, combined.

    All QEB generators acting on the *same* set of qubits are summed into a single
    anti-Hermitian generator (the OVP-CEO variant: one variational parameter and
    one shared entangling structure per group).  Because several exchange terms
    then ride a single CNOT ladder, CEO reaches a given accuracy with fewer
    two-qubit gates than the fermionic or bare-QEB pools.
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
               mapping: str = "jordan_wigner") -> PoolBase:
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
    return cls(n_spatial_orbitals, num_particles, mapping=mapping)
