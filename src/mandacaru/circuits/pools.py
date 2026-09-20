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
* **CEO** (:class:`CEOPool`) -- Coupled Exchange Operators (Ramôa *et al.*, npj
  Quantum Inf. **11**, 86, 2025): the qubit excitations acting on one set of
  spin-orbitals are combined into a single generator.  Every excitation on a
  given set is built from the same eight Pauli strings, so coupling them costs
  no extra entangling structure; a coupled double needs only four of those
  strings, which roughly halves its compiled CNOT count against ``qeb``.  Unlike
  the other pools it is built from **generalized** excitations -- without them a
  set carries one excitation and there is nothing to couple.

All pools are built from the spin-blocked spin-orbital ordering used throughout
Mandacaru (first ``M`` :math:`\alpha`, next ``M`` :math:`\beta`) and only include
excitations that conserve the spin projection :math:`S_z`.  ``fermionic``,
``qubit`` and ``qeb`` restrict the source and target orbitals to be occupied and
unoccupied in the reference; ``ceo`` does not (see :class:`CEOPool`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mapping import (Fermion, PauliSum, qubit_excitation,
                            resolve_mapping)
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
    members : tuple of PoolOperator
        The operators this one is a linear combination of, empty for a plain
        generator.  Only :class:`CEOPool` fills it: an OVP-CEO carries the qubit
        excitations it couples, which the growth step needs to decide between
        the one- and the multiple-parameter form (see :meth:`CEOPool.
        grown_operators`).
    """

    label: str
    generator: PauliSum
    support: tuple[int, ...]
    kind: str
    _matrix: np.ndarray | None = field(default=None, repr=False, compare=False)
    members: tuple = field(default=(), repr=False, compare=False)

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

#: Below this an excitation's energy derivative counts as zero, so it is not
#: worth an independent variational parameter (:meth:`CEOPool.grown_operators`).
#: Well under any ADAPT convergence threshold and well above round-off on the
#: matrix-vector products the gradient is assembled from.
GRADIENT_FLOOR = 1e-12


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
    supports_parity_reduced = False

    #: Whether every generator commutes with the number operators (N, Sz).
    conserves_particle_number = True

    def __init__(self, n_spatial_orbitals: int, num_particles: tuple[int, int],
                 mapping: str = "jordan_wigner"):
        self.n_spatial_orbitals = int(n_spatial_orbitals)
        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.mapping = resolve_mapping(mapping)
        #: Spin-orbital (mode) count; the register is two smaller when tapered.
        self.n_modes = 2 * self.n_spatial_orbitals
        self.n_qubits = self.n_modes
        if self.mapping == "parity_reduced":
            if not self.supports_parity_reduced:
                raise ValueError(
                    f"the {self.name!r} pool's generators do not commute with "
                    "the tapered parity symmetries, so the two-qubit "
                    "reduction would change them; use the 'fermionic', 'qeb' "
                    "or 'ceo' pool with mapping='parity_reduced'")
            self.n_qubits = self.n_modes - 2
        (self._occ, _virtual, self._singles,
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

    def grown_operators(self, selected: PoolOperator, gradient) -> list:
        """Operators to append when ``selected`` wins the gradient screening.

        One operator for every pool but :class:`CEOPool`, whose selected
        element may expand into the several independently parameterized
        excitations it couples.  ``gradient(op)`` returns the energy derivative
        of a candidate at the current state; it is supplied by the driver
        because only the driver knows the state and the matrix backend, and it
        is called only for the operators of the *selected* support.
        """
        return [selected]

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
    supports_parity_reduced = True

    def _build(self) -> list[PoolOperator]:
        ops: list[PoolOperator] = []
        for label, gen, support in self._fermionic_generators():
            pauli = gen.map_to_qubits(
                self.mapping, n_modes=self.n_modes,
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
    supports_parity_reduced = True

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
    r"""Coupled Exchange Operators (Ramoa *et al.*, npj Quantum Inf. **11**, 86, 2025).

    A CEO is a linear combination of the qubit excitations that act on **the same
    set of spin-orbitals**.  The construction rests on an algebraic coincidence
    (verified in ``test/test_pool_encoding.py``): every double QE on a given set
    of four spin-orbitals is a uniformly weighted combination of *the same eight
    Pauli strings*, differing only in the signs of the coefficients.  Combining
    them therefore costs no extra entangling structure, which is where the gate
    savings come from.

    **The excitations are generalized.**  A set of four spin-orbitals carries
    more than one excitation only if the source and target orbitals are *not*
    restricted to be occupied and unoccupied in the reference: with that
    restriction each set carries exactly one, every group is a singleton and the
    pool degenerates into ``qeb``.  So, following Sec. II B of the paper, this
    pool is built from **generalized** singles and doubles (:meth:`excitations`),
    unlike :class:`QEBPool`, which keeps the occupied-to-virtual restriction.
    The consequence is a much larger pool -- 660 operators against QEB's 92 for a
    frozen-core water -- screened for the same measurement cost, since every
    gradient is a linear combination of the same Pauli expectation values.

    **Unique excitations per set.** Of the three ways to pair four spin-orbitals,
    only those conserving :math:`S_z` survive: **two** when the set holds two
    :math:`\alpha` and two :math:`\beta` orbitals (Eqs. 8-9), **three** when all
    four have the same spin.  A pair of spin-orbitals carries exactly one single
    excitation.

    **What the pool contains** (the OVP-CEO set, Eqs. 23-24): for every set with
    :math:`k` excitations, all :math:`2\binom{k}{2}` sums and differences of
    pairs -- 2 operators for an opposite-spin set, 6 for a same-spin one -- plus
    every single excitation, which is trivially its own CEO.  An OVP-CEO of a
    double is a combination of only **four** Pauli strings where a QE needs
    eight, which is what its 9-CNOT circuit (against 13) exploits.

    **Growth** (:meth:`grown_operators`, the paper's modified step 3): the
    selected OVP-CEO is appended as one operator with one parameter when only
    one of its excitations has a non-zero gradient, and otherwise expands into
    the MVP-CEO -- those excitations with independent parameters.  The
    excitations on a shared set commute, so appending them consecutively
    realizes :math:`\exp(\sum_i \theta_i T_i)` exactly.

    **Not implemented:** the specialized 9- and 13-CNOT circuit syntheses of the
    paper's Figs. 5-9.  Mandacaru compiles every generator through the generic
    ``{cx, u}`` path, so the operator counts and parameter counts here follow the
    paper but the CNOT counts do not reach its figures.
    """

    name = "ceo"

    #: Ways to split four spin-orbitals into an ordered pair of pairs.  The third
    #: entry is the one that moves both same-spin orbitals together, so it is the
    #: one an opposite-spin set drops for violating S_z.
    _PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

    def _spin(self, mode: int) -> int:
        return int(mode) // self.n_spatial_orbitals

    def excitations(self) -> dict:
        """``{spin-orbital set: [PoolOperator, ...]}`` of generalized QEs.

        Generalized: every same-spin pair carries a single excitation and every
        four-orbital set carries its :math:`S_z`-conserving double excitations,
        whether or not the orbitals are occupied in the reference.
        """
        groups: dict[tuple[int, ...], list[PoolOperator]] = {}
        modes = range(self.n_modes)

        for i in modes:
            for a in range(i + 1, self.n_modes):
                if self._spin(i) != self._spin(a):
                    continue
                ops: list[PoolOperator] = []
                self._append_excitation(ops, f"S({i}->{a})", (a,), (i,),
                                        "qeb-single")
                if ops:
                    groups[(i, a)] = ops

        for p in modes:
            for q in range(p + 1, self.n_modes):
                for r in range(q + 1, self.n_modes):
                    for s in range(r + 1, self.n_modes):
                        quad = (p, q, r, s)
                        ops = []
                        for (x, y), (u, v) in self._PAIRINGS:
                            i, j, a, b = quad[x], quad[y], quad[u], quad[v]
                            if (self._spin(i) + self._spin(j)
                                    != self._spin(a) + self._spin(b)):
                                continue
                            self._append_excitation(
                                ops, f"D({i},{j}->{a},{b})", (a, b), (i, j),
                                "qeb-double")
                        if ops:
                            groups[quad] = ops
        return groups

    def _build(self) -> list[PoolOperator]:
        ops: list[PoolOperator] = []
        for modes, excitations in self.excitations().items():
            orbitals = ",".join(f"o{m}" for m in modes)
            if len(excitations) == 1:
                # A single excitation realizes the one viable exchange on its
                # orbitals, so it is already its own CEO (paper, Sec. II B 3).
                only = excitations[0]
                ops.append(PoolOperator(f"CEO[{orbitals}]{{{only.label}}}",
                                        only.generator, only.support, "ceo",
                                        members=(only,)))
                continue
            for x in range(len(excitations)):
                for y in range(x + 1, len(excitations)):
                    first, second = excitations[x], excitations[y]
                    for sign, mark in ((1.0, "+"), (-1.0, "-")):
                        generator = (first.generator
                                     + second.generator * sign).simplify()
                        if not generator.terms:
                            continue
                        label = (f"CEO[{orbitals}]"
                                 f"{{{first.label}{mark}{second.label}}}")
                        ops.append(PoolOperator(
                            label, generator, _support_of(generator), "ceo",
                            members=(first, second)))
        return ops

    def grown_operators(self, selected: PoolOperator, gradient) -> list:
        """The paper's modified step 3, applied to the selected OVP-CEO.

        Excitations whose gradient vanishes cannot change the energy at this
        state, so giving them their own parameter buys nothing: when exactly one
        of the coupled excitations is live, the OVP-CEO itself is appended (one
        parameter, the cheaper circuit).  Otherwise the live excitations are
        appended with independent parameters -- the MVP-CEO.  They commute, so
        appending them in sequence is exactly ``exp(sum_i theta_i T_i)``.
        """
        members = tuple(selected.members)
        if len(members) < 2:
            return [selected]
        live = [op for op in members if abs(gradient(op)) > GRADIENT_FLOOR]
        if len(live) < 2:
            return [selected]
        return live


class OVPCEOPool(CEOPool):
    r"""CEO with one variational parameter per growth step (OVP-CEO only).

    The same pool as :class:`CEOPool`; only the growth differs.  The paper's
    algorithm expands the selected operator into the MVP-CEO -- its coupled
    excitations with independent parameters -- whenever more than one of them
    has a non-zero gradient, and implements that as a single 13-CNOT circuit.
    Mandacaru has no such synthesis, so an expanded step compiles as two or
    three separate eight-string excitations and *costs* gates instead of saving
    them.  Keeping the one-parameter form throughout sidesteps that: every
    appended operator is a four-string combination with the cheap circuit.

    The paper considers this variant explicitly (Sec. II B 4 and Supplementary
    Sec. I): the trade is variational freedom, since the coupled excitations are
    then constrained to one shared parameter magnitude.  Measured on the LiH
    curve of ``examples/24_ADAPTVQE_LiH_IBM.py`` (STO-3G, 12 qubits) it costs
    nothing in energy and halves the gate count -- 104 CNOTs against 208 for
    ``qeb`` and 248 for the adaptive ``ceo`` -- which is why the hardware
    example uses it.
    """

    name = "ceo-ovp"

    def grown_operators(self, selected: PoolOperator, gradient) -> list:
        return [selected]


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #

_POOLS = {
    "fermionic": FermionicPool,
    "qubit": QubitPool,
    "qeb": QEBPool,
    "ceo": CEOPool,
    "ceo-ovp": OVPCEOPool,
}

# Friendly aliases.
_POOL_ALIASES = {
    "fermion": "fermionic",
    "uccsd": "fermionic",
    "qubit-adapt": "qubit",
    "pauli": "qubit",
    "qubit-excitation": "qeb",
    "ovp-ceo": "ceo-ovp",
    "coupled-exchange": "ceo",
}


def available_pools() -> list[str]:
    """Canonical pool names understood by :func:`build_pool`."""
    return list(_POOLS)


def build_pool(name: str, n_spatial_orbitals: int,
               num_particles: tuple[int, int],
               mapping: str = "jordan_wigner") -> PoolBase:
    return _build_pool(name, n_spatial_orbitals, num_particles, mapping)


def _build_pool(name: str, n_spatial_orbitals: int,
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
