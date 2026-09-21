# -*- coding: utf-8 -*-
# file: backends/measurement.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Measuring :math:`\langle H\rangle` from shots -- the hardware energy path.

A state-vector simulator hands back the full amplitude vector, so the energy is
one inner product.  **Real quantum hardware cannot do that.**  A QPU returns
*samples* of computational-basis bit-strings, so the energy of a qubit
Hamiltonian :math:`H = \sum_j c_j P_j` has to be assembled term by term from
measured expectation values,

.. math::

    \langle H\rangle = \sum_j c_j \langle P_j\rangle ,

and each :math:`\langle P_j\rangle` needs the register measured in that Pauli's
own eigenbasis.  Measuring one term per circuit is correct but ruinously
expensive: a modest active space has :math:`10^2`--:math:`10^4` terms.

The standard remedy, implemented here, is **qubit-wise commuting (QWC)
grouping**.  Two Pauli strings are QWC when, on every qubit where both act
non-trivially, they carry the *same* Pauli.  A QWC set can be measured by a
single circuit: rotate each qubit once into the basis its group prescribes,
measure everything, and read every term's expectation value out of the same
bit-strings.  :func:`qubit_wise_commuting_groups` builds those groups with a
greedy largest-first coloring, typically cutting the circuit count by one to
two orders of magnitude.

This module is deliberately provider-independent -- it only produces *basis
labels* and combines counts -- so any SDK backend (see
:mod:`mandacaru.backends.providers`) can drive a QPU with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mapping import PauliSum


def is_qubit_wise_commuting(a: str, b: str) -> bool:
    """True when Pauli strings ``a`` and ``b`` can be measured in one basis.

    The condition is per qubit: wherever both strings act non-trivially they must
    carry the *same* Pauli letter (identity always agrees).
    """
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def merge_basis(labels) -> str:
    """The single measurement basis covering a set of QWC Pauli strings.

    Qubit ``k`` takes the non-identity letter any member requires there, or
    ``"I"`` (measured in the ``Z`` basis, contributing nothing) if none does.

    Raises
    ------
    ValueError
        If the strings are not qubit-wise commuting.
    """
    labels = list(labels)
    if not labels:
        return ""
    basis = list(labels[0])
    for label in labels[1:]:
        for k, letter in enumerate(label):
            if letter == "I":
                continue
            if basis[k] in ("I", letter):
                basis[k] = letter
            else:
                raise ValueError(
                    f"Pauli strings {''.join(basis)!r} and {label!r} are not "
                    f"qubit-wise commuting on qubit {k}")
    return "".join(basis)


def qubit_wise_commuting_groups(hamiltonian: PauliSum,
                                drop_identity: bool = True):
    """Partition ``hamiltonian`` into QWC groups; return ``(groups, identity)``.

    ``groups`` is a list of ``(basis, [(pauli, coefficient), ...])`` -- one
    measurement circuit per entry, ``basis`` being the string of single-qubit
    bases to rotate into.  ``identity`` is the coefficient of the all-``I`` term,
    which needs no measurement and is added to the energy as a constant.

    The partition is greedy, largest-coefficient-first (so the terms that
    dominate the energy land in the earliest, largest groups).  Greedy coloring
    is not optimal -- finding the minimum number of groups is NP-hard -- but it
    is fast and close enough in practice.
    """
    simplified = hamiltonian.simplify()
    identity = 0.0
    terms = []
    for label, coeff in simplified.terms.items():
        if set(label) == {"I"}:
            identity += float(np.real(coeff))
            continue
        terms.append((label, complex(coeff)))
    if drop_identity is False and identity:
        terms.append(("I" * simplified.num_qubits, complex(identity)))

    # Largest |coefficient| first: the energetically important terms group early.
    terms.sort(key=lambda item: -abs(item[1]))

    groups: list[tuple[list[str], list[tuple[str, complex]]]] = []
    for label, coeff in terms:
        for members, payload in groups:
            if all(is_qubit_wise_commuting(label, other) for other in members):
                members.append(label)
                payload.append((label, coeff))
                break
        else:
            groups.append(([label], [(label, coeff)]))

    return [(merge_basis(members), payload) for members, payload in groups], \
        identity


def pauli_expectation_from_counts(label: str, counts: dict) -> float:
    r"""``<P>`` for one Pauli string from bit-string counts of its own basis.

    ``counts`` maps a measured bit-string (character ``k`` is qubit ``k``, the
    Mandacaru convention) to its number of occurrences, taken in a basis that
    diagonalizes ``label``.  Each shot contributes :math:`(-1)^{\text{parity}}`
    over the qubits where ``label`` acts non-trivially.
    """
    active = [k for k, letter in enumerate(label) if letter != "I"]
    if not active:
        return 1.0
    total = sum(counts.values())
    if total == 0:
        return 0.0
    accumulated = 0
    for bits, n in counts.items():
        parity = sum(int(bits[k]) for k in active) & 1
        accumulated += -n if parity else n
    return accumulated / total


def energy_from_group_counts(groups, identity: float, counts_per_group) -> float:
    r"""Assemble :math:`\langle H\rangle` from one count dictionary per QWC group.

    ``groups`` is the output of :func:`qubit_wise_commuting_groups` and
    ``counts_per_group`` the measured bit-string counts for each of them, in the
    same order.  Coefficients are Hermitian (real) for a physical Hamiltonian, so
    only the real part contributes.
    """
    energy = float(identity)
    for (_basis, payload), counts in zip(groups, counts_per_group):
        for label, coeff in payload:
            energy += float(np.real(coeff)) * \
                pauli_expectation_from_counts(label, counts)
    return energy


def shot_noise_estimate(hamiltonian: PauliSum, shots: int) -> float:
    r"""Rough standard error of a shot-based ``<H>`` at ``shots`` per group.

    Each :math:`\langle P_j\rangle` has variance at most 1, so summing
    independent terms bounds the error by
    :math:`\bigl(\sum_j |c_j|\bigr)/\sqrt{\text{shots}}` -- the usual
    coefficient-1-norm estimate.  Useful for choosing ``shots`` before paying for
    QPU time: chemical accuracy (1.6 mHa = 0.043 eV) on a 1-norm of 10 Ha
    (272 eV) needs
    :math:`\sim 4\times10^7` shots per group in the worst case, which is why
    hardware VQE needs error mitigation and smarter estimators.
    """
    one_norm = sum(abs(complex(c)) for label, c in
                   hamiltonian.simplify().terms.items() if set(label) != {"I"})
    return float(one_norm / np.sqrt(max(int(shots), 1)))


# --------------------------------------------------------------------------- #
# Pre-flight: what a measurement job will cost, before it is queued.
# --------------------------------------------------------------------------- #

#: Nominal two-qubit gate error used to turn a gate count into an expected
#: circuit fidelity.  A round number, not a calibration: the point of the
#: estimate is the order of magnitude, and the live value is on
#: ``backend.properties()``.
NOMINAL_2Q_ERROR = 3.0e-3

#: Below this expected fidelity a plan is reported as noise rather than signal.
#: At 0.1 an unmitigated expectation value retains a tenth of its amplitude,
#: which is already past the point where zero-noise extrapolation has anything
#: to extrapolate from (measured: 12-qubit LiH, 92 CZ, landed 0.5-0.8 Ha above
#: exact; a 24-qubit PAW-TZP ansatz transpiles to ~2,200 two-qubit gates, i.e.
#: a fidelity of 1e-3).
FIDELITY_WARNING = 0.1

#: Default ceilings for :meth:`MeasurementPlan.check`.  They exist to stop the
#: job that motivated them: LiH/PAW-TZP submitted 97,980 observables in ~21,000
#: measurement bases and the Runtime program died with "error code 1336;
#: Program runtime ran out of memory" after 37 minutes in the queue.  Each
#: limit is generous enough for a problem that can actually run and tight
#: enough to refuse that one.  ``None`` anywhere means "do not check".
DEFAULT_MEASUREMENT_BUDGET = {
    "observables": 50_000,
    "bases": 10_000,
    "circuit_instances": 2_000_000,
    "total_shots": None,
}

#: Keys :func:`resolve_measurement_budget` accepts.
BUDGET_KEYS = tuple(DEFAULT_MEASUREMENT_BUDGET)

#: How an energy is measured on a processor.
#:
#: ``"qwc"`` partitions the Hamiltonian's Pauli strings into qubit-wise
#: commuting groups -- ``O(M^3)`` measurement bases, no extra gates.
#: ``"double-factorized"`` rotates into each factor's own orbital basis and
#: reads occupations -- ``O(M)`` bases at the cost of a Givens network per
#: basis (:mod:`mandacaru.backends.factorization`).
MEASUREMENT_SCHEMES = ("qwc", "double-factorized")


def resolve_measurement_budget(budget) -> dict:
    """Normalize ``measurement_budget=`` into a full dict of ceilings.

    ``None`` / ``True`` is :data:`DEFAULT_MEASUREMENT_BUDGET`, ``False`` turns
    every check off, and a dict overrides the keys it names (a value of ``None``
    switching that one check off).  An unknown key raises, because a misspelled
    ceiling that silently does nothing is the failure this exists to prevent.
    """
    if budget is False:
        return dict.fromkeys(BUDGET_KEYS)
    if budget is None or budget is True:
        return dict(DEFAULT_MEASUREMENT_BUDGET)
    if not isinstance(budget, dict):
        raise TypeError(
            f"measurement_budget must be True, False or a dict of "
            f"{BUDGET_KEYS}, got {type(budget).__name__}")
    unknown = sorted(set(budget) - set(BUDGET_KEYS))
    if unknown:
        raise ValueError(
            f"unknown measurement_budget key(s) {unknown}; "
            f"use {BUDGET_KEYS}")
    resolved = dict(DEFAULT_MEASUREMENT_BUDGET)
    resolved.update(budget)
    return resolved


@dataclass(frozen=True)
class MeasurementPlan:
    """What one measurement submission will ask a processor to do.

    Everything here is computable **before** anything is queued -- the grouping
    is local, the transpilation is local, and the multipliers come from the
    estimator options -- which is the point: a job whose size is only discovered
    when the Runtime program runs out of memory has already cost its queue time.
    """

    n_qubits: int
    observables: int
    bases: int
    shots_per_basis: int
    noise_factors: int
    twirls: int
    two_qubit_gates: int | None = None
    circuit_depth: int | None = None
    one_norm_hartree: float = 0.0
    device: str = "?"
    includes_rdms: bool = False
    jobs: int = 1
    #: Bases the same energy would need under double factorization (``L + 1``),
    #: when the integrals were available to compute it.  Reported beside the
    #: qubit-wise count because the gap is the argument for implementing the
    #: basis-rotation circuits: measured on LiH/PAW, 21 -> 4, 93 -> 11,
    #: 1,600 -> 55 and 3,290 -> 73 bases at 4 / 8 / 20 / 24 spin orbitals.
    factorized_bases: int | None = None
    #: Which scheme this submission uses (:data:`MEASUREMENT_SCHEMES`).
    scheme: str = "qwc"

    @property
    def circuit_instances(self) -> int:
        """Distinct circuits the processor executes, ZNE and twirling included."""
        return self.bases * max(self.noise_factors, 1) * max(self.twirls, 1)

    @property
    def total_shots(self) -> int:
        """Shots across every circuit instance."""
        return self.circuit_instances * max(self.shots_per_basis, 0)

    @property
    def fidelity(self) -> float | None:
        """Expected circuit fidelity ``(1 - e)^n2q``, or ``None`` if unknown."""
        if self.two_qubit_gates is None:
            return None
        return float((1.0 - NOMINAL_2Q_ERROR) ** self.two_qubit_gates)

    @property
    def shot_noise_hartree(self) -> float:
        """1-norm bound on the standard error of ``<H>`` at this shot count."""
        if self.shots_per_basis <= 0:
            return 0.0
        return float(self.one_norm_hartree / np.sqrt(self.shots_per_basis))

    def fields(self) -> dict:
        """The plan as ordered ``KEY: value`` lines for the run log."""
        from ..units import HARTREE_TO_EV

        fidelity = self.fidelity
        fields = {
            "device": self.device,
            "scheme": self.scheme,
            "measured": ("energy and RDMs" if self.includes_rdms
                         else "energy only"),
            "qubits": self.n_qubits,
            "observables": self.observables,
            "measurement_bases": self.bases,
            "jobs": self.jobs,
            "shots_per_basis": self.shots_per_basis,
            "zne_noise_factors": self.noise_factors,
            "twirling_randomizations": self.twirls,
            "circuit_instances": self.circuit_instances,
            "total_shots": self.total_shots,
            "hamiltonian_one_norm_Ha": f"{self.one_norm_hartree:.6f}",
            "shot_noise_bound_eV":
                f"{self.shot_noise_hartree * HARTREE_TO_EV:.6f}",
        }
        if self.scheme == "double-factorized":
            # The factorized form has a 1-norm of its own, and it is *larger*
            # than the Pauli one: fewer bases, more shots in each.  Reporting
            # the Pauli norm here would understate the shot cost.
            fields["hamiltonian_one_norm_Ha"] = (
                f"{self.one_norm_hartree:.6f} (factorized; the Pauli 1-norm "
                f"is smaller)")
        if self.factorized_bases and self.scheme != "double-factorized":
            fields["double_factorized_bases"] = (
                f"{self.factorized_bases} (available: "
                f"measurement_scheme='double-factorized')")
        if self.two_qubit_gates is not None:
            fields["isa_two_qubit_gates"] = self.two_qubit_gates
            fields["isa_depth"] = self.circuit_depth
            fields["expected_fidelity"] = (
                f"{fidelity:.3e} (at a nominal {NOMINAL_2Q_ERROR:g} "
                f"two-qubit error)")
        return fields

    def warnings(self) -> list[str]:
        """Things worth saying that are not grounds for refusing the job."""
        notes = []
        fidelity = self.fidelity
        if fidelity is not None and fidelity < FIDELITY_WARNING:
            notes.append(
                f"the circuit transpiles to {self.two_qubit_gates} two-qubit "
                f"gates, an expected fidelity of {fidelity:.1e} at a nominal "
                f"{NOMINAL_2Q_ERROR:g} error per gate: the result will be "
                f"noise, and zero-noise extrapolation has nothing to "
                f"extrapolate from. Reduce the register (a smaller basis "
                f"size, mapping='parity_reduced', frozen_core) or the circuit "
                f"(pool='ceo-ovp', tetris=True, prune=True).")
        return notes

    def check(self, budget=None) -> None:
        """Raise when this plan exceeds ``budget`` (see the module defaults)."""
        limits = resolve_measurement_budget(budget)
        measured = {"observables": self.observables, "bases": self.bases,
                    "circuit_instances": self.circuit_instances,
                    "total_shots": self.total_shots}
        over = [(key, measured[key], limits[key]) for key in BUDGET_KEYS
                if limits.get(key) is not None and measured[key] > limits[key]]
        if not over:
            return
        detail = "; ".join(f"{key} {value:,} > {limit:,}"
                           for key, value, limit in over)
        raise MeasurementBudgetError(
            f"this measurement would exceed the budget ({detail}). "
            f"Plan: {self.observables:,} observables in {self.bases:,} "
            f"measurement bases, {self.circuit_instances:,} circuit "
            f"instances, {self.total_shots:,} shots on {self.device}. "
            f"Ask for the energy without forces (the RDM operators are the "
            f"O(M^4) part), shrink the register, lower resilience_level, or "
            f"raise measurement_budget= if you mean it.", self)


class MeasurementBudgetError(RuntimeError):
    """A measurement job refused before it was submitted; carries its plan."""

    def __init__(self, message: str, plan: MeasurementPlan):
        super().__init__(message)
        #: The :class:`MeasurementPlan` that was refused.
        self.plan = plan


def _resilience_multipliers(provider) -> tuple[int, int]:
    """``(zne_noise_factors, twirling_randomizations)`` of a provider's options.

    Both multiply the number of circuits the processor runs, so both belong in
    a size estimate; both are read defensively, because the options object is
    whatever the caller handed to Qiskit Runtime.
    """
    options = getattr(provider, "estimator_options", None) or {}
    if not isinstance(options, dict):                 # an Options dataclass
        options = {k: getattr(options, k) for k in ("resilience_level", "zne",
                                                    "twirling")
                   if getattr(options, k, None) is not None}

    def _get(section, key, default):
        block = options.get(section)
        if isinstance(block, dict):
            return block.get(key, default)
        return getattr(block, key, default) if block is not None else default

    level = int(options.get("resilience_level", 0) or 0)
    # Runtime's defaults: ZNE is on from level 2, gate twirling from level 1.
    factors = _get("zne", "noise_factors", None)
    noise_factors = len(factors) if factors else (3 if level >= 2 else 1)
    twirls = _get("twirling", "num_randomizations", None)
    if twirls in (None, "auto"):
        twirls = 32 if level >= 1 else 1
    enable = _get("twirling", "enable_gates", None)
    if enable is False:
        twirls = 1
    return max(int(noise_factors), 1), max(int(twirls), 1)


def measurement_plan(provider, n_qubits: int, labels, hamiltonian=None,
                     includes_rdms: bool = False, isa_circuit=None,
                     jobs: int = 1,
                     factorized_bases: int | None = None,
                     scheme: str = "qwc", bases: int | None = None,
                     observables: int | None = None,
                     one_norm: float | None = None) -> MeasurementPlan:
    """Cost of measuring ``labels`` for one state on ``provider``.

    ``labels`` are the Pauli strings that will be submitted; ``hamiltonian``
    (optional) supplies the coefficient 1-norm behind the shot-noise bound,
    ``isa_circuit`` a transpiled circuit whose two-qubit count and depth decide
    the fidelity estimate, and ``factorized_bases`` the number of bases a
    double-factorized measurement would need instead
    (:mod:`mandacaru.backends.factorization`).
    """
    labels = [str(label) for label in labels]
    identity = "I" * n_qubits
    payload = [label for label in labels if label != identity]
    if bases is None:
        bases = len(_greedy_bases(payload)) if payload else 0

    if one_norm is None:
        one_norm = 0.0
        if hamiltonian is not None:
            one_norm = sum(abs(complex(c)) for label, c
                           in hamiltonian.simplify().terms.items()
                           if set(label) != {"I"})

    two_q = depth = None
    if isa_circuit is not None:
        counts = isa_circuit.count_ops()
        two_q = int(sum(v for k, v in counts.items()
                        if k in ("cz", "cx", "ecr", "cnot")))
        depth = int(isa_circuit.depth())

    noise_factors, twirls = _resilience_multipliers(provider)
    return MeasurementPlan(
        n_qubits=int(n_qubits),
        observables=len(labels) if observables is None else int(observables),
        bases=int(bases),
        shots_per_basis=int(getattr(provider, "shots", 0) or 0),
        noise_factors=noise_factors, twirls=twirls,
        two_qubit_gates=two_q, circuit_depth=depth,
        one_norm_hartree=float(one_norm),
        device=str(getattr(provider, "device_spec", provider)),
        includes_rdms=bool(includes_rdms), jobs=int(jobs),
        scheme=str(scheme),
        factorized_bases=(None if factorized_bases is None
                          else int(factorized_bases)))


def _greedy_bases(labels) -> list[str]:
    """Greedy QWC bases covering ``labels`` -- the count, done in NumPy.

    :func:`qubit_wise_commuting_groups` returns the groups *and their terms*,
    which is what a shot-based energy needs; a plan needs only how many there
    are, and at 10^5 labels the per-term Python loop over groups is the slow
    part.  Comparing a label against every open basis at once, as rows of a
    byte array, turns that inner loop into one vectorized test.
    """
    labels = sorted(labels)
    if not labels:
        return []
    width = len(labels[0])
    rows = np.frombuffer("".join(labels).encode(), dtype=np.uint8)
    rows = rows.reshape(len(labels), width)
    ident = np.uint8(ord("I"))
    bases = np.empty((0, width), dtype=np.uint8)
    for row in rows:
        if len(bases):
            free = (bases == ident) | (row == ident)
            fits = np.flatnonzero((free | (bases == row)).all(axis=1))
        else:
            fits = ()
        if len(fits):
            g = fits[0]
            bases[g] = np.where(bases[g] == ident, row, bases[g])
        else:
            bases = np.vstack([bases, row])
    return ["".join(chr(c) for c in row) for row in bases]


def chunk_labels_by_basis(labels, max_bases: int) -> list[list[str]]:
    """Split ``labels`` into submissions of at most ``max_bases`` QWC bases.

    A chunk is a whole number of measurement bases, never a basis cut in half:
    the point of the grouping is that one circuit answers for every label in
    its group, so splitting a group would pay for the same circuit twice.

    ``max_bases <= 0`` (or a value that covers everything) returns a single
    chunk, which is the unsplit submission.
    """
    labels = [str(label) for label in labels]
    if max_bases is None or max_bases <= 0 or not labels:
        return [labels] if labels else []
    bases = _greedy_bases(labels)
    if len(bases) <= max_bases:
        return [labels]
    # Assign each label to the first basis that covers it -- the same order
    # the bases were opened in, so the assignment matches the grouping.
    buckets: list[list[str]] = [[] for _ in bases]
    for label in labels:
        for index, basis in enumerate(bases):
            if all(x == "I" or x == y for x, y in zip(label, basis)):
                buckets[index].append(label)
                break
    chunks = []
    for start in range(0, len(buckets), max_bases):
        chunk = [l for bucket in buckets[start:start + max_bases]
                 for l in bucket]
        if chunk:
            chunks.append(chunk)
    return chunks


def planned_jobs(labels, max_bases: int | None) -> int:
    """How many submissions :func:`chunk_labels_by_basis` would produce."""
    if max_bases is None or max_bases <= 0:
        return 1
    return max(len(chunk_labels_by_basis(labels, max_bases)), 1)
