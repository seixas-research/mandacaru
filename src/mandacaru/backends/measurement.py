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


#: Instructions that occupy wires without being gates, so they never count.
NON_GATE_INSTRUCTIONS = ("barrier", "measure", "reset", "delay", "snapshot")


def two_qubit_gate_count(circuit) -> int:
    """Two-qubit gates in a compiled circuit, counted by **arity**.

    Counting by name was the previous approach and it does not survive a change
    of instruction set architecture.  The list in use was
    ``("cz", "cx", "ecr", "cnot")``, which is right for an Eagle or a Heron
    compiled the ordinary way and **silently reports zero** for a Heron compiled
    with fractional gates, whose entangler is ``rzz``: a deep circuit would then
    come back with a perfect expected fidelity, which is the most misleading
    number this module could produce.

    An instruction acting on two wires is a two-qubit gate whatever it is
    called, so that is what is counted.  ``swap`` is included and counted once,
    which understates its cost (three entanglers) -- it does not survive
    transpilation to an IBM basis, and a count that is honest about its unit is
    better than a name list that can miss an entangler entirely.
    """
    total = 0
    for instruction in getattr(circuit, "data", ()):
        operation = getattr(instruction, "operation", None)
        name = str(getattr(operation, "name", "")).lower()
        if name in NON_GATE_INSTRUCTIONS:
            continue
        if len(getattr(instruction, "qubits", ())) == 2:
            total += 1
    return total


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
    which needs no measurement and is added to the energy as a constant.  With
    ``drop_identity=False`` the all-``I`` term is placed in a group instead
    and ``identity`` is ``0``: the constant is counted exactly once either way,
    so :func:`energy_from_group_counts` gives the same energy for both.

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
        identity = 0.0

    labels = [label for label, _c in terms]
    coefficients = dict(terms)
    groups = [(basis, [(labels[i], coefficients[labels[i]]) for i in members])
              for basis, members in qwc_partition(
                  labels, weights=[abs(c) for _l, c in terms])]
    return groups, identity


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
#: exact; a 24-qubit PAW-LCAO-TZP ansatz transpiles to ~2,200 two-qubit gates, i.e.
#: a fidelity of 1e-3).
FIDELITY_WARNING = 0.1

#: Default ceilings for :meth:`MeasurementPlan.check`.  They exist to stop the
#: job that motivated them: LiH/PAW-LCAO-TZP submitted 97,980 observables in ~21,000
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
    #: Shots each twirling randomization runs; the randomizations share
    #: :attr:`shots_per_basis` between them (Runtime's ``auto`` rule).
    shots_per_twirl: int | None = None
    two_qubit_gates: int | None = None
    circuit_depth: int | None = None
    one_norm_hartree: float = 0.0
    device: str = "?"
    includes_rdms: bool = False
    jobs: int = 1
    #: Bases the same energy would need under double factorization (``L + 1``),
    #: when the integrals were available to compute it.  Reported beside the
    #: qubit-wise count because the gap is the argument for implementing the
    #: basis-rotation circuits: measured on LiH/PAW-LCAO, 21 -> 4, 93 -> 11,
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
        """Shots across every circuit instance (an estimate: see
        :func:`resilience_multipliers`).

        Twirling randomizations divide a basis's shots among themselves, so
        each (basis, noise factor) pair costs ``twirls * shots_per_twirl``
        shots -- the requested count rounded up -- not ``twirls`` times it.
        """
        per = (self.shots_per_twirl if self.shots_per_twirl is not None
               else max(self.shots_per_basis, 0))
        per_instance_group = (max(self.twirls, 1) * per if self.twirls > 1
                              else max(self.shots_per_basis, 0))
        return (self.bases * max(self.noise_factors, 1) * per_instance_group)

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
            "shots_per_randomization": (self.shots_per_twirl
                                        if self.twirls > 1
                                        else self.shots_per_basis),
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
                f"size, mapping='parity_reduced', an active_space) or the "
                f"circuit "
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


#: Run-wide ceiling of a submission sequence on a **processor** (an ``ibm_*``
#: device, an AWS Braket QPU or simulator), on by default there: an
#: optimization on hardware submits a job per objective evaluation, so a
#: per-job check alone cannot keep a run inside a monthly quota (IBM's open
#: plan: 10 minutes).  ``jobs`` counts submissions, ``total_shots`` their
#: shots with the resilience multipliers; ``None`` switches one check off.
#: Only the job count is bounded by default: the size of each job is already
#: bounded by the measurement budget (:data:`DEFAULT_MEASUREMENT_BUDGET`), and
#: a default shot ceiling below what that budget admits would refuse a job
#: its own plan had just approved.
DEFAULT_HARDWARE_RUN_BUDGET = {"jobs": 250, "total_shots": None}
RUN_BUDGET_KEYS = tuple(DEFAULT_HARDWARE_RUN_BUDGET)


def resolve_run_budget(budget, hardware: bool) -> dict | None:
    """The run-wide limits a provider enforces, or ``None`` for none.

    ``None`` is the default: :data:`DEFAULT_HARDWARE_RUN_BUDGET` on a
    processor, nothing on a local or fake device (a rehearsal passes the same
    explicit budget to be refused the same way).  ``False`` switches it off;
    a dict sets :data:`RUN_BUDGET_KEYS`, keys left out keep their default.
    """
    if budget is False:
        return None
    if budget is None or budget is True:
        return (dict(DEFAULT_HARDWARE_RUN_BUDGET)
                if hardware or budget is True else None)
    if not isinstance(budget, dict):
        raise TypeError(
            f"run_budget must be None, False or a dict of {RUN_BUDGET_KEYS}, "
            f"got {type(budget).__name__}")
    unknown = sorted(set(budget) - set(RUN_BUDGET_KEYS))
    if unknown:
        raise ValueError(f"unknown run_budget key(s) {unknown}; use "
                         f"{RUN_BUDGET_KEYS}")
    return {**DEFAULT_HARDWARE_RUN_BUDGET, **budget}


def resolve_shots_per_energy(provider, hamiltonian) -> int:
    """Shots one energy of ``hamiltonian`` submits on ``provider`` (its
    ``shots_per_energy``, or 0 when it cannot say)."""
    method = getattr(provider, "shots_per_energy", None)
    return int(method(hamiltonian)) if callable(method) else 0


class RunBudgetExceeded(RuntimeError):
    """A submission refused because the run has spent its budget.

    Raised *before* the job is queued.  Whatever the run had reached -- the
    optimizer's best point, the checkpoint -- is kept by the caller.
    """


class MeasurementBudgetError(RuntimeError):
    """A measurement job refused before it was submitted; carries its plan."""

    def __init__(self, message: str, plan: MeasurementPlan):
        super().__init__(message)
        #: The :class:`MeasurementPlan` that was refused.
        self.plan = plan


#: Runtime's resilience level when the options leave it unset (the V2
#: Estimator's server default).
RUNTIME_DEFAULT_RESILIENCE_LEVEL = 1
#: ZNE's default noise factors, ``(1, 3, 5)``.
DEFAULT_ZNE_NOISE_FACTORS = 3


def _options_dict(options) -> dict:
    """``options`` as nested plain dicts, whatever the caller handed Runtime.

    An ``EstimatorOptions`` dataclass becomes a dict of its set fields, so a
    nested section (``resilience.zne``, ``twirling``) reads the same way as
    the dict spelling.  Runtime's ``Unset`` sentinel becomes a missing key.
    """
    if options is None:
        return {}
    if isinstance(options, dict):
        return {k: (_options_dict(v) if hasattr(v, "__dataclass_fields__")
                    else v) for k, v in options.items()}
    import dataclasses
    if dataclasses.is_dataclass(options):
        out = {}
        for f in dataclasses.fields(options):
            value = getattr(options, f.name)
            if type(value).__name__ == "UnsetType":
                continue
            out[f.name] = (_options_dict(value)
                           if dataclasses.is_dataclass(value)
                           or isinstance(value, dict) else value)
        return out
    return {}


def _section(options: dict, *path):
    block = options
    for key in path:
        block = block.get(key) if isinstance(block, dict) else None
        if block is None:
            return {}
    return block if isinstance(block, dict) else {}


def resilience_multipliers(provider) -> tuple[int, int, int]:
    """``(zne_noise_factors, randomizations, shots_per_randomization)``.

    Read from the provider's ``estimator_options`` with Runtime's own schema
    and precedence (qiskit-ibm-runtime ``EstimatorOptions``):

    * ZNE lives under ``resilience.zne``; an explicit
      ``resilience.zne_mitigation`` wins over the level default (on at level
      2 only), and the factor count is ``len(resilience.zne.noise_factors)``,
      three by default;
    * gate twirling (``twirling.enable_gates``, default on at level 2) and
      measurement twirling (``twirling.enable_measure``, default on from level
      1) both run randomized circuit instances;
    * the randomizations **share** the requested shots: with both counts on
      ``"auto"`` Runtime takes ``max(64, ceil(shots / 32))`` shots per
      randomization and ``ceil(shots / that)`` randomizations, and an explicit
      value of either fixes the other by ``ceil(shots / value)``.

    An unset ``resilience_level`` is Runtime's default (1) on an IBM
    processor; the local and fake estimators apply no mitigation, so there it
    counts as 0.  The result estimates what the request asks for; Runtime
    submits a *precision* and its learning circuits are extra, so billing is
    what the returned job metadata says.
    """
    options = _options_dict(getattr(provider, "estimator_options", None))
    shots = max(int(getattr(provider, "shots", 0) or 0), 0)
    level = options.get("resilience_level")
    if level is None:
        level = (RUNTIME_DEFAULT_RESILIENCE_LEVEL
                 if getattr(provider, "is_ibm_device", False) else 0)
    level = int(level)

    resilience = _section(options, "resilience")
    zne_on = resilience.get("zne_mitigation")
    if zne_on is None:
        zne_on = level >= 2
    factors = _section(options, "resilience", "zne").get("noise_factors")
    noise_factors = (len(factors) if factors else DEFAULT_ZNE_NOISE_FACTORS) \
        if zne_on else 1

    twirling = _section(options, "twirling")
    gates = twirling.get("enable_gates")
    gates = level >= 2 if gates is None else bool(gates)
    measure = twirling.get("enable_measure")
    measure = level >= 1 if measure is None else bool(measure)
    if not (gates or measure) or shots <= 0:
        return max(int(noise_factors), 1), 1, shots
    count = twirling.get("num_randomizations", "auto")
    per = twirling.get("shots_per_randomization", "auto")
    ceil = lambda a, b: -(-int(a) // int(b))                   # noqa: E731
    if count in (None, "auto") and per in (None, "auto"):
        per = max(64, ceil(shots, 32))
        count = ceil(shots, per)
    elif count in (None, "auto"):
        count = ceil(shots, per)
    elif per in (None, "auto"):
        per = ceil(shots, count)
    return max(int(noise_factors), 1), max(int(count), 1), max(int(per), 1)


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
        # The same partition the execution uses: weighted by the
        # Hamiltonian's coefficients when the labels are its terms (an
        # energy), lexical otherwise (RDM operators carry no weights).
        weights = None
        if hamiltonian is not None:
            terms = hamiltonian.simplify().terms
            if all(label in terms for label in payload):
                weights = [abs(complex(terms[label])) for label in payload]
        bases = len(_greedy_bases(payload, weights)) if payload else 0

    if one_norm is None:
        one_norm = 0.0
        if hamiltonian is not None:
            one_norm = sum(abs(complex(c)) for label, c
                           in hamiltonian.simplify().terms.items()
                           if set(label) != {"I"})

    two_q = depth = None
    if isa_circuit is not None:
        two_q = two_qubit_gate_count(isa_circuit)
        depth = int(isa_circuit.depth())

    noise_factors, twirls, per_twirl = resilience_multipliers(provider)
    return MeasurementPlan(
        n_qubits=int(n_qubits),
        observables=len(labels) if observables is None else int(observables),
        bases=int(bases),
        shots_per_basis=int(getattr(provider, "shots", 0) or 0),
        noise_factors=noise_factors, twirls=twirls,
        shots_per_twirl=per_twirl,
        two_qubit_gates=two_q, circuit_depth=depth,
        one_norm_hartree=float(one_norm),
        device=str(getattr(provider, "device_spec", provider)),
        includes_rdms=bool(includes_rdms), jobs=int(jobs),
        scheme=str(scheme),
        factorized_bases=(None if factorized_bases is None
                          else int(factorized_bases)))


def qwc_partition(labels, weights=None) -> list[tuple[str, list[int]]]:
    """The one qubit-wise-commuting partition every path uses.

    Returns ``[(basis, [index, ...]), ...]``: the measurement bases in the
    order they were opened and, for each, the positions in ``labels`` of the
    strings it measures.  Executing a Braket energy, chunking a Runtime
    submission and sizing a plan all read this, so the count a plan reports
    is the count that is executed.

    The policy is greedy first fit: labels are visited largest ``weight``
    first (the energetically important terms land in the earliest, largest
    groups), ties -- and every label when ``weights`` is ``None`` -- in
    lexical order, so the result does not depend on the input order.  A label
    joins the first open basis it commutes with qubit-wise; comparing against
    the merged basis is the same test as comparing against every member.
    Greedy coloring is not optimal (the minimum is NP-hard) but is fast: the
    per-label test is one vectorized comparison against every open basis.
    """
    labels = [str(label) for label in labels]
    if not labels:
        return []
    if weights is None:
        order = sorted(range(len(labels)), key=lambda i: labels[i])
    else:
        weights = [float(w) for w in weights]
        order = sorted(range(len(labels)),
                       key=lambda i: (-weights[i], labels[i]))
    width = len(labels[0])
    rows = np.frombuffer("".join(labels).encode(), dtype=np.uint8)
    rows = rows.reshape(len(labels), width)
    ident = np.uint8(ord("I"))
    bases = np.empty((max(len(labels), 1), width), dtype=np.uint8)
    open_count = 0
    members: list[list[int]] = []
    for i in order:
        row = rows[i]
        if open_count:
            block = bases[:open_count]
            fits = np.flatnonzero(((block == ident) | (row == ident)
                                   | (block == row)).all(axis=1))
        else:
            fits = ()
        if len(fits):
            g = int(fits[0])
            bases[g] = np.where(bases[g] == ident, row, bases[g])
            members[g].append(i)
        else:
            bases[open_count] = row
            members.append([i])
            open_count += 1
    return [("".join(chr(c) for c in bases[g]), members[g])
            for g in range(open_count)]


def _greedy_bases(labels, weights=None) -> list[str]:
    """The bases of :func:`qwc_partition` -- the count a plan needs."""
    return [basis for basis, _members in qwc_partition(labels, weights)]


def chunk_labels_by_basis(labels, max_bases: int,
                          weights=None) -> list[list[str]]:
    """Split ``labels`` into submissions of at most ``max_bases`` QWC bases.

    A chunk is a whole number of measurement bases of :func:`qwc_partition`,
    never a basis cut in half: the point of the grouping is that one circuit
    answers for every label in its group, so splitting a group would pay for
    the same circuit twice.

    ``max_bases <= 0`` (or a value that covers everything) returns a single
    chunk, which is the unsplit submission.
    """
    labels = [str(label) for label in labels]
    if max_bases is None or max_bases <= 0 or not labels:
        return [labels] if labels else []
    partition = qwc_partition(labels, weights)
    if len(partition) <= max_bases:
        return [labels]
    chunks = []
    for start in range(0, len(partition), max_bases):
        chunk = [labels[i] for _basis, members
                 in partition[start:start + max_bases] for i in members]
        if chunk:
            chunks.append(chunk)
    return chunks


def planned_jobs(labels, max_bases: int | None, weights=None) -> int:
    """How many submissions :func:`chunk_labels_by_basis` would produce."""
    if max_bases is None or max_bases <= 0:
        return 1
    return max(len(chunk_labels_by_basis(labels, max_bases, weights)), 1)
