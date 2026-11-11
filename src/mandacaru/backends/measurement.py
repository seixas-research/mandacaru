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


def qubit_wise_commuting_groups(hamiltonian: PauliSum, drop_identity: bool=True, *, resources=None, context=None):
    """Partition ``hamiltonian`` into QWC groups; return ``(groups, identity)``.

    ``groups`` is a list of ``(basis, [(pauli, coefficient), ...])`` -- one
    measurement circuit per entry, ``basis`` being the string of single-qubit
    bases to rotate into.  ``identity`` is the coefficient of the all-``I`` term,
    which needs no measurement and is added to the energy as a constant.  With
    ``drop_identity=False`` the all-``I`` term is placed in a group instead
    and ``identity`` is ``0``: the constant is counted exactly once either way.

    The partition is greedy, largest-coefficient-first (so the terms that
    dominate the energy land in the earliest, largest groups).  Greedy coloring
    is not optimal -- finding the minimum number of groups is NP-hard -- but it
    is fast and close enough in practice.
    """
    native = _context(resources, context)
    terms = [(label, complex(c).real, complex(c).imag) for label, c in hamiltonian.terms.items()]
    raw, identity = native.qwc_hamiltonian(terms, drop_identity=bool(drop_identity))
    groups = [(basis, [(label, complex(re, im)) for label, re, im in members])
              for basis, members in raw]
    return (groups, identity)


def pauli_expectation_from_counts(label: str, counts: dict, *, resources=None, context=None) -> float:
    """``<P>`` for one Pauli string from bit-string counts of its own basis.

    ``counts`` maps a measured bit-string (character ``k`` is qubit ``k``, the
    Mandacaru convention) to its number of occurrences, taken in a basis that
    diagonalizes ``label``.  Each shot contributes :math:`(-1)^{\\text{parity}}`
    over the qubits where ``label`` acts non-trivially.
    """
    return _context(resources, context).count_expectations(
        [label], [(str(k), float(v)) for k, v in counts.items()])[0]


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
    from .._native import shot_noise
    return shot_noise([(label, complex(c).real, complex(c).imag)
                       for label, c in hamiltonian.terms.items()], max(int(shots), 0))


#: Default ceilings of ``measurement_budget=``.  They exist to stop the
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


#: Run-wide ceiling of a submission sequence on a **processor** (an ``ibm_*``
#: device, an AWS Braket QPU or simulator), on by default there: an
#: optimization on hardware submits a job per objective evaluation, so a
#: per-job check alone cannot keep a run inside a monthly quota (IBM's open
#: plan: 10 minutes).  ``jobs`` counts submissions, ``total_shots`` their
#: shots with the resilience multipliers; ``None`` switches one check off.
#: Only the job count is bounded by default: the size of each job is already
#: bounded by the measurement budget (:data:`DEFAULT_MEASUREMENT_BUDGET`), and
#: a default shot ceiling below what that budget admits would refuse a job
#: the measurement budget had just approved.
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


class RunBudgetExceeded(RuntimeError):
    """A submission refused because the run has spent its budget.

    Raised *before* the job is queued.  Whatever the run had reached -- the
    optimizer's best point, the checkpoint -- is kept by the caller.
    """


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
    from .._native import twirled_instances
    automatic = lambda value: None if value in (None, "auto") else int(value)  # noqa: E731
    count, per = twirled_instances(
        shots, automatic(twirling.get("num_randomizations", "auto")),
        automatic(twirling.get("shots_per_randomization", "auto")))
    return max(int(noise_factors), 1), count, per


def _context(resources, context):
    """The caller's native context, or one built from ``resources``."""
    from .._native import Context
    return Context(**dict(resources or {})) if context is None else context


def qwc_partition(labels, weights=None, *, resources=None, context=None) -> list[tuple[str, list[int]]]:
    """The one qubit-wise-commuting partition every path uses.

    Returns ``[(basis, [index, ...]), ...]``: the measurement bases in the
    order they were opened and, for each, the positions in ``labels`` of the
    strings it measures.  Executing a Braket energy and chunking a Runtime
    submission both read this, so the count reported is the count executed.

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
    return _context(resources, context).qwc_groups(
        labels, None if weights is None else [float(w) for w in weights])


def chunk_labels_by_basis(labels, max_bases: int, weights=None, *, resources=None, context=None) -> list[list[str]]:
    """Split ``labels`` into submissions of at most ``max_bases`` QWC bases.

    A chunk is a whole number of measurement bases of :func:`qwc_partition`,
    never a basis cut in half: the point of the grouping is that one circuit
    answers for every label in its group, so splitting a group would pay for
    the same circuit twice.

    ``max_bases <= 0`` (or a value that covers everything) returns a single
    chunk, which is the unsplit submission.
    """
    labels = [str(label) for label in labels]
    native = _context(resources, context)
    limit = None if max_bases is None or max_bases <= 0 else int(max_bases)
    chunks = native.basis_chunks(labels, limit, None if weights is None else [float(w) for w in weights])
    return [[labels[i] for i in chunk] for chunk in chunks]
