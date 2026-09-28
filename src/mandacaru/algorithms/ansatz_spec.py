# -*- coding: utf-8 -*-
# file: algorithms/ansatz_spec.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The ``ansatz=`` option of ``method="vqe"``: named circuit templates.

A fixed-ansatz VQE is one solver whatever circuit it optimizes; the circuit
is an option, spelled like ``basis`` and ``optimizer``:

* a name -- ``"uccsd"`` (the default) or ``"hva"``;
* a dictionary giving the name and that template's options --
  ``{"name": "hva", "layers": 3, "evolution": "trotter", "steps": 2}``;
* a pre-built ansatz object, used as it is.

:func:`resolve_ansatz` validates a name or dictionary when the calculator is
constructed, and :func:`build_ansatz` builds the circuit once the solver has
its Hamiltonian and particle counts -- the Hamiltonian variational ansatz
needs the fermionic operator itself, since its layers are that operator's
one- and two-body parts (:mod:`mandacaru.circuits.hva`).

Options per template:

``"uccsd"`` (:class:`~mandacaru.circuits.ansatz.UCCSD`)
    ``include_singles`` (default ``True``) and ``trotter`` (default: ``True``
    exactly when circuits are executed or shots are measured, since a circuit
    prepares the ordered product of exponentials).
``"hva"`` (:class:`~mandacaru.circuits.hva.HamiltonianVariationalAnsatz`)
    ``layers`` (default 2), ``grouping`` (``"body_order"`` or
    ``"spin_resolved"``), ``groups`` (an explicit tuple of ``Fermion`` groups
    that sum to the Hamiltonian), ``evolution`` (``"exact"`` or
    ``"trotter"``), ``order`` and ``steps`` (product formula), ``seed_angle``
    (starting value of every angle) and ``reference`` (a mean-field result:
    its Hamiltonian and particle counts become the problem, and a UHF result
    prepares its actual unrestricted determinant).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..core.mapping import Fermion

#: The circuit templates ``ansatz=`` names, and the default.
ANSATZ_NAMES = ("uccsd", "hva")
DEFAULT_ANSATZ = "uccsd"

#: The options each template accepts in the dictionary form.
ANSATZ_OPTIONS = {
    "uccsd": ("include_singles", "trotter"),
    "hva": ("layers", "grouping", "groups", "evolution", "order", "steps",
            "seed_angle", "reference"),
}

#: Solver options the Hamiltonian variational ansatz cannot run with: it
#: needs its fermionic group decomposition, which a cached qubit Hamiltonian
#: does not carry, and it builds its own circuit.
HVA_REFUSED_OPTIONS = ("load_hamiltonian", "ansatz_builder", "shots")


@dataclass(frozen=True)
class AnsatzSpec:
    """A resolved ``ansatz=`` name or dictionary."""

    name: str
    options: Mapping = field(default_factory=dict)

    @property
    def evolution(self) -> str | None:
        """The HVA evolution policy, ``None`` for any other template."""
        if self.name != "hva":
            return None
        return str(self.options.get("evolution", "exact"))


def resolve_ansatz(spec) -> AnsatzSpec | None:
    """Normalize ``ansatz=`` to an :class:`AnsatzSpec` (``None``: an object).

    ``None`` is the default template.  A name or dictionary is validated here,
    at construction -- an unknown template or option fails before any
    integral is computed.  Anything else is taken to be a pre-built ansatz.
    """
    if spec is None:
        return AnsatzSpec(DEFAULT_ANSATZ)
    if isinstance(spec, AnsatzSpec):
        return spec
    if isinstance(spec, str):
        name, options = spec, {}
    elif isinstance(spec, Mapping):
        options = dict(spec)
        name = options.pop("name", None)
        if name is None:
            raise ValueError(
                "an ansatz dictionary needs a 'name', e.g. "
                "{'name': 'hva', 'layers': 3}")
    else:
        return None
    name = str(name).strip().lower()
    if name not in ANSATZ_NAMES:
        raise ValueError(f"unknown ansatz {name!r}; use one of "
                         f"{ANSATZ_NAMES} or pass an ansatz object")
    unknown = sorted(set(options) - set(ANSATZ_OPTIONS[name]))
    if unknown:
        raise ValueError(f"the {name!r} ansatz does not take {unknown}; "
                         f"its options are {list(ANSATZ_OPTIONS[name])}")
    if name == "hva":
        _check_hva_options(options)
    return AnsatzSpec(name, options)


def _check_hva_options(options: Mapping) -> None:
    """Validate the HVA options that need no Hamiltonian to check."""
    from ..circuits.hva import HVAEvolution
    from .mean_field import MeanFieldResult

    layers = options.get("layers", 2)
    if isinstance(layers, bool) or not isinstance(layers, Integral) \
            or layers < 1:
        raise ValueError("HVA layers must be a positive integer")
    grouping = options.get("grouping", "body_order")
    if grouping not in ("body_order", "spin_resolved"):
        raise ValueError("HVA grouping must be 'body_order' or 'spin_resolved'")
    if options.get("groups") is not None and grouping != "body_order":
        raise ValueError("choose either HVA groups or a grouping preset")
    HVAEvolution(options.get("evolution", "exact"), options.get("order", 2),
                 options.get("steps", 1))
    if not np.isfinite(options.get("seed_angle", 0.0)):
        raise ValueError("HVA seed_angle must be finite")
    reference = options.get("reference")
    if reference is not None and not isinstance(reference, MeanFieldResult):
        raise TypeError("the HVA reference must be a mean-field result "
                        "(Mandacaru(method='rhf' or 'uhf').result)")


def ansatz_name(ansatz) -> str | None:
    """The template name of a built ansatz, ``None`` for a custom class."""
    from ..circuits.ansatz import UCCSD
    from ..circuits.hva import HamiltonianVariationalAnsatz

    if isinstance(ansatz, HamiltonianVariationalAnsatz):
        return "hva"
    if isinstance(ansatz, UCCSD):
        return "uccsd"
    return None


def _same_fermion(first: Fermion, second: Fermion) -> bool:
    """Whether two orbital-basis Hamiltonians agree term by term."""
    if first.n_modes() != second.n_modes():
        return False
    keys = set(first.terms) | set(second.terms)
    return all(abs(first.terms.get(key, 0j) - second.terms.get(key, 0j)) < 1e-9
               for key in keys)


def reference_problem(spec: AnsatzSpec, hamiltonian=None, num_particles=None,
                      n_spatial_orbitals=None):
    """``(hamiltonian, num_particles, n_spatial_orbitals)`` with an HVA
    ``reference`` filled in and checked.

    A mean-field reference is a complete problem: its orbital-basis
    Hamiltonian and particle counts are used when not given, and must agree
    with them when they are.
    """
    reference = spec.options.get("reference") if spec.name == "hva" else None
    if reference is None:
        return hamiltonian, num_particles, n_spatial_orbitals
    if hamiltonian is None:
        hamiltonian = reference.fermion_hamiltonian
    elif not isinstance(hamiltonian, Fermion) or not _same_fermion(
            hamiltonian, reference.fermion_hamiltonian):
        raise ValueError("the HVA reference and the Hamiltonian use different "
                         "orbital-basis operators")
    if num_particles is None:
        num_particles = reference.num_particles
    elif tuple(num_particles) != tuple(reference.num_particles):
        raise ValueError("the HVA reference's particle counts disagree")
    if n_spatial_orbitals is None:
        n_spatial_orbitals = reference.n_spatial_orbitals
    elif n_spatial_orbitals != reference.n_spatial_orbitals:
        raise ValueError("the HVA reference's orbital count disagrees")
    return hamiltonian, num_particles, n_spatial_orbitals


def build_ansatz(spec: AnsatzSpec, *, hamiltonian, num_particles,
                 n_spatial_orbitals: int, mapping: str, taper: bool = False,
                 provider=None, shots: int = 0):
    """Build the circuit ``spec`` names for this problem.

    ``provider`` is the circuit provider the ansatz prepares its states on
    (``None`` for the internal state vector); ``taper`` asks the HVA to apply
    the Z2 sector its Hamiltonian groups conserve.
    """
    options = dict(spec.options)
    if spec.name == "uccsd":
        from ..circuits import UCCSD
        # A circuit realizes the ordered product, so the state matches the
        # executed circuit whenever one is run.
        trotter = options.get("trotter", provider is not None or bool(shots))
        return UCCSD(n_spatial_orbitals, num_particles, mapping=mapping,
                     include_singles=options.get("include_singles", True),
                     trotter=trotter, provider=provider)
    return _build_hva(options, hamiltonian=hamiltonian,
                      num_particles=num_particles,
                      n_spatial_orbitals=n_spatial_orbitals, mapping=mapping,
                      taper=taper, provider=provider)


def _build_hva(options: dict, *, hamiltonian, num_particles,
               n_spatial_orbitals: int, mapping: str, taper: bool, provider):
    """The Hamiltonian variational ansatz of ``hamiltonian``."""
    from ..circuits.hva import DEFAULT_SEED_ANGLE, HamiltonianVariationalAnsatz
    from .hartree_fock import UHFResult

    if not isinstance(hamiltonian, Fermion):
        raise TypeError("the HVA needs a Fermion Hamiltonian: its layers are "
                        "that operator's one- and two-body groups")
    if hamiltonian.n_modes() != 2 * n_spatial_orbitals:
        raise ValueError("the HVA Hamiltonian's modes and spatial orbitals "
                         "disagree")
    reference = options.get("reference")
    coefficients = None
    if reference is not None:
        reference_problem(AnsatzSpec("hva", options), hamiltonian,
                          num_particles, n_spatial_orbitals)
        if isinstance(reference.scf, UHFResult):
            # The occupied UHF columns in the Hamiltonian's spatial basis:
            # a spin-resolved Givens network prepares that determinant.
            scf, orbitals = reference.scf, reference.model_orbitals
            coefficients = (
                orbitals.conj().T
                @ scf.mo_coefficients_alpha[:, :num_particles[0]],
                orbitals.conj().T
                @ scf.mo_coefficients_beta[:, :num_particles[1]])
    return HamiltonianVariationalAnsatz(
        hamiltonian, num_particles, mapping=mapping,
        layers=options.get("layers", 2), groups=options.get("groups"),
        grouping=options.get("grouping", "body_order"),
        evolution=options.get("evolution", "exact"),
        order=options.get("order", 2), steps=options.get("steps", 1),
        taper=taper, provider=provider, occupied_coefficients=coefficients,
        seed_angle=options.get("seed_angle", DEFAULT_SEED_ANGLE))
