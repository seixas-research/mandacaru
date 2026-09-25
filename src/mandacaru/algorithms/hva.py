# -*- coding: utf-8 -*-
# file: algorithms/hva.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Fixed Hamiltonian variational ansatz through the ordinary VQE optimizer."""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np

from ..circuits.hva import HVAEvolution, HamiltonianVariationalAnsatz
from ..core.checkpoint import WavefunctionCheckpoint, fingerprint
from ..core.mapping import Fermion, PauliSum
from .dry_run import QubitEstimate
from .hartree_fock import UHFResult
from .mean_field import MeanFieldResult
from .vqe import VQE, VQEResult


def _same_fermion(first: Fermion, second: Fermion) -> bool:
    """Whether two orbital-basis Hamiltonians agree term by term."""
    if first.n_modes() != second.n_modes():
        return False
    keys = set(first.terms) | set(second.terms)
    return all(abs(first.terms.get(key, 0j)
                   - second.terms.get(key, 0j)) < 1e-9 for key in keys)


class HVA(VQE):
    """Optimize ordered Hamiltonian layers with the shared VQE machinery.

    Use through ``Mandacaru(method='hva', layers=...)``.  Geometry mode builds
    the normal MO-basis fermionic Hamiltonian.  Direct mode takes that
    ``Fermion`` together with ``num_particles`` and optionally
    ``n_spatial_orbitals``.  ``hva_groups`` selects a physical decomposition;
    the default is one-body and two-body Hamiltonian parts.

    ``evolution='exact'`` applies complete group exponentials locally.
    ``evolution='trotter'`` compiles first- or second-order product-formula
    blocks, enabling equivalent local and SDK simulator circuits.  The two
    evolution policies describe different finite-depth variational states.
    """

    citation_method = "hva"
    solver_label = "HVA"
    _default_sparse = "auto"
    _supports_tapering = True

    def __init__(self, hamiltonian: Fermion | None = None,
                 num_particles: tuple[int, int] | None = None,
                 n_spatial_orbitals: int | None = None, *, layers: int = 2,
                 hva_groups: tuple[Fermion, ...] | None = None,
                 grouping: str = "body_order",
                 evolution: str = "exact", order: int = 2, steps: int = 1,
                 seed_angle: float = 0.1,
                 reference: MeanFieldResult | None = None,
                 **driver_kwargs: object) -> None:
        """Configure a fixed HVA through Mandacaru's ordinary VQE pipeline.

        ``evolution='exact'`` uses complete sparse group exponentials.
        ``evolution='trotter'`` binds each logical group angle to ordered
        Pauli rotations with ``order`` 1 or 2 and ``steps`` repetitions.
        ``reference`` accepts a mean-field result whose Hamiltonian and
        particle counts must match this problem; a UHF result supplies its
        actual unrestricted Slater determinant.
        """
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
            raise ValueError("HVA layers must be a positive integer")
        policy = HVAEvolution(evolution, order, steps)
        if grouping not in ("body_order", "spin_resolved"):
            raise ValueError("HVA grouping must be 'body_order' or 'spin_resolved'")
        if hva_groups is not None and grouping != "body_order":
            raise ValueError("choose either hva_groups or a grouping preset")
        if not np.isfinite(seed_angle):
            raise ValueError("HVA seed_angle must be finite")
        unsupported = ("load_hamiltonian", "ansatz_builder", "shots")
        used = [name for name in unsupported if driver_kwargs.get(name)]
        if used:
            raise ValueError(
                "HVA requires its fermionic group decomposition and local "
                "optimization; it does not take "
                + ", ".join(f"{name}=" for name in used))
        if policy.mode == "exact" and driver_kwargs.get("execute_circuits"):
            raise ValueError("exact HVA group evolution has no generic circuit; "
                             "set evolution='trotter' for circuit execution")
        self.layers = int(layers)
        self.hva_groups = None if hva_groups is None else tuple(hva_groups)
        self.grouping = grouping
        self.policy = policy
        self.evolution = policy.mode
        self.order = int(policy.order)
        self.steps = int(policy.steps)
        self.seed_angle = float(seed_angle)
        if reference is not None:
            if not isinstance(reference, MeanFieldResult):
                raise TypeError("HVA reference must be a MeanFieldResult")
            if hamiltonian is None:
                hamiltonian = reference.fermion_hamiltonian
            elif not _same_fermion(hamiltonian, reference.fermion_hamiltonian):
                raise ValueError("HVA reference and Hamiltonian use different "
                                 "orbital-basis operators")
            if num_particles is None:
                num_particles = reference.num_particles
            elif tuple(num_particles) != reference.num_particles:
                raise ValueError("HVA reference particle counts disagree")
            if n_spatial_orbitals is None:
                n_spatial_orbitals = reference.n_spatial_orbitals
            elif n_spatial_orbitals != reference.n_spatial_orbitals:
                raise ValueError("HVA reference orbital count disagrees")
        self.reference = reference
        super().__init__(hamiltonian=None, ansatz=None, **driver_kwargs)
        if self.execute_circuits and self.evolution != "trotter":
            raise ValueError("HVA circuit execution needs evolution='trotter'")
        if hamiltonian is not None:
            if num_particles is None:
                raise ValueError("direct HVA needs num_particles=(n_alpha, n_beta)")
            orbitals = (hamiltonian.n_modes() // 2 if n_spatial_orbitals is None
                        else int(n_spatial_orbitals))
            if self.dry_run:
                self._dry_run_problem = (hamiltonian, num_particles, orbitals)
            else:
                self._configure(hamiltonian, num_particles, orbitals)
                self._built_from_hamiltonian = True
                self._maybe_write_references()
        elif num_particles is not None or n_spatial_orbitals is not None:
            raise ValueError("num_particles and n_spatial_orbitals need a direct "
                             "fermionic hamiltonian=")

    def _configure(self, hamiltonian: Fermion,
                   num_particles: tuple[int, int], n_orbitals: int) -> None:
        """Build the fixed HVA ansatz and let VQE map the same Hamiltonian."""
        if not isinstance(hamiltonian, Fermion):
            raise TypeError("HVA needs a Fermion Hamiltonian so its physical "
                            "one- and two-body groups are available")
        if hamiltonian.n_modes() != 2 * n_orbitals:
            raise ValueError("HVA Hamiltonian modes and spatial orbitals disagree")
        coefficients = None
        if self.reference is not None:
            if not _same_fermion(hamiltonian,
                                 self.reference.fermion_hamiltonian):
                raise ValueError("HVA reference and Hamiltonian use different "
                                 "orbital-basis operators")
            if tuple(num_particles) != self.reference.num_particles:
                raise ValueError("HVA reference particle counts disagree")
            if isinstance(self.reference.scf, UHFResult):
                scf = self.reference.scf
                orbitals = self.reference.model_orbitals
                coefficients = (
                    orbitals.conj().T
                    @ scf.mo_coefficients_alpha[:, :num_particles[0]],
                    orbitals.conj().T
                    @ scf.mo_coefficients_beta[:, :num_particles[1]])
        ansatz = HamiltonianVariationalAnsatz(
            hamiltonian, num_particles, mapping=self.mapping,
            layers=self.layers, groups=self.hva_groups,
            grouping=self.grouping,
            evolution=self.policy,
            taper=self.taper, provider=self.ansatz_provider(),
            occupied_coefficients=coefficients)
        self._preset_ansatz = ansatz
        self._taper_info = ansatz.taper_info
        if self._taper_info is None:
            if self.taper:
                warnings.warn(
                    "taper=True found no Z2 symmetry in the HVA Hamiltonian; "
                    "the run continues untapered", RuntimeWarning,
                    stacklevel=2)
            super()._configure(hamiltonian, num_particles, n_orbitals)
            return
        self.ansatz = ansatz
        self.num_particles = tuple(int(n) for n in num_particles)
        self.n_spatial_orbitals = int(n_orbitals)
        self._materialize_hamiltonian(self._taper_info.hamiltonian,
                                      ansatz.n_qubits)
        self._maybe_save_hamiltonian(self.num_particles, n_orbitals)
        self._maybe_dump_hamiltonian(self.num_particles, n_orbitals)
        self._configured = True

    def ansatz_problem(self, theta: np.ndarray | None = None
                       ) -> tuple[int, list[int], list[PauliSum],
                                  np.ndarray, PauliSum]:
        """Return the compiled circuit problem for a product-formula HVA."""
        if self.evolution != "trotter":
            raise ValueError("exact HVA has no generic circuit representation; "
                             "use evolution='trotter'")
        if theta is None:
            theta = self.result.optimal_parameters
        return (self.ansatz.n_qubits, self.ansatz.reference_qubits(),
                self.ansatz.pauli_generators,
                self.ansatz.circuit_parameters(theta), self.hamiltonian)

    def _ansatz_details(self) -> tuple[str, ...]:
        """Report HVA group order, evolution semantics, and reference choice."""
        ansatz = self.ansatz
        policy = ("exact sparse groups" if self.evolution == "exact"
                  else f"product formula order {self.order}, {self.steps} step(s)")
        reference = ("actual UHF Slater determinant"
                     if self.reference is not None
                     and isinstance(self.reference.scf, UHFResult)
                     else "occupation determinant")
        lines = (
            f"HVA: {self.layers} layer(s)  |  {ansatz.grouping} grouping  |  "
            f"{policy}",
            f"group order: {', '.join(ansatz.group_labels)}",
            f"HVA reference: {reference}",
        )
        if self._taper_info is not None:
            lines += (self._taper_info.summary(),)
        return lines

    def _checkpoint_record(
            self, ansatz: HamiltonianVariationalAnsatz,
            parameters: np.ndarray, energy_ha: float | None,
            status: dict[str, object], labels: list[str] | None = None,
            kinds: list[str] | None = None) -> WavefunctionCheckpoint:
        """Store exactly the generator stream that prepares this HVA state."""
        logical = np.asarray(parameters, dtype=float).ravel()
        expanded = (np.asarray(list(ansatz.preparation_angles) + logical.tolist())
                    if self.evolution == "exact"
                    else ansatz.circuit_parameters(logical))
        if labels is None:
            labels = ([f"prepare_{i + 1}"
                       for i in range(len(ansatz.preparation_angles))]
                      + (list(ansatz.parameter_names)
                         if self.evolution == "exact"
                         else list(ansatz.rotation_names)))
        if kinds is None:
            kinds = ["hva"] * expanded.size
        record = super()._checkpoint_record(
            ansatz, expanded, energy_ha, status, labels=labels, kinds=kinds)
        record.status["hva_logical_parameters"] = logical.tolist()
        record.metadata["hva"] = {
            "evolution": self.evolution, "order": self.order,
            "steps": self.steps, "layers": self.layers,
            "grouping": ansatz.grouping,
            "groups": [fingerprint(group) for group in ansatz.mapped_groups],
            "hamiltonian": fingerprint(self.hamiltonian),
            "reference_preparation": [fingerprint(generator)
                                      for generator in ansatz.preparation_generators],
            "reference_angles": list(ansatz.preparation_angles),
            "taper_symmetries": list(self._taper_info.symmetries)
            if self._taper_info is not None else [],
            "taper_signs": list(self._taper_info.signs)
            if self._taper_info is not None else [],
        }
        return record

    def _load_resume(
            self, ansatz: HamiltonianVariationalAnsatz
    ) -> WavefunctionCheckpoint | None:
        """Validate HVA physics, then restore logical rather than gate angles."""
        record = super()._load_resume(ansatz)
        if record is None:
            return None
        expected = {
            "evolution": self.evolution, "order": self.order,
            "steps": self.steps, "layers": self.layers,
            "grouping": ansatz.grouping,
            "groups": [fingerprint(group) for group in ansatz.mapped_groups],
            "hamiltonian": fingerprint(self.hamiltonian),
            "reference_preparation": [fingerprint(generator)
                                      for generator in ansatz.preparation_generators],
            "reference_angles": list(ansatz.preparation_angles),
            "taper_symmetries": list(self._taper_info.symmetries)
            if self._taper_info is not None else [],
            "taper_signs": list(self._taper_info.signs)
            if self._taper_info is not None else [],
        }
        if record.method != "hva" or record.metadata.get("hva") != expected:
            raise ValueError("HVA checkpoint differs in Hamiltonian, groups, "
                             "evolution policy, reference sector, or layers")
        logical = np.asarray(record.status.get("hva_logical_parameters"),
                             dtype=float).ravel()
        if logical.size != ansatz.num_parameters:
            raise ValueError("HVA checkpoint has the wrong logical angle count")
        if record.num_particles != self.num_particles \
                or record.n_spatial_orbitals != self.n_spatial_orbitals:
            raise ValueError("HVA checkpoint particle or orbital counts differ")
        saved = (np.asarray(list(ansatz.preparation_angles) + logical.tolist())
                 if self.evolution == "exact"
                 else ansatz.circuit_parameters(logical))
        if not np.allclose(record.parameters, saved, atol=1e-12, rtol=0):
            raise ValueError("HVA checkpoint gate angles disagree with its "
                             "logical parameters")
        record.parameters = logical
        return record

    def _check_resumed_generators(
            self, record: WavefunctionCheckpoint) -> None:
        """Reject a checkpoint whose ordered HVA operators changed."""
        mine = [generator.simplify().terms
                for generator in self.ansatz.pauli_generators]
        theirs = [generator.simplify().terms for generator in record.generators]
        if len(mine) != len(theirs) or any(
                a.keys() != b.keys()
                or any(abs(a[key] - b[key]) > 1e-10 for key in a)
                for a, b in zip(mine, theirs)):
            raise ValueError("HVA checkpoint has different ordered generators")

    def _deflated_ground(
            self, states: list[np.ndarray], beta: float, *,
            state_index: int = 0,
            initial_parameters: np.ndarray | None = None,
            restarts: int = 1, seed: int = 0
    ) -> tuple[float, np.ndarray, int, None]:
        """Seed fixed-layer deflation away from the zero-angle stationary point."""
        if initial_parameters is None:
            initial_parameters = np.full(self.ansatz.num_parameters,
                                         self.seed_angle)
        return super()._deflated_ground(
            states, beta, state_index=state_index,
            initial_parameters=initial_parameters, restarts=restarts, seed=seed)

    def run(self, initial_parameters: np.ndarray | None = None
            ) -> VQEResult | QubitEstimate:
        """Optimize from nonzero seed angles unless angles were supplied.

        For real Hamiltonians and a real Hartree-Fock determinant, the energy
        derivatives of ``exp(-i theta H_j)`` vanish at all-zero angles.  A
        small deterministic seed moves the fixed HVA off that stationary
        point while preserving reproducibility.
        """
        if initial_parameters is None and self._configured \
                and self.resume_path is None:
            initial_parameters = np.full(self.ansatz.num_parameters,
                                         self.seed_angle)
        return super().run(initial_parameters=initial_parameters)
