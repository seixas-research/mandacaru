# -*- coding: utf-8 -*-
# file: circuits/hva.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Hamiltonian variational ansatz with fixed, symmetry-respecting layers.

The default decomposition is the full one-body and two-body parts of a
number-conserving fermionic Hamiltonian.  One layer applies
``exp(-i theta_1 H_1)`` followed by ``exp(-i theta_2 H_2)``.  Custom groups
can describe a different physical decomposition, provided their sum is the
nonconstant Hamiltonian.  Sparse exponential actions prepare exact local
state vectors; an explicit product formula compiles the same fixed layers
into provider-independent Pauli rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import null_space
from scipy.sparse.linalg import expm_multiply

from ..backends.factorization import givens_decomposition
from ..core.mapping import (Fermion, FermionTerm, PauliSum,
                            reencode_pauli, reference_qubit_bits,
                            resolve_mapping)
from ..core.sector import pauli_masks
from ..core.tapering import SymmetryLeakError, TaperedRegister, taper_problem


@dataclass(frozen=True, slots=True)
class HVAEvolution:
    """Immutable exact or finite product-formula evolution policy.

    ``order`` and ``steps`` are retained in exact mode as explicit metadata,
    but affect the state only when ``mode='trotter'``.  The policy is written
    into checkpoints so a resumed run cannot reinterpret the same angles.
    """

    mode: str = "exact"
    order: int = 2
    steps: int = 1

    def __post_init__(self) -> None:
        """Reject invalid modes and counts before a Hamiltonian is built."""
        if self.mode not in ("exact", "trotter"):
            raise ValueError("HVA evolution must be 'exact' or 'trotter'")
        if isinstance(self.order, bool) or self.order not in (1, 2):
            raise ValueError("HVA product-formula order must be 1 or 2")
        if isinstance(self.steps, bool) or not isinstance(self.steps, Integral) \
                or self.steps < 1:
            raise ValueError("HVA product-formula steps must be a positive integer")


def _group_terms(group: Fermion, n_modes: int,
                 separate_spins: bool) -> None:
    """Validate finite, number-conserving terms on the declared mode space."""
    half = n_modes // 2
    for term, coefficient in group.terms.items():
        if not term:
            raise ValueError("HVA groups omit scalar terms (global phases)")
        if not (np.isfinite(coefficient.real) and np.isfinite(coefficient.imag)):
            raise ValueError("HVA group coefficients must be finite")
        if any(mode < 0 or mode >= n_modes for mode, _ in term):
            raise ValueError("an HVA group term lies outside the mode space")
        if sum(bool(dagger) for _, dagger in term) * 2 != len(term):
            raise ValueError("HVA groups must conserve particle number")
        if separate_spins:
            alpha = sum(1 if dagger else -1 for mode, dagger in term
                        if mode < half)
            if alpha:
                raise ValueError("parity_reduced HVA groups must conserve "
                                 "alpha and beta particle numbers separately")


def hamiltonian_groups(
        hamiltonian: Fermion, groups: tuple[Fermion, ...] | None = None,
        *, separate_spins: bool = False,
        grouping: str = "body_order"
) -> tuple[Fermion, ...]:
    """Validate or build a number-conserving decomposition of ``hamiltonian``.

    Constants are omitted because they contribute only a global phase.  Each
    supplied group must preserve particle number and their termwise sum must
    equal every nonconstant term of the Hamiltonian.  ``body_order`` groups
    quadratic and quartic terms separately; ``spin_resolved`` also separates
    alpha, beta, and mixed-spin blocks.  Explicit groups retain their order.
    """
    if not isinstance(hamiltonian, Fermion):
        raise TypeError("HVA needs a fermionic Hamiltonian")
    n_modes = hamiltonian.n_modes()
    if grouping not in ("body_order", "spin_resolved"):
        raise ValueError("HVA grouping must be 'body_order' or 'spin_resolved'")
    if groups is not None and grouping != "body_order":
        raise ValueError("choose either explicit hva_groups or a grouping preset")
    _group_terms(Fermion({term: coeff for term, coeff in
                          hamiltonian.terms.items() if term}, n_modes=n_modes),
                 n_modes, separate_spins)
    if groups is None:
        by_key: dict[tuple[int, tuple[int, ...]],
                     dict[FermionTerm, complex]] = {}
        for term, coefficient in sorted(hamiltonian.terms.items()):
            if not term:
                continue
            if len(term) not in (2, 4):
                raise ValueError(
                    "default HVA groups support one- and two-body "
                    "number-conserving Hamiltonians; pass hva_groups= for "
                    "another decomposition")
            spins = tuple(sorted(int(mode >= n_modes // 2)
                                 for mode, dagger in term if dagger))
            if grouping == "spin_resolved":
                other = tuple(sorted(int(mode >= n_modes // 2)
                                     for mode, dagger in term if not dagger))
                if spins != other:
                    raise ValueError("spin_resolved HVA grouping needs each "
                                     "term to conserve both spin populations")
            key = (len(term), spins if grouping == "spin_resolved" else ())
            by_key.setdefault(key, {})[term] = coefficient
        groups = tuple(Fermion(by_key[key], n_modes=n_modes)
                       for key in sorted(by_key))
    else:
        groups = tuple(groups)
        if not groups:
            raise ValueError("hva_groups must contain at least one operator")

    combined: dict[FermionTerm, complex] = {}
    for group in groups:
        if not isinstance(group, Fermion):
            raise TypeError("every HVA group must be a Fermion operator")
        if group.n_modes() != n_modes:
            raise ValueError("every HVA group must act on all Hamiltonian modes")
        if not any(abs(coefficient) > 1e-12
                   for coefficient in group.terms.values()):
            raise ValueError("an HVA group must contain a nonzero operator")
        _group_terms(group, n_modes, separate_spins)
        for term, coefficient in group.terms.items():
            combined[term] = combined.get(term, 0j) + coefficient
    keys = set(combined) | {term for term in hamiltonian.terms if term}
    if any(abs(combined.get(term, 0j)
               - hamiltonian.terms.get(term, 0j)) > 1e-9 for term in keys):
        raise ValueError("HVA groups must sum to the nonconstant Hamiltonian")
    return groups


def slater_preparation(
        alpha_occupied: np.ndarray, beta_occupied: np.ndarray, *,
        mapping: str, num_particles: tuple[int, int],
        taper_info: TaperedRegister | None = None
) -> tuple[tuple[PauliSum, ...], tuple[float, ...]]:
    """Compile a real UHF determinant into spin-resolved Givens rotations.

    The input columns are occupied alpha/beta orbital coefficients in the
    shared, orthonormal spatial basis of the Hamiltonian.  The first occupied
    columns of a completed orthogonal matrix define the desired determinant;
    its remaining columns are arbitrary.  Reversed negative Givens rotations
    prepare those columns from the occupation-ordered reference.  Signs left
    by the decomposition change only the determinant's global phase.
    """
    alpha = np.asarray(alpha_occupied)
    beta = np.asarray(beta_occupied)
    if alpha.ndim != 2 or beta.ndim != 2 or alpha.shape[0] != beta.shape[0]:
        raise ValueError("UHF occupied orbitals need matching spatial dimensions")
    m = alpha.shape[0]
    if (alpha.shape[1], beta.shape[1]) != tuple(num_particles):
        raise ValueError("UHF occupied orbital counts do not match the problem")
    if max((float(np.max(np.abs(array.imag))) if array.size else 0.0
            for array in (alpha, beta))) > 1e-10:
        raise ValueError("circuit UHF preparation requires real orbitals")
    canonical = resolve_mapping(mapping)
    generators: list[PauliSum] = []
    angles: list[float] = []
    for offset, occupied in ((0, alpha.real), (m, beta.real)):
        if not np.allclose(occupied.T @ occupied,
                           np.eye(occupied.shape[1]), atol=1e-8):
            raise ValueError("UHF occupied orbitals are not orthonormal")
        complement = null_space(occupied.T)
        full = np.column_stack((occupied, complement))
        rotations, _signs = givens_decomposition(full)
        for p, theta in reversed(rotations):
            if abs(theta) <= 1e-12:
                continue
            first, second = offset + p, offset + p + 1
            fermion = Fermion({((first, True), (second, False)): 1.0,
                               ((second, True), (first, False)): -1.0},
                              n_modes=2*m)
            mapped = fermion.map_to_qubits(
                canonical, n_modes=2*m,
                num_particles=num_particles if canonical == "parity_reduced"
                else None)
            if taper_info is not None:
                mapped = taper_info.taper_operator(mapped)
            generators.append(mapped)
            angles.append(-float(theta))
    return tuple(generators), tuple(angles)


def _check_slater_taper_sector(
        coefficients: tuple[np.ndarray, np.ndarray],
        info: TaperedRegister) -> None:
    """Require a UHF determinant to occupy one complete Z2 symmetry sector.

    For a Slater determinant, the expectation of a diagonal one-body parity
    transformation is the determinant of its occupied-orbital overlap matrix.
    This tests sector membership without allocating a full qubit state vector.
    """
    for tau, sign in zip(info.symmetries, info.signs):
        diagonal = np.asarray([-1 if letter == "Z" else 1 for letter in tau])
        half = diagonal.size // 2
        expectation = 1.0 + 0j
        for occupied, block in zip(coefficients,
                                   (diagonal[:half], diagonal[half:])):
            orbitals = np.asarray(occupied)
            expectation *= np.linalg.det(
                orbitals.conj().T @ (block[:, None] * orbitals))
        if abs(expectation - sign) > 1e-8:
            raise ValueError(
                "the actual UHF reference spans multiple Z2 symmetry "
                "sectors; use taper=False or the occupation determinant")


class HamiltonianVariationalAnsatz:
    r"""Fixed product of exponentials of Hamiltonian groups.

    Parameters
    ----------
    hamiltonian : Fermion
        Number-conserving model in the same orbital basis as the reference.
    num_particles : tuple[int, int]
        Alpha and beta occupations of the reference determinant.
    mapping : str
        Fermion-to-qubit encoding, including ``parity_reduced``.
    layers : int
        Positive number of repetitions of the ordered group sequence (default
        two; one layer cannot correlate some symmetry-adapted HF references).
    groups : tuple[Fermion, ...], optional
        Explicit physical decomposition; defaults to one-body then two-body.
    grouping : {"body_order", "spin_resolved"}
        Built-in grouping preset when ``groups`` is not supplied.
    evolution : {"exact", "trotter"}
        Full sparse group exponentials or a finite product formula.
    order, steps : int
        Product-formula order (one or two) and positive step count.
    taper : bool
        Apply one discovered Jordan-Wigner Z2 sector to Hamiltonian groups and
        reference; reject a group that changes the sector.
    provider : CircuitProvider, optional
        Local circuit provider for ``state`` in product-formula mode.
    occupied_coefficients : tuple[ndarray, ndarray], optional
        Occupied alpha and beta UHF columns in the Hamiltonian's spatial basis.
        A spin-resolved Givens network prepares this actual Slater determinant.

    Notes
    -----
    ``evolution='exact'`` uses sparse exponential actions.  With
    ``evolution='trotter'``, the same deterministic single-Pauli rotation
    schedule is used by the local simulator and the circuit providers.
    """

    def __init__(self, hamiltonian: Fermion,
                 num_particles: tuple[int, int], *, mapping: str,
                 layers: int = 2,
                 groups: tuple[Fermion, ...] | None = None,
                 grouping: str = "body_order",
                 evolution: str | HVAEvolution = "exact", order: int = 2,
                 steps: int = 1,
                 taper: bool = False, provider: object | None = None,
                 occupied_coefficients: tuple[np.ndarray, np.ndarray] | None = None
                 ) -> None:
        """Validate the model, map fixed groups, and prepare its reference."""
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
            raise ValueError("HVA layers must be a positive integer")
        self.layers = int(layers)
        self.mapping = resolve_mapping(mapping)
        self.n_modes = hamiltonian.n_modes()
        if self.n_modes < 2 or self.n_modes % 2:
            raise ValueError("HVA needs an even number of spin orbitals")
        self.n_spatial_orbitals = self.n_modes // 2
        self.num_particles = tuple(int(n) for n in num_particles)
        if len(self.num_particles) != 2 or any(
                n < 0 or n > self.n_spatial_orbitals
                for n in self.num_particles):
            raise ValueError("HVA particle counts do not fit the orbital basis")
        self.n_qubits = self.n_modes - (
            2 if self.mapping == "parity_reduced" else 0)
        if isinstance(evolution, HVAEvolution):
            if order != 2 or steps != 1:
                raise ValueError("pass either an HVAEvolution policy or "
                                 "separate order/steps")
            policy = evolution
        else:
            policy = HVAEvolution(evolution, order, steps)
        if taper and self.mapping != "jordan_wigner":
            raise ValueError("taper=True needs mapping='jordan_wigner'")
        if provider is not None and policy.mode != "trotter":
            raise ValueError("HVA circuit execution needs evolution='trotter'")
        self.policy = policy
        self.evolution = policy.mode
        self.order = int(policy.order)
        self.steps = int(policy.steps)
        self.provider = provider
        self.preparation = "product"
        self.circuit_serializable = policy.mode == "trotter"
        self.grouping = grouping if groups is None else "custom"
        self.groups = hamiltonian_groups(
            hamiltonian, groups,
            separate_spins=self.mapping == "parity_reduced",
            grouping=grouping)
        if not self.groups:
            raise ValueError("HVA needs at least one nonconstant Hamiltonian group")
        self.group_labels = tuple(self._group_label(group, index)
                                  for index, group in enumerate(self.groups))
        mapped_groups: list[PauliSum] = []
        for group in self.groups:
            mapped = group.map_to_qubits(
                self.mapping, n_modes=self.n_modes,
                num_particles=self.num_particles if self.mapping == "parity_reduced"
                else None)
            self._check_hermitian(mapped)
            mapped_groups.append(mapped)
        occupied = (tuple(range(self.num_particles[0]))
                    + tuple(range(self.n_spatial_orbitals,
                                  self.n_spatial_orbitals + self.num_particles[1])))
        bits = reference_qubit_bits(self.mapping, self.n_modes, occupied)
        self.taper_info: TaperedRegister | None = None
        if taper:
            full_h = hamiltonian.map_to_qubits(self.mapping,
                                                n_modes=self.n_modes)
            info = taper_problem(full_h, [-1j * group for group in mapped_groups],
                                 bits)
            if info is not None:
                if info.dropped:
                    raise ValueError(
                        "HVA cannot taper symmetry-changing group(s) "
                        f"{tuple(i + 1 for i in info.dropped)}: dropping a "
                        "fixed Hamiltonian group would change the ansatz")
                self.taper_info = info
                mapped_groups = [1j * generator for generator in info.generators]
                self.n_qubits = info.n_qubits
                bits = [int(qubit in info.occupied)
                        for qubit in range(self.n_qubits)]
        self.mapped_groups = tuple(mapped_groups)
        self._matrices = tuple((-1j * group.to_sparse_matrix()).tocsr()
                               for group in self.mapped_groups)
        self._occupied_qubits = tuple(i for i, bit in enumerate(bits) if bit)
        index = sum(int(bit) << (self.n_qubits - 1 - qubit)
                    for qubit, bit in enumerate(bits))
        self._reference = np.zeros(1 << self.n_qubits, dtype=complex)
        self._reference[index] = 1.0
        self.preparation_generators: tuple[PauliSum, ...] = ()
        self.preparation_angles: tuple[float, ...] = ()
        if occupied_coefficients is not None:
            if self.taper_info is not None:
                _check_slater_taper_sector(occupied_coefficients,
                                          self.taper_info)
            try:
                self.preparation_generators, self.preparation_angles = \
                    slater_preparation(
                        occupied_coefficients[0], occupied_coefficients[1],
                        mapping=self.mapping, num_particles=self.num_particles,
                        taper_info=self.taper_info)
            except SymmetryLeakError as error:
                raise ValueError(
                    "the UHF preparation does not conserve every tapered "
                    "symmetry; use taper=False or the occupation determinant"
                ) from error
            from ..core.checkpoint import prepare_state
            self._reference = prepare_state(
                self.n_qubits, self._occupied_qubits,
                self.preparation_generators, self.preparation_angles)
        self._rotations: tuple[tuple[str, float, int], ...] = ()
        if self.evolution == "trotter":
            self._rotations = self._compile_rotations()

    def _compile_rotations(self) -> tuple[tuple[str, float, int], ...]:
        """Expand canonical JW term order into mapped, tied Pauli rotations.

        The mapping is a Clifford basis change, so transforming each ordered
        Jordan-Wigner Pauli term preserves the finite-step product formula.
        Sorting labels *after* encoding would change a noncommuting term order
        and thus change the ansatz at fixed step count.
        """
        schedule: list[tuple[str, float, int]] = []
        identity = "I" * self.n_qubits
        canonical_groups: list[list[tuple[str, float]]] = []
        for physical, mapped_group in zip(self.groups, self.mapped_groups):
            jw = physical.map_to_qubits("jordan_wigner", n_modes=self.n_modes)
            terms: list[tuple[str, float]] = []
            reconstructed: dict[str, complex] = {}
            for source_label, coefficient in sorted(jw.terms.items()):
                source = PauliSum({source_label: coefficient},
                                  num_qubits=self.n_modes)
                encoded = reencode_pauli(
                    source, self.mapping,
                    num_particles=self.num_particles
                    if self.mapping == "parity_reduced" else None)
                if self.taper_info is not None:
                    encoded = self.taper_info.taper_operator(encoded)
                if len(encoded.terms) > 1:
                    raise RuntimeError("one Pauli rotation mapped to more "
                                       "than one target-register term")
                for label, value in encoded.terms.items():
                    reconstructed[label] = reconstructed.get(label, 0j) + value
                    if label != identity and abs(value.real) > 1e-12:
                        terms.append((label, float(value.real)))
            labels = set(reconstructed) | set(mapped_group.terms)
            if any(abs(reconstructed.get(label, 0j)
                       - mapped_group.terms.get(label, 0j)) > 1e-8
                   for label in labels):
                raise RuntimeError("canonical HVA Pauli rotations do not "
                                   "reconstruct the mapped group")
            canonical_groups.append(terms)
        for logical in range(self.num_parameters):
            terms = canonical_groups[logical % len(canonical_groups)]
            for _ in range(self.steps):
                if self.order == 1:
                    ordered = [(label, weight / self.steps)
                               for label, weight in terms]
                else:
                    ordered = ([(label, weight / (2 * self.steps))
                                for label, weight in terms]
                               + [(label, weight / (2 * self.steps))
                                  for label, weight in reversed(terms)])
                for label, weight in ordered:
                    if schedule and schedule[-1][0] == label \
                            and schedule[-1][2] == logical:
                        old = schedule[-1]
                        schedule[-1] = (label, old[1] + weight, logical)
                    else:
                        schedule.append((label, weight, logical))
        return tuple(schedule)

    def _group_label(self, group: Fermion, index: int) -> str:
        """Stable physical label for one fixed Hamiltonian group."""
        if self.grouping == "custom":
            return f"group_{index + 1}"
        term = next(iter(group.terms))
        body = "one_body" if len(term) == 2 else "two_body"
        if self.grouping == "body_order":
            return body
        spins = sorted(int(mode >= self.n_spatial_orbitals)
                       for mode, dagger in term if dagger)
        suffix = "".join("a" if spin == 0 else "b" for spin in spins)
        return f"{body}_{suffix}"

    @staticmethod
    def _check_hermitian(operator: PauliSum) -> None:
        """Reject a group that cannot generate unitary real-time evolution."""
        scale = max((abs(c) for c in operator.terms.values()), default=1.0)
        residual = max((abs(c.imag) for c in operator.terms.values()), default=0.0)
        if residual > 1e-9 * max(scale, 1.0):
            raise ValueError("each HVA Hamiltonian group must be Hermitian")

    @property
    def num_parameters(self) -> int:
        """One angle per group per layer."""
        return self.layers * len(self.groups)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Logical angles in layer-major, group-minor order."""
        return tuple(f"layer_{layer + 1}:{label}"
                     for layer in range(self.layers)
                     for label in self.group_labels)

    @property
    def rotation_names(self) -> tuple[str, ...]:
        """Name each compiled Pauli rotation by its controlling group angle."""
        names = self.parameter_names
        return tuple(f"{names[logical]}:rotation_{index + 1}"
                     for index, (_, _, logical) in enumerate(self._rotations))

    @property
    def pauli_generators(self) -> list[PauliSum]:
        """Ordered exact groups or circuit-compatible single-Pauli generators."""
        if self.evolution == "exact":
            return (list(self.preparation_generators)
                    + [-1j * self.mapped_groups[i % len(self.mapped_groups)]
                       for i in range(self.num_parameters)])
        return (list(self.preparation_generators)
                + [PauliSum({label: -1j * weight}, num_qubits=self.n_qubits)
                   for label, weight, _ in self._rotations])

    def reference_qubits(self) -> list[int]:
        """Set qubits of the occupation-determinant reference."""
        return list(self._occupied_qubits)

    def circuit_parameters(self, theta: np.ndarray) -> np.ndarray:
        """Expand logical group angles to the flattened circuit schedule."""
        angles = np.asarray(theta, dtype=float).ravel()
        if angles.size != self.num_parameters or not np.all(np.isfinite(angles)):
            raise ValueError(f"HVA needs {self.num_parameters} finite angles")
        if self.evolution != "trotter":
            raise ValueError("exact HVA has no generic Pauli-rotation circuit")
        return np.asarray(
            list(self.preparation_angles)
            + [angles[index] for _, _, index in self._rotations], dtype=float)

    def reference_state(self) -> np.ndarray:
        """The occupation-ordered reference determinant in this mapping."""
        return self._reference.copy()

    def state(self, theta: np.ndarray) -> np.ndarray:
        """Prepare the HVA state from the mean-field reference determinant."""
        if self.provider is not None:
            return self.provider.statevector(
                self.n_qubits, self.reference_qubits(),
                self.pauli_generators, self.circuit_parameters(theta))
        return self.evolve(theta, self._reference)

    def evolve(self, theta: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Apply all ordered HVA layers to one or more reference states."""
        angles = np.asarray(theta, dtype=float).ravel()
        if angles.size != self.num_parameters or not np.all(np.isfinite(angles)):
            raise ValueError(f"HVA needs {self.num_parameters} finite angles")
        out = np.asarray(references, dtype=complex)
        if out.ndim not in (1, 2) or out.shape[0] != self._reference.size:
            raise ValueError("reference state has the wrong register dimension")
        out = out.copy()
        if self.evolution == "exact":
            for angle, generator in zip(angles, self._matrices * self.layers):
                if angle:
                    out = expm_multiply(
                        angle * generator, out,
                        traceA=angle * generator.diagonal().sum())
            return out
        indices = np.arange(out.shape[0], dtype=np.int64)
        for label, weight, logical in self._rotations:
            phase = angles[logical] * weight
            if not phase:
                continue
            flip, parity_mask, n_y = pauli_masks(label)
            source = indices ^ flip
            parity = np.bitwise_count(source & parity_mask) & 1
            factors = (1j ** n_y) * (1 - 2 * parity.astype(np.int8))
            action = factors * out[source] if out.ndim == 1 \
                else factors[:, None] * out[source]
            out = np.cos(phase) * out - 1j * np.sin(phase) * action
        return out
