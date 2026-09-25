# -*- coding: utf-8 -*-
# file: algorithms/mean_field.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical RHF and UHF drivers behind ``Mandacaru(method=...)``.

The common geometry builder produces molecular integrals and an MO-basis
fermionic Hamiltonian.  It also performs the SCF that chooses that basis.
These drivers reuse its converged SCF result and never build a quantum ansatz,
operator pool, circuit, qubit Hamiltonian or state vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mapping import (Fermion, PauliSum, reference_qubit_bits,
                            resolve_mapping)
from ..core.sector import ParticleSector
from ..units import convert_energy
from .base import VariationalDriver
from .dry_run import QubitEstimate
from .hartree_fock import RHFResult, UHFResult


@dataclass(frozen=True)
class MeanFieldResult:
    """Classical SCF energy and the matching post-HF problem.

    Energies use ``energy_unit`` (eV by default).  ``scf`` retains the detailed
    RHF or UHF orbitals and electronic energy in Hartree.  The exported
    ``fermion_hamiltonian`` uses the same MO basis and active space as the
    quantum drivers.  :meth:`as_quantum_problem` supplies their direct-mode
    constructor options; the default quantum reference occupies the first
    ``n_alpha`` and ``n_beta`` orbitals in that basis.  For UHF this is the
    natural-orbital determinant, whose energy can differ from the UHF energy.
    """

    method: str
    optimal_energy: float
    reference_energy: float
    scf: RHFResult | UHFResult
    model_orbitals: np.ndarray
    fermion_hamiltonian: Fermion
    num_particles: tuple[int, int]
    n_spatial_orbitals: int
    energy_unit: str

    @property
    def success(self) -> bool:
        """Whether the underlying SCF converged."""
        return bool(self.scf.converged)

    @property
    def optimal_parameters(self) -> np.ndarray:
        """Empty array: classical SCF has no variational circuit angles."""
        return np.empty(0, dtype=float)

    @property
    def num_evaluations(self) -> int:
        """Number of SCF iterations, not quantum energy evaluations."""
        return int(self.scf.n_iterations)

    def in_units(self, units: str = "eV") -> float:
        """Return the converged total SCF energy in ``units``."""
        return float(convert_energy(self.optimal_energy, self.energy_unit, units))

    def as_quantum_problem(self) -> dict[str, object]:
        """Options for ``Mandacaru(method='adapt-vqe', **options)``.

        Direct mode reuses the MO-basis Hamiltonian without repeating the
        real-space integration or SCF.  Its Hartree-Fock reference determinant
        is the RHF solution, or the UHF natural-orbital reference for UHF.
        """
        return {"hamiltonian": self.fermion_hamiltonian,
                "num_particles": self.num_particles,
                "n_spatial_orbitals": self.n_spatial_orbitals,
                "initial_state": "hartree-fock"}

    def qubit_hamiltonian(self, mapping: str = "jordan_wigner") -> PauliSum:
        """Map the saved MO Hamiltonian for a post-HF quantum calculation."""
        canonical = resolve_mapping(mapping)
        return self.fermion_hamiltonian.map_to_qubits(
            canonical, n_modes=2 * self.n_spatial_orbitals,
            num_particles=self.num_particles if canonical == "parity_reduced"
            else None)

    def reference_state(self, mapping: str = "jordan_wigner") -> np.ndarray:
        """Encode the MO occupation determinant for Quantum Echoes or VQE.

        The RHF result's determinant is the SCF state.  UHF exports the common
        natural-orbital reference used by the post-HF Hamiltonian; the true
        unrestricted determinant has different alpha and beta orbital sets.
        """
        canonical = resolve_mapping(mapping)
        m = self.n_spatial_orbitals
        occupied = (tuple(range(self.num_particles[0]))
                    + tuple(range(m, m + self.num_particles[1])))
        bits = reference_qubit_bits(canonical, 2 * m, occupied)
        index = sum(int(bit) << (len(bits) - qubit - 1)
                    for qubit, bit in enumerate(bits))
        state = np.zeros(1 << len(bits), dtype=complex)
        state[index] = 1.0
        return state

    def scf_state(self, mapping: str = "jordan_wigner") -> np.ndarray:
        """Encode the actual converged RHF or UHF Slater determinant.

        RHF is one computational basis state.  UHF uses independent alpha and
        beta orbitals, so this expands each occupied-orbital wedge product in
        the shared natural-orbital basis of :attr:`fermion_hamiltonian`.  The
        amplitude of a determinant is the product of the corresponding alpha
        and beta overlap minors.  The full state vector is allocated only on
        request, for consumers such as Quantum Echoes.
        """
        if isinstance(self.scf, RHFResult):
            return self.reference_state(mapping)
        canonical = resolve_mapping(mapping)
        m = self.n_spatial_orbitals
        width = 2 * m - (2 if canonical == "parity_reduced" else 0)
        sector = ParticleSector(width, self.num_particles, canonical)
        alpha_coefficients = (self.model_orbitals.conj().T
                              @ self.scf.mo_coefficients_alpha[:,
                                                               :self.num_particles[0]])
        beta_coefficients = (self.model_orbitals.conj().T
                             @ self.scf.mo_coefficients_beta[:,
                                                              :self.num_particles[1]])
        alpha: dict[tuple[int, ...], complex] = {}
        beta: dict[tuple[int, ...], complex] = {}
        amplitudes = np.empty(sector.dim, dtype=complex)
        for index, occupations in enumerate(sector.occupations):
            alpha_occ = tuple(int(i) for i in np.flatnonzero(occupations[:m]))
            beta_occ = tuple(int(i) for i in np.flatnonzero(occupations[m:]))
            if alpha_occ not in alpha:
                alpha[alpha_occ] = np.linalg.det(
                    alpha_coefficients[list(alpha_occ), :])
            if beta_occ not in beta:
                beta[beta_occ] = np.linalg.det(
                    beta_coefficients[list(beta_occ), :])
            amplitudes[index] = alpha[alpha_occ] * beta[beta_occ]
        state = np.zeros(1 << width, dtype=complex)
        state[sector.indices] = amplitudes
        return state


class _MeanFieldDriver(VariationalDriver):
    """Classical SCF adapter using the shared geometry and integral pipeline."""

    classical_mean_field = True
    _kind: str = ""

    def _citation_config(self) -> dict[str, object]:
        """Cite the SCF method and basis without unused quantum components."""
        config = super()._citation_config()
        config.update(method=self._kind, mapping=None, optimizer=None,
                      backend_provider=None, execute_circuits=False)
        return config

    def __init__(self, **driver_kwargs: object) -> None:
        # These options require a circuit or a mapped Hamiltonian.  Refuse them
        # before the dry-run constructor probe can quietly accept and ignore one.
        unsupported = ("load_hamiltonian", "hamiltonian_builder", "taper",
                       "shots", "execute_circuits", "checkpoint", "resume",
                       "save_hamiltonian", "verbose_operators",
                       "verbose_hamiltonian", "optimizer", "device",
                       "backend_provider", "backend_options", "quenching",
                       "sparse", "run_options")
        used = [name for name in unsupported if driver_kwargs.get(name)]
        if used:
            raise ValueError(
                f"method={self._kind!r} is classical and does not take "
                + ", ".join(f"{name}=" for name in used))
        reductions = ("frozen_core", "frozen_orbitals", "active_orbitals",
                      "active_threshold")
        reduced = [name for name in reductions if driver_kwargs.get(name)]
        if reduced:
            raise ValueError(
                f"method={self._kind!r} reports the full SCF baseline; "
                + ", ".join(f"{name}=" for name in reduced)
                + " belongs on the subsequent quantum calculation")
        super().__init__(**driver_kwargs)
        self.ansatz = None
        self.fermion_hamiltonian: Fermion | None = None
        self._scf: RHFResult | UHFResult | None = None

    def _configure(
            self, hamiltonian: Fermion, num_particles: tuple[int, int],
            n_orbitals: int) -> None:
        """Adopt the SCF result and fermionic model without qubit mapping."""
        context = self._gradient_context or {}
        integrals = context.get("integrals")
        if integrals is None:
            raise ValueError(
                f"method={self._kind!r} needs molecular integrals from a "
                "geometry and basis; a cached or custom qubit Hamiltonian "
                "does not contain the spatial integrals needed for SCF")
        if getattr(integrals, "spin_orbit_coupling", None):
            raise NotImplementedError(
                "RHF/UHF baselines do not include spin-orbit coupling in their "
                "self-consistent Fock operator")
        if not isinstance(hamiltonian, Fermion):
            raise TypeError("mean-field export needs a fermionic Hamiltonian")
        full_particles = tuple(int(n) for n in num_particles)
        if self._kind == "rhf":
            if full_particles[0] != full_particles[1]:
                raise ValueError("RHF requires a closed shell with n_alpha = n_beta; "
                                 "use method='uhf' for an open shell")
            cached = getattr(integrals, "_last_rhf_result", None)
            self._scf = (cached[1] if cached is not None
                         and cached[0] == sum(full_particles)
                         else integrals.hartree_fock(sum(full_particles)))
        else:
            cached = getattr(integrals, "_last_uhf_result", None)
            self._scf = (cached[1] if cached is not None
                         and cached[0] == full_particles
                         else integrals.open_shell_hartree_fock(*full_particles))
            # The builder's automatic reference is RHF for an even count,
            # including a spin-polarized even count.  Always request the UHF
            # natural-orbital Hamiltonian so the exported reference and its
            # energy use exactly the same orbital basis.
            hamiltonian = integrals.molecular_hamiltonian(
                mo_basis=True, n_electrons=sum(full_particles),
                num_particles=full_particles, open_shell=True)
        self.fermion_hamiltonian = hamiltonian
        self.hamiltonian = hamiltonian
        self.num_particles = tuple(int(n) for n in num_particles)
        self.n_spatial_orbitals = int(n_orbitals)
        self.n_qubits = 2 * n_orbitals - (
            2 if self.mapping == "parity_reduced" else 0)
        self._configured = True

    def run(self) -> MeanFieldResult | QubitEstimate:
        """Return the converged classical energy and reusable MO Hamiltonian."""
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured or self._scf is None:
            raise RuntimeError("RHF/UHF needs an ASE geometry and basis first")
        self._check_kpts()
        if not self._scf.converged:
            raise RuntimeError(f"{self._kind.upper()} SCF did not converge")
        integrals = self._gradient_context["integrals"]
        constant = float(integrals.constant_energy + integrals.nuclear_repulsion)
        total = self._scf.electronic_energy + constant
        reference = (self._scf.reference_energy + constant
                     if isinstance(self._scf, UHFResult) else total)
        result = MeanFieldResult(
            method=self._kind, optimal_energy=self._to_energy_units(total),
            reference_energy=self._to_energy_units(reference), scf=self._scf,
            model_orbitals=np.asarray(integrals.mo_coefficients).copy(),
            fermion_hamiltonian=self.fermion_hamiltonian,
            num_particles=self.num_particles,
            n_spatial_orbitals=self.n_spatial_orbitals,
            energy_unit=self._energy_unit_label())
        if self.verbose:
            self._show_banner()
            print(f"{self._kind.upper()} SCF: E = {result.optimal_energy:+.8f} "
                  f"{result.energy_unit}, {result.num_evaluations} iterations")
        return result


class RHFDriver(_MeanFieldDriver):
    """Restricted Hartree-Fock baseline for an even, closed-shell system."""

    _kind = "rhf"


class UHFDriver(_MeanFieldDriver):
    """Unrestricted Hartree-Fock baseline for any spin occupation."""

    _kind = "uhf"
