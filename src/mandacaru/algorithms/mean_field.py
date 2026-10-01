# -*- coding: utf-8 -*-
# file: algorithms/mean_field.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical RHF, UHF and GHF drivers behind ``Mandacaru(method=...)``.

The Kohn-Sham driver (``method="dft"``, :mod:`mandacaru.algorithms.dft`) is
built on the same adapter and result type.

The common geometry builder produces molecular integrals and an MO-basis
fermionic Hamiltonian.  It also performs the SCF that chooses that basis.
These drivers reuse its converged SCF result and never build a quantum ansatz,
operator pool, circuit, qubit Hamiltonian or state vector.

``method="ghf"`` is the one mean field with spin-orbit coupling in its Fock
operator (:class:`~mandacaru.algorithms.hartree_fock.GHF`): its orbitals are
two-component spinors, and its exported Hamiltonian is written in them.

A run reports through the shared run log (:class:`~mandacaru.utils.logging.Logger`),
to the ``txt=`` file, standard output or both: ``[SYSTEM]``, ``[BASIS]`` and a
classical ``[ELECTRONS]`` -- no mapping, register or qubit Hamiltonian, since
none is built -- then ``[SCF SETUP]`` and ``[SCF SUMMARY]``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

import numpy as np

from ..core.mapping import (Fermion, PauliSum, reference_qubit_bits,
                            resolve_mapping)
from ..core.sector import ParticleSector
from ..units import convert_energy
from .base import VariationalDriver
from .dry_run import QubitEstimate
from .hartree_fock import (GHF, LEVEL_SHIFT, RHF, UHF, GHFResult, RHFResult,
                           UHFResult)

#: How each SCF is described in ``[SCF SETUP]``, and the solver entry point
#: whose defaults the geometry builder runs it with.
SCF_METHODS = {
    "rhf": ("restricted Hartree-Fock (closed shell)", RHF.run),
    "uhf": ("unrestricted Hartree-Fock (natural-orbital export)", UHF.solve),
    "ghf": ("generalized (spinor) Hartree-Fock", GHF.solve),
}


@dataclass(frozen=True)
class MeanFieldResult:
    """Classical SCF energy and the matching post-HF problem.

    Energies use ``energy_unit`` (eV by default).  ``scf`` retains the detailed
    RHF, UHF or GHF orbitals and electronic energy in Hartree.  The exported
    ``fermion_hamiltonian`` uses the same MO basis and active space as the
    quantum drivers.  :meth:`as_quantum_problem` supplies their direct-mode
    constructor options; the default quantum reference occupies the first
    ``n_alpha`` and ``n_beta`` orbitals in that basis.  For UHF this is the
    natural-orbital determinant, whose energy can differ from the UHF energy.
    For GHF the modes are spinors, not spin-orbitals of one spatial basis
    (:meth:`~mandacaru.algorithms.hartree_fock.GHF.mode_order`): the reference
    is the GHF determinant itself, and ``model_orbitals`` holds the
    ``(2M, 2M)`` spinor coefficients in mode order.  The spinor Hamiltonian
    does not conserve :math:`S_z` mode by mode even without spin-orbit
    coupling, so its quantum calculation needs ``pool="spin-orbit"``.
    """

    method: str
    optimal_energy: float
    reference_energy: float
    scf: RHFResult | UHFResult | GHFResult
    model_orbitals: np.ndarray
    fermion_hamiltonian: Fermion
    num_particles: tuple[int, int]
    n_spatial_orbitals: int
    energy_unit: str
    #: Wall-clock stages, resources and the total, as for the quantum drivers.
    timings: dict | None = None

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
        real-space integration or SCF.  Its reference determinant is the RHF
        solution, the UHF natural-orbital reference for UHF, or the Kohn-Sham
        determinant for DFT.
        """
        return {"hamiltonian": self.fermion_hamiltonian,
                "num_particles": self.num_particles,
                "n_spatial_orbitals": self.n_spatial_orbitals,
                "initial_state": "hartree-fock"}

    def _check_mapping(self, mapping: str) -> str:
        canonical = resolve_mapping(mapping)
        if canonical == "parity_reduced" and isinstance(self.scf, GHFResult):
            raise ValueError(
                "the GHF spinor Hamiltonian has no alpha-parity qubit to "
                "remove; use 'jordan_wigner', 'parity' or 'bravyi_kitaev'")
        return canonical

    def qubit_hamiltonian(self, mapping: str = "jordan_wigner") -> PauliSum:
        """Map the saved MO Hamiltonian for a post-HF quantum calculation."""
        canonical = self._check_mapping(mapping)
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
        canonical = self._check_mapping(mapping)
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
        if isinstance(self.scf, (RHFResult, GHFResult)):
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
    writes_output_log = True
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
        reductions = ("active_space",)
        reduced = [name for name in reductions if driver_kwargs.get(name)]
        if reduced:
            raise ValueError(
                f"method={self._kind!r} reports the full SCF baseline; "
                + ", ".join(f"{name}=" for name in reduced)
                + " belongs on the subsequent quantum calculation")
        super().__init__(**driver_kwargs)
        self.ansatz = None
        self.fermion_hamiltonian: Fermion | None = None
        self._scf: RHFResult | UHFResult | GHFResult | None = None

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

    def run(self, geometry=None,
            cell=None) -> MeanFieldResult | QubitEstimate:
        """Return the converged classical energy and reusable MO Hamiltonian.

        ``geometry`` (an ASE ``Atoms`` or a ``(symbols, positions)`` pair) and
        ``cell`` fill the log's ``[SYSTEM]`` block; the calculator passes them.
        """
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured or self._scf is None:
            raise RuntimeError("RHF/UHF needs an ASE geometry and basis first")
        self._check_kpts()
        timings, run_t0 = self._make_timings()
        self._show_banner()
        logger = None
        if self.log_targets:
            logger = self._open_header_logger(self.log_targets, geometry, cell)
        try:
            if logger is not None:
                logger.write_block("SCF SETUP", self._scf_setup_fields())
            result = self._result(timings, run_t0)
            if logger is not None:
                logger.write_block("SCF SUMMARY", self._scf_summary_fields(
                    result), framed=True)
        finally:
            if logger is not None:
                logger.close()
        if not self._scf.converged:
            raise RuntimeError(f"{self._kind.upper()} SCF did not converge")
        # The performance block closes the step, after the summary.  In
        # calculator mode `Mandacaru` writes it instead (`defer_performance`).
        self.write_performance(timings)
        return result

    def _result(self, timings, run_t0) -> MeanFieldResult:
        """The :class:`MeanFieldResult` of the SCF ``_configure`` ran."""
        integrals = self._gradient_context["integrals"]
        constant = float(integrals.constant_energy + integrals.nuclear_repulsion)
        total = self._scf.electronic_energy + constant
        reference = (self._scf.reference_energy + constant
                     if isinstance(self._scf, UHFResult) else total)
        orbitals = (self._scf.mode_coefficients
                    if isinstance(self._scf, GHFResult)
                    else integrals.mo_coefficients)
        result = MeanFieldResult(
            method=self._kind, optimal_energy=self._to_energy_units(total),
            reference_energy=self._to_energy_units(reference), scf=self._scf,
            model_orbitals=np.asarray(orbitals).copy(),
            fermion_hamiltonian=self.fermion_hamiltonian,
            num_particles=self.num_particles,
            n_spatial_orbitals=self.n_spatial_orbitals,
            energy_unit=self._energy_unit_label())
        self._finalize_timings(timings, run_t0)
        return replace(result, timings=timings.as_dict())

    # -- the run log ------------------------------------------------------ #

    def _log_title(self) -> str:
        """``RHF``, ``UHF`` or ``GHF``."""
        return self._kind.upper()

    def _run_kwargs(self, atoms) -> dict:
        """Forward the geometry to :meth:`run` for the log's ``[SYSTEM]``."""
        return {"geometry": atoms}

    def _electron_fields(self) -> dict:
        """The classical ``[ELECTRONS]`` block: grid, charge, spin, orbitals.

        No reference state, mapping, register or qubit Hamiltonian: the SCF
        builds none, and a line reading ``Jordan-Wigner`` would describe a
        calculation that did not happen.
        """
        return {
            **self._grid_fields(),
            "kinetic operator": self.kinetic or "finite difference",
            "k-points": self._kpts_label(),
            "charge": str(self.charge),
            "spin-polarized": self._spin_polarized_label(),
            "spatial orbitals": str(self.n_spatial_orbitals),
            "electrons (alpha, beta)": str(self.num_particles),
        }

    def _scf_setup_fields(self) -> dict:
        """``[SCF SETUP]``: which SCF ran and what it stops on."""
        label, entry = SCF_METHODS[self._kind]
        defaults = inspect.signature(entry).parameters
        integrals = self._gradient_context["integrals"]
        fields = {
            "scf_method": label,
            "max_iterations": defaults["max_iter"].default,
            "convergence_Hartree": (f"{defaults['tol'].default:g} (energy "
                                    f"change and largest density change)"),
            "acceleration": (f"DIIS; level shift {LEVEL_SHIFT:g} Hartree "
                             f"after the first energy rise"),
        }
        if self._kind == "ghf":
            fields["spin_orbit_coupling"] = str(bool(
                getattr(integrals, "spin_orbit_coupling", None)))
        fields["energy_unit"] = self._energy_unit_label()
        return fields

    def _scf_summary_fields(self, result: MeanFieldResult) -> dict:
        """``[SCF SUMMARY]``: the converged determinant and its energy."""
        unit = result.energy_unit
        scf = self._scf
        fields = {"converged": str(bool(scf.converged)),
                  f"optimal_energy_{unit}": f"{result.optimal_energy:.10f}"}
        if isinstance(scf, UHFResult):
            # The natural-orbital determinant the exported problem starts
            # from; its energy differs from the UHF one.
            fields[f"reference_energy_{unit}"] = \
                f"{result.reference_energy:.10f}"
            fields["spin_contamination"] = f"{scf.spin_contamination:.6f}"
        elif isinstance(scf, GHFResult):
            fields["kramers_pairing_Hartree"] = f"{scf.kramers_pairing:.3e}"
        elif scf.converged and 0 < scf.n_occupied < len(scf.mo_energies):
            to_unit = self._to_energy_units
            fields.update({
                f"homo_energy_{unit}": f"{to_unit(scf.homo_energy):.6f}",
                f"lumo_energy_{unit}": f"{to_unit(scf.lumo_energy):.6f}",
                f"homo_lumo_gap_{unit}": f"{to_unit(scf.homo_lumo_gap):.6f}"})
        fields["scf_iterations"] = result.num_evaluations
        return fields


class RHFDriver(_MeanFieldDriver):
    """Restricted Hartree-Fock baseline for an even, closed-shell system."""

    _kind = "rhf"


class UHFDriver(_MeanFieldDriver):
    """Unrestricted Hartree-Fock baseline for any spin occupation."""

    _kind = "uhf"


class GHFDriver(_MeanFieldDriver):
    """Generalized (spinor) Hartree-Fock, with spin-orbit coupling in the
    self-consistent Fock operator when the basis carries it.

    Started from the core guess, the screened core guess, and the RHF
    (closed shell) and UHF determinants written as spinors, so the result is
    at or below those determinants' energies in the same Hamiltonian.
    """

    _kind = "ghf"

    def _configure(
            self, hamiltonian: Fermion, num_particles: tuple[int, int],
            n_orbitals: int) -> None:
        """Solve GHF on the builder's integrals and export the spinor-basis
        Hamiltonian."""
        context = self._gradient_context or {}
        integrals = context.get("integrals")
        if integrals is None:
            raise ValueError(
                "method='ghf' needs molecular integrals from a geometry and "
                "basis; a cached or custom qubit Hamiltonian does not contain "
                "the spatial integrals needed for SCF")
        if self.mapping == "parity_reduced":
            raise ValueError(
                "the GHF spinor Hamiltonian has no alpha-parity qubit to "
                "remove; use 'jordan_wigner', 'parity' or 'bravyi_kitaev'")
        full_particles = tuple(int(n) for n in num_particles)
        n_electrons = sum(full_particles)
        M = integrals.n_orbitals
        self._scf = integrals.generalized_hartree_fock(*full_particles)
        hamiltonian = Fermion.from_integrals(self._scf.h_mo, self._scf.eri_mo)
        # The same constant molecular_hamiltonian carries, so the exported
        # problem's energies are total energies like RHF's and UHF's.
        constant = complex(integrals.constant_energy
                           + integrals.nuclear_repulsion)
        self.fermion_hamiltonian = hamiltonian + Fermion(
            {(): constant}, n_modes=2 * M)
        self.hamiltonian = self.fermion_hamiltonian
        # The GHF determinant fills the first ceil(N/2) modes of the first
        # half and floor(N/2) of the second (GHF.mode_order).
        self.num_particles = ((n_electrons + 1) // 2, n_electrons // 2)
        self.n_spatial_orbitals = int(M)
        self.n_qubits = 2 * M
        self._configured = True
