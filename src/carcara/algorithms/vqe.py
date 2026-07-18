# -*- coding: utf-8 -*-
# file: algorithms/vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Variational Quantum Eigensolver (VQE).

:class:`VQE` minimizes the Rayleigh quotient
:math:`E(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle` over the
parameters of a variational ansatz, returning the (approximate) ground-state
energy and optimal parameters.

This reference implementation is an **exact state-vector simulator**: the qubit
Hamiltonian is materialized as a dense matrix and the ansatz produces the exact
:math:`2^N` state vector, so the energy is the noiseless expectation value.

The class mirrors :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`:

* **direct mode** -- construct with a Hamiltonian and an ansatz and call
  :meth:`run`;
* **ASE calculator mode** -- construct with a ``basis`` (no Hamiltonian/ansatz),
  attach to an ``Atoms`` object (``atoms.calc = VQE(...)``) and let
  ``atoms.get_total_energy()`` build the Hamiltonian from the geometry, build a
  default UCCSD ansatz, and drive :meth:`run`, returning eV.

The ``optimizer`` may be named by string, a ``verbose`` run prints the qubit
Hamiltonian as Pauli strings and a timing / memory / cores summary, and the run
returns a :class:`VQEResult` shaped like
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQEResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter as _perf

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from ..backends.hardware import normalize_device, require_runnable
from ..core.mapping import Fermion, PauliSum
from ..optimizers.optim import Optimizer, resolve_optimizer
from ..units import from_hartree
from .adapt_vqe import format_pauli_sum


@dataclass
class VQEResult:
    """Result of a :class:`VQE` run.

    Shaped like :class:`~carcara.algorithms.adapt_vqe.ADAPTVQEResult`: it carries
    the optimal energy/parameters, the reference energy, the evaluation count and
    the full cost history, exposes the same convenience views
    (:attr:`num_parameters`, :attr:`energy_history`), and records the timing /
    cores / memory profile of the run.
    """

    optimal_energy: float                 # minimized energy <psi|H|psi>
    optimal_parameters: np.ndarray        # optimal ansatz parameters
    reference_energy: float               # energy of the ansatz reference state
    num_evaluations: int = 0              # cost-function evaluations
    history: list[float] = field(default_factory=list)   # cost per evaluation
    success: bool = True
    timings: dict | None = None           # per-stage wall time / cores / memory
    integration_profile: dict | None = None   # real-space integration profile

    @property
    def num_parameters(self) -> int:
        return len(self.optimal_parameters)

    @property
    def energy_history(self) -> list[float]:
        return list(self.history)

    @property
    def correlation_energy(self) -> float:
        """Energy lowered relative to the reference (``E - E_ref``)."""
        return self.optimal_energy - self.reference_energy

    def __repr__(self) -> str:
        return (f"VQEResult(energy={self.optimal_energy:.6f}, "
                f"n_params={self.num_parameters}, "
                f"nfev={self.num_evaluations}, success={self.success})")


class VQE(Calculator):
    """Variational Quantum Eigensolver on an exact state-vector backend.

    Also an ASE calculator: see the module docstring for the two usage modes.

    Parameters
    ----------
    hamiltonian : PauliSum or Fermion, optional
        The qubit Hamiltonian, or a fermionic Hamiltonian mapped with the
        ansatz's mapping.  Omit in calculator mode.
    ansatz : object, optional
        A parameterized ansatz exposing ``num_parameters``, ``n_qubits``,
        ``state(theta)`` and ``reference_state()`` (e.g.
        :class:`~carcara.circuits.ansatz.UCCSD`).  Omit in calculator mode; a
        UCCSD ansatz is then built from the geometry.
    optimizer : str or Optimizer
        Method name -- ``"COBYLA"`` (default), ``"Nelder-Mead"``, ``"BFGS"`` -- or
        a pre-built :class:`~carcara.optimizers.optim.Optimizer`.
    verbose : bool
        Print the qubit Hamiltonian (Pauli strings) and a timing / resources
        summary to standard output (default ``True``).
    basis, mapping, device, grid, h, kpts, charge, n_electrons, hamiltonian_builder, ansatz_builder, run_options :
        Calculator-mode options mirroring
        :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`: ``basis`` (a name or a
        ``{"name": ..., <options>}`` dict, including the periodic ``"PW"`` plane-wave
        family ``{"name": "PW", "energy_cutoff": 300}``) and grid resolution ``h``
        build the Hamiltonian from the ASE geometry (the grid is generated from
        ``atoms.cell`` unless ``grid`` is given), ``ansatz_builder`` overrides the
        default UCCSD factory.  ``kpts`` is a Monkhorst-Pack mesh resolved with ASE
        (Gamma-point only is runnable; see ``ADAPTVQE``).
    frozen_core : bool, str or int
        Frozen-core approximation (default ``False``, no freezing).  ``True`` /
        ``"auto"`` freezes the chemical noble-gas core; an integer freezes that
        many lowest molecular orbitals.  The frozen orbitals are removed from the
        active space (see :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`).
    frozen_orbitals : sequence of int, optional
        Explicit list of (doubly occupied) spatial MO indices to freeze; overrides
        ``frozen_core`` and names exactly which electrons are core vs active.
    """

    implemented_properties = ["energy", "free_energy"]
    _OPTIMIZERS = ("COBYLA", "Nelder-Mead", "BFGS")

    def __init__(self, hamiltonian=None, ansatz=None,
                 optimizer: str | Optimizer = "COBYLA", verbose: bool = True,
                 *, basis="FAO", mapping: str = "jordan_wigner",
                 device: str = "AER_simulator", grid=None, h: float = 0.20,
                 kpts=None, spin: bool = False,
                 initial_state: str | None = "hartree-fock",
                 charge: int = 0, n_electrons=None,
                 frozen_core=False, frozen_orbitals=None,
                 hamiltonian_builder=None, ansatz_builder=None,
                 run_options: dict | None = None, **calc_kwargs):
        Calculator.__init__(self, **calc_kwargs)

        self.verbose = bool(verbose)
        self.optimizer = resolve_optimizer(optimizer, allowed=self._OPTIMIZERS)
        self.basis = basis
        self.mapping = mapping
        self.device = normalize_device(device)
        self.grid = grid
        self.h = float(h)
        self.spin = bool(spin)
        self.frozen_core = frozen_core
        self.frozen_orbitals = frozen_orbitals

        from ._hamiltonian_from_atoms import (monkhorst_pack_kpts,
                                              resolve_initial_state)
        self.initial_state = resolve_initial_state(initial_state)
        # Monkhorst-Pack k-point mesh (ASE); Gamma-point only is runnable.
        self.kpts, self.kpts_gamma, self.kpoints = monkhorst_pack_kpts(kpts)
        self.charge = int(charge)
        self.n_electrons = n_electrons
        self.hamiltonian_builder = hamiltonian_builder
        self.ansatz_builder = ansatz_builder
        self.run_options = dict(run_options or {})
        self._integration_profile = None

        self._configured = False
        if hamiltonian is not None and ansatz is not None:
            self._configure(hamiltonian, ansatz)

    # -- setup ------------------------------------------------------------ #

    def _configure(self, hamiltonian, ansatz) -> None:
        """Materialize the dense Hamiltonian for a given ansatz (direct mode)."""
        self.ansatz = ansatz
        qubit_h = self._as_pauli_sum(hamiltonian, ansatz)
        if qubit_h.num_qubits != ansatz.n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the ansatz "
                f"has {ansatz.n_qubits}")
        self.hamiltonian = qubit_h
        self.n_qubits = ansatz.n_qubits
        self.mapping = getattr(ansatz, "mapping", self.mapping)
        # Hermitize away rounding noise; the expectation value is then real.
        h = qubit_h.to_matrix()
        self._h_matrix = 0.5 * (h + h.conj().T)
        self._configured = True

    def _check_kpts(self) -> None:
        """Reject a non-Gamma Monkhorst-Pack mesh (the engine is Gamma-point)."""
        if len(self.kpoints) > 1:
            raise NotImplementedError(
                f"a {self.kpts[0]}x{self.kpts[1]}x{self.kpts[2]} Monkhorst-Pack "
                f"mesh ({len(self.kpoints)} k-points) is not yet supported: the "
                "real-space engine solves a Gamma-point (molecular) problem.  Use "
                "kpts=(1, 1, 1) or kpts=None.")

    def _kpts_label(self) -> str:
        n1, n2, n3 = self.kpts
        centred = ", Gamma-centred" if self.kpts_gamma else ""
        if len(self.kpoints) == 1:
            return f"Gamma ({n1}x{n2}x{n3} Monkhorst-Pack)"
        return (f"{len(self.kpoints)} k-points ({n1}x{n2}x{n3} "
                f"Monkhorst-Pack{centred})")

    @staticmethod
    def _as_pauli_sum(hamiltonian, ansatz) -> PauliSum:
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            mapping = getattr(ansatz, "mapping", "jordan_wigner")
            return hamiltonian.map_to_qubits(mapping, n_modes=ansatz.n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    def _default_ansatz(self, n_spatial_orbitals, num_particles):
        """Build the default UCCSD ansatz for calculator mode."""
        if self.ansatz_builder is not None:
            return self.ansatz_builder(n_spatial_orbitals, num_particles,
                                       self.mapping)
        from ..circuits import UCCSD
        return UCCSD(n_spatial_orbitals, num_particles, mapping=self.mapping)

    # -- ASE calculator interface ---------------------------------------- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """ASE hook: build the Hamiltonian + ansatz from ``atoms`` and run VQE.

        Stores the ground-state energy (eV, per ASE convention) in
        :attr:`results` and the full :class:`VQEResult` in :attr:`vqe_result`.
        """
        require_runnable(self.device)
        self._wall_start = _perf()       # wall clock spans integration + run
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms

        if not self._configured:
            if self.hamiltonian_builder is not None:
                hamiltonian, num_particles, n_orb = self.hamiltonian_builder(atoms)
            else:
                from ._hamiltonian_from_atoms import build_basis_hamiltonian
                hamiltonian, num_particles, n_orb, profile = \
                    build_basis_hamiltonian(atoms, self.basis, self.grid, self.h,
                                            self.charge, self.n_electrons,
                                            spin=self.spin,
                                            frozen_core=self.frozen_core,
                                            frozen_orbitals=self.frozen_orbitals)
                self._integration_profile = profile
            ansatz = self._default_ansatz(n_orb, num_particles)
            self._configure(hamiltonian, ansatz)

        result = self.run(**self.run_options)
        self.vqe_result = result

        energy_ev = float(from_hartree(result.optimal_energy, "eV"))
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

    # -- energy ----------------------------------------------------------- #

    def energy(self, theta) -> float:
        """Expectation value ``<psi(theta)| H |psi(theta)>`` (real)."""
        psi = self.ansatz.state(theta)
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def reference_energy(self) -> float:
        """Energy of the ansatz reference state (all parameters zero)."""
        psi = self.ansatz.reference_state()
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def run(self, initial_parameters=None) -> VQEResult:
        """Optimize the parameters and return the ground-state estimate.

        The optimizer and the ``verbose`` flag come from the constructor; the
        only argument is ``initial_parameters`` (which the constructor does not
        carry), defaulting to all-zero (the reference state).
        """
        if not self._configured:
            raise RuntimeError(
                "VQE has no Hamiltonian/ansatz; construct it with both, or use it "
                "as an ASE calculator with a `basis`")
        self._check_kpts()
        from ..utils.profiling import Timings
        from ..integrals import _backend

        n = self.ansatz.num_parameters
        x0 = (np.zeros(n) if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        timings = Timings(n_cores=_backend.num_threads(),
                          backend="C (OpenMP)" if _backend.HAS_C_BACKEND
                          else "NumPy")
        run_t0 = self.__dict__.pop("_wall_start", None)
        if run_t0 is None:
            run_t0 = _perf()

        ref_energy = self.reference_energy()
        if self.verbose:
            from ..utils import banner
            banner.show()
            self._print_header(ref_energy)

        with timings.time("parameter optimization"):
            result = self.optimizer.minimize(self.energy, x0)

        if self._integration_profile is not None:
            for name, secs in self._integration_profile.get("stages_s", {}).items():
                timings.add(f"integration: {name}", secs)
        timings.wall_time = _perf() - run_t0

        vqe_result = VQEResult(
            optimal_energy=result.fun,
            optimal_parameters=result.x,
            reference_energy=ref_energy,
            num_evaluations=result.nfev,
            history=result.history,
            success=result.success,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile)

        if self.verbose:
            self._print_summary(vqe_result, timings)
        return vqe_result

    # -- standard-output trace ------------------------------------------- #

    def _print_header(self, ref_energy: float) -> None:
        """Print the run banner and the qubit Hamiltonian as Pauli strings."""
        rule = "=" * 70
        print(rule)
        print(f"VQE  |  mapping: {self.mapping}  |  {self.n_qubits} qubits  |  "
              f"optimizer: {self.optimizer.method}  |  device: {self.device}")
        print(f"ansatz: {type(self.ansatz).__name__}  |  "
              f"parameters: {self.ansatz.num_parameters}  |  "
              f"k-points: {self._kpts_label()}")
        print(f"spin-polarized: {self.spin}  |  "
              f"initial state: {self.initial_state}")
        print(rule)
        n_terms = len(self.hamiltonian.simplify().terms)
        print(f"Qubit Hamiltonian ({n_terms} Pauli terms):")
        print(format_pauli_sum(self.hamiltonian))
        print(f"Reference (all-zero) energy = {ref_energy:+.8f} Ha")
        print(rule)

    def _print_summary(self, result: VQEResult, timings=None) -> None:
        """Print the closing summary: result line plus timings / resources."""
        rule = "=" * 70
        print(rule)
        status = "converged" if result.success else "did not converge"
        print(f"VQE finished ({status}): "
              f"E = {result.optimal_energy:+.8f} Ha, "
              f"{result.num_parameters} parameters, {result.num_evaluations} "
              f"evaluations")
        if timings is not None:
            print(timings.format_report())
        print(rule)
