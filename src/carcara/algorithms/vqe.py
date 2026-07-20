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

Beyond the ground state, :meth:`VQE.energy_levels` returns the low-lying
**molecular energy levels** (ground + excited states) by variational quantum
deflation -- see :mod:`carcara.algorithms.deflation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..optimizers.optim import Optimizer
from .base import VariationalDriver
from .deflation import DeflationMixin, deflation_penalty


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


class VQE(DeflationMixin, VariationalDriver):
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
        Method name -- one of ``"SPSA"``, ``"COBYLA"`` (default),
        ``"Nelder-Mead"``, ``"SLSQP"``, ``"Adam"``, ``"L-BFGS-B"`` -- or a
        pre-built :class:`~carcara.optimizers.optim.Optimizer`.
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
    save_hamiltonian : bool or str
        Write the qubit Hamiltonian (Pauli strings) to disk once built (default
        ``False``); ``True`` uses ``"hamiltonian"`` plus the extension of
        ``hamiltonian_format``.
    hamiltonian_format : {"parquet", "json"}
        Format written by ``save_hamiltonian`` (default ``"parquet"``);
        ``"json"`` needs no Parquet engine.  Loading auto-detects the format.
    load_hamiltonian : str, optional
        Read the qubit Hamiltonian from such a file, **skipping the molecular
        integrals and the fermion-to-qubit mapping**.  The file's recorded
        ``num_particles`` / ``n_spatial_orbitals`` are enough to rebuild the
        default UCCSD ansatz, so no geometry is needed.
    backend_provider : str
        SDK used to construct / execute the ansatz circuits -- ``"qiskit"``
        (default), ``"braket"`` or ``"cirq"``; see
        :mod:`carcara.backends.providers`.
    execute_circuits : bool, optional
        Prepare states by executing the compiled circuit on the provider's local
        simulator (default: ``True`` for ``"braket"``/``"cirq"``, ``False`` for
        ``"qiskit"``).  A circuit realizes the **Trotter** UCCSD product, so the
        default ansatz is built with ``trotter=True`` when this is on.
    quenching : bool
        Dynamic parametrization (default ``True``).  ``True`` optimizes all
        parameters jointly.  ``False`` sweeps them one at a time -- parameter
        ``k`` alone, with ``0..k-1`` frozen at their optimized values -- the
        fixed-ansatz analogue of ADAPT-VQE's frozen-parameter growth.
    """

    _result_attr = "vqe_result"
    _default_sparse = False

    def __init__(self, hamiltonian=None, ansatz=None,
                 optimizer: str | Optimizer = "COBYLA", verbose: bool = True,
                 *, basis="FAO", mapping: str = "jordan_wigner",
                 device: str = "AER_simulator", grid=None, h: float = 0.20,
                 kpts=None, spin: bool = False,
                 initial_state: str | None = "hartree-fock",
                 charge: int = 0, n_electrons=None,
                 frozen_core=False, frozen_orbitals=None,
                 hamiltonian_builder=None, ansatz_builder=None,
                 save_hamiltonian: bool | str = False,
                 load_hamiltonian: str | None = None,
                 hamiltonian_format: str = "parquet",
                 backend_provider: str | None = None,
                 execute_circuits: bool | None = None,
                 backend_options: dict | None = None, shots: int = 0,
                 quenching: bool = True,
                 run_options: dict | None = None, **calc_kwargs):
        super().__init__(optimizer=optimizer, mapping=mapping, basis=basis,
                         device=device, grid=grid, h=h, kpts=kpts, spin=spin,
                         initial_state=initial_state, charge=charge,
                         n_electrons=n_electrons, frozen_core=frozen_core,
                         frozen_orbitals=frozen_orbitals,
                         hamiltonian_builder=hamiltonian_builder,
                         save_hamiltonian=save_hamiltonian,
                         load_hamiltonian=load_hamiltonian,
                         hamiltonian_format=hamiltonian_format,
                         backend_provider=backend_provider,
                         execute_circuits=execute_circuits,
                         backend_options=backend_options, shots=shots,
                         quenching=quenching,
                         run_options=run_options, verbose=verbose, **calc_kwargs)
        self.ansatz_builder = ansatz_builder
        self._preset_ansatz = ansatz

        # A cached Hamiltonian carries num_particles / n_spatial_orbitals, so it
        # is a complete problem specification: the default UCCSD ansatz can be
        # built from it with no geometry, no integrals and no mapping step.
        if hamiltonian is None and self.load_hamiltonian is not None:
            hamiltonian, num_particles, n_orbitals = \
                self._load_hamiltonian_record()
            if ansatz is None and (num_particles is None or n_orbitals is None):
                raise ValueError(
                    f"{self.load_hamiltonian!r} does not record num_particles / "
                    "n_spatial_orbitals, so the default UCCSD ansatz cannot be "
                    "rebuilt from it; pass an explicit `ansatz`")
            self._configure(hamiltonian, num_particles, n_orbitals)
            self._built_from_hamiltonian = True
        # Direct mode: a Hamiltonian and ansatz were supplied at construction.
        elif hamiltonian is not None and ansatz is not None:
            self._configure(hamiltonian, None, None)
            self._built_from_hamiltonian = True

    # -- setup ------------------------------------------------------------ #

    def _configure(self, hamiltonian, num_particles, n_orbitals) -> None:
        """Adopt / build the ansatz for ``hamiltonian`` and materialize it.

        In direct mode the ansatz supplied at construction is used; in calculator
        mode a default UCCSD ansatz is built from ``num_particles`` / ``n_orbitals``.
        """
        ansatz = (self._preset_ansatz if self._preset_ansatz is not None
                  else self._default_ansatz(n_orbitals, num_particles))
        self.ansatz = ansatz
        self.mapping = getattr(ansatz, "mapping", self.mapping)
        qubit_h = self._as_pauli_sum(hamiltonian, ansatz.n_qubits)
        self._materialize_hamiltonian(qubit_h, ansatz.n_qubits)
        self._maybe_save_hamiltonian(
            getattr(ansatz, "num_particles", num_particles),
            getattr(ansatz, "n_spatial_orbitals", n_orbitals))
        self._configured = True

    def _default_ansatz(self, n_spatial_orbitals, num_particles):
        """Build the default UCCSD ansatz for calculator mode.

        With circuit execution enabled the ansatz is built in **Trotter** form and
        bound to the provider: a quantum circuit realizes the product of
        single-generator exponentials, not the exact exponential of the summed
        cluster operator.
        """
        if self.ansatz_builder is not None:
            return self.ansatz_builder(n_spatial_orbitals, num_particles,
                                       self.mapping)
        from ..circuits import UCCSD
        provider = self.circuit_provider()
        return UCCSD(n_spatial_orbitals, num_particles, mapping=self.mapping,
                     trotter=provider is not None, provider=provider)

    # -- energy ----------------------------------------------------------- #

    def energy_at(self, theta) -> float:
        """Expectation value ``<psi(theta)| H |psi(theta)>`` for parameters ``theta``.

        The parameter-space cost function.  (The base class's ``energy(psi)`` is the
        state-vector expectation value; ``energy_at`` prepares the state first --
        or, with ``shots > 0``, measures ``<H>`` on hardware instead.)
        """
        return self.ansatz_energy(self.ansatz, theta)

    def reference_energy(self) -> float:
        """Energy of the ansatz reference state (all parameters zero)."""
        return self.energy(self.ansatz.reference_state())

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

        n = self.ansatz.num_parameters
        x0 = (np.zeros(n) if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        timings, run_t0 = self._make_timings()
        ref_energy = self.reference_energy()
        if self.verbose:
            self._show_banner()
            self._print_header(ref_energy)

        with timings.time("parameter optimization"):
            result = self._optimize_all(self.energy_at, x0)

        self._finalize_timings(timings, run_t0)

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

    # -- excited states / energy levels (DeflationMixin hook) ------------- #

    def _deflated_ground(self, states, beta, *, state_index: int = 0,
                         initial_parameters=None, restarts: int = 1,
                         seed: int = 0):
        r"""Lowest state of the fixed ansatz orthogonal to ``states`` (deflation).

        Minimizes ``<psi(theta)|H|psi(theta)> + beta * sum_j |<psi_j|psi>|^2`` over
        the fixed ansatz, from the reference (ground) or seeded random restarts
        (excited states); the reported energy is the bare expectation value.
        Called per level by :meth:`~carcara.algorithms.deflation.DeflationMixin.energy_levels`.
        """
        n = self.ansatz.num_parameters
        base_x0 = (np.zeros(n) if initial_parameters is None
                   else np.asarray(initial_parameters, dtype=float).ravel())
        if base_x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {base_x0.size}")
        rng = np.random.default_rng(seed + int(state_index))

        def cost(theta):
            psi = self.ansatz.state(theta)
            return self.energy(psi) + deflation_penalty(psi, states, beta)

        # First attempt from the warm start (ground) / reference; further restarts
        # (and every excited-state restart) from seeded random points.
        best = None
        total_evals = 0
        for r in range(max(1, int(restarts))):
            x0 = base_x0 if (state_index == 0 and r == 0) else \
                rng.uniform(-np.pi, np.pi, size=n)
            result = self._optimize_all(cost, x0)
            total_evals += result.nfev
            if best is None or result.fun < best.fun:
                best = result
        psi = self.ansatz.state(best.x)
        return self.energy(psi), psi, total_evals, None

    # -- standard-output trace ------------------------------------------- #

    def _print_header(self, ref_energy: float) -> None:
        """Print the run configuration banner.

        The Hamiltonian's Pauli-string expansion is not printed (see
        :meth:`~carcara.algorithms.adapt_vqe.ADAPTVQE._print_header`); only its
        term count is.
        """
        rule = "=" * 70
        print(rule)
        print(f"VQE  |  mapping: {self.mapping}  |  {self.n_qubits} qubits  |  "
              f"optimizer: {self.optimizer.method}  |  device: {self.device}")
        print(f"ansatz: {type(self.ansatz).__name__}  |  "
              f"parameters: {self.ansatz.num_parameters}  |  "
              f"k-points: {self._kpts_label()}")
        print(f"spin-polarized: {self.spin}  |  "
              f"initial state: {self.initial_state}")
        print(f"backend provider: {self.backend_provider}  |  circuit execution: "
              f"{self.execute_circuits}  |  quenching: {self.quenching}")
        print(rule)
        n_terms = len(self.hamiltonian.simplify().terms)
        print(f"Qubit Hamiltonian: {n_terms} Pauli terms")
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
