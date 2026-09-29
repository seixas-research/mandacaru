# -*- coding: utf-8 -*-
# file: algorithms/vasqa.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""VASQA: the Variational, Adaptive and Stochastic Quantum Algorithm.

:class:`VASQA` searches the *structure* of a product-of-exponentials ansatz
with a Markov chain (Markov Chain Ansatz Search, :mod:`~mandacaru.algorithms.mcas`)
and relaxes each proposed structure with VQE.  Where ADAPT-VQE only ever
appends the largest-gradient operator, the chain can also delete, replace and
swap operators, so an early choice can be revised.

Each step

1. draws an architecture move -- ``insert``, ``delete``, ``replace`` or
   ``swap`` -- at a uniform position; a new operator (``insert``,
   ``replace``) is drawn from a softmax of the pool gradients
   :math:`|\langle[H, A_\mu]\rangle|` at the current state
   (``proposal="gradient"``, the default) or uniformly
   (``proposal="uniform"``);
2. rebuilds the ansatz :math:`\prod_k e^{\theta_k A_{\mu_k}}|\mathrm{HF}\rangle`
   for the proposed operator sequence and minimizes its energy with the
   classical optimizer, starting from the current angles carried over by the
   move (a new occurrence starts at zero);
3. accepts the proposal with the Metropolis-Hastings probability
   :math:`\min\{1, e^{-\beta\,\Delta F} q(C|C')/q(C'|C)\}` on the cost
   :math:`F(C) = \widehat E(C) + \lambda_L L`.

Interpretation
--------------
With ``warm_start=True`` (default) the energy of an architecture depends on
the angles it was reached with, so the chain is a **stochastic optimization**
(simulated annealing over ansatz structures), not a sampler of a fixed
distribution.  With ``warm_start=False`` every architecture is optimized from
zero angles, its cost is a fixed function of the sequence (memoized), and at a
fixed ``temperature`` the chain leaves
:math:`\pi(C) \propto e^{-\beta \widehat F(C)}` invariant -- for the optimizer's
:math:`\widehat E(C)`, which a local optimizer does not certify to be the
global minimum over the angles.  The gradient proposal keeps that property:
its distribution at ``C`` is computed from ``C``'s own relaxed state, so it is
a fixed function of ``C`` too, and the reverse probability is evaluated at the
proposed state.

The reported state (``result.optimal_energy``, ``result.operators``,
``result.optimal_parameters`` and the solver's ``ansatz``) is the **lowest-cost**
architecture evaluated, rejected proposals included; the lowest-energy one and
the chain's final state are reported beside it.  Learned proposals are not
implemented.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..circuits.profiling import CircuitMetrics
from ..optimizers.optim import DEFAULT_OPTIMIZER
from ..units import convert_energy, to_hartree
from .mcas import (MOVES, ProposalKernel, TemperatureSchedule,
                   gradient_softmax,
                   log_acceptance, metropolis_accept)
from .pool_driver import PoolDriver

if TYPE_CHECKING:
    from ..optimizers.optim import Optimizer

#: Default architecture temperature, in **eV** whatever ``atomic_units`` says:
#: a geometric anneal that starts where a typical correlation-energy step is
#: accepted about half the time and ends where only sub-meV increases survive.
DEFAULT_TEMPERATURE = {"initial": 0.1, "final": 1e-3}

#: How new operators are drawn: ``"gradient"``, a softmax of the pool
#: gradients at the current state, or ``"uniform"``.
PROPOSALS = ("gradient", "uniform")

#: Default softmax temperature of the gradient proposal, relative to the
#: largest gradient: the steepest operator is ``e^5 ~ 150`` times as likely
#: as one with zero gradient, which still leaves the rest of the pool within
#: reach.
DEFAULT_PROPOSAL_TEMPERATURE = 0.2


@dataclass
class MCASStep:
    """One step of the chain: the proposal, the decision and the state after.

    ``proposed`` and ``current`` are pool-index tuples; energies and costs are
    in the result's :attr:`VASQAResult.energy_unit`.  A rejected step leaves
    ``current`` equal to the previous step's, and is kept: the chain's
    occupation includes it.
    """

    step: int
    move: str
    action: str
    temperature: float
    proposed: tuple[int, ...]
    proposed_energy: float
    proposed_cost: float
    log_q_forward: float
    log_q_reverse: float
    log_acceptance: float
    accepted: bool
    current: tuple[int, ...]
    current_energy: float
    current_cost: float
    num_evaluations: int
    optimizer_success: bool


@dataclass
class VASQAResult:
    """Result of a VASQA run.

    ``optimal_*`` and :attr:`operators` describe the lowest-**cost**
    architecture evaluated; with ``length_penalty=0`` that is also the
    lowest-energy one, otherwise ``best_energy*`` may differ.  ``final_*`` is
    where the chain stopped.  Every energy is in :attr:`energy_unit`.
    """

    optimal_energy: float
    optimal_parameters: np.ndarray
    reference_energy: float
    operators: list[str]
    architecture: tuple[int, ...]
    optimal_cost: float
    best_energy: float
    best_energy_operators: list[str]
    best_energy_parameters: np.ndarray
    final_energy: float
    final_operators: list[str]
    final_parameters: np.ndarray
    steps: list[MCASStep] = field(default_factory=list)
    acceptance_by_move: dict = field(default_factory=dict)
    num_architectures: int = 0
    #: Pool-gradient screenings the gradient proposal spent (``0`` for the
    #: uniform one): ``len(pool)`` expectation values each.
    num_screenings: int = 0
    num_evaluations: int = 0
    optimizer_steps: int = 0
    optimizer_failures: list = field(default_factory=list)
    seed: int | None = None
    metrics: CircuitMetrics | None = None
    timings: dict | None = None
    integration_profile: dict | None = None
    energy_unit: str = "eV"

    @property
    def num_operators(self) -> int:
        return len(self.operators)

    @property
    def num_parameters(self) -> int:
        return len(self.optimal_parameters)

    @property
    def correlation_energy(self) -> float:
        """Energy lowered relative to the reference (``E - E_ref``)."""
        return self.optimal_energy - self.reference_energy

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals accepted (``0`` for a run of no steps)."""
        if not self.steps:
            return 0.0
        return sum(s.accepted for s in self.steps) / len(self.steps)

    @property
    def energy_history(self) -> list[float]:
        """The chain's current energy after every step."""
        return [s.current_energy for s in self.steps]

    def in_units(self, units: str = "eV") -> float:
        """The optimal energy converted to ``units`` (``"eV"`` or ``"Ha"``)."""
        return float(convert_energy(self.optimal_energy, self.energy_unit,
                                    units))

    def __repr__(self) -> str:
        return (f"VASQAResult(energy={self.optimal_energy:.6f} "
                f"{self.energy_unit}, operators={self.num_operators}, "
                f"steps={len(self.steps)}, "
                f"acceptance={self.acceptance_rate:.2f})")


@dataclass
class _Evaluated:
    """An architecture with its optimized angles, energy and cost (Hartree)."""

    architecture: tuple[int, ...]
    parameters: np.ndarray
    energy: float
    cost: float
    nfev: int = 0
    nit: int = 0
    success: bool = True
    message: str = ""
    #: The distribution new operators are drawn from *at this state*
    #: (``None``: uniform).  The forward move uses the current state's, the
    #: reverse the proposed state's.
    operator_probabilities: np.ndarray | None = None


class VASQA(PoolDriver):
    """Markov Chain Ansatz Search with VQE relaxation, on the state vector.

    **The internal layer**: reached only through the calculator,
    ``Mandacaru(method="vasqa", ...)``, which forwards every option here.
    The problem setup -- ``hamiltonian`` or ``basis``, ``pool``, ``mapping``,
    ``sparse``, ``sector``, ``taper``, ``active_space`` and the rest -- is the
    one ADAPT-VQE takes (:class:`~mandacaru.algorithms.pool_driver.PoolDriver`).

    Parameters
    ----------
    pool : PoolBase or str
        The operator pool the architectures are drawn from -- ``"fermionic"``
        (default), ``"qubit"``, ``"qeb"``, ``"ceo"``, ``"ceo-ovp"`` or
        ``"spin-orbit"``.  ``"ceo-ovp"`` differs from ``"ceo"`` only in how
        ADAPT-VQE grows it, which a chain does not do: here the two are the
        same set of operators.
    max_steps : int
        Chain length: proposals drawn, each one VQE optimization (default
        ``200``).
    min_length, max_length : int
        Allowed number of operators in the ansatz, inclusive (defaults ``0``
        and ``20``).  The chain starts from the empty ansatz, the reference
        itself, so ``min_length`` must be ``0``.
    move_weights : dict, optional
        Relative weight of ``"insert"``, ``"delete"``, ``"replace"`` and
        ``"swap"`` (default all ``1``); a move left out is never proposed, but
        ``insert`` and ``delete`` are both required.
    temperature : float or dict
        Architecture temperature in the run's energy unit (eV; Hartree with
        ``atomic_units=True``).  A number is a fixed-temperature chain, ``0``
        the greedy limit (accept only decreases); ``{"initial": T0, "final":
        T1}`` anneals geometrically from ``T0`` to ``T1`` over ``max_steps``.
        The default, :data:`DEFAULT_TEMPERATURE`, is 0.1 -> 0.001 eV in
        either unit convention.  It weights circuit
        descriptions by their cost and is not a physical temperature.
    length_penalty : float
        :math:`\\lambda_L`, the cost of one operator in the run's energy unit
        (default ``0``: energy only).  With independent angles an insertion
        can never raise the optimal energy, so without a penalty the chain
        drifts to ``max_length``.
    warm_start : bool
        Start each optimization from the current angles carried over by the
        move (default ``True``).  ``False`` starts every architecture from
        zero angles and memoizes its energy, which makes the cost a fixed
        function of the architecture (see the module docstring).
    proposal : {"gradient", "uniform"}
        How ``insert`` and ``replace`` draw their new operator (default
        ``"gradient"``).  ``"gradient"`` is a softmax of the pool gradients
        :math:`|g_\\mu| = |2\\,\\mathrm{Re}\\langle H\\psi|A_\\mu\\psi\\rangle|` at the
        current state (:func:`~mandacaru.algorithms.mcas.gradient_softmax`):
        the operators ADAPT-VQE would pick are proposed most often, and the
        rest keep a positive probability.  The gradient is the one for
        appending at the end of the circuit, used for every slot -- a
        heuristic for *which* operator, not a derivative at each position --
        and the reverse probability is evaluated at the proposed state, so
        the Metropolis-Hastings ratio stays exact.  One pool screening per
        proposal.  ``"uniform"`` draws every operator with ``1/M``.
    proposal_temperature : float
        Softmax temperature of the gradient proposal, relative to the largest
        gradient (default :data:`DEFAULT_PROPOSAL_TEMPERATURE`): the steepest
        operator is ``exp(1/proposal_temperature)`` times as likely as one
        with zero gradient.  Small values approach ADAPT's greedy choice,
        large ones the uniform proposal.
    seed : int, optional
        Seed of the chain's random stream (proposals and acceptance).
    profile : bool
        Compile and profile the final ansatz (default ``True``).  Only the
        reported architecture is compiled, not every proposal.
    """

    _default_sparse = "auto"
    _supports_spin_orbit = True

    citation_method = "vasqa"
    #: The name in the ``[SYSTEM]`` block's title.
    log_title = "VASQA"
    #: ``run()`` writes the ``txt=`` log: ADAPT-VQE's blocks, with a
    #: ``[MARKOV CHAIN]`` table where ADAPT has ``[ITERATIONS]``.
    writes_output_log = True
    #: A checkpoint stores one grown ansatz; a chain's state (current, best,
    #: random stream) is not one, so ``checkpoint=`` / ``resume=`` are refused.
    supports_checkpoints = False

    def __init__(self,
                 hamiltonian=None,
                 pool="fermionic",
                 basis="HAO",
                 num_particles=None,
                 n_spatial_orbitals=None,
                 optimizer: str | Optimizer = DEFAULT_OPTIMIZER,
                 mapping: str = "jordan_wigner",
                 device: str = "AER_simulator",
                 max_steps: int = 200,
                 min_length: int = 0,
                 max_length: int = 20,
                 move_weights: dict | None = None,
                 temperature=None,
                 length_penalty: float = 0.0,
                 warm_start: bool = True,
                 proposal: str = "gradient",
                 proposal_temperature: float = DEFAULT_PROPOSAL_TEMPERATURE,
                 seed: int | None = None,
                 profile: bool = True,
                 verbose: bool = True,
                 sparse: bool | str = "auto",
                 sector: bool | str = "auto",
                 atomic_units: bool = False,
                 **driver_kwargs):
        super().__init__(pool=pool, profile=profile, sector=sector,
                         optimizer=optimizer, mapping=mapping, basis=basis,
                         device=device, verbose=verbose, sparse=sparse,
                         atomic_units=atomic_units, **driver_kwargs)
        self.max_steps = int(max_steps)
        if self.max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps!r}")
        if int(min_length) != 0:
            raise ValueError(
                "min_length must be 0: the chain starts from the empty "
                "ansatz (the reference state)")
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.move_weights = move_weights
        # Validated now, with a placeholder pool size: the real kernel needs
        # the pool, which may not exist before the first geometry.
        ProposalKernel(2, move_weights, self.min_length, self.max_length)
        #: ``None`` means :data:`DEFAULT_TEMPERATURE`, which is in eV.
        self.temperature = temperature
        TemperatureSchedule(DEFAULT_TEMPERATURE if temperature is None
                            else temperature, self.max_steps)
        self.length_penalty = float(length_penalty)
        if not (math.isfinite(self.length_penalty)
                and self.length_penalty >= 0.0):
            raise ValueError(f"length_penalty must be finite and >= 0, got "
                             f"{length_penalty!r}")
        if not isinstance(warm_start, bool):
            raise ValueError(f"warm_start must be True or False, got "
                             f"{warm_start!r}")
        self.warm_start = warm_start
        self.proposal = str(proposal).strip().lower()
        if self.proposal not in PROPOSALS:
            raise ValueError(f"unknown proposal {proposal!r}; use one of "
                             f"{PROPOSALS}")
        self.proposal_temperature = float(proposal_temperature)
        gradient_softmax([1.0, 0.0], self.proposal_temperature)   # validates
        self.seed = None if seed is None else int(seed)
        self._adopt_problem(hamiltonian, num_particles, n_spatial_orbitals)

    # -- the evaluator ---------------------------------------------------- #

    def _ansatz_for(self, architecture):
        """A fresh ansatz holding ``architecture``'s operators in order."""
        ansatz = self._new_ansatz()
        for index in architecture:
            ansatz.append(self._pool_ops[index])
        return ansatz

    def _cost(self, energy_ha: float, length: int) -> float:
        """:math:`F = E + \\lambda_L L`, in Hartree; ``inf`` for a NaN energy."""
        if not math.isfinite(energy_ha):
            return math.inf
        return energy_ha + self._length_penalty_ha * length

    def _evaluate(self, architecture, x0) -> _Evaluated:
        """Relax ``architecture`` with VQE from angles ``x0``."""
        architecture = tuple(architecture)
        if not self.warm_start:
            cached = self._cache.get(architecture)
            if cached is not None:
                # The memoized evaluation is the architecture's fixed cost;
                # it costs nothing to look up.
                return _Evaluated(cached.architecture,
                                  cached.parameters.copy(), cached.energy,
                                  cached.cost, operator_probabilities=(
                                      cached.operator_probabilities))
        ansatz = self._ansatz_for(architecture)
        if not architecture:
            energy = self.energy(ansatz.reference_state())
            evaluated = _Evaluated(architecture, np.zeros(0), energy,
                                   self._cost(energy, 0))
        else:
            result = self._optimize_all(
                lambda t: self.ansatz_energy(ansatz, t),
                np.asarray(x0, dtype=float))
            energy = float(result.fun)
            evaluated = _Evaluated(
                architecture, np.asarray(result.x, dtype=float), energy,
                self._cost(energy, len(architecture)), int(result.nfev),
                int(result.nit or 0), bool(result.success),
                str(result.message))
        if self.proposal == "gradient":
            # The proposal out of this state: a softmax of the pool gradients
            # at its relaxed angles.  Needed whether or not it is accepted --
            # a rejected proposal's distribution is the reverse probability.
            psi = (ansatz.state(evaluated.parameters) if architecture
                   else ansatz.reference_state())
            evaluated.operator_probabilities = gradient_softmax(
                self._analytic_gradients(psi), self.proposal_temperature)
            self._screenings += 1
        if not self.warm_start:
            self._cache[architecture] = evaluated
        return evaluated

    # -- the chain -------------------------------------------------------- #

    def run(self, geometry=None, cell=None) -> VASQAResult:
        """Run the chain for ``max_steps`` proposals and report the best state.

        ``geometry`` (an ASE ``Atoms`` or a ``(symbols, positions)`` pair) and
        ``cell`` fill the ``[SYSTEM]`` block of the run log, as for ADAPT-VQE;
        the calculator passes the geometry it evaluates.
        """
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured:
            raise RuntimeError(
                "VASQA has no Hamiltonian; construct it with one, or use it "
                "as an ASE calculator with a `hamiltonian_builder`")
        self._check_kpts()
        timings, run_t0 = self._make_timings()
        unit = self._energy_unit_label()
        labels = [op.label for op in self._pool_ops]
        # The log's action column names operators by their short form; the
        # full labels are on the result.
        short = [getattr(op, "short_label", op.label) for op in self._pool_ops]

        # An empty pool (one electron in one orbital) has nothing to place:
        # the reference is the answer, as it is for ADAPT-VQE, and the chain
        # takes no step.
        kernel = (ProposalKernel(len(self._pool_ops), self.move_weights,
                                 self.min_length, self.max_length)
                  if self._pool_ops else None)
        n_steps = self.max_steps if kernel is not None else 0
        schedule = TemperatureSchedule(
            _temperature_in_hartree(DEFAULT_TEMPERATURE, "eV")
            if self.temperature is None
            else _temperature_in_hartree(self.temperature, unit),
            self.max_steps)
        self._length_penalty_ha = float(to_hartree(self.length_penalty, unit))
        self._cache: dict = {}
        #: Pool-gradient screenings, one per new state under the gradient
        #: proposal: part of the search cost, so it is reported.
        self._screenings = 0
        rng = np.random.default_rng(self.seed)

        self._show_banner()
        with timings.time("parameter optimization"):
            current = self._evaluate((), np.zeros(0))
        reference_energy = current.energy
        logger = self._make_logger(self.log_targets, geometry, cell,
                                   reference_energy, schedule)

        best_cost = best_energy = current
        seen = {current.architecture}
        steps: list[MCASStep] = []
        tried = {move: 0 for move in MOVES}
        taken = {move: 0 for move in MOVES}
        total_evals = total_steps = 0
        failures: list[tuple[int, str]] = []

        try:
            for step in range(1, n_steps + 1):
                beta = schedule.beta(step - 1)
                action = kernel.sample(current.architecture, rng,
                                       current.operator_probabilities)
                if action is None:
                    # Unreachable while insert has weight and max_length > 0,
                    # which the kernel enforces; a silent self-loop would
                    # hide it.
                    raise RuntimeError(
                        f"no architecture move is valid at "
                        f"{current.architecture!r}")
                proposed_arch = action.apply(current.architecture)
                x0 = (action.transfer(current.parameters) if self.warm_start
                      else np.zeros(len(proposed_arch)))
                with timings.time("parameter optimization"):
                    proposed = self._evaluate(proposed_arch, x0)
                total_evals += proposed.nfev
                total_steps += proposed.nit
                if not proposed.success:
                    failures.append((step, proposed.message))
                seen.add(proposed.architecture)

                log_forward = kernel.log_q(proposed_arch, current.architecture,
                                           current.operator_probabilities)
                log_reverse = kernel.log_q(current.architecture, proposed_arch,
                                           proposed.operator_probabilities)
                log_alpha = log_acceptance(beta, current.cost, proposed.cost,
                                           log_forward, log_reverse)
                accepted = metropolis_accept(log_alpha, rng)
                previous = current
                tried[action.move] += 1
                if accepted:
                    taken[action.move] += 1
                    current = proposed
                if proposed.cost < best_cost.cost:
                    best_cost = proposed
                if proposed.energy < best_energy.energy:
                    best_energy = proposed

                temperature = float(self._to_energy_units(
                    schedule.temperature(step - 1)))
                steps.append(MCASStep(
                    step=step, move=action.move,
                    action=action.describe(labels),
                    temperature=temperature,
                    proposed=proposed.architecture,
                    proposed_energy=self._to_energy_units(proposed.energy),
                    proposed_cost=self._to_energy_units(proposed.cost),
                    log_q_forward=log_forward, log_q_reverse=log_reverse,
                    log_acceptance=log_alpha, accepted=accepted,
                    current=current.architecture,
                    current_energy=self._to_energy_units(current.energy),
                    current_cost=self._to_energy_units(current.cost),
                    num_evaluations=proposed.nfev,
                    optimizer_success=proposed.success))
                if logger is not None:
                    logger.write_chain_step(
                        step=step, move=action.move,
                        length=len(proposed.architecture),
                        energy=self._to_energy_units(proposed.energy),
                        delta_energy=self._to_energy_units(
                            proposed.energy - previous.energy),
                        current_energy=self._to_energy_units(current.energy),
                        temperature=temperature, log_acceptance=log_alpha,
                        accepted=accepted, energy_unit=unit,
                        optimizer_steps=(proposed.nit if proposed.nfev
                                         else None),
                        action=action.describe(short))

            # The reported state is what the calculator's forces, densities
            # and measured energies are taken from: the ansatz and the
            # parameters have to describe the same architecture.
            self.ansatz = self._ansatz_for(best_cost.architecture)
            with timings.time("circuit profiling"):
                metrics = self._profile(self.ansatz)

            def names(evaluated):
                return [labels[i] for i in evaluated.architecture]

            result = VASQAResult(
                optimal_energy=self._to_energy_units(best_cost.energy),
                optimal_parameters=best_cost.parameters,
                reference_energy=self._to_energy_units(reference_energy),
                operators=names(best_cost),
                architecture=best_cost.architecture,
                optimal_cost=self._to_energy_units(best_cost.cost),
                best_energy=self._to_energy_units(best_energy.energy),
                best_energy_operators=names(best_energy),
                best_energy_parameters=best_energy.parameters,
                final_energy=self._to_energy_units(current.energy),
                final_operators=names(current),
                final_parameters=current.parameters,
                steps=steps,
                acceptance_by_move={move: (taken[move], tried[move])
                                    for move in MOVES},
                num_architectures=len(seen),
                num_screenings=self._screenings,
                num_evaluations=total_evals,
                optimizer_steps=total_steps,
                optimizer_failures=failures,
                seed=self.seed,
                metrics=metrics,
                integration_profile=self._integration_profile,
                energy_unit=unit)
            if logger is not None:
                self._write_summary(logger, result)
        finally:
            if logger is not None:
                logger.close()

        if failures:
            warnings.warn(
                f"the inner optimizer reported no convergence in "
                f"{len(failures)} of {len(steps)} proposals; their energies "
                f"are kept as the optimizer left them.  Last message: "
                f"{failures[-1][1]!r}", RuntimeWarning, stacklevel=2)
        self._finalize_timings(timings, run_t0)
        result.timings = timings.as_dict()
        # The performance block closes the step, after the summary.
        self.write_performance(timings)
        return result

    # -- the run log ------------------------------------------------------ #

    def _make_logger(self, targets, geometry, cell, reference_energy,
                     schedule):
        """Open the run log and write every block before the chain.

        ``[SYSTEM]``, ``[BASIS]`` and ``[ELECTRONS]`` are ADAPT-VQE's
        (:meth:`~mandacaru.algorithms.pool_driver.PoolDriver._open_header_logger`);
        ``[OPTIMIZATION SETUP]`` carries the chain's settings.  ``None`` when
        the run reports nowhere.
        """
        if not targets:
            return None
        logger = self._open_header_logger(targets, geometry, cell)
        unit = self._energy_unit_label()
        to_unit = self._to_energy_units
        if schedule.annealed:
            temperature = (f"{to_unit(schedule.initial):g} -> "
                           f"{to_unit(schedule.final):g} (geometric)")
        else:
            temperature = f"{to_unit(schedule.initial):g}"
        weights = ProposalKernel(max(len(self._pool_ops), 2),
                                 self.move_weights, self.min_length,
                                 self.max_length).move_weights
        shots = (f"{self.shots}" if self.shots
                 else "0 (exact expectation values)")
        logger.write_optimizer_setup(
            optimizer_method=self.optimizer.method,
            reference_energy=to_unit(reference_energy), energy_unit=unit,
            extra={
                "pool": getattr(self.pool, "name", "?"),
                "pool_class": self.pool.__class__.__name__,
                "pool_size": len(self._pool_ops),
                "proposal": (
                    f"gradient softmax (tau {self.proposal_temperature:g} of "
                    f"max |grad|), Metropolis-Hastings"
                    if self.proposal == "gradient"
                    else "uniform, Metropolis-Hastings"),
                "max_steps": self.max_steps,
                "ansatz_length": f"{self.min_length} to {self.max_length}",
                "move_weights": ", ".join(f"{move} {weight:g}" for move, weight
                                          in weights.items()),
                f"architecture_temperature_{unit}": temperature,
                f"length_penalty_{unit}": f"{self.length_penalty:g}",
                "warm_start": str(self.warm_start),
                "seed": str(self.seed),
                "energy_column": "relaxed energy of the proposed ansatz",
                "dE_column": ("proposed minus current energy, before the "
                              "acceptance test"),
                "reoptimize_all_parameters": str(self.quenching),
                "state_vector_backend": self._backend_description(),
                "device": str(self.device),
                "backend_provider": str(self.backend_provider),
                "circuit_execution": str(self.execute_circuits),
                "shots": shots,
                "circuit_profiling": (
                    "True (the reported ansatz only)" if self.profile else
                    "False (no gate counts: pass profile=True)")})
        return logger

    def _write_summary(self, logger, result: VASQAResult) -> None:
        """The ``[VARIATIONAL QUANTUM SUMMARY]`` of the lowest-cost ansatz."""
        unit = result.energy_unit
        extra = {
            f"optimal_cost_{unit}": f"{result.optimal_cost:.10f}",
            f"best_energy_{unit}": f"{result.best_energy:.10f}",
            f"final_energy_{unit}": f"{result.final_energy:.10f}",
            "final_num_operators": len(result.final_operators),
            "chain_steps": len(result.steps),
            "acceptance_rate": f"{result.acceptance_rate:.4f}",
            **{f"accepted_{move}": f"{taken} of {tried}" for move, (taken, tried)
               in result.acceptance_by_move.items()},
            "architectures_evaluated": result.num_architectures,
            "gradient_screenings": result.num_screenings,
            "optimizer_failures": len(result.optimizer_failures),
            # The table names each proposal, not the sequence that won, so
            # the reported ansatz is written here, in order.
            "operators": ", ".join(result.operators) or "(none)",
        }
        logger.write_summary(
            converged=None, optimal_energy=result.optimal_energy,
            energy_unit=unit, reference_energy=result.reference_energy,
            correlation_energy=result.correlation_energy,
            num_operators=result.num_operators,
            num_parameters=result.num_parameters,
            num_evaluations=result.num_evaluations,
            optimizer_steps=result.optimizer_steps, metrics=result.metrics,
            extra=extra)


def _temperature_in_hartree(temperature, unit: str):
    """The ``temperature`` option converted from ``unit`` to Hartree."""
    if isinstance(temperature, dict):
        return {key: float(to_hartree(float(value), unit))
                for key, value in temperature.items()}
    return float(to_hartree(float(temperature), unit))
