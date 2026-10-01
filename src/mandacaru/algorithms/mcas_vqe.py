# -*- coding: utf-8 -*-
# file: algorithms/mcas_vqe.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""MCAS-VQE: Markov Chain Ansatz Search with the Variational Quantum
Eigensolver.

:class:`MCASVQE` searches the *structure* of a product-of-exponentials ansatz
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
the chain's final state are reported beside it.  When the operator
probabilities come from a trained model instead, the method is VALQA
(:mod:`~mandacaru.algorithms.valqa`); both share the chain,
:class:`MarkovChainSearch`, and both can record their proposals for training
(``record=DIR``, :mod:`~mandacaru.algorithms.proposal_data`).
"""

from __future__ import annotations

import math
import uuid
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..circuits.profiling import CircuitMetrics
from ..units import convert_energy, to_hartree
from .mcas import (MOVES, REPLACE_STARTS, Action, ProposalKernel,
                   TemperatureSchedule, edit_distance, gradient_softmax,
                   log_acceptance, metropolis_accept)
from .orbital_tracking import (DEFAULT_TRANSFER_THRESHOLD, OrbitalSnapshot,
                               match_orbitals, occupation_blocks,
                               orbital_overlap, transfer_ansatz)
from .pool_driver import PoolDriver
from .proposal_data import EditRecorder, ProblemRecord, resolve_record

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
    in the result's :attr:`MCASVQEResult.energy_unit`.  A rejected step leaves
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
class MCASVQEResult:
    """Result of a MCAS-VQE run.

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
    #: The architecture the chain started from: empty, unless ``transfer=True``
    #: carried the previous geometry's ansatz over.
    start_operators: list = field(default_factory=list)
    #: Why the chain started where it did (see ``transfer``).
    start: str = "empty ansatz"
    #: :func:`~mandacaru.algorithms.mcas.edit_distance` from the start to the
    #: reported architecture.
    edit_distance_from_start: int = 0
    #: Insertions relaxed and recorded beside the chain (``screen_insertions``).
    num_screened_insertions: int = 0
    #: What a periodic rebuild (``rebuild_every``) kept; ``None`` without one.
    rebuild: str | None = None

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
        return (f"{type(self).__name__}(energy={self.optimal_energy:.6f} "
                f"{self.energy_unit}, operators={self.num_operators}, "
                f"steps={len(self.steps)}, "
                f"acceptance={self.acceptance_rate:.2f})")


@dataclass
class _Inherited:
    """What the previous geometry's chain hands on: its reported ansatz, the
    orbitals it was expressed in, and its place in the trajectory.

    Copies only -- never the previous solver itself, which would keep every
    earlier geometry alive along a long trajectory.
    """

    operators: list
    angles: np.ndarray
    pool_labels: list
    symbols: list | None
    #: The previous molecular orbitals, or why there are none (a string).
    orbitals: object
    #: The trajectory the previous chain belonged to, and its step there.
    trajectory: str | None = None
    step: int = 0
    #: The problem (Hamiltonian, reference, pool) the previous chain searched.
    problem: ProblemRecord | None = None


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
    #: The pool gradients at this state (``None`` when not screened).
    gradients: np.ndarray | None = None


class MarkovChainSearch(PoolDriver):
    """Markov Chain Ansatz Search with VQE relaxation, on the state vector.

    The chain MCAS-VQE and VALQA share; they differ only in the distribution
    ``insert`` and ``replace`` draw their new operator from
    (:meth:`_operator_distribution`).  **The internal layer**: reached only
    through the calculator, ``Mandacaru(method="mcas-vqe" | "valqa", ...)``,
    which forwards every option here.  The problem setup -- ``hamiltonian`` or ``basis``, ``pool``, ``mapping``,
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
        and ``20``).  The chain can always fall back to the empty ansatz,
        the reference itself, so ``min_length`` must be ``0``.
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
    replace_start : str
        The angle the operator a ``replace`` puts in starts from, with
        ``warm_start``: ``"zero"`` (default) or ``"inherit"``, the angle of
        the operator it replaces.
    transfer : bool
        Along an ASE trajectory (a relaxation, a scan, dynamics), start each
        geometry's chain from the ansatz the previous geometry reported --
        its operators and, with ``warm_start``, its angles -- instead of
        from the empty ansatz (default ``False``).  The transferred ansatz is
        relaxed at the new geometry first and kept as the start only if its
        cost is below the reference's.  The molecular orbitals are followed
        between the geometries (:mod:`~mandacaru.algorithms.orbital_tracking`):
        operators are renamed where two orbitals changed places and angles
        change sign where an orbital did.  The chain starts empty instead,
        and the run log's ``start`` line says why, when the previous geometry
        was another system or pool, an orbital's best match falls below
        ``transfer_threshold``, an operator cannot be followed through the
        matching, the ansatz is longer than ``max_length``, or the relaxed
        ansatz does not beat the reference.  Where the chain starts does not
        change what it samples, only how soon it gets there.
    rebuild_every : int, optional
        With ``transfer``, balance exploiting the carried ansatz against
        exploring: at every ``rebuild_every``-th geometry of a trajectory,
        after the transferred chain's ``transfer_steps``, a fresh chain from
        the empty ansatz searches for the full ``max_steps``, and the
        lowest-cost state of both is reported -- so a basin carried along a
        trajectory is challenged at least that often (default ``None``:
        never).  The result's ``rebuild`` and the run log's summary say
        which one won.
    transfer_threshold : float
        Smallest overlap :math:`|\\langle\\phi_p(X_n)|\\phi_{\\pi(p)}(X_{n+1})
        \\rangle|` between a previous orbital and its match at which the
        ansatz is still transferred (default
        :data:`~mandacaru.algorithms.orbital_tracking.DEFAULT_TRANSFER_THRESHOLD`).
    transfer_steps : int, optional
        Chain length when the chain starts from a transferred ansatz
        (default ``max_steps``).  A transferred start is usually close to
        the new geometry's answer, so a short local search is what saves
        quantum evaluations over rebuilding; a chain that had to start empty
        still takes ``max_steps``.  The temperature schedule spans the steps
        actually taken.
    proposal_temperature : float
        Softmax temperature of the gradient proposal, relative to the largest
        gradient (default :data:`DEFAULT_PROPOSAL_TEMPERATURE`): the steepest
        operator is ``exp(1/proposal_temperature)`` times as likely as one
        with zero gradient.  Small values approach ADAPT's greedy choice,
        large ones the uniform proposal.
    screen_insertions : int
        At every state the chain visits for the first time, also relax this
        many insertions and record them, flagged ``screened`` (default
        ``0``).  They are drawn without replacement from the state's own
        operator distribution -- the choice the proposal faces there -- each
        at a random slot, from a random stream of their own, so the chain's
        moves and acceptances are unchanged.  They are not chain moves: no
        acceptance test, no effect on the reported state.  What they buy is
        several measured insertions per state, which is what judges a
        proposal's choice *within* a state
        (:func:`~mandacaru.algorithms.proposal_model.assess_data_volume`).
        Their evaluations count in ``num_evaluations``.  Needs a ``record``
        directory.
    record : str, path or bool, optional
        Directory to append every proposal to -- rejected ones included --
        with the problem it searched (:mod:`~mandacaru.algorithms.proposal_data`):
        the training data of VALQA's learned proposal.  ``None`` (default)
        records into the shared store ``MANDACARU_PROPOSAL_DATA`` when that
        variable is set and records nothing otherwise; ``True`` requires the
        store; ``False`` records nothing.  Recording needs the
        pool gradients at every new state, so a ``"uniform"`` MCAS-VQE run that
        records also screens the pool (counted in ``num_screenings``).
    seed : int, optional
        Seed of the chain's random stream (proposals and acceptance).
    profile : bool
        Compile and profile the final ansatz (default ``True``).  Only the
        reported architecture is compiled, not every proposal.
    """

    _default_sparse = "auto"
    _supports_spin_orbit = True

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
                 optimizer: str | Optimizer | None = None,
                 mapping: str = "jordan_wigner",
                 device: str = "AER_simulator",
                 max_steps: int = 200,
                 min_length: int = 0,
                 max_length: int = 20,
                 move_weights: dict | None = None,
                 temperature=None,
                 length_penalty: float = 0.0,
                 warm_start: bool = True,
                 replace_start: str = "zero",
                 transfer: bool = False,
                 transfer_steps: int | None = None,
                 transfer_threshold: float = DEFAULT_TRANSFER_THRESHOLD,
                 rebuild_every: int | None = None,
                 screen_insertions: int = 0,
                 proposal_temperature: float = DEFAULT_PROPOSAL_TEMPERATURE,
                 record=None,
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
                "min_length must be 0: the chain starts from, or falls back "
                "to, the empty ansatz (the reference state)")
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
        if replace_start not in REPLACE_STARTS:
            raise ValueError(f"replace_start must be one of {REPLACE_STARTS}, "
                             f"got {replace_start!r}")
        self.replace_start = replace_start
        if not isinstance(transfer, bool):
            raise ValueError(f"transfer must be True or False, got "
                             f"{transfer!r}")
        self.transfer = transfer
        self.transfer_steps = (self.max_steps if transfer_steps is None
                               else int(transfer_steps))
        if self.transfer_steps < 0:
            raise ValueError(f"transfer_steps must be >= 0, got "
                             f"{transfer_steps!r}")
        self.rebuild_every = (None if rebuild_every is None
                              else int(rebuild_every))
        if self.rebuild_every is not None and self.rebuild_every < 1:
            raise ValueError(f"rebuild_every must be >= 1 or None, got "
                             f"{rebuild_every!r}")
        self.transfer_threshold = float(transfer_threshold)
        if not 0.0 <= self.transfer_threshold <= 1.0:
            raise ValueError(f"transfer_threshold must be in [0, 1], got "
                             f"{transfer_threshold!r}")
        #: Where the chain started and why (the setup block's ``start``).
        self._start_description = "empty ansatz"
        #: The previous geometry's reported ansatz (operator labels, angles),
        #: handed over by the calculator (:meth:`inherit_ansatz`).
        self._inherited: _Inherited | None = None
        #: This chain's trajectory (a new one unless it continues the
        #: previous geometry's), its step there, and the problem it searched.
        self._trajectory = uuid.uuid4().hex[:12]
        self._step = 0
        self._problem: ProblemRecord | None = None
        self._linked = False
        #: The evaluated insertions, kept only when ``_collects_insertions``.
        self._insertions: list[dict] = []
        self.proposal_temperature = float(proposal_temperature)
        gradient_softmax([1.0, 0.0], self.proposal_temperature)   # validates
        self.record = resolve_record(record)
        self.screen_insertions = int(screen_insertions)
        if self.screen_insertions < 0:
            raise ValueError(f"screen_insertions must be >= 0, got "
                             f"{screen_insertions!r}")
        if self.screen_insertions and self.record is None:
            raise ValueError(
                "screen_insertions records what it measures: give a record "
                "directory (record=DIR, or set MANDACARU_PROPOSAL_DATA)")
        self._recorder: EditRecorder | None = None
        self.seed = None if seed is None else int(seed)
        self._adopt_problem(hamiltonian, num_particles, n_spatial_orbitals)

    # -- transfer between geometries -------------------------------------- #

    def inherit_ansatz(self, previous) -> None:
        """Take over the ansatz ``previous`` -- the solver of the preceding
        geometry -- reported, when ``transfer=True``.

        The ASE calculator calls this on each new geometry's solver before
        running it.  Only a chain of the same method passes its ansatz on,
        with the molecular orbitals it was expressed in.
        """
        result = getattr(previous, "result", None)
        if type(previous) is not type(self) \
                or not isinstance(result, MCASVQEResult):
            return
        context = getattr(previous, "_gradient_context", None) or {}
        pool = getattr(previous, "pool", None)
        self._inherited = _Inherited(
            operators=list(result.operators),
            angles=np.asarray(result.optimal_parameters, dtype=float),
            pool_labels=[op.label for op in previous._pool_ops],
            symbols=getattr(previous, "_basis_symbols", None),
            orbitals=(OrbitalSnapshot.from_integrals(
                context.get("integrals"),
                getattr(pool, "n_spatial_orbitals", 0)) if self.transfer
                else "not needed without transfer"),
            trajectory=previous._trajectory, step=previous._step,
            problem=previous._problem)

    def _link_trajectory(self, labels) -> bool:
        """Join the previous geometry's trajectory when it searched the same
        system and pool; start a new one otherwise.  Returns whether linked."""
        inherited = self._inherited
        linked = (inherited is not None
                  and inherited.symbols == getattr(self, "_basis_symbols",
                                                   None)
                  and inherited.pool_labels == list(labels))
        if linked:
            self._trajectory = inherited.trajectory or self._trajectory
            self._step = inherited.step + 1
        return linked

    def _previous_problem(self) -> ProblemRecord | None:
        """The previous geometry's problem when this chain continues its
        trajectory and it differs from this one's."""
        inherited = self._inherited
        if not self._linked or inherited.problem is None:
            return None
        if self._problem is not None \
                and inherited.problem.key == self._problem.key:
            return None
        return inherited.problem

    def _transferred_start(self, labels) -> tuple[tuple, np.ndarray, str]:
        """The architecture and angles the chain starts from, and why."""
        empty = (), np.zeros(0)
        if not self.transfer:
            return *empty, "empty ansatz"
        inherited = self._inherited
        if inherited is None:
            return *empty, "empty ansatz (transfer: no previous geometry)"
        refused = "empty ansatz (transfer refused: {})".format
        if not self._linked:
            # The same label names a different excitation in another
            # molecule, charge or basis.
            return *empty, refused("another system or pool than the previous "
                                   "geometry's")
        operators, angles = inherited.operators, inherited.angles
        context = getattr(self, "_gradient_context", None) or {}
        integrals = context.get("integrals")
        current = OrbitalSnapshot.from_integrals(
            integrals, getattr(self.pool, "n_spatial_orbitals", 0))
        if isinstance(inherited.orbitals, str) or isinstance(current, str):
            reason = (inherited.orbitals if isinstance(inherited.orbitals, str)
                      else current)
            tracking = f"orbitals not tracked: {reason}"
        else:
            overlap = orbital_overlap(inherited.orbitals, current,
                                      integrals.grid)
            match = match_orbitals(overlap, occupation_blocks(
                len(overlap), self.pool.num_particles))
            if match.confidence < self.transfer_threshold:
                return *empty, refused(
                    f"smallest matched orbital overlap "
                    f"{match.confidence:.3f} < transfer_threshold "
                    f"{self.transfer_threshold:g}")
            moved = transfer_ansatz(operators, angles, match, len(overlap),
                                    labels)
            if isinstance(moved, str):
                return *empty, refused(moved)
            operators, angles = moved
            tracking = match.describe()
        if len(operators) > self.max_length:
            return *empty, refused(f"{len(operators)} operators > max_length "
                                   f"{self.max_length}")
        index = {label: i for i, label in enumerate(labels)}
        architecture = tuple(index[label] for label in operators)
        x0 = angles if self.warm_start else np.zeros(len(architecture))
        return architecture, x0, (
            f"previous geometry's ansatz, {len(architecture)} operator(s), "
            f"{'angles carried' if self.warm_start else 'angles from zero'}; "
            f"{tracking}")

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
                                      cached.operator_probabilities),
                                  gradients=cached.gradients)
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
        if not architecture:
            self._reference_ha = evaluated.energy
        if self._draws_from_gradients or self._recorder is not None:
            # The proposal out of this state, from the pool gradients at its
            # relaxed angles.  Needed whether or not it is accepted -- a
            # rejected proposal's distribution is the reverse probability.
            psi = (ansatz.state(evaluated.parameters) if architecture
                   else ansatz.reference_state())
            evaluated.gradients = self._analytic_gradients(psi)
            self._screenings += 1
        evaluated.operator_probabilities = self._operator_distribution(
            evaluated)
        if not self.warm_start:
            self._cache[architecture] = evaluated
        return evaluated

    # -- the proposal: what MCAS-VQE and VALQA define ------------------------- #

    #: Whether the operator distribution needs the pool gradients.
    _draws_from_gradients = True
    #: Whether the proposal needs the run's problem record (a learned model
    #: is bound to it); recording needs it regardless.
    _needs_problem = False

    def _gradient_distribution(self, evaluated: _Evaluated) -> np.ndarray:
        """The gradient softmax at ``evaluated``'s relaxed state."""
        return gradient_softmax(evaluated.gradients,
                                self.proposal_temperature)

    def _operator_distribution(self, evaluated: _Evaluated):
        """The distribution new operators are drawn from at ``evaluated``
        (``None``: uniform).  A fixed function of the state, so the
        Metropolis-Hastings ratio stays exact."""
        raise NotImplementedError

    def _prepare_proposal(self, problem: ProblemRecord | None) -> None:
        """Set the proposal up for this run's problem (``None``: empty pool)."""

    def _proposal_setup(self) -> dict:
        """The ``[OPTIMIZATION SETUP]`` lines describing the proposal."""
        raise NotImplementedError

    def _result_extras(self) -> dict:
        """Fields the result class adds to :class:`MCASVQEResult`."""
        return {}

    _result_class = MCASVQEResult

    def _problem_record(self) -> ProblemRecord:
        """The Hamiltonian, reference and pool this run searches."""
        return ProblemRecord.from_problem(
            self.hamiltonian, self._new_ansatz().reference_qubits(),
            self._pool_ops)

    def _open_recorder(self, problem, geometry) -> EditRecorder | None:
        if self.record is None or problem is None:
            return None
        from ..version import __version__
        previous = self._previous_problem()
        recorder = EditRecorder(self.record, problem, run={
            "run": uuid.uuid4().hex[:12], "method": self.log_title,
            "mandacaru": __version__, "formula": _formula(geometry),
            "pool": getattr(self.pool, "name", "?"),
            "proposal": self._proposal_setup()["proposal"],
            "model": self._model_version(),
            "warm_start": self.warm_start,
            "length_penalty": self._length_penalty_ha,
            "optimizer": self.optimizer.method, "seed": self.seed,
            "trajectory": self._trajectory, "geometry_step": self._step,
            "previous_problem": None if previous is None else previous.key})
        if previous is not None:
            # Training conditions on it, so it goes into the same store.
            previous.save(recorder.directory)
        return recorder

    def _model_version(self) -> str | None:
        """The version of the model drawing proposals, if any."""
        return None

    def _keep_row(self, row: dict) -> None:
        """Record an evaluated edit, and keep an insertion for the next
        geometry when the method asks for it."""
        if self._recorder is not None:
            self._recorder.write(row)
        if self._collects_insertions and row["move"] == "insert":
            before = self._previous_problem()
            self._insertions.append({
                **row, "problem": self._problem.key,
                "previous_problem": None if before is None else before.key})

    def _screen(self, state: _Evaluated, step: int, beta: float,
                rng: np.random.Generator) -> tuple[int, int, int]:
        """Relax and record ``screen_insertions`` insertions at ``state``;
        returns ``(insertions, function evaluations, optimizer steps)``."""
        size = len(self._pool_ops)
        p = (np.full(size, 1.0 / size)
             if state.operator_probabilities is None
             else np.asarray(state.operator_probabilities, dtype=float))
        k = min(self.screen_insertions, int(np.count_nonzero(p > 0.0)))
        nfev = nit = 0
        for mu in rng.choice(size, size=k, replace=False, p=p / p.sum()):
            action = Action("insert",
                            int(rng.integers(len(state.architecture) + 1)),
                            operator=int(mu))
            x0 = (action.transfer(state.parameters, self.replace_start)
                  if self.warm_start
                  else np.zeros(len(state.architecture) + 1))
            evaluated = self._evaluate(action.apply(state.architecture), x0)
            nfev += evaluated.nfev
            nit += evaluated.nit
            row = self._edit_row(step, action, state, evaluated, None, None,
                                 None, None, beta)
            row.update(screened=True, accepted=None)
            self._keep_row(row)
        return k, nfev, nit

    #: Whether the chain keeps its evaluated insertions in memory for the
    #: next geometry (VALQA's update between geometries).
    _collects_insertions = False

    def _edit_row(self, step, action, source: _Evaluated,
                  proposed: _Evaluated, log_forward, log_reverse,
                  log_alpha, accepted, beta) -> dict:
        """One ``edits.jsonl`` row: the state before, the edit, the outcome."""
        mu = action.operator
        probs = source.operator_probabilities
        drawn = (None if mu is None
                 else float(probs[mu]) if probs is not None
                 else 1.0 / len(self._pool_ops))
        return {
            "step": step, "move": action.move, "position": action.position,
            "operator": mu, "other": action.other,
            "replaced": (source.architecture[action.position]
                         if action.move == "replace" else None),
            "source": list(source.architecture),
            "proposed": list(proposed.architecture),
            "source_energy": source.energy,
            "reference_energy": self._reference_ha,
            "delta_energy": proposed.energy - source.energy,
            "delta_cost": proposed.cost - source.cost,
            "source_gradients": (None if source.gradients is None
                                 else [float(g) for g in source.gradients]),
            "operator_probability": drawn,
            "log_q_forward": log_forward, "log_q_reverse": log_reverse,
            "log_acceptance": log_alpha, "accepted": bool(accepted),
            "temperature": 0.0 if math.isinf(beta) else 1.0 / beta,
            "nfev": proposed.nfev, "optimizer_success": proposed.success,
            "screened": False}

    # -- the chain -------------------------------------------------------- #

    def run(self, geometry=None, cell=None) -> MCASVQEResult:
        """Run the chain for ``max_steps`` proposals and report the best state.

        ``geometry`` (an ASE ``Atoms`` or a ``(symbols, positions)`` pair) and
        ``cell`` fill the ``[SYSTEM]`` block of the run log, as for ADAPT-VQE;
        the calculator passes the geometry it evaluates.
        """
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured:
            raise RuntimeError(
                "MCAS-VQE has no Hamiltonian; construct it with one, or use it "
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
        temperature_ha = (_temperature_in_hartree(DEFAULT_TEMPERATURE, "eV")
                          if self.temperature is None
                          else _temperature_in_hartree(self.temperature, unit))
        self._length_penalty_ha = float(to_hartree(self.length_penalty, unit))
        self._cache: dict = {}
        #: Pool-gradient screenings, one per new state under the gradient
        #: proposal: part of the search cost, so it is reported.
        self._screenings = 0
        self._reference_ha = None
        rng = np.random.default_rng(self.seed)
        problem = self._problem_record() if kernel is not None and (
            self.record is not None or self._needs_problem) else None
        self._problem = problem
        self._insertions = []        # this run's, not a previous run's
        self._linked = self._link_trajectory(labels)
        self._prepare_proposal(problem)
        logger = None
        self._recorder = self._open_recorder(problem, geometry)
        # Everything after the recorder opens is inside the try, so the
        # edit file is closed whatever fails, the first evaluation included.
        try:
            self._show_banner()
            with timings.time("parameter optimization"):
                current = self._evaluate((), np.zeros(0))
            reference_energy = current.energy
            reference = current
            start_architecture, x0, start = self._transferred_start(labels)
            seen = {current.architecture}
            total_evals = total_steps = 0
            failures: list[tuple[int, str]] = []
            if start_architecture and kernel is None:
                start = "empty ansatz (empty pool)"
            elif start_architecture:
                with timings.time("parameter optimization"):
                    transferred = self._evaluate(start_architecture, x0)
                total_evals += transferred.nfev
                total_steps += transferred.nit
                seen.add(transferred.architecture)
                if not transferred.success:
                    failures.append((0, transferred.message))
                if transferred.cost < current.cost:
                    current = transferred
                else:
                    start = (f"empty ansatz (transfer refused: the previous "
                             f"geometry's ansatz relaxed to a cost "
                             f"{transferred.cost - current.cost:+.3e} Ha "
                             f"above the reference's)")
            # Every rebuild_every-th geometry of a trajectory, a fresh full
            # search challenges the transferred ansatz, which stays a
            # candidate for the reported state.
            # Every rebuild_every-th geometry of a trajectory, a fresh full
            # search from the empty ansatz follows the transferred chain, and
            # the better state of the two is reported: the carried basin is
            # challenged at least that often.
            rebuild = bool(self.rebuild_every and self._linked
                           and self._step % self.rebuild_every == 0
                           and current.architecture)
            if rebuild:
                start += (f"; periodic rebuild (every {self.rebuild_every} "
                          f"geometries): a fresh search from the empty "
                          f"ansatz follows")
            self._start_description = start
            if current.architecture:          # a transferred start
                n_steps = self.transfer_steps
            schedule = TemperatureSchedule(temperature_ha, max(n_steps, 1))
            logger = self._make_logger(self.log_targets, geometry, cell,
                                       reference_energy, schedule)

            start_state = current
            best_cost = best_energy = current
            steps: list[MCASStep] = []
            tried = {move: 0 for move in MOVES}
            taken = {move: 0 for move in MOVES}

            screened_states: set = set()
            screened = 0
            # Its own stream: screening must not move the chain's draws.
            screen_rng = np.random.default_rng(
                None if self.seed is None else [self.seed, 1])

            def segment(current, length, schedule, offset):
                """Run ``length`` proposals from ``current``; steps are
                numbered after ``offset``.  Returns where the chain ends."""
                nonlocal total_evals, total_steps, screened, best_cost, \
                    best_energy
                for i in range(1, length + 1):
                    step = offset + i
                    beta = schedule.beta(i - 1)
                    if (self.screen_insertions and self._recorder is not None
                            and current.architecture not in screened_states
                            and len(current.architecture) < self.max_length):
                        screened_states.add(current.architecture)
                        with timings.time("parameter optimization"):
                            k, nfev, nit = self._screen(current, step, beta,
                                                        screen_rng)
                        screened += k
                        total_evals += nfev
                        total_steps += nit
                    action = kernel.sample(current.architecture, rng,
                                           current.operator_probabilities)
                    if action is None:
                        # Unreachable while insert has weight and
                        # max_length > 0, which the kernel enforces; a silent
                        # self-loop would hide it.
                        raise RuntimeError(
                            f"no architecture move is valid at "
                            f"{current.architecture!r}")
                    proposed_arch = action.apply(current.architecture)
                    x0 = (action.transfer(current.parameters,
                                          self.replace_start)
                          if self.warm_start
                          else np.zeros(len(proposed_arch)))
                    with timings.time("parameter optimization"):
                        proposed = self._evaluate(proposed_arch, x0)
                    total_evals += proposed.nfev
                    total_steps += proposed.nit
                    if not proposed.success:
                        failures.append((step, proposed.message))
                    seen.add(proposed.architecture)

                    log_forward = kernel.log_q(
                        proposed_arch, current.architecture,
                        current.operator_probabilities)
                    log_reverse = kernel.log_q(
                        current.architecture, proposed_arch,
                        proposed.operator_probabilities)
                    log_alpha = log_acceptance(
                        beta, current.cost, proposed.cost, log_forward,
                        log_reverse)
                    accepted = metropolis_accept(log_alpha, rng)
                    if self._recorder is not None or self._collects_insertions:
                        self._keep_row(self._edit_row(
                            step, action, current, proposed, log_forward,
                            log_reverse, log_alpha, accepted, beta))
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
                        schedule.temperature(i - 1)))
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
                            current_energy=self._to_energy_units(
                                current.energy),
                            temperature=temperature, log_acceptance=log_alpha,
                            accepted=accepted, energy_unit=unit,
                            optimizer_steps=(proposed.nit if proposed.nfev
                                             else None),
                            action=action.describe(short))
                return current

            current = segment(current, n_steps, schedule, 0)
            challenger = None
            if rebuild:
                challenger = best_cost
                current = segment(reference, self.max_steps,
                                  TemperatureSchedule(temperature_ha,
                                                      max(self.max_steps, 1)),
                                  n_steps)

            # The reported state is what the calculator's forces, densities
            # and measured energies are taken from: the ansatz and the
            # parameters have to describe the same architecture.
            self.ansatz = self._ansatz_for(best_cost.architecture)
            with timings.time("circuit profiling"):
                metrics = self._profile(self.ansatz)

            def names(evaluated):
                return [labels[i] for i in evaluated.architecture]

            result = self._result_class(
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
                num_screened_insertions=screened,
                num_evaluations=total_evals,
                optimizer_steps=total_steps,
                optimizer_failures=failures,
                seed=self.seed,
                metrics=metrics,
                integration_profile=self._integration_profile,
                energy_unit=unit,
                start_operators=names(start_state), start=start,
                rebuild=_rebuild_outcome(challenger, best_cost),
                edit_distance_from_start=edit_distance(
                    start_state.architecture, best_cost.architecture),
                **self._result_extras())
            if logger is not None:
                self._write_summary(logger, result)
        finally:
            if logger is not None:
                logger.close()
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None

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
                **self._proposal_setup(),
                "recorded_edits": (str(self.record) if self.record is not None
                                   else "none"),
                "screened_insertions_per_state": self.screen_insertions,
                "max_steps": (f"{self.max_steps} ({self.transfer_steps} from "
                              f"a transferred ansatz)" if self.transfer
                              else self.max_steps),
                "ansatz_length": f"{self.min_length} to {self.max_length}",
                "move_weights": ", ".join(f"{move} {weight:g}" for move, weight
                                          in weights.items()),
                f"architecture_temperature_{unit}": temperature,
                f"length_penalty_{unit}": f"{self.length_penalty:g}",
                "warm_start": str(self.warm_start),
                "replace_start": self.replace_start,
                "start": self._start_description,
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

    def _write_summary(self, logger, result: MCASVQEResult) -> None:
        """The ``[VARIATIONAL QUANTUM SUMMARY]`` of the lowest-cost ansatz."""
        unit = result.energy_unit
        extra = {
            f"optimal_cost_{unit}": f"{result.optimal_cost:.10f}",
            f"best_energy_{unit}": f"{result.best_energy:.10f}",
            f"final_energy_{unit}": f"{result.final_energy:.10f}",
            "final_num_operators": len(result.final_operators),
            "chain_steps": len(result.steps),
            "acceptance_rate": f"{result.acceptance_rate:.4f}",
            "edit_distance_from_start": result.edit_distance_from_start,
            **({"periodic_rebuild": result.rebuild} if result.rebuild
               else {}),
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


def _rebuild_outcome(challenger, best) -> str | None:
    """Which ansatz a periodic rebuild reported, or ``None`` without one."""
    if challenger is None:
        return None
    if best is challenger:
        return "the transferred chain's ansatz was kept"
    return (f"the rebuilt ansatz won, by "
            f"{1000 * (challenger.cost - best.cost):.3f} mHa in cost")


def _temperature_in_hartree(temperature, unit: str):
    """The ``temperature`` option converted from ``unit`` to Hartree."""
    if isinstance(temperature, dict):
        return {key: float(to_hartree(float(value), unit))
                for key, value in temperature.items()}
    return float(to_hartree(float(temperature), unit))


def _formula(geometry) -> str | None:
    """The chemical formula of a run's geometry, if it has one."""
    if geometry is None:
        return None
    if hasattr(geometry, "get_chemical_formula"):
        return geometry.get_chemical_formula()
    try:
        from ase import Atoms
        return Atoms(list(geometry[0])).get_chemical_formula()
    except Exception:
        return None


class MCASVQE(MarkovChainSearch):
    """MCAS-VQE: the chain with a gradient-softmax or uniform proposal.

    Takes every option of :class:`MarkovChainSearch` and

    Parameters
    ----------
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
    """

    citation_method = "mcas-vqe"
    #: The name in the ``[SYSTEM]`` block's title.
    log_title = "MCAS-VQE"
    _needs_problem = False

    def __init__(self, hamiltonian=None, proposal: str = "gradient",
                 **chain_options):
        self.proposal = str(proposal).strip().lower()
        if self.proposal not in PROPOSALS:
            raise ValueError(f"unknown proposal {proposal!r}; use one of "
                             f"{PROPOSALS}")
        super().__init__(hamiltonian, **chain_options)

    @property
    def _draws_from_gradients(self) -> bool:
        return self.proposal == "gradient"

    def _operator_distribution(self, evaluated):
        if self.proposal == "gradient":
            return self._gradient_distribution(evaluated)
        return None

    def _proposal_setup(self) -> dict:
        return {"proposal": (
            f"gradient softmax (tau {self.proposal_temperature:g} of "
            f"max |grad|), Metropolis-Hastings"
            if self.proposal == "gradient"
            else "uniform, Metropolis-Hastings")}
