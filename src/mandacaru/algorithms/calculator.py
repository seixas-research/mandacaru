# -*- coding: utf-8 -*-
# file: algorithms/calculator.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The unified ASE calculator for every molecular variational method.

:class:`Mandacaru` is the single user-facing entry point for running a
variational quantum simulation: the eigensolver is selected by the ``method``
argument and every method-specific option is forwarded to it.  It reports the
**energy** for any method and, for the atom-centered bases, the analytic
**nuclear forces** (Hellmann-Feynman **plus** Pulay, see
:mod:`mandacaru.algorithms.forces`), so any ASE optimizer -- ``BFGS``, ``LBFGS``,
``FIRE``, ``QuasiNewton`` -- can drive a geometry optimization whose energies
come from a quantum variational eigensolver:

.. code-block:: python

    from ase.build import molecule
    from ase.optimize import BFGS
    from mandacaru.algorithms import Mandacaru

    water = molecule("H2O")
    water.center(vacuum=3.0)          # the cell is the real-space box
    water.calc = Mandacaru(method="adapt-vqe",
                           basis="FAO",
                           h=0.30,
                           frozen_core=True)
    BFGS(water).run(fmax=0.05)

The run result of the most recent evaluation is available uniformly on
:attr:`Mandacaru.result`, whatever the method
(``VQEResult`` / ``ADAPTVQEResult`` / the subspace results).
The calculator also exposes the two non-ASE entry points of the underlying
solvers: :meth:`run` (direct mode, e.g. from a cached Hamiltonian via
``load_hamiltonian=``) and :meth:`energy_levels` (excited states by variational
deflation).

Why the grid is frozen for forces
---------------------------------
This is the one subtlety that makes forces meaningful in a real-space code.  For
a plain energy the integration grid is regenerated for each geometry, centered
on the molecule -- exactly what the single-point examples want.  During a
relaxation that would make the grid *move with the atoms*, adding a spurious
"grid drag" to the energy surface: the computed forces (taken at fixed grid)
would then disagree with the finite difference of the energies, and the
optimizer would chase an artifact.

Whenever forces are requested, :class:`Mandacaru` therefore builds the
grid **once**, from the initial geometry's ``atoms.cell``, and reuses it for
every subsequent geometry.  Energies along the trajectory are then all
evaluated on one common grid, which is exactly the condition under which the
analytic gradient is the derivative of the reported energy.  The cell is the
box -- there is no padding argument on the calculator -- so size it in the
geometry (``atoms.center(vacuum=3.0)``, an explicit ``atoms.cell``, or the
``Lattice`` of an extended-XYZ file) generously enough that no atom
approaches the box edge during the relaxation; a geometry without a cell
raises ``ValueError``.  An explicit ``grid=`` is always used verbatim (and
frozen).
"""

from __future__ import annotations

import atexit
import sys
import warnings
from time import perf_counter as _perf

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..units import DEFAULT_GRID_SPACING

#: The default method: ADAPT-VQE, everywhere a ``method`` is not given.
DEFAULT_METHOD = "adapt-vqe"

#: Stable method names accepted by ``method=``.
STABLE_METHODS = ("vqe", "adapt-vqe", "subspace-vqe", "subspace-adapt-vqe")

#: Kept as the historical name of the stable list.
METHODS = STABLE_METHODS

# Methods registered by packages outside the stable API (see
# :func:`register_method`); nothing here names them.
_REGISTERED: dict[str, type] = {}


def register_method(name: str, solver_class: type) -> None:
    """Make ``Mandacaru(method=name)`` build ``solver_class``.

    The hook a package outside the stable API uses to plug its solvers into
    the unified calculator without the stable code knowing them by name.  The
    class must be a :class:`~mandacaru.algorithms.base.VariationalDriver`.
    """
    key = str(name).strip().lower()
    if key in STABLE_METHODS:
        raise ValueError(f"{name!r} is a stable method and cannot be replaced")
    _REGISTERED[key] = solver_class


def experimental_methods() -> tuple[str, ...]:
    """Names registered by :func:`register_method` (imported packages only)."""
    return tuple(_REGISTERED)


def available_methods() -> tuple[str, ...]:
    """Every method name ``method=`` accepts right now."""
    return STABLE_METHODS + experimental_methods()


def _method_key(name) -> str:
    """Spelling-insensitive key: ``"adapt-vqe"``, ``"adaptvqe"``, ``"ADAPT_VQE"``
    and ``"Adapt VQE"`` all name the same method."""
    return "".join(ch for ch in str(name).strip().lower() if ch not in "-_ ")


def resolve_method(name: str):
    """Return ``(canonical_name, solver_class)`` for a method spec.

    The stable solvers come from :mod:`mandacaru.algorithms`; any other name
    must have been registered with :func:`register_method` first (the
    :mod:`mandacaru.experimental` package does so when it is imported).
    """
    key = _method_key(name)
    known = {_method_key(method): method
             for method in (*STABLE_METHODS, *_REGISTERED)}
    canonical = known.get(key)
    if canonical in STABLE_METHODS:
        # The solver classes are the internal layer: this is the one place
        # they are reached from, and ``Mandacaru(method=...)`` the one way in.
        from .adapt_vqe import ADAPTVQE
        from .subspace import SubspaceADAPTVQE, SubspaceVQE
        from .vqe import VQE
        return canonical, {"vqe": VQE, "adapt-vqe": ADAPTVQE,
                           "subspace-vqe": SubspaceVQE,
                           "subspace-adapt-vqe": SubspaceADAPTVQE}[canonical]
    if canonical is not None:
        return canonical, _REGISTERED[canonical]
    raise ValueError(
        f"unknown method {name!r}; use one of {available_methods()}")


#: Whether standard output has already been put in line-buffered mode.
_STDOUT_LINE_BUFFERED = False


def _watchable_stdout() -> None:
    """Make a **redirected** standard output show each line as it is written.

    With the trace routed to a log file the terminal gets only what an ASE
    optimizer prints -- one line of about sixty bytes per step.  Python
    block-buffers a redirected stream and ASE's ``Optimizer.log`` does not flush
    after writing, so ``python run.py > run.log`` would show nothing for minutes
    at a time and a running relaxation would be indistinguishable from a hung
    one.  Line buffering costs nothing at that rate; an interactive terminal is
    line-buffered already, so this changes only the redirected case.

    Done once per process, and defensively: a replaced ``sys.stdout`` (a test
    harness's, a notebook's) may not support reconfiguration, and failing to
    make output prettier must never fail a calculation.
    """
    global _STDOUT_LINE_BUFFERED
    if _STDOUT_LINE_BUFFERED:
        return
    _STDOUT_LINE_BUFFERED = True
    try:
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


#: Largest orbital-rotation residual (Hartree) the RDM gradient accepts quietly.
#:
#: The residual is ``max |dE/dkappa|`` with the density matrices held fixed, and
#: it vanishes at **both** ends of the correlation range: for the Hartree-Fock
#: determinant (RHF is orbital-stationary) and for the exact ground state of the
#: same orbital space (a full CI there is invariant under any rotation of those
#: orbitals).  Measured on H2O / PAW-SZ at h = 0.25: 1.2e-9 Ha for the HF
#: determinant, 1.8e-9 for the sector FCI, and 1.0e-2 for the ADAPT state that
#: stalls 5.4e-4 Ha above it.  So a nonzero residual says the ansatz stopped
#: short of that exact state -- to *first* order in the state error, where the
#: energy only shows it to second.
ORBITAL_RESPONSE_TOLERANCE = 1e-3

#: Remove the spurious net force before reporting it (see
#: :meth:`Mandacaru._project_translation`).
#:
#: ``"auto"`` -- the default -- projects whenever the identity it enforces
#: actually holds: a non-periodic system of two or more atoms.  That is not a
#: cosmetic choice.  The exact force of a free molecule sums to zero, so
#: subtracting the mean is an **orthogonal projection onto a subspace that
#: contains the true answer**, and such a projection can only reduce the
#: distance to it -- the reported force is never made worse and is usually
#: better.  What it gives up is the identity ``forces == -dE/dR`` of the
#: *discretized* energy, which is why the unprojected array stays available in
#: ``force_result.details["forces_unprojected"]`` and the unprojected residual
#: in ``details["translational_residual"]``: a finite-difference check has to
#: compare against those.
DEFAULT_PROJECT_TRANSLATION = "auto"

#: Accepted values of ``project_translation``.
PROJECT_TRANSLATION_CHOICES = (True, False, "auto")

#: A free-standing molecule feels no net force: translation is a symmetry of
#: the exact energy, so ``sum_A F_A`` is zero and whatever comes out instead is
#: pure discretization artifact (the grid "egg-box").  The check is free -- the
#: forces are already computed -- and rigorous, so it is the cheapest honest
#: measure of whether a force is usable for geometry.  A warning is raised when
#: the residual exceeds this many eV/Angstrom **and** this fraction of the
#: largest force; a sharp all-electron core on a coarse grid can put hundreds
#: of eV/Angstrom here while every other diagnostic stays quiet.
TRANSLATIONAL_RESIDUAL_TOLERANCE = 0.05
TRANSLATIONAL_RESIDUAL_FRACTION = 0.05


class Mandacaru(Calculator):
    """ASE calculator running the variational method named by ``method``.

    Parameters
    ----------
    method : str
        Which variational eigensolver evaluates the energy -- ``"adapt-vqe"``
        (the default), ``"vqe"``, or the subspace-search variants
        ``"subspace-vqe"`` / ``"subspace-adapt-vqe"``.  ADAPT-VQE is the
        practical choice for anything beyond a couple of orbitals: a fixed UCCSD
        ansatz becomes very slow past ~8 qubits.  A method registered through
        :func:`register_method` is accepted by name as well.
    basis : str or dict
        Basis family, as for the solvers (default ``"FAO"``); accepts a
        ``{"name": ..., <options>}`` dict, including the periodic plane-wave
        family (energy only -- plane waves carry no forces) and the
        pseudopotential families ``"NCPP"`` / ``"ONCVPSP"`` / ``"PAW"``
        (``{"name": "PAW", "size": "DZP"}``), which replace the all-electron
        problem by a valence-only one.
    h : float
        Grid spacing in Angstrom (default ``0.20``), used both for the
        per-geometry grid built from ``atoms.cell`` and for the frozen force
        grid.  The box itself is always ``atoms.cell`` -- the geometry must
        carry one (``atoms.center(vacuum=...)``, ``atoms.cell = ...`` or an
        extended-XYZ ``Lattice``), or a ``ValueError`` is raised.
    grid : Grid, optional
        An explicit grid, used verbatim (and frozen) for every evaluation.
    trace : bool, optional
        Whether the solver prints its full run trace (configuration header,
        per-iteration table, timings) to **standard output**.  ``None`` (the
        default) decides automatically: **off** when ``output=<path>`` gives the
        detail a destination, **on** when it does not.  With it off, standard
        output carries only the evolution of energies and forces an ASE
        optimizer prints there (``Step Time Energy fmax``) -- the same split
        GPAW makes with ``txt=``.  ``True`` / ``False`` force it either way; a
        single-point run with the trace off reports nothing to the terminal (the
        energy is the return value, and the log file has the rest).
    measurement_provider : CircuitProvider, optional
        Measure the optimized state instead of reading the local state vector:
        the ansatz is still optimized locally, then every Pauli string of the
        Hamiltonian and of the RDM operators is measured on this provider in
        one Estimator job per geometry (e.g. ``QiskitProvider(device=
        "ibm_fez", shots=4096)``), and both the ASE energy and the forces come
        from those expectation values.  Small registers only (see
        :data:`~mandacaru.algorithms.rdm.MAX_PAULI_RDM_MODES`).  The last
        measurement is on :attr:`measurement`.
    include_pulay : bool
        Include the Pulay (basis-motion) force terms (default ``True``).  Setting
        it to ``False`` gives the bare Hellmann-Feynman force; for an atom-centered
        basis that is **not** the gradient of the energy and will not relax to the
        right geometry -- it is exposed for analysis, not for production.
    project_translation : {"auto", True, False}
        Subtract the mean force from every atom before reporting, so the forces
        sum to zero.  ``"auto"`` (the **default**) does it wherever the identity
        it enforces holds: a non-periodic system of two or more atoms.

        A free molecule's exact forces *do* sum to zero, so removing the mean is
        an orthogonal projection onto a subspace that contains the true force --
        it can only move the reported force closer to it, never further away.
        What it gives up is the identity ``forces == -dE/dR`` of the
        *discretized* energy, which is genuinely not translation invariant: so a
        finite-difference check must compare against
        ``force_result.unprojected``, and the residual that was removed stays on
        ``details["translational_residual"]`` (with the mean on
        ``details["translation_removed"]``).  Use ``False`` when you need the
        raw gradient.

        It fixes only the *translational* component of the grid's egg-box: the
        force differences that bend and stretch the molecule keep whatever error
        the spacing gives them, so it stops a relaxation drifting across the grid
        without making a coarse grid trustworthy.  Under periodic boundary
        conditions the sum over a cell's atoms is not this identity, so nothing
        is projected there and an explicit ``True`` says so.
    force_method : {"rdm", "scf-response"}
        How the nuclear gradient is taken.  ``"rdm"`` (default) differentiates
        the energy expression the solver actually reported, holding the reduced
        density matrices and molecular orbitals fixed
        (:func:`~mandacaru.algorithms.pseudo_forces.pseudo_nuclear_gradient`): it
        is complex-safe, covers pseudopotential projectors, augmented overlaps
        and compensation charges when the family has them, refills a frozen
        core, and reports its orbital-response residual in
        ``force_result.details["orbital_gradient"]``.  ``"scf-response"`` is the
        legacy path through a real-arithmetic differentiated SCF
        (:func:`~mandacaru.algorithms.forces.nuclear_gradient`); it assumes a
        closed-shell real reference and does not carry the pseudopotential
        terms, and is kept only for comparison.
    hellmann_feynman : {"analytic", "by-parts"}
        How the electron-nucleus force term is evaluated.  Keep the default
        ``"analytic"``: it is exactly the derivative of the reported energy and
        is verified against finite differences to 0.04 % for H2.

        ``"by-parts"`` is an independent formulation (the derivative moved onto
        the density) provided for **diagnosis only** -- it does *not* cure the
        heavy-atom force problem and is not the gradient of the computed energy.
        See :func:`~mandacaru.algorithms.forces.hellmann_feynman_gradient` for the
        measurements.
    charge, frozen_core, frozen_orbitals, mapping, optimizer, pool, ... :
        Forwarded to the solver selected by ``method``.
    dry_run : bool
        Forwarded too: with ``dry_run=True`` every evaluation only *estimates*
        the qubit requirements (no integrals, no mapping, no circuits), stores
        them on :attr:`dry_run_result` and reports ``NaN`` energies.  For a
        one-off estimate without changing the calculator, use :meth:`dry_run`.

    Notes
    -----
    Every ``calculate`` runs a full variational optimization, so an ASE
    relaxation step costs one complete solver run.  The converged result of the
    most recent evaluation is available on :attr:`result`, the solver instance
    on :attr:`solver`, and the force breakdown on :attr:`force_result`.

    With ``output=<path>`` (forwarded to the solver) every step **appends** its
    own block to that one file -- the iteration table, then the forces of that
    geometry -- so the whole trajectory is in one log rather than the last step
    overwriting the rest; see :mod:`mandacaru.utils.logging`.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, method: str = DEFAULT_METHOD, *, basis="FAO",
                 h: float = DEFAULT_GRID_SPACING, grid=None,
                 include_pulay: bool = True, force_method: str = "rdm",
                 project_translation: bool = DEFAULT_PROJECT_TRANSLATION,
                 hellmann_feynman: str = "analytic", orbital_delta=None,
                 scf_iterations: int = 40,
                 measurement_provider=None, trace: bool | None = None,
                 **solver_kwargs):
        Calculator.__init__(self)
        self.method, self._solver_class = resolve_method(method)
        self.basis = basis
        self.h = float(h)
        self.include_pulay = bool(include_pulay)
        # `1 in (True, False, "auto")` is True in Python, so the check is by
        # type: an int is a typo for a boolean, not a boolean.
        if not (isinstance(project_translation, bool)
                or project_translation == "auto"):
            raise ValueError(
                f"project_translation must be one of "
                f"{PROJECT_TRANSLATION_CHOICES}, got {project_translation!r}")
        self.project_translation = project_translation
        if str(force_method) not in ("rdm", "scf-response"):
            raise ValueError(f"force_method must be 'rdm' or 'scf-response', "
                             f"got {force_method!r}")
        self.force_method = str(force_method)
        self.hellmann_feynman = str(hellmann_feynman)
        self.orbital_delta = orbital_delta
        self.scf_iterations = int(scf_iterations)
        if "verbose" in solver_kwargs:
            # Refused loudly rather than ignored: `verbose` was the solver's own
            # flag and silencing it left a run that looked as though it produced
            # nothing.  `trace=` is the supported control, and it routes the
            # detail rather than discarding it.
            raise TypeError(
                "Mandacaru() does not take `verbose`: use `trace=` to control the "
                "standard-output trace (None = automatic: off when `output=` "
                "routes the detail to a file, on otherwise; True / False force "
                "it).  The structured per-iteration log is written with "
                "`output=<path>`, and the operator pool and Hamiltonian with "
                "`verbose_operators=` / `verbose_hamiltonian=`.")
        if trace is not None and not isinstance(trace, bool):
            raise TypeError(f"trace must be True, False or None, got {trace!r}")
        self.trace = trace
        self.solver_kwargs = dict(solver_kwargs)
        self._check_solver_options()
        self.measurement_provider = measurement_provider
        #: Energy, RDMs and expectation values of the last measured state.
        self.measurement = None

        # An explicit grid is frozen from the start; otherwise the grid is only
        # frozen once forces are requested (see the module docstring).
        self._grid = grid
        #: :class:`~mandacaru.algorithms.forces.ForceResult` of the most recent step.
        self.force_result = None
        #: One entry per geometry at which forces were computed -- the
        #: trajectory, as :meth:`write_optimization_summary` reports it.
        self.trajectory: list[dict] = []
        self._summary_written = False
        self._exit_hook = False
        # Options are validated *now*, by building the solver once with its
        # dry-run switch on: every constructor check runs, but nothing is
        # configured -- no pool, no Hamiltonian matrix, not even the terms of a
        # cache file.  The real solver is built on first use (:attr:`solver`),
        # so a calculator that is only asked for a `dry_run()` never
        # materializes the problem it is estimating.
        self._solver = None
        self._solver_is_fresh = False
        probe = self._make_solver(grid=self._grid, dry_run=True)
        if not self.solver_kwargs.get("dry_run", False):
            probe._check_shots_for_hardware()    # the one check a dry run waives

    # -- solver delegation ------------------------------------------------- #

    @property
    def solver(self):
        """The solver instance -- built on first use, replaced by every
        geometry evaluation.

        For a direct-mode problem (``hamiltonian=`` / ``load_hamiltonian=``)
        building it configures the problem -- pool, Hamiltonian matrix -- so
        ``calc.n_qubits``, ``calc.pool`` ... are available without a run; an
        invalid *problem* (a non-Hermitian operator, say) is refused at that
        point, an invalid *option* already by the constructor.
        """
        if self._solver is None:
            self._solver = self._make_solver(grid=self._grid)
            self._solver_is_fresh = True
        return self._solver

    @solver.setter
    def solver(self, value) -> None:
        self._solver, self._solver_is_fresh = value, False

    @property
    def result(self):
        """Run result of the most recent evaluation (``None`` before any run)."""
        return getattr(self._solver, "result", None)

    @property
    def dry_run_result(self):
        """:class:`~mandacaru.algorithms.dry_run.QubitEstimate` of the last dry run."""
        return getattr(self._solver, "dry_run_result", None)

    def dry_run(self, atoms=None):
        """Estimate the qubit requirements **without running** anything.

        Counts the basis functions of ``atoms`` (or of the attached geometry),
        resolves the charge / spin / frozen-core bookkeeping and the mapping
        exactly as a run would, and returns a
        :class:`~mandacaru.algorithms.dry_run.QubitEstimate` -- no integrals, no
        Hamiltonian, no circuits.  In direct mode (``load_hamiltonian=``) the
        cached file's header is all that is read.
        """
        if atoms is None:
            atoms = self.atoms
        # Built with the solver's own dry-run switch on, so constructing it
        # cannot configure the real problem (pool, Hamiltonian matrix): an
        # estimate that materialized what it estimates would be pointless.
        solver = self._make_solver(grid=self._grid, dry_run=True)
        estimate = solver.estimate_qubits(atoms)
        solver.dry_run_result = estimate
        self.solver = solver
        return estimate

    def __getattr__(self, name):
        # Only reached for names Mandacaru itself does not define: the solver's
        # own attributes (``pool``, ``ansatz``, ``kpoints``, ``energy_at``...)
        # are readable on the calculator, which is the single entry point.
        if "_solver" in self.__dict__ and not name.startswith("__"):
            try:
                return getattr(self.solver, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def _require_solver(self):
        if getattr(self.solver, "hamiltonian", None) is None:
            raise RuntimeError(
                "the calculator has not been evaluated yet; attach it to an "
                "Atoms object and get an energy, or call run()")
        return self.solver

    @property
    def hamiltonian(self):
        """Qubit Hamiltonian (:class:`~mandacaru.core.mapping.PauliSum`) of the last evaluation."""
        return self._require_solver().hamiltonian

    @property
    def n_qubits(self) -> int:
        """Qubit count of the last evaluation's active space."""
        return int(self._require_solver().n_qubits)

    @property
    def num_particles(self):
        """``(n_alpha, n_beta)`` of the last evaluation."""
        return self._require_solver().num_particles

    @property
    def mapping(self) -> str:
        """Fermion-to-qubit mapping of the last evaluation."""
        return self._require_solver().mapping

    #: Options whose whole purpose is to write a file, mapped to the capability
    #: a solver has to declare for them to do anything.
    REPORTING_OPTIONS = {"output": "writes_output_log",
                         "checkpoint": "supports_checkpoints",
                         "resume": "supports_checkpoints"}

    def _check_solver_options(self) -> None:
        """Refuse options the selected method would accept and ignore.

        Two ways an option can go nowhere.  It may not be a parameter of the
        solver at all -- ASE's ``Calculator`` keeps unknown keywords as
        *parameters*, so ``Mandacaru(method="vqe", output=...)`` used to compute an
        energy, write no file, and (because a path had been given) print no trace
        either: a run that reported nothing anywhere.  Or the solver may accept
        it and not act on it, which is the subspace methods' ``run()`` and the
        ``output`` / ``checkpoint`` / ``resume`` options it never reaches.
        """
        import inspect

        accepted: set[str] = set()
        for klass in self._solver_class.__mro__:
            init = klass.__dict__.get("__init__")
            if init is not None:
                accepted |= set(inspect.signature(init).parameters)
        unknown = sorted(set(self.solver_kwargs) - accepted)
        if unknown:
            raise TypeError(
                f"method {self.method!r} ({self._solver_class.__name__}) does "
                f"not take {', '.join(repr(name) for name in unknown)}; the "
                f"option would be kept as an inert ASE parameter.  Accepted "
                f"options are the solver's own constructor keywords.")
        ignored = [name for name, capability in self.REPORTING_OPTIONS.items()
                   if self.solver_kwargs.get(name) is not None
                   and not getattr(self._solver_class, capability, False)]
        if ignored:
            raise NotImplementedError(
                f"method {self.method!r} ({self._solver_class.__name__}) does "
                f"not write {', '.join(repr(name) for name in ignored)}: its "
                f"run() does not go through that machinery, so the option would "
                f"be silently ignored.  Use method='adapt-vqe' for the "
                f"structured log and checkpoints, or drop the option.")

    def _show_trace(self) -> bool:
        """Whether the solver prints its full trace to standard output.

        Automatic by default, on the same principle as GPAW's ``txt=``: with
        ``output=<path>`` the detail has a destination, so standard output is
        left to the **evolution of energies and forces** -- which is what an ASE
        optimizer prints there, in ASE's own ``Step Time Energy fmax`` format.
        Without ``output=`` the trace is the only report there is, so it is
        printed.  ``trace=True`` / ``False`` overrides either way.
        """
        if self.trace is not None:
            return self.trace
        # Routed away from the terminal only when the detail really lands in a
        # file: a solver that does not write one must not be silenced.
        return not (self.solver_kwargs.get("output") is not None
                    and getattr(self._solver_class, "writes_output_log", False))

    def _make_solver(self, grid, **overrides):
        options = {**self.solver_kwargs, **overrides}
        return self._solver_class(basis=self.basis, grid=grid, h=self.h,
                                  verbose=self._show_trace(), **options)

    def run(self, **run_kwargs):
        """Run the solver in **direct mode** (no geometry) and return its result.

        Direct mode needs a complete problem specification in the constructor
        keywords -- typically ``load_hamiltonian=`` (a cached qubit Hamiltonian),
        or an explicit ``hamiltonian`` with its companions.  Keyword arguments
        are forwarded to the solver's ``run``.
        """
        if not self._solver_is_fresh:       # reuse one built only to be read
            self._solver = self._make_solver(grid=self._grid)
        self._solver_is_fresh = False
        outcome = self.solver.run(**run_kwargs)
        if getattr(self.solver, "dry_run", False):
            # A dry run returns its estimate; nothing ran, so there is no result.
            self.solver.dry_run_result = outcome
        else:
            self.solver.result = outcome
        return outcome

    def interaction_energy(self, atoms, fragments, charges=None, **overrides):
        """``E(complex) - sum E(fragments)`` on one shared grid, with this
        calculator's method, basis and solver options.

        See :func:`~mandacaru.algorithms.interaction.interaction_energy`; the
        complex charge is this calculator's ``charge`` unless overridden.
        """
        from .interaction import interaction_energy

        options = dict(self.solver_kwargs)
        charge = int(options.pop("charge", 0))
        options.update(overrides)
        charge = int(options.pop("charge", charge))
        return interaction_energy(atoms, fragments, charges, charge=charge,
                                  method=self.method, basis=self.basis,
                                  h=self.h, grid=self._grid,
                                  **options)

    def energy_levels(self, num_states: int = 2, **solver_kwargs):
        """Excited states by variational deflation (see :mod:`mandacaru.algorithms.deflation`).

        Delegates to the current solver.  In calculator mode, evaluate an energy
        first (``atoms.get_potential_energy()``) so the Hamiltonian exists; in
        direct mode (``load_hamiltonian=``) it can be called immediately.
        """
        return self.solver.energy_levels(num_states, **solver_kwargs)

    # -- the frozen force grid --------------------------------------------- #

    def _frozen_grid(self, atoms):
        """Build the frozen integration grid once, from the *initial* geometry.

        The box is the geometry's own ``atoms.cell`` (centered on the
        molecule), exactly the grid a single-point energy would use -- there
        is no extra padding; a geometry without a cell raises ``ValueError``.
        """
        if self._grid is not None:
            return self._grid
        from ._hamiltonian_from_atoms import grid_from_cell

        self._grid = grid_from_cell(atoms, self.h)
        return self._grid

    @staticmethod
    def _require_atom_centered_basis(basis):
        """Reject the plane-wave family before any expensive work.

        Plane waves do not move with the nuclei, so they generate no Pulay
        forces and the gradient machinery does not apply.  Checking up front
        means the user finds out immediately instead of after a full
        variational run.
        """
        from ._hamiltonian_from_atoms import PER_ELEMENT, resolve_basis

        name, _options = resolve_basis(basis)
        if name == PER_ELEMENT:
            return                        # atom-centered by construction
        if name.upper().replace("-", "").replace(" ", "") in ("PW", "PLANEWAVE"):
            raise NotImplementedError(
                "nuclear forces need an atom-centered basis whose orbitals move "
                "with the nuclei; the plane-wave ('PW') family does not "
                "qualify. Use 'FAO', 'NAO', 'GTO', '6-31G(d)' or a "
                "pseudopotential family ('NCPP', 'ONCVPSP', 'PAW').")

    # -- ASE hook ---------------------------------------------------------- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """Run the variational solver; compute forces when they are requested.

        Sets ``results["energy"]`` / ``results["free_energy"]`` (eV) and, when
        forces are requested, ``results["forces"]`` (eV/Angstrom, ASE sign
        convention).
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms
        step_t0 = _perf()
        if not self._show_trace():
            # The terminal now carries only ASE's one line per step, which
            # nothing flushes: keep a redirected stream watchable.
            _watchable_stdout()

        want_forces = "forces" in properties
        # A previous step's breakdown must never survive a new geometry.
        self.force_result = None
        if want_forces:
            self._require_atom_centered_basis(self.basis)

        # Forces need one common grid along the whole trajectory; a plain
        # energy uses the solver's own per-geometry grid unless one was given.
        grid = self._frozen_grid(atoms) if want_forces else self._grid
        solver = self._make_solver(grid=grid)
        # This step's performance block is written here, once the gradient and
        # any measurement have been timed too (see :meth:`_log_performance`).
        solver.defer_performance = True
        energy_ev = self._single_point(solver, atoms)
        self.solver = solver

        stages: dict[str, float] = {}
        measured = None
        if self.measurement_provider is not None and not solver.dry_run:
            t0 = _perf()
            measured = self._measure(solver)
            stages["measurement (provider)"] = _perf() - t0
            energy_ev = measured["energy_eV"]

        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        if measured is not None:
            self._log_measurement(solver, measured)

        if want_forces and solver.dry_run:
            # A dry run computes nothing to differentiate.
            self.results["forces"] = np.full((len(atoms), 3), np.nan)
            return
        if want_forces:
            self._check_force_support(solver)
            t0 = _perf()
            if measured is None:
                self.force_result = self._forces(solver)
            else:
                self.force_result = self._forces(
                    solver, rdms=measured["rdms"],
                    reference_energy=measured["energy_hartree"])
            stages["nuclear gradient (forces)"] = _perf() - t0
            self.results["forces"] = self.force_result.forces
            self._log_forces(solver, atoms)
        step_seconds = _perf() - step_t0
        self._log_performance(solver, stages, step_seconds)
        if want_forces and not solver.dry_run:
            self._record_step(atoms, energy_ev, step_seconds)
        # The optimizer's line for the *previous* step is written after its
        # calculate returns, so this is what pushes it out on a stream that
        # could not be reconfigured.
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def _single_point(self, solver, atoms):
        """Attach the solver to a copy of ``atoms`` and get the energy in eV."""
        work = atoms.copy()
        work.calc = solver
        return float(work.get_potential_energy())

    # -- forces ------------------------------------------------------------ #

    def _check_force_support(self, solver) -> None:
        """Refuse force requests the derivative does not actually cover.

        The gradient differentiates *this* energy expression; a configuration
        it does not model must fail here rather than return a plausible number.
        """
        context = getattr(solver, "_gradient_context", None) or {}
        integrals = context.get("integrals")
        legacy = self.force_method == "scf-response"

        kinetic = getattr(integrals, "kinetic", "fd")
        if kinetic != "fd":
            raise NotImplementedError(
                f"forces differentiate the finite-difference Laplacian, but "
                f"the integrals use kinetic={kinetic!r}; rebuild the "
                f"calculator with kinetic='fd' (the default) for a gradient.")

        if getattr(solver, "shots", 0) and self.measurement_provider is None:
            raise NotImplementedError(
                "with shots > 0 the energy is measured but the forces would "
                "come from the internally simulated state, so they would not "
                "be the gradient of the reported energy; pass "
                "measurement_provider= to measure the density matrices too, "
                "or use shots=0.")

        if not legacy:
            return
        # The legacy path differentiates a real-arithmetic closed-shell replica
        # of the SCF -- see `algorithms/_jax_energy.py`.
        particles = getattr(solver, "num_particles", None)
        if particles is not None and particles[0] != particles[1]:
            raise NotImplementedError(
                f"force_method='scf-response' assumes a closed-shell "
                f"reference, but this run is open shell "
                f"(num_particles={tuple(particles)}); use the default "
                f"force_method='rdm'.")
        basis = getattr(integrals, "basis", None) or []
        if any(getattr(function, "l", 0) > 0 for function in basis):
            warnings.warn(
                "force_method='scf-response' runs a real-arithmetic replica of "
                "the SCF, but this basis carries l > 0 functions whose "
                "molecular orbitals are complex: the gradient is not "
                "consistent with the reported energy.  The default "
                "force_method='rdm' is complex-safe.",
                RuntimeWarning, stacklevel=3)

    def _measure(self, solver):
        """Energy and RDMs of the optimized state, from ``measurement_provider``.

        Every Pauli string the qubit Hamiltonian and the spin-conserving RDM
        operators need is measured in **one** PUB (one job); the energy and the
        RDMs are assembled from the same expectation values, so the energy the
        forces are checked against is the measured one.
        """
        from ..units import HARTREE_TO_EV
        from .rdm import rdm_qubit_operators, rdms_from_expectations

        reduced = solver.mapping == "parity_reduced"
        n_qubits = int(solver.n_qubits)
        n_modes = n_qubits + (2 if reduced else 0)
        ones, twos = rdm_qubit_operators(n_modes, solver.mapping,
                                         num_particles=solver.num_particles)
        hamiltonian = solver.hamiltonian
        identity = "I" * n_qubits
        labels = sorted({label for op in (hamiltonian, *ones.values(),
                                          *twos.values())
                         for label in op.terms} - {identity})
        provider = self.measurement_provider
        values, stds = provider.expectation_values(*solver.ansatz_problem()[:4],
                                                   labels)
        values[identity], stds[identity] = 1.0, 0.0
        energy = float(np.real(sum(complex(c) * values[label]
                                   for label, c in hamiltonian.terms.items())))
        job = getattr(provider, "last_job", None)
        self.measurement = {
            "energy_hartree": energy, "energy_eV": energy * HARTREE_TO_EV,
            "rdms": rdms_from_expectations(n_modes, ones, twos, values),
            "expectation_values": values, "stds": stds,
            "job_id": job.job_id() if job is not None else None}
        return self.measurement

    def _state_rdms(self, solver):
        """Spin-orbital RDMs of the converged state vector."""
        from .rdm import (one_rdm, pauli_expectations, rdm_qubit_operators,
                          rdms_from_expectations, two_rdm)

        psi = self._converged_state(solver)
        n_qubits = int(solver.n_qubits)
        if solver.mapping == "parity_reduced":
            # A tapered register has no ladder operators of its own: its RDM
            # elements are expectation values of the tapered qubit operators.
            n_modes = n_qubits + 2
            ones, twos = rdm_qubit_operators(n_modes, solver.mapping,
                                             num_particles=solver.num_particles)
            labels = {label for op in (*ones.values(), *twos.values())
                      for label in op.terms}
            return rdms_from_expectations(n_modes, ones, twos,
                                          pauli_expectations(psi, labels))
        sector = getattr(solver, "_sector", None)
        return (one_rdm(psi, n_qubits, solver.mapping, sector=sector),
                two_rdm(psi, n_qubits, solver.mapping, sector=sector))

    def _forces(self, solver, rdms=None, reference_energy=None):
        """Analytic nuclear gradient of the converged (or measured) state."""
        from .forces import nuclear_gradient

        context = getattr(solver, "_gradient_context", None)
        if context is None:
            raise NotImplementedError(
                "nuclear forces need an atom-centered basis whose integrals are "
                "available; the plane-wave ('PW') family does not qualify. Use "
                "an atom-centered basis such as 'FAO', 'GTO' or '6-31G(d)'.")

        gamma, gamma2 = self._state_rdms(solver) if rdms is None else rdms
        frozen = context.get("frozen") or ()
        if frozen:
            # The gradient contracts against the *full* integrals, so the
            # inert core has to be put back into the density matrices.
            from .rdm import expand_frozen_core

            gamma, gamma2 = expand_frozen_core(
                gamma, gamma2, frozen, len(context["integrals"].basis))

        if self.force_method == "rdm":
            # The complex-safe RDM gradient: projector coupling, augmented
            # overlap and compensation charges when the family has them, the
            # bare -Z/r when it does not.  It differentiates the same energy
            # expression the solver reported, for every atom-centered basis.
            from .pseudo_forces import (ENERGY_CHECK_TOLERANCE,
                                        pseudo_nuclear_gradient)
            result = pseudo_nuclear_gradient(
                context["integrals"], gamma, gamma2,
                atom_of_orbital=context["atom_of_orbital"],
                orbital_delta=self.orbital_delta,
                include_pulay=self.include_pulay)
            if reference_energy is None and not getattr(solver, "shots", 0):
                # `optimal_energy` is the scalar every result type has -- the
                # ground state, the one `_converged_state` returns -- whereas
                # `in_units()` is the whole spectrum for the subspace solvers.
                reference_energy = float(solver._from_energy_units(
                    solver.result.optimal_energy, "Ha"))
            if reference_energy is not None:
                reported = reference_energy
                rebuilt = result.details["energy_hartree"]
                if abs(reported - rebuilt) > ENERGY_CHECK_TOLERANCE:
                    raise RuntimeError(
                        "the energy rebuilt from the RDMs and molecular orbitals "
                        f"({rebuilt:.10f} Ha) differs from the solver's "
                        f"({reported:.10f} Ha); the force would not be the "
                        "gradient of the reported energy")
            residual = float(result.details.get("orbital_gradient", 0.0) or 0.0)
            if residual > ORBITAL_RESPONSE_TOLERANCE:
                warnings.warn(
                    f"the state is not stationary with respect to orbital "
                    f"rotations (max |dE/dkappa| = {residual:.2e} Ha): the "
                    f"gradient holds the molecular orbitals fixed, so it "
                    f"misses an orbital-response term of that size.  The "
                    f"residual is zero for the Hartree-Fock determinant and "
                    f"zero for the exact ground state of the same orbital "
                    f"space, so it measures how far the ansatz stopped short "
                    f"of that state -- and it is first order in that error "
                    f"where the energy is second, so tightening "
                    f"gradient_tolerance moves it very little once the run "
                    f"reports converged.  What moves it is expressivity: a "
                    f"different `pool`, or a growth that has not stalled "
                    f"(check result.converged against the final screening "
                    f"gradient).",
                    RuntimeWarning, stacklevel=3)
            self._check_translational_invariance(result)
            if self.project_translation and self._projection_applies(result):
                self._project_translation(result)
            return result

        legacy = nuclear_gradient(
            context["integrals"], gamma, gamma2,
            n_electrons=context["n_electrons"],
            atom_of_orbital=context["atom_of_orbital"],
            frozen=context["frozen"], orbital_delta=self.orbital_delta,
            scf_iterations=self.scf_iterations,
            include_pulay=self.include_pulay,
            hellmann_feynman=self.hellmann_feynman)
        self._check_translational_invariance(legacy)
        if self.project_translation and self._projection_applies(legacy):
            self._project_translation(legacy)
        return legacy

    def _log_forces(self, solver, atoms) -> None:
        """Append the step's forces to the run's ``output.txt``, when there is one.

        The solver logs the geometry's energies and closes its own log before
        the gradient is even computed, so the forces are appended afterwards --
        under the iteration table of the step they belong to.  A relaxation's
        log then reads as alternating energy and force blocks, one pair per
        geometry, which is what makes the convergence followable in the file
        rather than only in the terminal.
        """
        path = getattr(solver, "output", None)
        if path is None or self.force_result is None:
            return
        from ..utils.logging import append_forces

        result = self.force_result
        details = result.details
        extra = {"force_method": self.force_method,
                 "include_pulay": self.include_pulay,
                 "pulay_fraction": f"{result.pulay_fraction:.6f}",
                 "n_electrons": f"{result.n_electrons:.6f}"}
        # The two residuals that say whether this gradient can be trusted: the
        # orbital response it neglects and the net force the grid invents.
        for key in ("orbital_gradient", "translational_residual"):
            if details.get(key) is not None:
                extra[key] = f"{float(details[key]):.6e}"
        projected = bool(details.get("translation_projected"))
        if projected:
            removed = np.asarray(details["translation_removed"], dtype=float)
            extra["translation_projected"] = True
            extra["translation_removed"] = "[" + " ".join(
                f"{value:+.8f}" for value in removed) + "]"
        append_forces(path, atoms.get_chemical_symbols(), result.forces,
                      hellmann_feynman=result.hellmann_feynman,
                      pulay=result.pulay, extra=extra,
                      unprojected=result.unprojected if projected else None)

    def _log_measurement(self, solver, measured) -> None:
        """Record the measured energy -- the one ASE returns -- in the log.

        Without this the file's only energy is the variational one, which the
        measurement *replaced*: on a mocked provider the summary read
        -27.2113862460 eV while ``get_potential_energy()`` returned the measured
        value, with nothing in the file to say which was which.  Both numbers are
        meaningful, so the block names them both and says which one ASE reported.
        """
        path = getattr(solver, "output", None)
        if path is None:
            return
        from ..backends.providers import qpu_usage
        from ..utils.logging import append_block

        provider = self.measurement_provider
        stds = measured.get("stds") or {}
        largest = max((abs(float(v)) for v in stds.values()), default=None)
        result = getattr(solver, "result", None)
        fields = {
            "reported_by": "ASE get_potential_energy()",
            "source": "measurement_provider",
            "energy_eV": f"{measured['energy_eV']:.10f}",
            "energy_hartree": f"{measured['energy_hartree']:.10f}",
            # The optimization's own value, kept under its own name: the state
            # was optimized on the local state vector, then measured.
            "variational_energy_eV": None if result is None else
                f"{solver._from_energy_units(result.optimal_energy, 'eV'):.10f}",
            "pauli_expectations": len(measured.get("expectation_values") or {}),
            "largest_std": None if largest is None else f"{largest:.6e}",
        }
        fields.update(qpu_usage(provider))
        append_block(path, "MEASUREMENT", fields)

    # -- the relaxation as a whole ----------------------------------------- #

    def _record_step(self, atoms, energy_ev: float, seconds: float) -> None:
        """Keep this geometry in :attr:`trajectory`, for the closing summary."""
        forces = np.asarray(self.force_result.forces, dtype=float)
        self.trajectory.append({
            "energy": float(energy_ev),
            "max_force": float(np.linalg.norm(forces, axis=1).max()),
            # The unprojected sum: the grid artifact, whether or not the
            # reported force had it removed (see `project_translation`).
            "net_force": float(self.force_result.details.get(
                "translational_residual",
                np.abs(forces.sum(axis=0)).max())),
            "wall_time_s": float(seconds),
            "com": np.asarray(atoms.get_center_of_mass(), dtype=float),
            "positions": np.asarray(atoms.get_positions(), dtype=float),
            "symbols": list(atoms.get_chemical_symbols()),
        })
        self._arm_exit_hook()

    def _arm_exit_hook(self) -> None:
        """Write the closing summary at interpreter exit, once.

        ASE never tells a calculator that a relaxation is over -- the optimizer
        just stops calling it -- so there is no in-band moment at which to close
        the log.  Interpreter exit is the one signal that always arrives, and a
        script's relaxation is over by then.  Call
        :meth:`write_optimization_summary` explicitly to close the log earlier
        (in a notebook, or before doing something else with the same file); this
        hook then finds it already written and does nothing.
        """
        if self._exit_hook or self._log_path() is None:
            return
        self._exit_hook = True
        atexit.register(self._write_summary_at_exit)

    def _write_summary_at_exit(self) -> None:
        """The ``atexit`` body: close the log **without** claiming success.

        Exit handlers also run after an unhandled exception, so reaching this
        point says only that the process ended -- not that the relaxation
        finished, and certainly not that it converged.  (Verified: raising after
        two recorded steps exited 1 and still produced a footer reading
        "finished after 2 geometry steps".)  The footer therefore states that
        completion was never signaled and points at the call that would have
        signaled it; ``write_optimization_summary`` is the only path that can say
        *converged*, because only its caller has the optimizer's verdict.
        """
        try:
            steps = len(self.trajectory)
            self.write_optimization_summary(status=(
                f"completion not signaled: the process exited after {steps} "
                f"geometry step(s), which does not establish that the "
                f"optimization finished or converged (call "
                f"write_optimization_summary(optimizer=...) to record the "
                f"verdict)"))
        except Exception:
            # Interpreter shutdown, a deleted temporary directory, a read-only
            # file: the calculation itself finished long ago and must not be
            # reported as having failed because its log could not be closed.
            pass

    def _log_path(self):
        """The ``output=`` log the steps are being written to, or ``None``."""
        return self.solver_kwargs.get("output")

    def write_optimization_summary(self, atoms=None, fmax: float | None = None,
                                   optimizer=None, force: bool = False,
                                   status: str | None = None) -> bool:
        """Close the log with the relaxation's summary and completion blocks.

        Returns whether anything was written.  Nothing is written for a single
        geometry (there is no trajectory to summarize), when no ``output=`` log
        was given, or when the summary is already there -- so calling it twice,
        or calling it *and* letting the exit hook run, writes one summary.

        Parameters
        ----------
        atoms : Atoms, optional
            The relaxed geometry; defaults to the last one evaluated.
        fmax : float, optional
            The optimizer's force threshold, if you want the footer to state
            convergence against it rather than just reporting the final force.
        optimizer : ase.optimize.Optimizer, optional
            Read ``fmax`` from an ASE optimizer that has already run.  ASE gives
            a calculator no way to reach the optimizer driving it, so handing it
            over is the only way the footer can know what "converged" meant.
        force : bool
            Write even for a single geometry.
        status : str, optional
            The footer verbatim -- for a workflow that knows the completion
            reason (interrupted, cancelled, budget exhausted).  The
            interpreter-exit hook uses it to say that completion was *not*
            signaled, since reaching exit does not establish it.

        Notes
        -----
        The block summarizes the sequence of geometries at which forces were
        computed.  That is a relaxation in every practical case; a force-carrying
        *scan* would be reported the same way.
        """
        converged = None
        if optimizer is not None:
            if fmax is None:
                fmax = getattr(optimizer, "fmax", None)
            # The optimizer's own verdict, which accounts for constraints and
            # cell filters; a bare force-threshold comparison does not.
            try:
                converged = bool(optimizer.converged())
            except Exception:
                converged = None
        from ..utils.logging import append_optimization_summary

        path = self._log_path()
        if path is None or self._summary_written:
            return False
        # An empty trajectory has nothing to summarize -- not even under
        # `force`, which is about a *single* geometry, not about none.
        if not self.trajectory or (len(self.trajectory) < 2 and not force):
            return False
        last = self.trajectory[-1]
        symbols = last["symbols"] if atoms is None else \
            list(atoms.get_chemical_symbols())
        positions = last["positions"] if atoms is None else atoms.get_positions()
        append_optimization_summary(path, self.trajectory, symbols=symbols,
                                    positions=positions, fmax=fmax,
                                    converged=converged, status=status)
        # Only after the write: marking it first meant a failed write (a full
        # disk, a vanished directory) suppressed the summary permanently, with
        # every retry returning False.
        self._summary_written = True
        return True

    def _log_performance(self, solver, stages, wall_time_s) -> None:
        """Append this step's ``[PERFORMANCE]`` block to the run's ``output.txt``.

        The block covers the **whole** step, which is why the calculator writes
        it and not the solver: on a real relaxation the nuclear gradient is the
        largest single stage -- a six-step water relaxation in PAW-SZ at
        h = 0.10 spends 62 % of its 325 s on forces against 19 % on the
        variational optimization -- so a block closed when the solver finished
        would account for the smaller part of the time.  ``stages`` carries what
        this method's caller timed -- the gradient, and the provider measurement
        when there was one -- on top of the solver's own.
        """
        path = getattr(solver, "output", None)
        if path is None:
            return
        from ..utils.logging import append_performance
        from ..utils.profiling import Timings, backend_cores

        reported = (getattr(solver.result, "timings", None) or {}) \
            if solver.result is not None else {}
        timings = Timings(n_cores=reported.get("n_cores", backend_cores()),
                          backend=reported.get("backend"))
        for name, seconds in (reported.get("stages_s") or {}).items():
            timings.add(name, seconds)
        # The solver's own wall clock covers the stages it timed plus the
        # Hamiltonian construction it does not; recording it as a stage keeps
        # `untimed_s` about *this* step rather than hiding the difference.
        solver_wall = reported.get("wall_time_s")
        for name, seconds in stages.items():
            timings.add(name, seconds)
        timings.wall_time = float(wall_time_s)

        accounting = dict(solver.qpu_accounting(
            stages.get("measurement (provider)")))
        if self.measurement_provider is not None:
            from ..backends.providers import qpu_usage
            accounting.update(qpu_usage(self.measurement_provider,
                                        stages.get("measurement (provider)")))
        extra = {"solver_wall_time_s": (None if solver_wall is None
                                        else round(float(solver_wall), 4))}
        extra = {k: v for k, v in extra.items() if v is not None}
        extra.update(accounting)
        append_performance(path, stages=timings.stages,
                           wall_time_s=timings.wall_time,
                           resources=timings.resources(),
                           extra=extra or None)

    def _projection_applies(self, result) -> bool:
        """Whether the free-molecule identity ``sum_A F_A = 0`` holds here.

        Two atoms or more (a single atom's grid re-centers on it, so its
        residual is zero by construction and says nothing) and no periodic
        direction.  Under periodic boundary conditions the sum over the cell's
        atoms is not the free-molecule identity, so there is nothing to enforce;
        an explicit ``project_translation=True`` there is reported rather than
        silently ignored.
        """
        if np.asarray(result.forces).shape[0] < 2:
            return False
        if self.atoms is not None and bool(np.any(self.atoms.pbc)):
            if self.project_translation is True:
                warnings.warn(
                    "project_translation=True is ignored for a periodic "
                    "system: the sum of forces over a cell's atoms is not the "
                    "free-molecule identity this enforces.",
                    RuntimeWarning, stacklevel=4)
            return False
        return True

    def _project_translation(self, result):
        """Subtract the mean force, so the molecule cannot drift.

        The exact energy of a free molecule is translation invariant, so
        ``sum_A F_A = 0`` is an identity, not an approximation.  Subtracting the
        mean is therefore an orthogonal projection onto a subspace that
        *contains* the exact force, and an orthogonal projection can only move a
        vector closer to anything already in that subspace: the reported force is
        never made worse by this and is usually made better.

        What it gives up is the identity ``forces == -dE/dR`` of the discretized
        energy, which genuinely is not translation invariant -- so the
        unprojected array is kept in ``details["forces_unprojected"]`` for a
        finite-difference check to compare against.  And it fixes only the
        *translational* component of the egg-box: the force differences that bend
        and stretch the molecule keep whatever error the spacing gives them, so
        this stops a relaxation walking across the grid without making a coarse
        grid trustworthy.
        """
        forces = np.asarray(result.forces, dtype=float)
        mean = forces.mean(axis=0)
        result.details["forces_unprojected"] = forces.copy()
        result.forces = forces - mean
        result.details["translation_projected"] = True
        result.details["translation_removed"] = mean.copy()
        return result

    @staticmethod
    def _check_translational_invariance(result) -> float:
        """Warn when the net force is not zero; return the residual (eV/Ang).

        Translating a free-standing molecule does not change its exact energy,
        so the forces must sum to zero.  On a real-space grid they do not: the
        discretized energy is not translation invariant, and the leftover is
        the same egg-box that a relaxation walks the molecule along.  The
        residual is recorded on ``details["translational_residual"]`` either
        way, so it can be checked without catching a warning.

        A sharp all-electron core is where this matters: LiH in the ``FAO``
        basis at ``h = 0.25`` reports ~480 eV/Angstrom of net force -- larger
        than any real force in the problem -- while its energy, its RDMs and
        its orbital-response residual all look healthy.
        """
        forces = np.asarray(result.forces, dtype=float)
        if forces.shape[0] < 2:
            # One atom: the grid re-centers on it, so the residual is zero by
            # construction and says nothing.
            return 0.0
        residual = float(np.abs(forces.sum(axis=0)).max())
        largest = float(np.abs(forces).max())
        result.details["translational_residual"] = residual
        if (residual > TRANSLATIONAL_RESIDUAL_TOLERANCE
                and residual > TRANSLATIONAL_RESIDUAL_FRACTION * largest):
            warnings.warn(
                f"the forces do not sum to zero (net |sum F| = {residual:.3g} "
                f"eV/Angstrom against a largest force of {largest:.3g}): a "
                f"free molecule feels no net force, so this is grid artifact, "
                f"not physics.  The energy is not translation invariant on "
                f"this grid -- typically a basis function too sharp for the "
                f"spacing (an all-electron core).  Refine h, or use a "
                f"pseudopotential basis (PAW / ONCVPSP / NCPP), which removes "
                f"the core rather than trying to sample it.",
                RuntimeWarning, stacklevel=3)
        return residual

    @staticmethod
    def _converged_state(solver):
        """The optimized state vector of whichever solver just ran."""
        result = getattr(solver, "result", None)
        if result is None:
            raise RuntimeError("the solver has not been run yet")
        # A subspace run optimizes one unitary over several references and sorts
        # the resulting levels: its ground state is a *stored* vector, not
        # U(theta) applied to the Hartree-Fock reference (that is whichever
        # branch the HF determinant evolved into, which need not be the lowest).
        states = getattr(result, "states", None)
        if states is not None and len(states):
            return np.asarray(states[0], dtype=complex)
        parameters = np.asarray(result.optimal_parameters, dtype=float)
        ansatz = getattr(solver, "ansatz", None)
        if ansatz is not None:                                   # fixed ansatz
            return ansatz.state(parameters)
        # Adaptive solvers rebuild the grown ansatz from the selected operators.
        from ..circuits.adapt_ansatz import AdaptAnsatz

        ansatz = AdaptAnsatz(solver.n_qubits, solver.pool.occupied_orbitals,
                             solver.mapping,
                             sparse=getattr(solver, "_sparse", False))
        labels = {op.label: op for op in solver._pool_ops}
        for label in result.operators:
            ansatz.append(labels[label])
        return ansatz.state(parameters)

    # -- convenience ------------------------------------------------------- #

    def get_force_breakdown(self):
        """``(hellmann_feynman, pulay)`` gradients of the last step (eV/Angstrom).

        Useful for showing how much of the force the Hellmann-Feynman term alone
        accounts for -- for an atom-centered basis, typically not enough.
        """
        if self.force_result is None:
            raise RuntimeError("no forces have been computed yet")
        return (self.force_result.hellmann_feynman, self.force_result.pulay)

    def __repr__(self) -> str:
        return (f"Mandacaru(method={self.method!r}, "
                f"basis={self.basis!r}, h={self.h})")
