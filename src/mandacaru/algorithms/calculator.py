# -*- coding: utf-8 -*-
# file: algorithms/calculator.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The unified ASE calculator for classical mean field and variational methods.

:class:`Mandacaru` is the single user-facing entry point for running a
classical mean-field or variational quantum calculation: the solver is selected
by the ``method`` argument and every method-specific option is forwarded to it.
It reports the
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
                           basis="HAO",
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
from contextlib import contextmanager
from time import perf_counter as _perf
from typing import TYPE_CHECKING

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .bloch import BLOCH_METHODS
from ..units import DEFAULT_GRID_SPACING

if TYPE_CHECKING:
    from .forces import ForceResult

#: The default method: ADAPT-VQE, everywhere a ``method`` is not given.
DEFAULT_METHOD = "adapt-vqe"

#: Stable method names accepted by ``method=``.
#: The periodic names are declared by the module that implements them, so
#: there is one place to change them.
STABLE_METHODS = ("rhf", "uhf", "vqe", "hva", "adapt-vqe",
                  "subspace-vqe", "subspace-adapt-vqe",
                  *BLOCH_METHODS)

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
        from .bloch import _bloch_drivers
        from .hva import HVA
        from .mean_field import RHFDriver, UHFDriver
        from .subspace import SubspaceADAPTVQE, SubspaceVQE
        from .vqe import VQE
        classes = {"rhf": RHFDriver, "uhf": UHFDriver,
                   "vqe": VQE, "hva": HVA, "adapt-vqe": ADAPTVQE,
                   "subspace-vqe": SubspaceVQE,
                   "subspace-adapt-vqe": SubspaceADAPTVQE}
        # The periodic drivers are these same solvers over the Born-von Karman
        # supercell, so they are built from them rather than duplicated.
        classes.update(_bloch_drivers())
        return canonical, classes[canonical]
    if canonical is not None:
        return canonical, _REGISTERED[canonical]
    raise ValueError(
        f"unknown method {name!r}; use one of {available_methods()}")


#: Whether standard output has already been put in line-buffered mode.
_STDOUT_LINE_BUFFERED = False


def _resolve_population(spec):
    """Normalize the ``population=`` option to a method name or ``None``.

    ``True`` means the default partition, a string names one, and
    ``None`` / ``False`` switch the analysis off.
    """
    from .charges import PARTITION_METHODS

    if spec is None or spec is False:
        return None
    name = PARTITION_METHODS[0] if spec is True else str(spec).strip().lower()
    if name not in PARTITION_METHODS:
        raise ValueError(f"unknown population analysis {spec!r}; use one of "
                         f"{PARTITION_METHODS}, True or None")
    return name


def _spin_moment(gamma, n_spatial_orbitals: int, frozen=(),
                 active=None) -> float:
    """``N_alpha - N_beta`` of an active spin-orbital RDM.

    ``n_spatial_orbitals`` is the **total** count (the basis size), not the
    active one: :func:`~mandacaru.algorithms.volumetric.spin_resolved_rdm`
    refills the frozen core into both channels.  The core is doubly occupied
    so it contributes exactly nothing to the difference; refilling it anyway
    is cheaper than asserting the cancellation.
    """
    from .volumetric import spin_resolved_rdm

    D_alpha, D_beta = spin_resolved_rdm(gamma, int(n_spatial_orbitals), frozen,
                                        active)
    return float(np.real(np.trace(D_alpha) - np.trace(D_beta)))


class MeasurementFailed(RuntimeError):
    """A measurement submission failed; the optimized state is still on the calculator."""


#: Largest imaginary part still treated as round-off in the reality test.
REAL_TOLERANCE = 1e-12


def _is_real_operator(operator) -> bool:
    r"""Whether a :class:`PauliSum` is a **real matrix**.

    A Pauli string with ``n`` letters ``Y`` is :math:`i^{n}` times a real
    matrix, because ``Y`` is ``i`` times a real antisymmetric one.  So the term
    ``c P`` is real exactly when ``c i^n`` is: an **even** number of ``Y``
    needs a real coefficient, an **odd** number an imaginary one.
    """
    for label, coeff in getattr(operator, "terms", {}).items():
        value = complex(coeff)
        part = value.imag if label.count("Y") % 2 == 0 else value.real
        if abs(part) > REAL_TOLERANCE:
            return False
    return True


def _real_problem(solver, hamiltonian) -> bool:
    r"""Whether every odd-``Y`` Pauli expectation of this state is exactly zero.

    The reference determinant is a computational-basis state, hence real, and
    :math:`e^{\theta A}` keeps it real when every generator ``A`` is a real
    matrix.  For a real :math:`\psi` and a string ``P`` with an odd number of
    ``Y``, :math:`P = i^{\mathrm{odd}} R` with ``R`` real *antisymmetric*, so
    :math:`\langle\psi|P|\psi\rangle = i^{\mathrm{odd}}\,\psi^T R \psi
    = 0` **exactly** -- not approximately.

    Both halves are checked rather than assumed: a complex basis (``l > 0``
    harmonics at low symmetry) makes the Hamiltonian complex, and a foreign
    ansatz may carry generators that are not real.  The test costs a pass over
    operators that are already in memory.
    """
    if not _is_real_operator(hamiltonian):
        return False
    ansatz = getattr(solver, "ansatz", None)
    # `pauli_generators` is the qubit form every ansatz exposes for the circuit
    # backends; a driver whose ansatz has none cannot be shown to be real.
    generators = getattr(ansatz, "pauli_generators", None)
    if generators is None:
        return False
    for generator in (generators() if callable(generators) else generators):
        if not _is_real_operator(generator):
            return False
    return True


def _planned_jobs(provider, labels) -> int:
    """How many jobs the submission will be split into (see ``max_bases_per_job``)."""
    from ..backends.measurement import planned_jobs

    return planned_jobs(labels, getattr(provider, "max_bases_per_job", None))


def _job_id(provider):
    """Identifier of ``provider``'s most recent job, or ``None``.

    Read defensively: a local primitive has no job, and a provider that cannot
    answer must not fail a measurement that has already returned its numbers.
    """
    job = getattr(provider, "last_job", None)
    if job is None:
        return None
    try:
        return str(job.job_id())
    except Exception:
        return None


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
#: orbitals).  Measured on H2O / PAW-LCAO-SZ at h = 0.25: 1.2e-9 Ha for the HF
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
        Which solver evaluates the energy -- ``"adapt-vqe"`` (the default),
        classical ``"rhf"`` / ``"uhf"``, fixed-layer ``"hva"``, ``"vqe"``,
        or the subspace-search variants
        ``"subspace-vqe"`` / ``"subspace-adapt-vqe"``.  ADAPT-VQE is the
        practical choice for anything beyond a couple of orbitals: a fixed UCCSD
        ansatz becomes very slow past ~8 qubits.  A method registered through
        :func:`register_method` is accepted by name as well.
    basis : str or dict
        Basis family, as for the solvers (default ``"HAO"``); accepts a
        ``{"name": ..., <options>}`` dict, including the periodic plane-wave
        family (energy only -- plane waves carry no forces) and the
        pseudopotential families ``"NCPP"`` / ``"ONCVPSP"`` / ``"PAW-LCAO"``
        (``{"name": "PAW-LCAO", "size": "DZP"}``), which replace the all-electron
        problem by a valence-only one.
    h : float
        Grid spacing in Angstrom (default ``0.20``), used both for the
        per-geometry grid built from ``atoms.cell`` and for the frozen force
        grid.  The box itself is always ``atoms.cell`` -- the geometry must
        carry one (``atoms.center(vacuum=...)``, ``atoms.cell = ...`` or an
        extended-XYZ ``Lattice``), or a ``ValueError`` is raised.
    grid : Grid, optional
        An explicit grid, used verbatim (and frozen) for every evaluation.
    txt : str, optional
        Path of the structured run log -- ``[SYSTEM]``, ``[BASIS]``,
        ``[ELECTRONS]``, ``[OPTIMIZATION SETUP]``, ``[ITERATIONS]``, the
        summary, the forces and the performance block.  ``None`` (the default)
        writes no file, and the **same blocks** are printed to standard output
        instead; they are one document rendered once, so the screen and the
        file never drift apart.  Forwarded to the solver.
    trace : bool, optional
        Whether that report also goes to **standard output**.  ``None`` (the
        default) decides automatically: **off** when ``txt=<path>`` gives it a
        destination, **on** when it does not.  With it off, standard output
        carries only the evolution of energies and forces an ASE optimizer
        prints there (``Step Time Energy fmax``).  ``True`` with a ``txt=``
        file writes both; ``False`` without one makes a single-point run report
        nothing to the terminal (the energy is still the return value).
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
    optimizer : str, dict or Optimizer
        The classical optimizer, forwarded to the solver.  A **method name**
        (``"SLSQP"``, the default, ``"COBYLA"``, ``"Nelder-Mead"``, ``"SPSA"``,
        ``"L-BFGS"``, ``"BFGS"``) takes the library's budget and tolerance; a
        **dict** sets them without importing anything::

            Mandacaru(method="adapt-vqe",
                      basis="HAO",
                      optimizer={"method": "SLSQP",
                                 "maxiter": 2000,
                                 "tol": 1e-12})

        Keys left out keep their defaults, and ``options`` / ``seed`` are
        accepted too (:data:`~mandacaru.optimizers.optim.OPTIMIZER_KEYS`); an
        unknown key is refused as the typo it is.  A pre-built
        :class:`~mandacaru.optimizers.Optimizer` works as well and is what the
        dict builds.
    charge, frozen_core, frozen_orbitals, mapping, pool, ... :
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

    With ``txt=<path>`` (forwarded to the solver) every step **appends** its
    own block to that one file -- the iteration table, then the forces of that
    geometry -- so the whole trajectory is in one log rather than the last step
    overwriting the rest; see :mod:`mandacaru.utils.logging`.
    """

    implemented_properties = ["energy", "free_energy", "forces",
                              "charges", "magmoms", "magmom"]

    def __init__(self, method: str = DEFAULT_METHOD, *, basis="HAO",
                 h: float = DEFAULT_GRID_SPACING, grid=None,
                 include_pulay: bool = True, force_method: str = "rdm",
                 project_translation: bool = DEFAULT_PROJECT_TRANSLATION,
                 hellmann_feynman: str = "analytic", orbital_delta=None,
                 scf_iterations: int = 40,
                 measurement_provider=None, measurement_budget=None,
                 measurement_scheme: str = "qwc",
                 trace: bool | None = None,
                 population=None, **solver_kwargs):
        Calculator.__init__(self)
        self.method, self._solver_class = resolve_method(method)
        if self.method in ("rhf", "uhf") and measurement_provider is not None:
            raise ValueError("classical RHF/UHF has no quantum state to measure; "
                             "omit measurement_provider=")
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
        # Population analysis on every evaluation, so a written file carries
        # it.  Off by default: the partition costs a grid pass (and a basin
        # search, for Bader) that a relaxation has no use for.
        self.population = _resolve_population(population)
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
                "standard-output trace (None = automatic: off when `txt=` "
                "routes the report to a file, on otherwise; True / False force "
                "it).  The structured per-iteration log is written with "
                "`txt=<path>`, and the operator pool and Hamiltonian with "
                "`verbose_operators=` / `verbose_hamiltonian=`.")
        if trace is not None and not isinstance(trace, bool):
            raise TypeError(f"trace must be True, False or None, got {trace!r}")
        self.trace = trace
        self.solver_kwargs = dict(solver_kwargs)
        self._check_solver_options()
        self.measurement_provider = measurement_provider
        # Validated here so a misspelled ceiling fails at construction rather
        # than being discovered when it does not stop the job it was meant to.
        from ..backends.measurement import (MEASUREMENT_SCHEMES,
                                            resolve_measurement_budget)
        resolve_measurement_budget(measurement_budget)
        self.measurement_budget = measurement_budget
        scheme = str(measurement_scheme).strip().lower().replace("_", "-")
        if scheme not in MEASUREMENT_SCHEMES:
            raise ValueError(
                f"unknown measurement_scheme {measurement_scheme!r}; use one "
                f"of {MEASUREMENT_SCHEMES}")
        self.measurement_scheme = scheme
        #: Energy, RDMs and expectation values of the last measured state.
        self.measurement = None
        #: :class:`~mandacaru.backends.measurement.MeasurementPlan` of the last
        #: submission -- kept even when the job failed, so a retry can be sized.
        self.measurement_plan = None

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
        """Last Hamiltonian: a ``Fermion`` for RHF/UHF, otherwise a ``PauliSum``."""
        return self._require_solver().hamiltonian

    @property
    def n_qubits(self) -> int:
        """Register width a quantum run would use for the last model."""
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
    REPORTING_OPTIONS = {"txt": "writes_output_log",
                         "checkpoint": "supports_checkpoints",
                         "resume": "supports_checkpoints"}

    def band_structure(self, *args, **kwargs):
        """The solver's band structure when it defines one, else ASE's.

        ASE's ``Calculator`` already defines ``band_structure``, so it shadows
        the solver's and ``__getattr__`` never fires for it; the periodic
        drivers are reached explicitly here.  Every other solver attribute is
        delegated by ``__getattr__`` as usual.
        """
        from ase.calculators.calculator import Calculator as _ASECalculator

        own = getattr(type(self.solver), "band_structure", None)
        if own is not None and own is not _ASECalculator.band_structure:
            return own(self.solver, *args, **kwargs)
        return _ASECalculator.band_structure(self, *args, **kwargs)

    def _check_solver_options(self) -> None:
        """Refuse options the selected method would accept and ignore.

        Two ways an option can go nowhere.  It may not be a parameter of the
        solver at all -- ASE's ``Calculator`` keeps unknown keywords as
        *parameters*, so ``Mandacaru(method="vqe", txt=...)`` used to compute an
        energy, write no file, and (because a path had been given) print no trace
        either: a run that reported nothing anywhere.  Or the solver may accept
        it and not act on it, which is the subspace methods' ``run()`` and the
        ``txt`` / ``checkpoint`` / ``resume`` options it never reaches.
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

        Automatic by default: with ``txt=<path>`` the report has a destination,
        so standard output is left to the **evolution of energies and forces** --
        which is what an ASE optimizer prints there, in ASE's own ``Step Time
        Energy fmax`` format. Without ``txt=`` standard output is the only
        destination there is, so the blocks go there.  ``trace=True`` / ``False``
        overrides either way; ``True`` alongside a ``txt=`` file writes the
        report to both.
        """
        if self.trace is not None:
            return self.trace
        # Routed away from the terminal only when the report really lands in a
        # file: a solver that does not write one must not be silenced.
        return not (self.solver_kwargs.get("txt") is not None
                    and getattr(self._solver_class, "writes_output_log", False))

    def set_atoms(self, atoms):
        """ASE calls this on ``atoms.calc = calc``; remember the geometry.

        Kept in a private attribute, deliberately **not** in ``self.atoms``.
        ASE decides whether a result may be reused by comparing ``self.atoms``
        with the geometry it is asked about, so filling it in at attach time
        would announce a result that has not been computed: the next
        ``get_potential_energy()`` would read back the previous geometry's
        energy.  Reattaching one calculator to a displaced copy is exactly how
        a finite-difference force is taken, so that would be silent and wrong.

        It exists so a periodic driver can answer ``calc.bands(...)`` before
        anything is evaluated -- the bands need the primitive cell, not a
        completed run.  ``ase.Atoms.calc``'s setter calls this when the
        calculator defines it (verified against ASE 3.29); nothing depends on
        that, since a driver that never receives the geometry raises a message
        naming the explicit ``atoms=`` argument instead.
        """
        self._attached_atoms = atoms.copy()

    def _make_solver(self, grid, **overrides):
        options = {**self.solver_kwargs, **overrides}
        solver = self._solver_class(basis=self.basis, grid=grid, h=self.h,
                                    verbose=self._show_trace(), **options)
        # A periodic driver keeps the primitive cell aside (its own `atoms`
        # becomes the supercell once it runs); hand over the attached geometry
        # so the band structure is available without a calculation.
        # Read through __dict__, not getattr: an attribute this class does not
        # define reaches __getattr__, which delegates to `self.solver` -- and
        # building that solver lands back here, recursing forever.
        attached = self.__dict__.get("_attached_atoms")
        if attached is None:
            attached = self.__dict__.get("atoms")
        if attached is not None and hasattr(solver, "_primitive"):
            solver._primitive = attached.copy()
        return solver

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

        Not used by the periodic methods: there the grid **is** the cell, built
        with ``periodic=True`` and commensurate with the k-point mesh, so it is
        already frozen across geometries and this bounding-box grid would fail
        the commensurability check.
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
                "qualify. Use 'HAO', 'NAO', 'GTO', '6-31G(d)' or a "
                "pseudopotential family ('NCPP', 'ONCVPSP', 'PAW-LCAO').")

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
        if want_forces and getattr(self._solver_class, "classical_mean_field",
                                   False):
            raise NotImplementedError(
                "RHF/UHF nuclear forces need self-consistent orbital response; "
                "the variational-state gradient cannot be used for a "
                "classical SCF result")
        if want_forces:
            self._require_atom_centered_basis(self.basis)

        # Forces need one common grid along the whole trajectory; a plain
        # energy uses the solver's own per-geometry grid unless one was given.
        # A *periodic* method needs no freezing: its grid is the cell itself,
        # built commensurate with it, so it is already the same at every
        # geometry as long as the cell does not change -- and the molecular
        # bounding-box grid would not even span the cell.
        periodic = getattr(self._solver_class, "periodic_hamiltonian", False)
        grid = (self._frozen_grid(atoms)
                if want_forces and not periodic else self._grid)
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
            # RDMs are measured only when the forces need them: they are an
            # O(M^4) set of extra observables (LiH/PAW-LCAO-TZP: 97,980 Pauli
            # strings against the Hamiltonian's 12,736) that an energy-only
            # evaluation would throw away, and the job that carries them is
            # what exhausts the Runtime program's memory.
            measured = self._measure(solver, rdms=want_forces)
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
        if self.population is not None and not solver.dry_run:
            t0 = _perf()
            self._store_population()
            stages["population analysis"] = _perf() - t0
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

    def _check_force_support(self, solver: object) -> None:
        """Refuse force requests the derivative does not actually cover.

        The gradient differentiates *this* energy expression; a configuration
        it does not model must fail here rather than return a plausible number.
        """
        if getattr(solver, "classical_mean_field", False):
            raise NotImplementedError(
                "RHF/UHF nuclear forces need self-consistent orbital response; "
                "the variational-state gradient cannot be used for a "
                "classical SCF result")
        context = getattr(solver, "_gradient_context", None) or {}
        integrals = context.get("integrals")
        legacy = self.force_method == "scf-response"

        periodic = getattr(solver, "periodic_hamiltonian", False)
        kinetic = getattr(integrals, "kinetic", "fd")
        if kinetic != "fd" and not (periodic and not legacy):
            # The rdm gradient rebuilds T with `integrals.kinetic`, so it
            # differentiates whatever operator the energy used.  The periodic
            # path *requires* the spectral operator -- the finite-difference
            # stencil has a wall at the box edge and is not periodic -- so
            # refusing it there would refuse periodic forces outright.  The
            # gate stays for the legacy differentiated-SCF path, which builds
            # its own stencil, and for molecules, where lifting it has not
            # been validated against a finite difference.
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

        if context.get("deleted"):
            if legacy:
                raise NotImplementedError(
                    "active-space forces require force_method='rdm'; the "
                    "scf-response path does not differentiate the selected "
                    "virtual space")
            if not self.include_pulay:
                raise NotImplementedError(
                    "active-space forces require include_pulay=True because "
                    "the selected orbital space moves with the nuclei")
            if periodic:
                raise NotImplementedError(
                    "periodic active-space forces need a displaced periodic "
                    "Hamiltonian and are not implemented")

        if not legacy:
            return
        if getattr(integrals, "split_local_potential", False):
            # The legacy path rebuilds `T + V_grid + C D C^dagger` itself and
            # differentiates the local potential on the grid.  With the local
            # channel range-separated, the grid holds only its long-range half
            # and the short-range half is an atom-centered quadrature the legacy
            # kernels know nothing about -- it would differentiate an energy
            # several Hartree away from the reported one.
            raise NotImplementedError(
                "force_method='scf-response' cannot differentiate a "
                "range-separated local potential: this family integrates the "
                "short-range half of it on atom-centered spheres "
                "(`short_range_local`).  Use the default force_method='rdm', "
                "or set `exact_local_potential = False` on the integrals class "
                "to put the whole potential back on the grid.")
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

    def _measure(self, solver: object, rdms: bool = True) -> dict:
        """Energy (and RDMs) of the optimized state, from ``measurement_provider``.

        With ``rdms=True`` every Pauli string the qubit Hamiltonian and the
        spin-conserving RDM operators need is measured in **one** PUB, and the
        energy and the RDMs are assembled from the same expectation values --
        so the energy the forces are checked against is the measured one.

        With ``rdms=False`` -- an energy-only evaluation, which is every
        ``get_potential_energy()`` that is not part of a force request -- the
        Hamiltonian goes to the provider as a **single weighted observable**
        and nothing else is measured.  The RDM operators are an ``O(M^4)`` set
        that would be built (34 s at 24 qubits) and submitted only to be
        discarded: on LiH/PAW-LCAO-TZP that is 97,980 Pauli strings instead of
        12,736, and the Runtime Estimator returns one result array *per
        observable*, which under ZNE is what runs it out of memory.
        """
        from ..units import HARTREE_TO_EV
        from .rdm import rdm_qubit_operators, rdms_from_expectations

        provider = self.measurement_provider
        if not rdms:
            hamiltonian = solver.hamiltonian
            identity = "I" * int(solver.n_qubits)
            factorized = self._factorized_scheme(solver)
            if factorized is not None:
                return self._measure_factorized(solver, factorized)
            self._plan_measurement(
                solver, [l for l in hamiltonian.terms if l != identity],
                hamiltonian=hamiltonian, includes_rdms=False)
            with self._measurement_failure(solver):
                energy = float(provider.energy(*solver.ansatz_problem()[:4],
                                               hamiltonian))
            self.measurement = {
                "energy_hartree": energy,
                "energy_eV": energy * HARTREE_TO_EV,
                "rdms": None, "expectation_values": None, "stds": None,
                "observables": 1, "job_id": _job_id(provider)}
            return self.measurement

        reduced = solver.mapping == "parity_reduced"
        n_qubits = int(solver.n_qubits)
        n_modes = n_qubits + (2 if reduced else 0)
        taper_info = getattr(solver, "_taper_info", None)
        if taper_info is not None:
            n_modes += taper_info.removed
        ones, twos = rdm_qubit_operators(n_modes, solver.mapping,
                                         num_particles=solver.num_particles,
                                         taper_info=taper_info)
        hamiltonian = solver.hamiltonian
        identity = "I" * n_qubits
        labels = sorted({label for op in (hamiltonian, *ones.values(),
                                          *twos.values())
                         for label in op.terms} - {identity})
        # A real Hamiltonian acting on a real state gives every Pauli string
        # with an odd number of Y a vanishing expectation value: <Y> is the
        # imaginary part of an amplitude product, and an odd count leaves the
        # whole string anti-Hermitian under complex conjugation.  Half the RDM
        # labels are of that kind (measured: 472 of 981 on LiH/PAW-LCAO-DZ), so
        # leaving them out is an exact halving of the job, not an
        # approximation -- provided the premise holds, which is checked.
        assumed_zero: set[str] = set()
        if _real_problem(solver, hamiltonian):
            assumed_zero = {l for l in labels if l.count("Y") % 2}
            labels = [l for l in labels if l not in assumed_zero]

        self._plan_measurement(solver, labels, hamiltonian=hamiltonian,
                               includes_rdms=True)
        with self._measurement_failure(solver):
            values, stds = provider.expectation_values(
                *solver.ansatz_problem()[:4], labels)
        for label in assumed_zero:
            values[label], stds[label] = 0.0, 0.0
        values[identity], stds[identity] = 1.0, 0.0
        energy = float(np.real(sum(complex(c) * values[label]
                                   for label, c in hamiltonian.terms.items())))
        self.measurement = {
            "energy_hartree": energy, "energy_eV": energy * HARTREE_TO_EV,
            "rdms": rdms_from_expectations(n_modes, ones, twos, values),
            "expectation_values": values, "stds": stds,
            "observables": len(labels),
            "assumed_zero": len(assumed_zero), "job_id": _job_id(provider)}
        return self.measurement

    def _factorized_scheme(self, solver):
        """The factorization to measure through, or ``None`` for the QWC path.

        ``measurement_scheme="double-factorized"`` is refused rather than
        silently ignored when the run cannot support it -- a direct-mode
        problem has no integrals, and a tapered register has no occupation
        picture to read.
        """
        if self.measurement_scheme != "double-factorized":
            return None
        if solver.mapping != "jordan_wigner":
            raise NotImplementedError(
                f"measurement_scheme='double-factorized' reads occupations off "
                f"the computational basis, which is the Jordan-Wigner picture; "
                f"mapping={solver.mapping!r} does not have one.")
        factorized = self.factorization(solver)
        if factorized is None:
            raise NotImplementedError(
                "measurement_scheme='double-factorized' needs the molecular "
                "integrals the Hamiltonian was built from, and this run has "
                "none (a direct-mode problem or a loaded Hamiltonian carries "
                "only the qubit operator).")
        return factorized

    def _measure_factorized(self, solver, factorized):
        """Energy through the Givens-rotation scheme: ``L + 1`` PUBs, one job."""
        from ..backends.factorization import (factorized_energy,
                                              measurement_problems)
        from ..units import HARTREE_TO_EV

        provider = self.measurement_provider
        problem = solver.ansatz_problem()[:4]
        problems = measurement_problems(factorized, *problem,
                                        mapping=solver.mapping)
        self._plan_measurement(solver, [], hamiltonian=solver.hamiltonian,
                               includes_rdms=False, problems=problems,
                               factorization=factorized)
        with self._measurement_failure(solver):
            energy = float(factorized_energy(provider, problem, factorized,
                                             mapping=solver.mapping))
        self.measurement = {
            "energy_hartree": energy, "energy_eV": energy * HARTREE_TO_EV,
            "rdms": None, "expectation_values": None, "stds": None,
            "observables": len(problems), "scheme": "double-factorized",
            "job_id": _job_id(provider)}
        return self.measurement

    # -- pre-flight and failure ------------------------------------------- #

    def _plan_measurement(self, solver, labels, hamiltonian=None,
                          includes_rdms: bool = False, problems=None,
                          factorization=None):
        """Size the submission, log it as ``[MEASUREMENT PLAN]``, enforce the budget.

        Everything the plan reports is computed locally -- the grouping, the
        transpilation, the estimator's own resilience multipliers -- so a job
        that cannot succeed is refused *before* it is queued.  The one that
        motivated this spent 37 minutes in a queue and then died with "Program
        runtime ran out of memory", discarding a converged optimization.
        """
        from ..backends.measurement import measurement_plan

        provider = self.measurement_provider
        # The circuit that is actually submitted: for the factorized scheme
        # that is the ansatz *plus* the longest Givens network, which is what
        # the fidelity estimate has to be made against.
        widest = (max(problems, key=lambda p: len(p[2])) if problems
                  else solver.ansatz_problem()[:4])
        isa = None
        try:
            isa = provider._transpiled(provider.build(*widest[:4]),
                                       int(solver.n_qubits))[0]
        except Exception:
            # A provider without a transpiler, or a backend that cannot be
            # reached: the plan is still worth having without the gate counts.
            isa = None
        if problems is not None:
            plan = measurement_plan(
                provider, int(solver.n_qubits), [], hamiltonian=hamiltonian,
                includes_rdms=includes_rdms, isa_circuit=isa, jobs=1,
                scheme="double-factorized", bases=len(problems),
                observables=len(problems),
                one_norm=factorization.one_norm,
                factorized_bases=factorization.measurement_bases)
        else:
            plan = measurement_plan(
                provider, int(solver.n_qubits), labels,
                hamiltonian=hamiltonian, includes_rdms=includes_rdms,
                isa_circuit=isa, jobs=_planned_jobs(provider, labels),
                factorized_bases=self._factorized_bases(solver))
        self.measurement_plan = plan
        self._log_plan(solver, plan)
        for note in plan.warnings():
            warnings.warn(note, RuntimeWarning, stacklevel=3)
        plan.check(self.measurement_budget)
        return plan

    def factorization(self, solver=None):
        """The double-factorized form of this run's Hamiltonian, or ``None``.

        Needs the molecular integrals, so it exists exactly when the run built
        a basis (not in direct mode, not for a loaded Hamiltonian).  Cheap --
        an ``eigh`` of an ``M^2 x M^2`` matrix, 0.1 s at 24 spin orbitals.

        The **constant** is whatever the driver's qubit Hamiltonian carries
        beyond the electronic operator (the nuclear repulsion): a scalar, read
        off rather than recomputed, so the factorized energy is on the same
        zero as every other energy the run reports.
        """
        solver = self.solver if solver is None else solver
        context = getattr(solver, "_gradient_context", None)
        if not context or context.get("integrals") is None:
            return None
        from ..backends.factorization import double_factorization
        from ..core.hamiltonian import (molecular_orbital_integrals,
                                        spin_block_integrals)
        from ..core.mapping import Fermion

        h_mo, eri_mo, _orbitals = molecular_orbital_integrals(
            context["integrals"], context["n_electrons"])
        h_so, g_so = spin_block_integrals(h_mo, eri_mo)
        electronic = Fermion.from_integrals(h_so, g_so).map_to_qubits(
            solver.mapping)
        offset = (solver.hamiltonian + (-1.0) * electronic).simplify()
        identity = "I" * int(solver.n_qubits)
        constant = float(np.real(offset.terms.get(identity, 0.0)))
        residue = {label: c for label, c in offset.terms.items()
                   if label != identity and abs(complex(c)) > 1e-10}
        if residue:
            # The factorization describes the electronic operator; anything
            # else in the driver's Hamiltonian would be silently dropped.
            return None
        return double_factorization(h_so, g_so, constant=constant)

    def _factorized_bases(self, solver):
        """Bases a double-factorized measurement would need, if computable."""
        try:
            factorized = self.factorization(solver)
        except Exception:
            # A diagnostic line is never worth failing a measurement for.
            return None
        return None if factorized is None else factorized.measurement_bases

    def _log_plan(self, solver, plan) -> None:
        """Write the ``[MEASUREMENT PLAN]`` block, when the run has a log."""
        path = solver.log_targets
        if not path:
            return
        from ..utils.logging import append_block

        append_block(path, "MEASUREMENT PLAN", plan.fields())

    @contextmanager
    def _measurement_failure(self, solver):
        """Re-raise a failed submission without discarding the optimization.

        A ``RuntimeJobFailureError`` used to propagate out of ``calculate()``
        with nothing kept: the converged ansatz, which had cost minutes, went
        with it.  :attr:`solver` and :attr:`measurement_plan` are already set
        by the time anything is submitted, so the state survives and
        :meth:`remeasure` can retry it with other options.
        """
        try:
            yield
        except Exception as exc:
            plan = self.measurement_plan
            detail = "" if plan is None else (
                f" The submission was {plan.observables:,} observables in "
                f"{plan.bases:,} measurement bases, {plan.circuit_instances:,} "
                f"circuit instances, {plan.total_shots:,} shots on "
                f"{plan.device}.")
            raise MeasurementFailed(
                f"the measurement job failed ({type(exc).__name__}: {exc})."
                f"{detail} The optimized state is kept: calc.solver and "
                f"calc.measurement_plan are intact, and calc.remeasure() "
                f"retries without re-optimizing.") from exc

    def remeasure(self, provider=None, rdms: bool | None = None, **overrides):
        """Measure the **already optimized** state again, without re-running it.

        The point of keeping the solver when a job fails: a measurement can be
        retried with fewer observables, a lower resilience level or a different
        device, at no cost in optimization time.

        Parameters
        ----------
        provider : CircuitProvider, optional
            Measure on this provider instead of ``measurement_provider``; it
            becomes the calculator's provider, so a later step uses it too.
        rdms : bool, optional
            Measure the RDM operators as well.  The default repeats whatever
            the last measurement did, which for an energy-only run is one
            weighted observable.
        **overrides
            Attributes set on the provider before submitting (``shots``,
            ``estimator_options``, ``device``...), so a retry does not need a
            new provider object.
        """
        solver = self.solver
        if solver is None or getattr(solver, "result", None) is None:
            raise RuntimeError(
                "nothing has been optimized yet: ask for an energy "
                "(atoms.get_potential_energy()) before remeasuring")
        if provider is not None:
            self.measurement_provider = provider
        if self.measurement_provider is None:
            raise RuntimeError(
                "remeasure() needs a provider: pass one, or construct the "
                "calculator with measurement_provider=")
        for name, value in overrides.items():
            setattr(self.measurement_provider, name, value)
        if rdms is None:
            rdms = (self.measurement or {}).get("rdms") is not None
        measured = self._measure(solver, rdms=rdms)
        self.results["energy"] = measured["energy_eV"]
        self.results["free_energy"] = measured["energy_eV"]
        self._log_measurement(solver, measured)
        return measured

    def _state_rdms(self, solver: object, psi: np.ndarray | None = None,
                    two_body: bool = True
                    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Spin-orbital RDMs of a state vector (the converged one by default).

        ``two_body=False`` returns ``(gamma, None)``: a one-particle picture of
        the state (a density, a natural orbital) needs only the one-body RDM,
        and building the two-body one costs ``M**4`` inner products it would
        throw away.
        """
        from .rdm import (one_rdm, pauli_expectations, rdm_qubit_operators,
                          rdms_from_expectations, two_rdm)

        if psi is None:
            psi = self._converged_state(solver)
        n_qubits = int(solver.n_qubits)
        taper_info = getattr(solver, "_taper_info", None)
        if solver.mapping == "parity_reduced" or taper_info is not None:
            # A tapered register has no ladder operators of its own: its RDM
            # elements are expectation values of the tapered qubit operators.
            n_modes = n_qubits + (2 if solver.mapping == "parity_reduced" else
                                  taper_info.removed)
            ones, twos = rdm_qubit_operators(n_modes, solver.mapping,
                                             num_particles=solver.num_particles,
                                             taper_info=taper_info,
                                             two_body=two_body)
            labels = {label for op in (*ones.values(), *twos.values())
                      for label in op.terms}
            gamma, gamma2 = rdms_from_expectations(
                n_modes, ones, twos, pauli_expectations(psi, labels))
            return (gamma, gamma2) if two_body else (gamma, None)
        sector = getattr(solver, "_sector", None)
        return (one_rdm(psi, n_qubits, solver.mapping, sector=sector),
                two_rdm(psi, n_qubits, solver.mapping, sector=sector)
                if two_body else None)

    @staticmethod
    def _fold_to_primitive(solver, result):
        r"""Bring a supercell gradient back to the primitive cell.

        The periodic gradient is computed on the Born-von Karman supercell, so
        it has one row per *supercell* atom, while the calculator is attached
        to the primitive cell and ASE expects ``len(atoms)`` rows.  The two are
        related by what the driver already does to the energy: it reports
        :math:`E_{\text{cell}} = E_{\text{super}}/N_c`, and moving one
        primitive atom moves **all** :math:`N_c` of its images together, so

        .. math::

            \frac{\partial E_{\text{cell}}}{\partial \mathbf R_\nu}
              = \frac{1}{N_c} \sum_{\text{images } i \text{ of } \nu}
                \frac{\partial E_{\text{super}}}{\partial \mathbf R_i} ,

        the **mean** over the images.  For a perfect repetition every image
        carries the same gradient and the mean is that common value; the mean
        is the right answer in general, and its spread is a useful diagnostic
        of how well the supercell respects its own translation symmetry.

        ``atoms.repeat`` lays the supercell out cell by cell, so image ``c`` of
        primitive atom ``nu`` is supercell atom ``c * n_primitive + nu`` -- the
        same indexing :meth:`_BlochMixin._bloch_creation` uses for the orbitals.
        """
        cells = int(getattr(solver, "n_supercells", 1))
        if cells <= 1:
            return result

        def fold(array):
            array = np.asarray(array)
            per_cell = array.shape[0] // cells
            return array.reshape(cells, per_cell, 3).mean(axis=0)

        spread = None
        raw = np.asarray(result.forces)
        per_cell = raw.shape[0] // cells
        if per_cell:
            stacked = raw.reshape(cells, per_cell, 3)
            spread = float(np.abs(stacked - stacked.mean(axis=0)).max())

        details = dict(result.details)
        details.update({
            "supercell_forces": raw,
            "image_spread": spread,
            "n_cells": cells,
            "forces_unprojected": fold(details.get("forces_unprojected", raw)),
        })
        if "ion_gradient" in details:
            details["ion_gradient"] = fold(details["ion_gradient"])
        folded = fold(raw)
        details["translational_residual"] = float(
            np.abs(folded.sum(axis=0)).max())
        return type(result)(
            forces=folded,
            hellmann_feynman=fold(result.hellmann_feynman),
            pulay=fold(result.pulay),
            gradient=fold(result.gradient),
            n_electrons=result.n_electrons,
            details=details)

    def _forces(self, solver: object,
                rdms: tuple[np.ndarray, np.ndarray] | None = None,
                reference_energy: float | None = None) -> "ForceResult":
        """Gradient of the converged or measured state in eV/Angstrom.

        Full-space AO derivatives are analytic.  A deleted virtual space adds
        the finite-difference response of the selected MO projector.
        """
        from .forces import nuclear_gradient

        context = getattr(solver, "_gradient_context", None)
        if context is None:
            raise NotImplementedError(
                "nuclear forces need an atom-centered basis whose integrals are "
                "available; the plane-wave ('PW') family does not qualify. Use "
                "an atom-centered basis such as 'HAO', 'GTO' or '6-31G(d)'.")

        gamma, gamma2 = self._state_rdms(solver) if rdms is None else rdms
        active_gamma, active_gamma2 = gamma, gamma2
        frozen = context.get("frozen") or ()
        if frozen or context.get("deleted"):
            # The gradient contracts against the *full* integrals, so the
            # inert core must be refilled and deleted virtuals left empty.
            from .rdm import expand_frozen_core

            gamma, gamma2 = expand_frozen_core(
                gamma, gamma2, frozen, len(context["integrals"].basis),
                active=context.get("active"))

        if getattr(solver, "periodic_hamiltonian", False):
            # A crystal: the electron feels the Ewald potential of the whole
            # ion lattice and the ion-ion term is the Ewald energy, so the
            # molecular gradient would differentiate two different operators.
            from .periodic_forces import periodic_nuclear_gradient
            from .pseudo_forces import ENERGY_CHECK_TOLERANCE

            if self.force_method != "rdm":
                raise NotImplementedError(
                    f"force_method={self.force_method!r} has no periodic "
                    "implementation; the legacy differentiated-SCF path "
                    "rebuilds -Z/r potentials of its own, which is not what a "
                    "lattice-summed Hamiltonian was built from.  Use "
                    "force_method='rdm'.")
            result = periodic_nuclear_gradient(
                context["integrals"], gamma, gamma2,
                atom_of_orbital=context["atom_of_orbital"],
                orbital_delta=self.orbital_delta,
                include_pulay=self.include_pulay)
            if reference_energy is None and not getattr(solver, "shots", 0):
                reference_energy = float(solver._from_energy_units(
                    solver.result.optimal_energy, "Ha"))
            if reference_energy is not None:
                rebuilt = result.details["energy_hartree"]
                if abs(reference_energy - rebuilt) > ENERGY_CHECK_TOLERANCE:
                    raise RuntimeError(
                        "the energy rebuilt from the RDMs and molecular "
                        f"orbitals ({rebuilt:.10f} Ha) differs from the "
                        f"solver's ({reference_energy:.10f} Ha); the force "
                        "would not be the gradient of the reported energy")
            residual = float(result.details.get("orbital_gradient", 0.0) or 0.0)
            if residual > ORBITAL_RESPONSE_TOLERANCE:
                warnings.warn(
                    "the state is not stationary with respect to orbital "
                    f"rotations (max |dE/dkappa| = {residual:.1e} Ha); the "
                    "neglected orbital-response term is first order in that "
                    "residual.  A larger pool usually reduces it.",
                    RuntimeWarning, stacklevel=2)
            return self._fold_to_primitive(solver, result)

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
            if context.get("deleted"):
                from .active_space_forces import (
                    ACTIVE_SPACE_STEP_ANGSTROM, active_space_gradient)

                total_gradient = active_space_gradient(
                    self.atoms, solver, active_gamma, active_gamma2)
                response = total_gradient - result.gradient
                result.pulay = result.pulay + response
                result.gradient = total_gradient
                result.forces = -total_gradient
                result.details["active_space_response"] = response
                result.details["active_space_step_angstrom"] = \
                    ACTIVE_SPACE_STEP_ANGSTROM
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
            if residual > ORBITAL_RESPONSE_TOLERANCE and not context.get("deleted"):
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
        """Append the step's forces to the run's report, wherever it goes.

        The solver logs the geometry's energies and closes its own log before
        the gradient is even computed, so the forces are appended afterwards --
        under the iteration table of the step they belong to.  A relaxation's
        log then reads as alternating energy and force blocks, one pair per
        geometry, which is what makes the convergence followable in the file
        rather than only in the terminal.
        """
        path = solver.log_targets
        if not path or self.force_result is None:
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
        path = solver.log_targets
        if not path:
            return
        from ..backends.providers import qpu_usage
        from ..utils.logging import append_block

        provider = self.measurement_provider
        stds = measured.get("stds") or {}
        largest = max((abs(float(v)) for v in stds.values()), default=None)
        # An energy-only measurement has no RDMs and no per-string values; the
        # block says so rather than reporting an empty expectation set.
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
            "observables_submitted": measured.get("observables"),
            "measured": "energy and RDMs" if measured.get("rdms") is not None
                        else "energy only (no forces requested)",
            "pauli_expectations": len(measured.get("expectation_values") or {})
                                  or None,
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
        if self._exit_hook or not self._log_targets():
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

    def _log_targets(self):
        """Every destination the steps are being reported to (possibly none).

        The ``txt=`` file and, when the trace is on, standard output -- the
        same pair the solver's
        :meth:`~mandacaru.algorithms.base.VariationalDriver.log_targets` names,
        resolved here because the blocks this class appends are written after
        the solver has closed its own logger.
        """
        from ..utils.logging import STDOUT

        path = self.solver_kwargs.get("txt")
        targets = () if path is None else (path,)
        return targets + ((STDOUT,) if self._show_trace() else ())

    def write_optimization_summary(self, atoms=None, fmax: float | None = None,
                                   optimizer=None, force: bool = False,
                                   status: str | None = None) -> bool:
        """Close the log with the relaxation's summary and completion blocks.

        Returns whether anything was written.  Nothing is written for a single
        geometry (there is no trajectory to summarize), when the run reports
        nowhere at all, or when the summary is already there -- so calling it twice,
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
            reason (interrupted, canceled, budget exhausted).  The
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

        path = self._log_targets()
        if not path or self._summary_written:
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
        """Append this step's ``[PERFORMANCE]`` block to the run's report.

        The block covers the **whole** step, which is why the calculator writes
        it and not the solver: on a real relaxation the nuclear gradient is the
        largest single stage -- a six-step water relaxation in PAW-LCAO-SZ at
        h = 0.10 spends 62 % of its 325 s on forces against 19 % on the
        variational optimization -- so a block closed when the solver finished
        would account for the smaller part of the time.  ``stages`` carries what
        this method's caller timed -- the gradient, and the provider measurement
        when there was one -- on top of the solver's own.
        """
        path = solver.log_targets
        if not path:
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

        A sharp all-electron core is where this matters: LiH in the ``HAO``
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
                f"pseudopotential basis (PAW-LCAO / ONCVPSP / NCPP), which removes "
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

    # -- real-space visualization ------------------------------------------ #

    def _volumetric_context(self):
        """``(solver, integrals, frozen, active)`` of the last evaluation, or refuse.

        A picture on a grid needs basis functions to draw.  A direct-mode
        problem (``hamiltonian=`` / ``load_hamiltonian=``) has a qubit operator
        and nothing else, and the plane-wave family has no atom-centered basis
        the grid machinery uses -- both are refused here, by name, instead of
        producing an empty or misleading file.
        """
        solver = self.solver
        if getattr(solver, "result", None) is None:
            raise RuntimeError(
                "nothing has been solved yet: attach the calculator to an "
                "Atoms object and ask for an energy "
                "(atoms.get_potential_energy()) before writing a cube file")
        context = getattr(solver, "_gradient_context", None)
        if not context or context.get("integrals") is None:
            raise NotImplementedError(
                "a real-space picture needs the basis functions the "
                "Hamiltonian was built from, and this run has none: a "
                "direct-mode problem (hamiltonian= / load_hamiltonian=) "
                "carries only a qubit operator, and the plane-wave ('PW') "
                "family is not sampled on the real-space grid.  Run the same "
                "geometry with an atom-centered basis ('HAO', 'NAO', 'GTO', "
                "'6-31G(d)', 'PAW-LCAO', ...) to get a density.")
        return (solver, context["integrals"], context.get("frozen") or (),
                context.get("active"))

    def _volumetric_state(self, solver, state):
        """The state vector a picture is drawn from.

        ``state`` is an integer index -- ``0`` is the converged ground state,
        higher indices the further levels a subspace run stored -- or an
        explicit state vector, which is how a deflation run's excited states
        (``calc.energy_levels(3).states[1]``) are drawn.
        """
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
            return np.asarray(state, dtype=complex).ravel()
        index = int(state)
        if index == 0:
            return self._converged_state(solver)
        states = getattr(solver.result, "states", None)
        if states is None or index >= len(states):
            raise ValueError(
                f"state={index} is not available: method {self.method!r} "
                f"reports {0 if states is None else len(states)} stored state "
                f"vectors.  Use a subspace method, or pass the state vector "
                f"itself (calc.energy_levels(n).states[i]).")
        return np.asarray(states[index], dtype=complex).ravel()

    def volumetric_field(self, quantity: str = "density", index: int = 0, *,
                         state=0, component: str = "auto", grid=None):
        """One real-space quantity of the converged state, as an array.

        See :mod:`mandacaru.algorithms.volumetric` for what each ``quantity``
        means -- and for why a one-particle reduction, rather than "the
        wavefunction", is what a volumetric file can hold.  Returns a
        :class:`~mandacaru.algorithms.volumetric.VolumetricField`, which carries
        the data, the grid, the nuclei, the integrated charge and a
        :meth:`~mandacaru.algorithms.volumetric.VolumetricField.write` method.
        """
        from .volumetric import volumetric_field

        solver, integrals, frozen, active = self._volumetric_context()
        psi = self._volumetric_state(solver, state)
        gamma, _ = self._state_rdms(solver, psi=psi, two_body=False)
        numbers = (None if self.atoms is None
                   else self.atoms.get_atomic_numbers())
        return volumetric_field(
            integrals, gamma, quantity=quantity, index=index, frozen=frozen,
            num_particles=solver.num_particles, component=component,
            grid=grid, numbers=numbers, active=active)

    def write_cube(self, path, quantity: str = "density", index: int = 0, *,
                   state=0, component: str = "auto", grid=None, format=None,
                   comment=None):
        """Write a volumetric file of the converged state and return the field.

        ``path``'s extension picks the format: ``.cube`` (Gaussian cube --
        VESTA, VMD, Avogadro) or ``.xsf`` (XCrySDen; VESTA reads it too).
        Everything in a cube file is in Bohr, and a density is in e/Bohr^3.

        .. code-block:: python

            atoms.calc = Mandacaru(method="adapt-vqe",
                                   basis={"name": "PAW-LCAO", "size": "DZP"},
                                   h=0.20)
            atoms.get_potential_energy()
            atoms.calc.write_cube("density.cube")
            atoms.calc.write_cube("spin.cube", quantity="spin_density")
            atoms.calc.write_cube("no0.cube", quantity="natural_orbital",
                                  index=0)

        The returned
        :class:`~mandacaru.algorithms.volumetric.VolumetricField` reports the
        integrated charge, the natural-orbital occupation and -- for PAW-LCAO -- the
        augmentation charge the smooth density on the grid does not carry.
        """
        volumetric = self.volumetric_field(quantity, index, state=state,
                                           component=component, grid=grid)
        volumetric.write(path, format=format, comment=comment)
        return volumetric

    def natural_orbitals(self, *, state=0, grid=None):
        """Natural orbitals and occupations of the converged state.

        The eigen-decomposition of the spin-summed one-particle density matrix:
        occupations in ``[0, 2]`` summing to the electron count, with the
        orbitals in both the atomic- and molecular-orbital bases.  Occupations
        that are not 2 or 0 are the correlation the variational state carries
        beyond a single determinant.
        """
        from .volumetric import state_natural_orbitals

        solver, integrals, frozen, active = self._volumetric_context()
        psi = self._volumetric_state(solver, state)
        gamma, _ = self._state_rdms(solver, psi=psi, two_body=False)
        return state_natural_orbitals(integrals, gamma, frozen=frozen,
                                      grid=grid, active=active)

    # -- population analysis ----------------------------------------------- #

    def _population_method(self, method) -> str:
        """The partition a getter should use: the one asked for, else the one
        ``population=`` configured, else the default."""
        from .charges import PARTITION_METHODS

        if method is not None:
            return method
        return self.population or PARTITION_METHODS[0]

    def get_stress(self, atoms=None, *, voigt: bool = True,
                   strain: float = None):
        r"""Stress tensor of a periodic calculation, in eV/Angstrom^3.

        :math:`\sigma_{\alpha\beta} = \frac{1}{\Omega}\,
        \partial E / \partial \varepsilon_{\alpha\beta}` by a symmetric
        strain applied to the **cell, the atoms and the grid together** --
        twelve full Hamiltonian rebuilds, six components either side of zero.

        The state is held fixed, exactly as the force does: the RDMs and the
        molecular orbitals are the converged ones, and the orbital-response
        residual reported by :attr:`force_result` covers both.  Straining the
        grid with the cell is deliberate; holding the grid fixed would fold the
        change in discretization into the answer, and the two are not
        separable.

        Parameters
        ----------
        atoms : Atoms, optional
            Geometry; defaults to the attached one.
        voigt : bool
            Return ASE's 6-vector ``(xx, yy, zz, yz, xz, xy)`` (the default)
            rather than the 3x3 tensor.
        strain : float, optional
            Half-amplitude of the symmetric strain; default
            :data:`~mandacaru.algorithms.periodic_forces.DEFAULT_STRAIN`.

        Raises
        ------
        NotImplementedError
            For a molecular method, which has no cell to strain.
        """
        from ..units import from_hartree
        from .periodic_forces import (DEFAULT_STRAIN, VOIGT, periodic_stress,
                                      strained_energy)
        from .pseudo_forces import spatial_rdms

        solver = self.solver
        if not getattr(solver, "periodic_hamiltonian", False):
            raise NotImplementedError(
                f"method {self.method!r} is not periodic, so there is no cell "
                "to strain and no stress to report.")
        if atoms is not None:
            self.calculate(atoms, properties=("energy",))
        context = getattr(solver, "_gradient_context", None) or {}
        integrals = context.get("integrals")
        if integrals is None:
            raise ValueError(
                "the stress needs the integrals the Hamiltonian was built "
                "from; run a geometry first (atoms.get_potential_energy()).")

        if context.get("deleted"):
            raise NotImplementedError(
                "the stress tensor with a truncated virtual space "
                "(active_orbitals=) is not implemented.  Deleting a virtual "
                "orbital is exact for the energy at a fixed geometry, but the "
                "selection itself moves with the nuclei: which orbitals are "
                "kept, and -- for active_selection='mp2' -- the rotation that "
                "defines them, both depend on the geometry, and that "
                "dependence is a response term the Hellmann-Feynman and Pulay "
                "sums here do not contain.  The result would be the "
                "derivative of a different energy than the one reported, "
                "which is worse than not having it.  Use the full virtual "
                "space for a relaxation, or a frozen core alone "
                "(frozen_core=), whose orbitals are fixed by the reference.")

        gamma, gamma2 = self._state_rdms(solver, two_body=True)
        frozen = context.get("frozen") or ()
        if frozen:
            from .rdm import expand_frozen_core
            gamma, gamma2 = expand_frozen_core(
                gamma, gamma2, frozen, len(integrals.basis))
        D, Gamma = spatial_rdms(gamma, gamma2, len(integrals.basis))
        n_electrons = context.get("n_electrons")

        def build(strain_matrix):
            return strained_energy(integrals, D, Gamma,
                                   integrals.mo_coefficients, strain_matrix,
                                   n_electrons=n_electrons)

        tensor = periodic_stress(
            build, integrals.cell,
            strain=DEFAULT_STRAIN if strain is None else float(strain))
        # Hartree/Bohr^3 -> eV/Angstrom^3.
        tensor = tensor * from_hartree(1.0, "eV") / 0.52917721092 ** 3
        self.results["stress"] = (np.array([tensor[a, b] for a, b in VOIGT])
                                  if voigt else tensor)
        return self.results["stress"]

    def finite_size_correction(self, scheme: str = "mpc", *, state=None,
                               shells=None):
        r"""What the exchange-correlation hole's periodic images cost.

        A Born-von Karman supercell makes every electron's exchange-correlation
        hole interact with ``N_c - 1`` copies of itself.  That is the leading
        finite-size error of a neutral cell once the Madelung term is
        accounted for, and it decays only as ``1/Omega``.  This evaluates it
        directly from the converged state, by contracting the **hole**
        ``Gamma - D (x) D`` against the difference between the periodic kernel
        the run used and the Wigner-Seitz minimum-image bare ``1/r``:

        .. math::

            \Delta E = \tfrac{1}{2} \sum_{pqrs}
              (f - g^{\text{E}})_{pqrs}\,(\Gamma - D \otimes D)_{pqrs} .

        Returns a
        :class:`~mandacaru.core.mpc.FiniteSizeCorrection`; **add** its
        ``correction`` to the reported energy to get the model-periodic-Coulomb
        (MPC) energy, which it also carries as ``mpc_energy``.

        This is a *diagnostic*: the variational energy is untouched, and
        nothing that was pinned moves.  To minimize the MPC energy instead,
        build the Hamiltonian with the MPC kernel -- see the ``interaction``
        option of :class:`~mandacaru.core.periodic.PeriodicIntegrals`.

        Two schemes are available and **neither is the default answer** --
        they measure different things and the choice is still open:

        ``"mpc"``
            Model periodic Coulomb.  Contracts the hole against the difference
            between the periodic kernel and the Wigner-Seitz minimum-image bare
            ``1/r``.  **Measured to overlap with the Madelung term this code
            already subtracts**: 86 % of it on a 1x1x1 simple-cubic hydrogen
            cell and 92 % on 2x2x2, scaling as ``Omega^(-1/3)`` rather than
            ``1/Omega``.  Read it as a diagnostic of that overlap, not as a
            correction to add on top of ``constant_energy``.
        ``"ccmh"``
            Chiesa-Ceperley-Martin-Holzmann.  The term the discrete k-sum omits
            at ``k = 0``, taken from the computed structure factor.  Additive
            and orthogonal to the Madelung constant, but it assumes the
            small-``k`` quadratic regime is resolved; it warns when it is not,
            which a cell surrounded by vacuum generally is not.
        ``"kzk"``
            Kwee-Zhang-Krakauer.  The difference between the infinite-size and
            size-dependent LDA exchange-correlation energies on the run's own
            density.  A functional of the density alone, so it leaves every
            two-electron integral -- and every Pulay term -- untouched.  Its
            jellium fit assumes a cubic cell and a density the fit has seen;
            :class:`~mandacaru.core.kzk.KZKCorrection` records the cell's
            deviation from cubic and the ``r_s`` range that contributed.

        Parameters
        ----------
        scheme : {"mpc", "ccmh"}
            Which correction to evaluate.
        state : array_like, optional
            A state vector to analyze instead of the converged one.
        shells : int, optional
            For ``"mpc"``, lattice shells searched for the minimum image
            (default :data:`~mandacaru.core.mpc.MINIMUM_IMAGE_SHELLS`); for
            ``"ccmh"``, reciprocal-lattice shells extrapolated to ``k = 0``
            (default :data:`~mandacaru.core.ccmh.DEFAULT_SHELLS`).

        Raises
        ------
        ValueError
            If the method is not periodic.  The correction is defined against
            a periodic kernel, so for a molecule there is nothing to remove.
        """
        from ..core.mpc import (MINIMUM_IMAGE_SHELLS, TruncatedCoulombSolver,
                                exchange_correlation_hole_energy)
        from .pseudo_forces import spatial_rdms

        scheme = str(scheme).strip().lower()
        if scheme not in ("mpc", "ccmh", "kzk"):
            raise ValueError(
                f"unknown finite-size scheme {scheme!r}; choose 'mpc' (model "
                "periodic Coulomb), 'ccmh' (Chiesa-Ceperley-Martin-Holzmann) "
                "or 'kzk' (Kwee-Zhang-Krakauer).")
        solver = self.solver
        if not getattr(solver, "periodic_hamiltonian", False):
            raise ValueError(
                f"method {self.method!r} is not periodic, so its electrons "
                "have no images and there is no exchange-correlation hole "
                "self-interaction to remove.  The correction is defined "
                "against the periodic Coulomb kernel; a molecule is already "
                "computed with the isolated one.")

        context = getattr(solver, "_gradient_context", None) or {}
        integrals = context.get("integrals")
        if integrals is None:
            raise ValueError(
                "the correction needs the integrals the Hamiltonian was built "
                "from; run a geometry first (atoms.get_potential_energy()).")
        if scheme in ("mpc", "ccmh") and (context.get("frozen")
                                          or context.get("deleted")):
            what = ("a frozen core" if context.get("frozen")
                    else "a truncated virtual space (active_orbitals=)")
            raise NotImplementedError(
                f"{what} is not supported by scheme {scheme!r}: the pair "
                "density would be contracted over the active orbitals only, "
                "while the electrons outside them carry a hole of their own.  "
                "Use scheme='kzk', which reads the density and refills what is "
                "missing.")

        # Cited at use: which correction a run reaches for is not knowable
        # when the driver is built.
        solver._cite(*{"mpc": ("Fraser1996", "Williamson1997"),
                       "ccmh": ("Chiesa2006",),
                       "kzk": ("Kwee2008",)}[scheme])

        two_body = scheme in ("mpc", "ccmh")
        gamma, gamma2 = self._state_rdms(solver, psi=state, two_body=two_body)
        D = Gamma = None
        if two_body:
            D, Gamma = spatial_rdms(gamma, gamma2, len(integrals.basis))

        # result.optimal_energy is the SUPERCELL energy in the result's own
        # unit (eV), while the driver reports per primitive cell.  Convert
        # once, here, rather than letting either scope leak into the record.
        from ..units import to_hartree

        n_cells = int(getattr(solver, "n_supercells", 1))
        result = getattr(solver, "result", None)
        energy_per_cell = None
        if result is not None:
            total = getattr(result, "optimal_energy", None)
            if total is not None:
                unit = getattr(result, "energy_unit", "eV")
                energy_per_cell = float(to_hartree(total, unit)) / n_cells
        volume = abs(float(np.linalg.det(integrals.cell)))

        if scheme == "kzk":
            from ..core.kzk import kzk_correction
            from .volumetric import volumetric_field

            # The same density path the cube writer uses, so a frozen core, a
            # tapered register and the PAW augmentation are all handled once.
            field = volumetric_field(
                integrals, gamma, quantity="density",
                frozen=context.get("frozen") or (),
                active=context.get("active"),
                num_particles=getattr(solver, "num_particles", None),
                n_spatial_orbitals=len(integrals.basis))
            return kzk_correction(
                field.data, integrals.grid.dV, integrals.cell,
                n_cells=n_cells, energy_per_cell=energy_per_cell)

        if scheme == "ccmh":
            from ..core.ccmh import (DEFAULT_SHELLS, ccmh_correction,
                                     structure_factor)

            structure = structure_factor(integrals, D, Gamma)
            return ccmh_correction(
                structure, volume=volume, n_cells=n_cells,
                shells=DEFAULT_SHELLS if shells is None else int(shells),
                energy_per_cell=energy_per_cell)

        grid = integrals.grid
        truncated = TruncatedCoulombSolver(
            grid.shape, step=grid.step,
            shells=MINIMUM_IMAGE_SHELLS if shells is None else int(shells))
        # The same orbitals, the same grid, the same Loewdin/MO rotation --
        # only the kernel differs, so the two tensors subtract term by term.
        eri_periodic = integrals.two_body_with_kernel(None)
        eri_truncated = integrals.two_body_with_kernel(truncated)

        return exchange_correlation_hole_energy(
            eri_periodic, eri_truncated, D, Gamma, volume=volume,
            n_cells=n_cells, energy_per_cell=energy_per_cell)

    def atomic_partition(self, method: str = "hirshfeld", *, state=0,
                         grid=None):
        """Split the converged density between the atoms.

        Returns an
        :class:`~mandacaru.algorithms.charges.AtomicPartition` carrying the
        per-atom populations, charges and magnetic moments of one partition,
        plus a :meth:`~mandacaru.algorithms.charges.AtomicPartition.summary`
        to print.  :meth:`get_charges` and :meth:`get_magnetic_moments` are
        the ASE-facing views of it; this is the whole thing, computed once.

        Parameters
        ----------
        method : {"hirshfeld", "voronoi", "bader"}
            The partition.  **An atom in a molecule has no boundary**, so
            which one is chosen is a convention and the three disagree by
            design -- see :mod:`mandacaru.algorithms.charges`.
        state : int or ndarray
            Which state to analyze, for a subspace run (0 = ground), or an
            explicit state vector.
        grid : Grid, optional
            Partition on this grid rather than the calculation's own.  A Bader
            basin boundary is resolved to one grid spacing, so it is the
            partition that most repays a finer one.
        """
        from .charges import partition_state

        solver, integrals, frozen, active = self._volumetric_context()
        psi = self._volumetric_state(solver, state)
        gamma, _ = self._state_rdms(solver, psi=psi, two_body=False)
        numbers = (None if self.atoms is None
                   else self.atoms.get_atomic_numbers())
        return partition_state(integrals, gamma, method=method, frozen=frozen,
                               numbers=numbers, grid=grid, active=active)

    def _store_population(self) -> None:
        """Put ``charges`` / ``magmoms`` / ``magmom`` in ``results``.

        What makes them land in a written file: ASE's extxyz writer turns the
        per-atom entries of ``calc.results`` into **columns** and the scalars
        into **header** keys, so with ``population=`` set,
        ``ase.io.write("out.extxyz", atoms)`` carries the charges and the
        local moments beside the positions and the total moment beside the
        energy -- no extra call at write time.

        A failure here must not lose the energy the step just computed, so it
        is a warning: the analysis is a report, the energy is the result.
        """
        try:
            partition = self.atomic_partition(self.population)
        except Exception as error:                       # noqa: BLE001
            warnings.warn(
                f"the {self.population} population analysis failed, so the "
                f"charges and magnetic moments are not in the results (the "
                f"energy and forces are unaffected): {error}",
                RuntimeWarning, stacklevel=2)
            return
        self.results["charges"] = partition.charges
        self.results["magmoms"] = partition.magnetic_moments
        self.results["magmom"] = partition.total_magnetic_moment

    def get_charges(self, atoms=None, method: str | None = None):
        """Partial charges (e), one per atom -- the ASE property ``charges``.

        ``q_A = Z_A - N_A`` with ``Z_A`` the charge the Hamiltonian carries
        (the *valence* charge for a pseudopotential run, so ``q_A`` is the
        physical partial charge either way) and ``N_A`` the electrons
        ``method`` assigns to the atom.  They sum to the system's total charge
        whichever partition is used; only the split between atoms is a
        convention.  See :meth:`atomic_partition`.
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        partition = self.atomic_partition(self._population_method(method))
        self.results["charges"] = partition.charges
        return partition.charges

    def get_magnetic_moments(self, atoms=None, method: str | None = None):
        """Local magnetic moments (Bohr magnetons), one per atom.

        The spin density ``n_alpha - n_beta`` integrated over each atom's
        share, with the same weights :meth:`get_charges` uses -- so a
        closed-shell state gives zeros and the moments of an open-shell one
        sum to :meth:`get_total_magnetic_moment` whatever the partition.
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        partition = self.atomic_partition(self._population_method(method))
        self.results["magmoms"] = partition.magnetic_moments
        return partition.magnetic_moments

    def get_total_magnetic_moment(self, atoms=None):
        """Total magnetic moment (Bohr magnetons) of the converged state.

        ``N_alpha - N_beta``, read off the traces of the two spin blocks of
        the one-particle density matrix.  **It takes no partition**: the
        integral of the spin density over all space is fixed by the state, and
        only its division between atoms is a convention.  It is the *measured*
        value, not the ``magmoms`` the geometry was given -- they agree
        because the ansatz conserves ``S_z``, and a disagreement would mean
        the run did not solve the sector it was asked for.
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        solver, integrals, frozen, active = self._volumetric_context()
        gamma, _ = self._state_rdms(solver, two_body=False)
        moment = _spin_moment(gamma, len(integrals.basis), frozen, active)
        self.results["magmom"] = moment
        return moment

    def get_magnetic_moment(self, atoms=None):
        """ASE's name for :meth:`get_total_magnetic_moment`."""
        return self.get_total_magnetic_moment(atoms)

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
