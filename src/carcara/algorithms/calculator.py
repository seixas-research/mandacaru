# -*- coding: utf-8 -*-
# file: algorithms/calculator.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The unified ASE calculator for every molecular variational method.

:class:`Carcara` is the single user-facing entry point for running a
variational quantum simulation: the eigensolver is selected by the ``method``
argument and every method-specific option is forwarded to it.  It reports the
**energy** for any method and, for the atom-centered bases, the analytic
**nuclear forces** (Hellmann-Feynman **plus** Pulay, see
:mod:`carcara.algorithms.forces`), so any ASE optimizer -- ``BFGS``, ``LBFGS``,
``FIRE``, ``QuasiNewton`` -- can drive a geometry optimization whose energies
come from a quantum variational eigensolver:

.. code-block:: python

    from ase.build import molecule
    from ase.optimize import BFGS
    from carcara.algorithms import Carcara

    water = molecule("H2O")
    water.center(vacuum=3.0)          # the cell is the real-space box
    water.calc = Carcara(method="adapt-vqe",
                         basis="FAO",
                         h=0.30,
                         frozen_core=True,
                         verbose=False)
    BFGS(water).run(fmax=0.05)

The run result of the most recent evaluation is available uniformly on
:attr:`Carcara.result`, whatever the method
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

Whenever forces are requested, :class:`Carcara` therefore builds the
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

import warnings

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

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
    """Make ``Carcara(method=name)`` build ``solver_class``.

    The hook a package outside the stable API uses to plug its solvers into
    the unified calculator without the stable code knowing them by name.  The
    class must be a :class:`~carcara.algorithms.base.VariationalDriver`.
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


def resolve_method(name: str):
    """Return ``(canonical_name, solver_class)`` for a method spec.

    The stable solvers come from :mod:`carcara.algorithms`; any other name
    must have been registered with :func:`register_method` first (the
    :mod:`carcara.experimental` package does so when it is imported).
    """
    key = str(name).strip().lower()
    if key in STABLE_METHODS:
        from . import ADAPTVQE, VQE, SubspaceADAPTVQE, SubspaceVQE
        return key, {"vqe": VQE, "adapt-vqe": ADAPTVQE,
                     "subspace-vqe": SubspaceVQE,
                     "subspace-adapt-vqe": SubspaceADAPTVQE}[key]
    if key in _REGISTERED:
        return key, _REGISTERED[key]
    raise ValueError(
        f"unknown method {name!r}; use one of {available_methods()}")


#: Pseudopotential families whose forces come from
#: :func:`~carcara.algorithms.pseudo_forces.pseudo_nuclear_gradient`.
PSEUDO_GRADIENT_FAMILIES = ("paw", "oncvpsp")
#: Largest orbital-rotation residual (Hartree) the RDM gradient accepts quietly.
ORBITAL_RESPONSE_TOLERANCE = 1e-3

#: Remove the spurious net force before reporting it (see
#: :meth:`Carcara._project_translation`).  Off by default: it changes reported
#: forces, and a large residual is a signal about the grid that should be seen
#: rather than silently absorbed.
DEFAULT_PROJECT_TRANSLATION = False

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


class Carcara(Calculator):
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
    measurement_provider : CircuitProvider, optional
        Measure the optimized state instead of reading the local state vector:
        the ansatz is still optimized locally, then every Pauli string of the
        Hamiltonian and of the RDM operators is measured on this provider in
        one Estimator job per geometry (e.g. ``QiskitProvider(device=
        "ibm_fez", shots=4096)``), and both the ASE energy and the forces come
        from those expectation values.  Small registers only (see
        :data:`~carcara.algorithms.rdm.MAX_PAULI_RDM_MODES`).  The last
        measurement is on :attr:`measurement`.
    include_pulay : bool
        Include the Pulay (basis-motion) force terms (default ``True``).  Setting
        it to ``False`` gives the bare Hellmann-Feynman force; for an atom-centered
        basis that is **not** the gradient of the energy and will not relax to the
        right geometry -- it is exposed for analysis, not for production.
    project_translation : bool
        Subtract the mean force from every atom before reporting, so the
        forces sum to zero (default ``False``; non-periodic systems only).
        A free molecule's exact forces *do* sum to zero, so this enforces a
        symmetry rather than hiding an error -- but it removes only the
        translational part of the grid's egg-box, so it stops a relaxation
        drifting across the grid without making a coarse grid trustworthy.
        The unprojected residual is always on
        ``force_result.details["translational_residual"]``.
    force_method : {"rdm", "scf-response"}
        How the nuclear gradient is taken.  ``"rdm"`` (default) differentiates
        the energy expression the solver actually reported, holding the reduced
        density matrices and molecular orbitals fixed
        (:func:`~carcara.algorithms.pseudo_forces.pseudo_nuclear_gradient`): it
        is complex-safe, covers pseudopotential projectors, augmented overlaps
        and compensation charges when the family has them, refills a frozen
        core, and reports its orbital-response residual in
        ``force_result.details["orbital_gradient"]``.  ``"scf-response"`` is the
        legacy path through a real-arithmetic differentiated SCF
        (:func:`~carcara.algorithms.forces.nuclear_gradient`); it assumes a
        closed-shell real reference and does not carry the pseudopotential
        terms, and is kept only for comparison.
    hellmann_feynman : {"analytic", "by-parts"}
        How the electron-nucleus force term is evaluated.  Keep the default
        ``"analytic"``: it is exactly the derivative of the reported energy and
        is verified against finite differences to 0.04 % for H2.

        ``"by-parts"`` is an independent formulation (the derivative moved onto
        the density) provided for **diagnosis only** -- it does *not* cure the
        heavy-atom force problem and is not the gradient of the computed energy.
        See :func:`~carcara.algorithms.forces.hellmann_feynman_gradient` for the
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
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, method: str = DEFAULT_METHOD, *, basis="FAO",
                 h: float = 0.20, grid=None,
                 include_pulay: bool = True, force_method: str = "rdm",
                 project_translation: bool = DEFAULT_PROJECT_TRANSLATION,
                 hellmann_feynman: str = "analytic", orbital_delta=None,
                 scf_iterations: int = 40, verbose: bool = True,
                 measurement_provider=None,
                 **solver_kwargs):
        Calculator.__init__(self)
        self.method, self._solver_class = resolve_method(method)
        self.basis = basis
        self.h = float(h)
        self.include_pulay = bool(include_pulay)
        self.project_translation = bool(project_translation)
        if str(force_method) not in ("rdm", "scf-response"):
            raise ValueError(f"force_method must be 'rdm' or 'scf-response', "
                             f"got {force_method!r}")
        self.force_method = str(force_method)
        self.hellmann_feynman = str(hellmann_feynman)
        self.orbital_delta = orbital_delta
        self.scf_iterations = int(scf_iterations)
        self.verbose = bool(verbose)
        self.solver_kwargs = dict(solver_kwargs)
        self.measurement_provider = measurement_provider
        #: Energy, RDMs and expectation values of the last measured state.
        self.measurement = None

        # An explicit grid is frozen from the start; otherwise the grid is only
        # frozen once forces are requested (see the module docstring).
        self._grid = grid
        #: :class:`~carcara.algorithms.forces.ForceResult` of the most recent step.
        self.force_result = None
        #: The solver instance of the most recent evaluation.
        self.solver = None

    # -- solver delegation ------------------------------------------------- #

    @property
    def result(self):
        """Run result of the most recent evaluation (``None`` before any run)."""
        return getattr(self.solver, "result", None)

    @property
    def dry_run_result(self):
        """:class:`~carcara.algorithms.dry_run.QubitEstimate` of the last dry run."""
        return getattr(self.solver, "dry_run_result", None)

    def dry_run(self, atoms=None):
        """Estimate the qubit requirements **without running** anything.

        Counts the basis functions of ``atoms`` (or of the attached geometry),
        resolves the charge / spin / frozen-core bookkeeping and the mapping
        exactly as a run would, and returns a
        :class:`~carcara.algorithms.dry_run.QubitEstimate` -- no integrals, no
        Hamiltonian, no circuits.  In direct mode (``load_hamiltonian=``) the
        cached file's header is all that is read.
        """
        if atoms is None:
            atoms = self.atoms
        solver = self._make_solver(grid=self._grid)
        estimate = solver.estimate_qubits(atoms)
        solver.dry_run_result = estimate
        self.solver = solver
        return estimate

    def _require_solver(self):
        if self.solver is None:
            raise RuntimeError(
                "the calculator has not been evaluated yet; attach it to an "
                "Atoms object and get an energy, or call run()")
        return self.solver

    @property
    def hamiltonian(self):
        """Qubit Hamiltonian (:class:`~carcara.core.mapping.PauliSum`) of the last evaluation."""
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

    def _make_solver(self, grid):
        return self._solver_class(basis=self.basis, grid=grid, h=self.h,
                                  verbose=self.verbose, **self.solver_kwargs)

    def run(self, **run_kwargs):
        """Run the solver in **direct mode** (no geometry) and return its result.

        Direct mode needs a complete problem specification in the constructor
        keywords -- typically ``load_hamiltonian=`` (a cached qubit Hamiltonian),
        or an explicit ``hamiltonian`` with its companions.  Keyword arguments
        are forwarded to the solver's ``run``.
        """
        self.solver = self._make_solver(grid=self._grid)
        self.solver.result = self.solver.run(**run_kwargs)
        return self.solver.result

    def interaction_energy(self, atoms, fragments, charges=None, **overrides):
        """``E(complex) - sum E(fragments)`` on one shared grid, with this
        calculator's method, basis and solver options.

        See :func:`~carcara.algorithms.interaction.interaction_energy`; the
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
                                  verbose=self.verbose, **options)

    def energy_levels(self, num_states: int = 2, **solver_kwargs):
        """Excited states by variational deflation (see :mod:`carcara.algorithms.deflation`).

        Delegates to the current solver.  In calculator mode, evaluate an energy
        first (``atoms.get_potential_energy()``) so the Hamiltonian exists; in
        direct mode (``load_hamiltonian=``) it can be called immediately.
        """
        if self.solver is None:
            self.solver = self._make_solver(grid=self._grid)
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

        want_forces = "forces" in properties
        # A previous step's breakdown must never survive a new geometry.
        self.force_result = None
        if want_forces:
            self._require_atom_centered_basis(self.basis)

        # Forces need one common grid along the whole trajectory; a plain
        # energy uses the solver's own per-geometry grid unless one was given.
        grid = self._frozen_grid(atoms) if want_forces else self._grid
        solver = self._make_solver(grid=grid)
        energy_ev = self._single_point(solver, atoms)
        self.solver = solver

        measured = None
        if self.measurement_provider is not None and not solver.dry_run:
            measured = self._measure(solver)
            energy_ev = measured["energy_eV"]

        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

        if want_forces and solver.dry_run:
            # A dry run computes nothing to differentiate.
            self.results["forces"] = np.full((len(atoms), 3), np.nan)
            return
        if want_forces:
            self._check_force_support(solver)
            if measured is None:
                self.force_result = self._forces(solver)
            else:
                self.force_result = self._forces(
                    solver, rdms=measured["rdms"],
                    reference_energy=measured["energy_hartree"])
            self.results["forces"] = self.force_result.forces

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

        reduced = bool(getattr(solver, "two_qubit_reduction", False))
        n_qubits = int(solver.n_qubits)
        n_modes = n_qubits + (2 if reduced else 0)
        ones, twos = rdm_qubit_operators(n_modes, solver.mapping,
                                         two_qubit_reduction=reduced,
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
        if getattr(solver, "two_qubit_reduction", False):
            # A tapered register has no ladder operators of its own: its RDM
            # elements are expectation values of the tapered qubit operators.
            n_modes = n_qubits + 2
            ones, twos = rdm_qubit_operators(n_modes, solver.mapping,
                                             two_qubit_reduction=True,
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
                reference_energy = solver.result.in_units("Ha")
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
                    f"misses an orbital-response term of that size.  Converge "
                    f"the solver further (a larger max_iterations / smaller "
                    f"gradient_tolerance) for trustworthy forces.",
                    RuntimeWarning, stacklevel=3)
            self._check_translational_invariance(result)
            if self.project_translation:
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
        if self.project_translation:
            self._project_translation(legacy)
        return legacy

    def _project_translation(self, result):
        """Subtract the mean force, so the molecule cannot drift.

        The exact energy of a free molecule is translation invariant, so
        ``sum_A F_A = 0`` is an identity, not an approximation.  Removing the
        mean therefore *enforces* a symmetry the discretized energy broke; it
        is the translational component of the grid's egg-box error and nothing
        else.  What it does not do is fix the rest of the egg-box: the force
        differences that bend and stretch the molecule keep whatever error the
        spacing gives them, so this is a way to stop a relaxation walking off
        across the grid, not a substitute for a grid fine enough to trust.

        Only for a non-periodic system: under periodic boundary conditions the
        net force on the cell contents is not a free-molecule identity.
        """
        forces = np.asarray(result.forces, dtype=float)
        periodic = self.atoms is not None and bool(np.any(self.atoms.pbc))
        if forces.shape[0] < 2 or periodic:
            return result
        mean = forces.mean(axis=0)
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
        return (f"Carcara(method={self.method!r}, "
                f"basis={self.basis!r}, h={self.h})")

