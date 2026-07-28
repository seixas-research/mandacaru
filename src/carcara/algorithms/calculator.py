# -*- coding: utf-8 -*-
# file: algorithms/calculator.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The unified ASE calculator for every molecular variational method.

:class:`QuantumCalculator` is the single user-facing entry point for running a
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
    from carcara.algorithms import QuantumCalculator

    water = molecule("H2O")
    water.center(vacuum=3.0)
    water.calc = QuantumCalculator(method="adapt-vqe", basis="FAO", h=0.30,
                                   frozen_core=True, verbose=False)
    BFGS(water).run(fmax=0.05)

The run result of the most recent evaluation is available uniformly on
:attr:`QuantumCalculator.result`, whatever the method
(``VQEResult`` / ``ADAPTVQEResult`` / ``VASQEResult`` / the subspace results).
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

Whenever forces are requested, :class:`QuantumCalculator` therefore builds the
grid **once**, from the initial geometry plus ``vacuum`` padding, and reuses it
for every subsequent geometry.  Energies along the trajectory are then all
evaluated on one common grid, which is exactly the condition under which the
analytic gradient is the derivative of the reported energy.  Keep the padding
generous enough that no atom approaches the box edge during the relaxation.  An
explicit ``grid=`` is always used verbatim (and frozen).
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

#: Method names accepted by ``method=``.
METHODS = ("vqe", "adapt-vqe", "vasqe",
           "subspace-vqe", "subspace-adapt-vqe", "subspace-vasqe")


def resolve_method(name: str):
    """Return ``(canonical_name, solver_class)`` for a method spec."""
    key = str(name).strip().lower()
    if key not in METHODS:
        raise ValueError(f"unknown method {name!r}; use one of {METHODS}")
    from . import (ADAPTVQE, VASQE, VQE, SubspaceADAPTVQE, SubspaceVASQE,
                   SubspaceVQE)
    return key, {"vqe": VQE, "adapt-vqe": ADAPTVQE, "vasqe": VASQE,
                 "subspace-vqe": SubspaceVQE,
                 "subspace-adapt-vqe": SubspaceADAPTVQE,
                 "subspace-vasqe": SubspaceVASQE}[key]


class QuantumCalculator(Calculator):
    """ASE calculator running the variational method named by ``method``.

    Parameters
    ----------
    method : str
        Which variational eigensolver evaluates the energy -- ``"vqe"``,
        ``"adapt-vqe"`` (default), ``"vasqe"``, or the subspace-search variants
        ``"subspace-vqe"`` / ``"subspace-adapt-vqe"`` / ``"subspace-vasqe"``.
        ADAPT-VQE is the practical choice for anything beyond a couple of
        orbitals: a fixed UCCSD ansatz becomes very slow past ~8 qubits.
    basis : str or dict
        Basis family, as for the solvers (default ``"FAO"``); accepts a
        ``{"name": ..., <options>}`` dict, including the periodic plane-wave
        family (energy only -- plane waves carry no forces).
    h : float
        Grid spacing in Angstrom (default ``0.20``), used both for the
        per-geometry grid built from ``atoms.cell`` and for the frozen force
        grid.
    vacuum : float
        Padding in Angstrom added around the initial geometry when the frozen
        force grid is built (default ``3.0``).  That grid is fixed for the whole
        trajectory, so this must accommodate any expansion during a relaxation.
    grid : Grid, optional
        An explicit grid, used verbatim (and frozen) for every evaluation.
    include_pulay : bool
        Include the Pulay (basis-motion) force terms (default ``True``).  Setting
        it to ``False`` gives the bare Hellmann-Feynman force; for an atom-centered
        basis that is **not** the gradient of the energy and will not relax to the
        right geometry -- it is exposed for analysis, not for production.
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

    Notes
    -----
    Every ``calculate`` runs a full variational optimization, so an ASE
    relaxation step costs one complete solver run.  The converged result of the
    most recent evaluation is available on :attr:`result`, the solver instance
    on :attr:`solver`, and the force breakdown on :attr:`force_result`.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, method: str = "adapt-vqe", *, basis="FAO",
                 h: float = 0.20, vacuum: float = 3.0, grid=None,
                 include_pulay: bool = True,
                 hellmann_feynman: str = "analytic", orbital_delta=None,
                 scf_iterations: int = 40, verbose: bool = True,
                 **solver_kwargs):
        Calculator.__init__(self)
        self.method, self._solver_class = resolve_method(method)
        self.basis = basis
        self.h = float(h)
        self.vacuum = float(vacuum)
        self.include_pulay = bool(include_pulay)
        self.hellmann_feynman = str(hellmann_feynman)
        self.orbital_delta = orbital_delta
        self.scf_iterations = int(scf_iterations)
        self.verbose = bool(verbose)
        self.solver_kwargs = dict(solver_kwargs)

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
        """Build the frozen integration grid once, from the *initial* geometry."""
        if self._grid is not None:
            return self._grid
        from ..integrals import Grid

        positions = atoms.get_positions()
        extent = positions.max(axis=0) - positions.min(axis=0)
        box = float(np.max(extent) + 2.0 * self.vacuum)
        self._grid = Grid(center=positions.mean(axis=0), box_size=box, h=self.h)
        return self._grid

    @staticmethod
    def _require_atom_centered_basis(basis):
        """Reject the plane-wave family before any expensive work.

        Plane waves do not move with the nuclei, so they generate no Pulay
        forces and the gradient machinery does not apply.  Checking up front
        means the user finds out immediately instead of after a full
        variational run.
        """
        from ._hamiltonian_from_atoms import resolve_basis

        name, _options = resolve_basis(basis)
        if name.upper().replace("-", "").replace(" ", "") in ("PW", "PLANEWAVE"):
            raise NotImplementedError(
                "nuclear forces need an atom-centered basis whose orbitals move "
                "with the nuclei; the plane-wave ('PW') family does not "
                "qualify. Use 'FAO', 'NAO', 'GTO' or '6-31G(d)'.")

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
        if want_forces and not self.solver_kwargs.get("pseudopotentials"):
            self._require_atom_centered_basis(self.basis)

        # Forces need one common grid along the whole trajectory; a plain
        # energy uses the solver's own per-geometry grid unless one was given.
        grid = self._frozen_grid(atoms) if want_forces else self._grid
        solver = self._make_solver(grid=grid)
        energy_ev = self._single_point(solver, atoms)

        self.solver = solver
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

        if want_forces:
            self.force_result = self._forces(solver)
            self.results["forces"] = self.force_result.forces

    def _single_point(self, solver, atoms):
        """Attach the solver to a copy of ``atoms`` and get the energy in eV."""
        work = atoms.copy()
        work.calc = solver
        return float(work.get_potential_energy())

    # -- forces ------------------------------------------------------------ #

    def _forces(self, solver):
        """Analytic nuclear gradient of the converged state."""
        from .forces import nuclear_gradient
        from .rdm import one_rdm, two_rdm

        context = getattr(solver, "_gradient_context", None)
        if context is None:
            raise NotImplementedError(
                "nuclear forces need an atom-centered basis whose integrals are "
                "available; the plane-wave ('PW') family does not qualify. Use "
                "an atom-centered basis such as 'FAO', 'GTO' or '6-31G(d)'.")

        psi = self._converged_state(solver)
        n_qubits = int(solver.n_qubits)
        gamma = one_rdm(psi, n_qubits, solver.mapping)
        gamma2 = two_rdm(psi, n_qubits, solver.mapping)

        return nuclear_gradient(
            context["integrals"], gamma, gamma2,
            n_electrons=context["n_electrons"],
            atom_of_orbital=context["atom_of_orbital"],
            frozen=context["frozen"], orbital_delta=self.orbital_delta,
            scf_iterations=self.scf_iterations,
            include_pulay=self.include_pulay,
            hellmann_feynman=self.hellmann_feynman)

    @staticmethod
    def _converged_state(solver):
        """The optimized state vector of whichever solver just ran."""
        result = getattr(solver, "result", None)
        if result is None:
            raise RuntimeError("the solver has not been run yet")
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
        """``(hellmann_feynman, pulay)`` gradients of the last step (Ha/Bohr).

        Useful for showing how much of the force the Hellmann-Feynman term alone
        accounts for -- for an atom-centered basis, typically not enough.
        """
        if self.force_result is None:
            raise RuntimeError("no forces have been computed yet")
        return (self.force_result.hellmann_feynman, self.force_result.pulay)

    def __repr__(self) -> str:
        return (f"QuantumCalculator(method={self.method!r}, "
                f"basis={self.basis!r}, h={self.h})")
