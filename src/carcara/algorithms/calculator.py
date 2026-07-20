# -*- coding: utf-8 -*-
# file: algorithms/calculator.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""ASE calculator with **energy and forces** for geometry optimization.

The variational drivers (:class:`~carcara.algorithms.VQE`,
:class:`~carcara.algorithms.ADAPTVQE`, :class:`~carcara.algorithms.VASQE`) are
already ASE calculators, but they report the energy only -- enough for a single
point, not for a relaxation.  :class:`QuantumCalculator` wraps a driver and adds
the missing property: the analytic nuclear gradient (Hellmann-Feynman **plus**
Pulay, see :mod:`carcara.algorithms.forces`).  With that, any ASE optimizer --
``BFGS``, ``LBFGS``, ``FIRE``, ``QuasiNewton`` -- drives a geometry optimization
whose energies come from a quantum variational eigensolver:

.. code-block:: python

    from ase.build import molecule
    from ase.optimize import BFGS
    from carcara.algorithms import QuantumCalculator

    water = molecule("H2O")
    water.center(vacuum=3.0)
    water.calc = QuantumCalculator(driver="adapt-vqe", basis="FAO", h=0.30,
                                   frozen_core=True)
    BFGS(water).run(fmax=0.05)

Why the grid is frozen
----------------------
This is the one subtlety that makes forces meaningful in a real-space code.  The
drivers normally regenerate the integration grid for each geometry, centred on
the molecule.  During a relaxation that would make the grid *move with the
atoms*, adding a spurious "grid drag" to the energy surface: the computed forces
(taken at fixed grid) would then disagree with the finite difference of the
energies, and the optimizer would chase an artefact.

:class:`QuantumCalculator` therefore builds the grid **once**, from the initial
geometry plus ``vacuum`` padding, and reuses it for every subsequent geometry.
Energies along the trajectory are then all evaluated on one common grid, which is
exactly the condition under which the analytic gradient is the derivative of the
reported energy.  Keep the padding generous enough that no atom approaches the
box edge during the relaxation.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..units import from_hartree

#: Driver names accepted by ``driver=``.
DRIVERS = ("vqe", "adapt-vqe", "vasqe")

_DRIVER_ALIASES = {
    "vqe": "vqe",
    "adapt": "adapt-vqe", "adapt-vqe": "adapt-vqe", "adaptvqe": "adapt-vqe",
    "vasqe": "vasqe",
}


def resolve_driver(name: str):
    """Return ``(canonical_name, driver_class)`` for a driver spec."""
    key = str(name).strip().lower()
    canonical = _DRIVER_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"unknown driver {name!r}; use one of {DRIVERS}")
    from . import ADAPTVQE, VASQE, VQE
    return canonical, {"vqe": VQE, "adapt-vqe": ADAPTVQE,
                       "vasqe": VASQE}[canonical]


class QuantumCalculator(Calculator):
    """ASE calculator returning the variational energy **and** analytic forces.

    Parameters
    ----------
    driver : str
        Which variational eigensolver evaluates the energy -- ``"vqe"``,
        ``"adapt-vqe"`` (default) or ``"vasqe"``.  ADAPT-VQE is the practical
        choice for anything beyond a couple of orbitals: a fixed UCCSD ansatz
        becomes very slow past ~8 qubits.
    basis : str or dict
        Basis family, as for the drivers (default ``"FAO"``).  Must be an
        atom-centred family -- the plane-wave basis does not move with the nuclei
        and is rejected.
    h : float
        Grid spacing in Angstrom for the frozen grid (default ``0.25``).
    vacuum : float
        Padding in Angstrom added around the initial geometry when the grid is
        built (default ``3.0``).  The grid is fixed for the whole trajectory, so
        this must accommodate any expansion during the relaxation.
    grid : Grid, optional
        An explicit grid, used verbatim instead of being generated.
    include_pulay : bool
        Include the Pulay (basis-motion) force terms (default ``True``).  Setting
        it to ``False`` gives the bare Hellmann-Feynman force; for an atom-centred
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
    charge, frozen_core, frozen_orbitals, mapping, optimizer, ... :
        Forwarded to the driver.  ``verbose`` defaults to ``False`` here, since a
        relaxation calls the driver many times.

    Notes
    -----
    Every ``calculate`` runs a full variational optimization, so an ASE
    relaxation step costs one complete VQE/ADAPT-VQE run.  The converged driver
    result of the most recent step is available on :attr:`driver_result`, and the
    force breakdown on :attr:`force_result`.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, driver: str = "adapt-vqe", *, basis="FAO",
                 h: float = 0.25, vacuum: float = 3.0, grid=None,
                 include_pulay: bool = True,
                 hellmann_feynman: str = "analytic", orbital_delta=None,
                 scf_iterations: int = 40, verbose: bool = False,
                 **driver_kwargs):
        Calculator.__init__(self)
        self.driver_name, self._driver_class = resolve_driver(driver)
        self.basis = basis
        self.h = float(h)
        self.vacuum = float(vacuum)
        self.include_pulay = bool(include_pulay)
        self.hellmann_feynman = str(hellmann_feynman)
        self.orbital_delta = orbital_delta
        self.scf_iterations = int(scf_iterations)
        self.verbose = bool(verbose)
        self.driver_kwargs = dict(driver_kwargs)

        if not self.driver_kwargs.get("pseudopotentials"):
            self._require_atom_centred_basis(basis)

        self._grid = grid
        #: Driver result of the most recent evaluation.
        self.driver_result = None
        #: :class:`~carcara.algorithms.forces.ForceResult` of the most recent step.
        self.force_result = None
        #: The driver instance of the most recent evaluation.
        self.driver = None

    @staticmethod
    def _require_atom_centred_basis(basis):
        """Reject the plane-wave family up front, before any expensive work.

        Plane waves do not move with the nuclei, so they generate no Pulay
        forces and the gradient machinery does not apply.  Checking here means
        the user finds out immediately instead of after a full variational run.
        """
        from ._hamiltonian_from_atoms import resolve_basis

        name, _options = resolve_basis(basis)
        if name.upper().replace("-", "").replace(" ", "") in ("PW", "PLANEWAVE"):
            raise NotImplementedError(
                "nuclear forces need an atom-centred basis whose orbitals move "
                "with the nuclei; the plane-wave ('PW') family does not "
                "qualify. Use 'FAO', 'NAO', 'GTO' or '6-31G(d)'.")

    # -- the frozen grid --------------------------------------------------- #

    def _ensure_grid(self, atoms):
        """Build the integration grid once, from the *initial* geometry."""
        if self._grid is not None:
            return self._grid
        from ..integrals import Grid

        positions = atoms.get_positions()
        extent = positions.max(axis=0) - positions.min(axis=0)
        box = float(np.max(extent) + 2.0 * self.vacuum)
        self._grid = Grid(center=positions.mean(axis=0), box_size=box, h=self.h)
        return self._grid

    # -- ASE hook ---------------------------------------------------------- #

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        """Run the variational solver and compute the analytic nuclear gradient.

        Sets ``results["energy"]`` / ``results["free_energy"]`` (eV) and
        ``results["forces"]`` (eV/Angstrom, ASE sign convention).
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms

        grid = self._ensure_grid(atoms)
        driver = self._driver_class(basis=self.basis, grid=grid,
                                    verbose=self.verbose, **self.driver_kwargs)
        energy_ev = self._single_point(driver, atoms)

        self.driver = driver
        self.driver_result = getattr(driver, driver._result_attr, None)
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

        if "forces" in properties:
            self.force_result = self._forces(driver)
            self.results["forces"] = self.force_result.forces

    def _single_point(self, driver, atoms):
        """Attach the driver to a copy of ``atoms`` and get the energy in eV."""
        work = atoms.copy()
        work.calc = driver
        return float(work.get_potential_energy())

    # -- forces ------------------------------------------------------------ #

    def _forces(self, driver):
        """Analytic nuclear gradient of the converged state."""
        from .forces import nuclear_gradient
        from .rdm import one_rdm, two_rdm

        context = getattr(driver, "_gradient_context", None)
        if context is None:
            raise NotImplementedError(
                "nuclear forces need an atom-centred basis whose integrals are "
                "available; the plane-wave ('PW') family does not qualify. Use "
                "an atom-centred basis such as 'FAO', 'GTO' or '6-31G(d)'.")

        psi = self._converged_state(driver)
        n_qubits = int(driver.n_qubits)
        gamma = one_rdm(psi, n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, n_qubits, driver.mapping)

        return nuclear_gradient(
            context["integrals"], gamma, gamma2,
            n_electrons=context["n_electrons"],
            atom_of_orbital=context["atom_of_orbital"],
            frozen=context["frozen"], orbital_delta=self.orbital_delta,
            scf_iterations=self.scf_iterations,
            include_pulay=self.include_pulay,
            hellmann_feynman=self.hellmann_feynman)

    @staticmethod
    def _converged_state(driver):
        """The optimized state vector of whichever driver just ran."""
        result = getattr(driver, driver._result_attr, None)
        if result is None:
            raise RuntimeError("the driver has not been run yet")
        parameters = np.asarray(result.optimal_parameters, dtype=float)
        ansatz = getattr(driver, "ansatz", None)
        if ansatz is not None:                                   # fixed ansatz
            return ansatz.state(parameters)
        # Adaptive drivers rebuild the grown ansatz from the selected operators.
        from ..circuits.adapt_ansatz import AdaptAnsatz

        ansatz = AdaptAnsatz(driver.n_qubits, driver.pool.occupied_orbitals,
                             driver.mapping,
                             sparse=getattr(driver, "_sparse", False))
        labels = {op.label: op for op in driver._pool_ops}
        for label in result.operators:
            ansatz.append(labels[label])
        return ansatz.state(parameters)

    # -- convenience ------------------------------------------------------- #

    def get_force_breakdown(self):
        """``(hellmann_feynman, pulay)`` gradients of the last step (Ha/Bohr).

        Useful for showing how much of the force the Hellmann-Feynman term alone
        accounts for -- for an atom-centred basis, typically not enough.
        """
        if self.force_result is None:
            raise RuntimeError("no forces have been computed yet")
        return (self.force_result.hellmann_feynman, self.force_result.pulay)

    def __repr__(self) -> str:
        return (f"QuantumCalculator(driver={self.driver_name!r}, "
                f"basis={self.basis!r}, h={self.h}, "
                f"include_pulay={self.include_pulay})")
