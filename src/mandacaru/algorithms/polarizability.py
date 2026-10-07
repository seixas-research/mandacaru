# -*- coding: utf-8 -*-
# file: algorithms/polarizability.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Static dipole polarizability of a molecule by finite fields.

In a uniform field :math:`\mathbf F` the energy and the dipole of a molecule
expand as

.. math::

    E(\mathbf F) = E_0 - \boldsymbol\mu_0\cdot\mathbf F
                   - \tfrac12\,\mathbf F\cdot\alpha\,\mathbf F - \dots,
    \qquad
    \mu_i(\mathbf F) = \mu_{0,i} + \sum_j \alpha_{ij} F_j + \dots

:func:`polarizability` runs the calculation at :math:`\pm F` along each axis
(``Mandacaru(electric_field=...)``) and differentiates the dipole by central
differences, :math:`\alpha_{ij} = [\mu_i(F\hat e_j) - \mu_i(-F\hat e_j)]/2F`;
the error is :math:`O(F^2)` (the second hyperpolarizability).  The energy's
second difference :math:`-[E(F) + E(-F) - 2E_0]/F^2` gives the diagonal
independently -- for a variational method (Hartree-Fock, Kohn-Sham) the two
agree, because the field couples through the same dipole matrices the dipole
is computed from.  Seven runs, all on one grid built from the molecule.

.. code-block:: python

    from ase.build import molecule
    from mandacaru.algorithms import polarizability

    water = molecule("H2O")
    water.center(vacuum=4.0)
    result = polarizability(water, method="dft", xc="pbe",
                            basis={"name": "PAW-LCAO", "size": "TZP"})
    result.isotropic             # Bohr^3 (atomic units)
    result.in_units("angstrom^3")

A crystal's polarization is a Berry phase, not a dipole; it is refused here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import BOHR_TO_ANGSTROM, DEFAULT_GRID_SPACING, HARTREE_TO_EV

#: Field strength (Hartree per e Bohr, 0.10 V/Angstrom) of the finite
#: differences: small enough that the hyperpolarizability's O(F^2) error is
#: far below a percent for a small molecule, large enough that the dipole
#: change stands well above the SCF convergence.
DEFAULT_FIELD = 2e-3

#: Units :meth:`Polarizability.in_units` converts the tensor to.
POLARIZABILITY_UNITS = {"bohr^3": 1.0, "au": 1.0,
                        "angstrom^3": BOHR_TO_ANGSTROM ** 3}


@dataclass
class Polarizability:
    """The static polarizability of one molecule, in atomic units (Bohr^3).

    ``tensor[i, j]`` is :math:`\\partial\\mu_i/\\partial F_j` from the dipoles;
    ``energy_diagonal[i]`` is :math:`-\\partial^2E/\\partial F_i^2` from the
    energies, the independent check of the diagonal.
    """

    tensor: np.ndarray
    energy_diagonal: np.ndarray
    dipole: np.ndarray                 # zero-field dipole, e Bohr
    field: float                       # Hartree per e Bohr
    method: str
    results: list = field(default_factory=list)   # the seven run results

    @property
    def isotropic(self) -> float:
        """The mean polarizability :math:`\\mathrm{Tr}\\,\\alpha/3` (Bohr^3)."""
        return float(np.trace(self.tensor) / 3.0)

    @property
    def anisotropy(self) -> float:
        r"""The polarizability anisotropy :math:`\Delta\alpha` (Bohr^3):
        :math:`\sqrt{\tfrac12[3\,\mathrm{Tr}(\alpha^2) - (\mathrm{Tr}\,\alpha)^2]}`
        of the symmetrized tensor."""
        a = 0.5 * (self.tensor + self.tensor.T)
        value = 0.5 * (3.0 * np.trace(a @ a) - np.trace(a) ** 2)
        return float(np.sqrt(max(value, 0.0)))

    def in_units(self, unit: str = "angstrom^3") -> np.ndarray:
        """The tensor in ``"bohr^3"`` (``"au"``) or ``"angstrom^3"``."""
        key = str(unit).strip().lower()
        if key not in POLARIZABILITY_UNITS:
            raise ValueError(f"unknown unit {unit!r}; use one of "
                             f"{sorted(POLARIZABILITY_UNITS)}")
        return self.tensor * POLARIZABILITY_UNITS[key]


def polarizability(atoms, *, field: float = DEFAULT_FIELD,
                   method: str = "dft", basis="HAO",
                   h: float = DEFAULT_GRID_SPACING,
                   grid=None, **solver_kwargs) -> Polarizability:
    """The static dipole polarizability of the molecule ``atoms``.

    Parameters
    ----------
    atoms : ase.Atoms
        A molecule (no periodic direction).
    field : float
        The finite-difference field strength (Hartree per e Bohr).
    method, basis, h, grid, **solver_kwargs :
        Forwarded to every run, as for any ``Mandacaru`` calculation.  The
        seven runs share one grid, built from ``atoms`` unless ``grid`` is
        given.
    """
    from .calculator import Mandacaru
    from .interaction import _shared_grid

    if bool(np.any(atoms.get_pbc())):
        raise NotImplementedError(
            "a crystal's polarization is a Berry phase, not a dipole: the "
            "finite-field polarizability is for molecules")
    if "electric_field" in solver_kwargs:
        raise ValueError("polarizability applies the fields itself; do not "
                         "pass electric_field")
    field = float(field)
    if not np.isfinite(field) or field <= 0.0:
        raise ValueError(f"field must be a positive number; got {field!r}")
    h = float(h)
    # Energies come back in eV unless the runs are in atomic units.
    to_hartree = 1.0 if solver_kwargs.get("atomic_units") else 1.0 / HARTREE_TO_EV
    method_key = str(method).strip().lower()
    shared = _shared_grid(atoms, h, grid, method_key)

    def run(vector):
        geometry = atoms.copy()
        geometry.calc = Mandacaru(
            method=method, basis=basis, h=h, grid=shared,
            electric_field=None if vector is None else tuple(vector),
            **solver_kwargs)
        energy = geometry.get_potential_energy() * to_hartree
        geometry.calc.get_dipole_moment()
        dipole = np.asarray(geometry.calc.dipole_result.total, dtype=float)
        result = getattr(geometry.calc.solver, "result", None)
        return energy, dipole, result

    e0, mu0, r0 = run(None)
    tensor = np.zeros((3, 3))
    diagonal = np.zeros(3)
    results = [r0]
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = field
        e_plus, mu_plus, r_plus = run(step)
        e_minus, mu_minus, r_minus = run(-step)
        tensor[:, axis] = (mu_plus - mu_minus) / (2.0 * field)
        diagonal[axis] = -(e_plus + e_minus - 2.0 * e0) / field ** 2
        results += [r_plus, r_minus]
    return Polarizability(tensor=tensor, energy_diagonal=diagonal,
                          dipole=mu0, field=field, method=method_key,
                          results=results)
