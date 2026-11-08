# -*- coding: utf-8 -*-
# file: test/test_paw_exact_local.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The range-separated local potential: the force is the derivative of the energy.

PAW-LCAO keeps only the long-range half of its local channel on the grid -- the
potential of a Gaussian ion, which the grid resolves -- and integrates the
short-range remainder on atom-centered spherical quadratures (in Rust,
``paw/local_split.rs``). The test here is about the force being the derivative
of the energy the solver reported.  The sphere matrices, their translation invariance and the
comparison with the all-grid form belonged to the molecular engine that
integrated the local potential in Python; that engine is native now
(``paw/molecular``), and its quadratures are tested in ``crates``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

# The classical optimizer used below, with its budget and tolerance written
# out.  `tol` is deliberately looser than the library default (1e-12): these
# tests compare an analytic gradient against a finite difference of the *same*
# state, so they need a converged state, not a converged screening gradient --
# and 1e-12 costs 15x the wall time here for no change in what is measured.
LBFGS = {"method": "L-BFGS", "maxiter": 2000, "tol": 1e-8}

SZ = {"name": "PAW-LCAO", "size": "SZ"}


def water(cell=8.0):
    """C2v water with the two O-H bonds along +y and +z.

    Swapping y and z maps the molecule onto itself and exchanges the hydrogens,
    and the cubic grid shares that symmetry, so exact forces must obey it -- and
    the oxygen sphere (4.3 Bohr at grid_spacing = 0.25 Angstrom) holds *both* hydrogens,
    which is what makes this the interesting case for a sphere quadrature: most
    of what it integrates is not centered on its own atom.
    """
    c = cell / 2
    return Atoms("OHH", positions=[[c, c, c], [c, c + 1.0, c], [c, c, c + 1.0]],
                 cell=[cell] * 3)


# --------------------------------------------------------------------------- #
# The split itself.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Forces.
# --------------------------------------------------------------------------- #

def water_calculator(grid_spacing=0.25):
    return Mandacaru(method="adapt-vqe", basis=SZ, grid_spacing=grid_spacing, pool="fermionic",
                     optimizer=LBFGS, max_iterations=60,
                     convergence={"gradient": 1e-4}, profile=False)


@pytest.fixture(scope="module")
def h2o_forces():
    atoms = water()
    atoms.calc = water_calculator()
    forces = atoms.get_forces()
    return (atoms, forces, atoms.calc.result.forces(project_translation=False))


@pytest.mark.slow
class TestForces:
    def test_p_valence_force_is_the_derivative_of_the_energy(self, h2o_forces):
        """Central difference of the calculator's own energy on its frozen grid.

        The oxygen sphere holds both hydrogens, so this exercises the
        cross-center part of the quadrature derivative -- the part a
        single-atom test would miss.  Only one component is differenced, to keep
        the test inside the suite's time budget; the symmetry test below covers
        the others.
        """
        atoms, _forces, raw = h2o_forces
        step, atom, axis = 0.004, 0, 1
        energies = []
        for sign in (1, -1):
            moved = atoms.copy()
            moved.positions[atom, axis] += sign * step
            moved.calc = atoms.calc
            energies.append(moved.get_potential_energy())
        numerical = -(energies[0] - energies[1]) / (2 * step)
        # The raw gradient: the finite difference is of the discretized energy,
        # which is not translation invariant, while `project_translation="auto"`
        # has taken that component out of the reported force.
        assert raw[atom, axis] == pytest.approx(numerical, abs=2e-2)

    def test_forces_respect_the_c2v_symmetry(self, h2o_forces):
        """The two O-H bonds lie along +y and +z and the cubic grid shares that
        symmetry, so a correct derivative of the sphere integrals keeps it."""
        _atoms, forces, _raw = h2o_forces
        assert forces[0, 1] == pytest.approx(forces[0, 2], abs=5e-3)
        assert forces[1, 1] == pytest.approx(forces[2, 2], abs=5e-3)
        assert abs(forces[0, 1]) > 1.0
        assert np.abs(forces[:, 0]).max() < 1e-3
