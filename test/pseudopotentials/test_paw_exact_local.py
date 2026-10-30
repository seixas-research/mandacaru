# -*- coding: utf-8 -*-
# file: test/test_paw_exact_local.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The range-separated local potential: exactness and forces.

PAW-LCAO keeps only the long-range half of its local channel on the grid -- the
potential of a Gaussian ion, which the grid resolves -- and integrates the
short-range remainder on atom-centered spherical quadratures
(:mod:`mandacaru.pseudopotentials.local_split`).  The two halves add up to the
potential the all-grid form used, so the tests here are about *exactness* of the
split, and about the force being the derivative of the energy the solver
reported.  The sphere matrices, their translation invariance and the
comparison with the all-grid form belonged to the molecular engine that
integrated the local potential in Python; that engine is native now
(``paw/molecular``), and its quadratures are tested in ``crates``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.pseudopotentials import local_split as ls

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

class TestRangeSeparation:
    def test_the_two_halves_add_up_to_the_local_potential(self):
        """``v^lr + v^sr == v`` by construction, at every radius.

        The short-range part is evaluated as a *difference* of the dataset's own
        ``local_potential`` and the Gaussian-ion potential, never from the
        asymptotic ``-Z erfc(r)/r``, so the identity is exact whatever the
        shipped table's tail actually does (it sits ~5e-7 Hartree off -Z/r).
        """
        from mandacaru.pseudopotentials.paw import get_paw

        radii = np.linspace(0.0, 12.0, 4001)
        for symbol in ("H", "O"):
            dataset = get_paw(symbol)
            sigma = 0.66
            total = (ls.long_range_potential(dataset.valence_charge, sigma, radii)
                     + ls.short_range_potential(dataset, sigma, radii))
            assert np.abs(total - dataset.local_potential(radii)).max() < 1e-12

    def test_the_long_range_part_is_finite_at_the_origin(self):
        """``-Z erf(r/sqrt(2) sigma)/r -> -Z sqrt(2/pi)/sigma``."""
        sigma = 0.5
        value = ls.long_range_potential(3.0, sigma, np.array([0.0]))[0]
        assert value == pytest.approx(-3.0 * np.sqrt(2.0 / np.pi) / sigma)
        # and it joins the tail smoothly: -Z/r far out.
        far = ls.long_range_potential(3.0, sigma, np.array([10.0]))[0]
        assert far == pytest.approx(-0.3, rel=1e-12)

    def test_the_short_range_part_dies_inside_its_sphere(self):
        """What the quadrature truncates, in its two separable pieces.

        Beyond the dataset's local cutoff the potential is the ionic tail, so
        ``v^sr = -Z erfc(r/sqrt2 sigma)/r`` plus whatever the *table* itself
        deviates from ``-Z/r``.  The first is the design's own quantity and is
        ``erfc(6.5/sqrt2) = 7.3e-11`` at the sphere's edge; the second is a
        property of the shipped, decimated library (measured at up to 4.4e-6
        Hartree near 2 Bohr for a *non-relativistic* oxygen, falling to ~2e-7
        by 8 Bohr) and is not
        something the split introduces -- the all-grid form integrates the same
        table.  Both are checked, separately, so a regression in either is
        attributable.

        The bound is 2e-5 rather than 1e-5 because the datasets are
        scalar-relativistic by default: oxygen measures 1.34e-5 against 4.4e-6
        for the same dataset generated with ``relativity="none"``, and
        hydrogen is 7.7e-7 either way.  That factor of three is the
        ``r**gamma`` cusp of a relativistic s state, which a uniform radial
        grid does not resolve -- the limitation recorded in
        :mod:`mandacaru.basis.relativity`.  It was 5.8e-4 until the reference
        atom was made to finish on the *same* Numerov orbitals the generators
        pseudize; a hundredfold of this number was the atom and the
        pseudization disagreeing about where oxygen's 2s sits.
        """
        from scipy.special import erfc

        from mandacaru.pseudopotentials.paw import get_paw

        for symbol in ("H", "O"):
            dataset = get_paw(symbol)
            charge = dataset.valence_charge
            for sigma in (0.3, 0.5, 0.8):
                radius = ls.short_range_radius(dataset, sigma)
                assert radius >= ls.local_cutoff(dataset)
                outside = np.linspace(radius, radius + 6.0, 200)
                erfc_tail = (charge * erfc(outside / (np.sqrt(2.0) * sigma))
                             / outside)
                assert erfc_tail.max() < 1e-9
                table = (ls.short_range_potential(dataset, sigma, outside)
                         + erfc_tail)
                assert np.abs(table).max() < 2e-5


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
