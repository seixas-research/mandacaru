# -*- coding: utf-8 -*-
# file: test/algorithms/test_crystal_forces.py

# This code is part of Mandacaru.
# MIT License

"""Forces and stress of a periodic Kohn-Sham crystal.

Each is checked against a central difference of the **self-consistent** free
energy on the same grid -- the derivative the analytic gradient (and the
fixed-state strain) claims to be.  A crystal gradient costs about a minute
even for two atoms at Gamma (the Bloch sums of a 7M-function extended basis),
so everything that computes one is marked ``slow``: run them with
``pytest -m slow test/algorithms/test_crystal_forces.py``.
"""

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.algorithms.crystal_forces import (crystal_gradient,
                                                 crystal_stress)
from mandacaru.algorithms.periodic_dft import PeriodicKohnSham
from mandacaru.integrals import Grid
from mandacaru.integrals import exchange_correlation as xc_grid
from mandacaru.pseudopotentials.periodic_paw import build_crystal
from mandacaru.units import HARTREE_TO_EV, to_bohr

OPTIONS = {"size": "SZ", "filter": 300}
SMEARING = {"method": "fermi-dirac", "width": 0.1}


def _lih():
    """Rocksalt LiH with H pushed off its site: every atom feels a force."""
    atoms = bulk("LiH", "rocksalt", a=4.0)
    atoms.positions[1] += [0.06, -0.03, 0.04]
    return atoms


def _solve(atoms, h, kpts=None, functional="lda", grid=None,
           symmetry=False):
    crystal, context = build_crystal(atoms, h, OPTIONS, kpts=kpts, grid=grid,
                                     symmetry=symmetry)
    constant = float(sum(d.one_center_energy for d in crystal.datasets)
                     + sum(xc_grid.core_correction_offset(d)
                           for d in crystal.datasets))
    solver = PeriodicKohnSham(crystal, context["n_electrons"], functional,
                              smearing=SMEARING, constant=constant)
    result = solver.run(tol=1e-11, density_tol=1e-9)
    assert result.converged
    return solver, result


def _strained(atoms, epsilon, shape):
    moved = atoms.copy()
    cell = atoms.cell.array @ (np.eye(3) + epsilon).T
    moved.set_cell(cell, scale_atoms=True)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                h=np.linalg.norm(cell, axis=1) / np.asarray(shape),
                units="angstrom", cell=cell, periodic=True)
    return moved, grid


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_the_force_is_the_derivative_of_the_free_energy(functional):
    """Hellmann-Feynman + Pulay against a central difference of F (one
    component of each atom), on a fixed grid: 1e-5 eV/Angstrom.  Si on a
    2x2x2 mesh reached 8e-7 (LDA), 8e-7 (PBE) and 4e-6 (r2SCAN)."""
    atoms = _lih()
    h = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, _result = _solve(atoms, h, functional=functional)
    gradient = crystal_gradient(solver)
    assert np.all(np.abs(gradient.hellmann_feynman) > 0.0)
    assert np.any(np.abs(gradient.pulay) > 1e-4)
    step = 1e-3                                                 # Angstrom
    for atom, d in ((1, 0), (0, 2)):
        values = []
        for sign in (+1.0, -1.0):
            moved = atoms.copy()
            moved.positions[atom, d] += sign * step
            values.append(_solve(moved, h, functional=functional,
                                 grid=solver.crystal.grid)[1].free_energy)
        numeric = (values[0] - values[1]) / (2.0 * step) * HARTREE_TO_EV
        assert gradient.gradient[atom, d] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.slow
def test_a_k_mesh_force_is_the_derivative_of_the_free_energy():
    """Si on a 2x2x2 mesh: the k-point phases of the Bloch sums and the
    time-reversed weights, which a Gamma-only check cannot see."""
    from ase.build import bulk as build
    atoms = build("Si", "diamond", a=5.43)
    atoms.positions[1] += [0.05, 0.02, -0.03]
    kpts = {"size": (2, 2, 2), "gamma": True}
    h = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, _result = _solve(atoms, h, kpts=kpts)
    gradient = crystal_gradient(solver)
    step = 1e-3
    values = []
    for sign in (+1.0, -1.0):
        moved = atoms.copy()
        moved.positions[1, 0] += sign * step
        values.append(_solve(moved, h, kpts=kpts,
                             grid=solver.crystal.grid)[1].free_energy)
    numeric = (values[0] - values[1]) / (2.0 * step) * HARTREE_TO_EV
    assert gradient.gradient[1, 0] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.slow
def test_the_stress_is_the_strain_derivative_of_the_free_energy():
    """The fixed-state strain derivative against self-consistent strained
    crystals (same node counts, pinned filter): a diagonal and a shear
    component to 1e-5 relative."""
    atoms = _lih()
    h = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, _result = _solve(atoms, h)
    stress = crystal_stress(solver, atoms, OPTIONS, "paw-lcao", None)
    volume = abs(np.linalg.det(to_bohr(atoms.cell.array, "angstrom")))
    step = 1e-3
    for a, b in ((0, 0), (0, 1)):
        values = []
        for sign in (+1.0, -1.0):
            epsilon = np.zeros((3, 3))
            epsilon[a, b] += 0.5 * sign * step
            epsilon[b, a] += 0.5 * sign * step
            moved, grid = _strained(atoms, epsilon,
                                    solver.crystal.grid.shape)
            values.append(_solve(moved, h, grid=grid)[1].free_energy)
        numeric = (values[0] - values[1]) / (2.0 * step) / volume
        assert stress[a, b] == pytest.approx(numeric, rel=1e-5, abs=1e-9)


class TestTheCalculator:
    @pytest.fixture(scope="class")
    def lih(self):
        atoms = _lih()
        atoms.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO",
                                                    **OPTIONS},
                               h=float(np.linalg.norm(atoms.cell[0])) / 10,
                               smearing=SMEARING, trace=False)
        atoms.get_potential_energy()
        return atoms

    @pytest.mark.slow
    def test_forces_reach_ase_with_their_breakdown(self, lih):
        forces = lih.get_forces()
        assert forces.shape == (2, 3)
        result = lih.calc.force_result
        assert np.allclose(result.forces, forces)
        assert np.allclose(result.hellmann_feynman + result.pulay,
                           result.gradient)

    def test_the_free_energy_is_what_ase_relaxes(self, lih):
        """Forces and stress are derivatives of F, not of E(sigma -> 0)."""
        scf = lih.calc.result.scf
        assert lih.get_potential_energy(force_consistent=True) == \
            pytest.approx(scf.free_energy * HARTREE_TO_EV)
        assert lih.get_potential_energy() == pytest.approx(
            lih.calc.result.optimal_energy)

    def test_stress_reaches_ase_in_voigt_order(self, lih, monkeypatch):
        """The calculator's side of the stress: units (Ha/Bohr^3 to
        eV/Angstrom^3) and ASE's Voigt order, with the solver's tensor
        replaced by a known one -- the tensor itself is checked against
        strained crystals above (twelve rebuilds would not fit 180 s here).
        """
        from mandacaru.units import BOHR_TO_ANGSTROM
        tensor = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
        monkeypatch.setattr(lih.calc.solver, "crystal_stress",
                            lambda strain=None: tensor * 1e-4)
        stress = lih.get_stress()
        scale = 1e-4 * HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3
        assert np.allclose(stress, np.array([1, 2, 3, 4, 5, 6]) * scale)
        assert np.allclose(lih.calc.get_stress(voigt=False), tensor * scale)


@pytest.mark.slow
def test_the_irreducible_k_points_give_the_full_mesh_result():
    """Forces and stress from the irreducible k-points, symmetrized, equal
    the time-reversal mesh's: Si with one atom moved along [111] keeps a
    C3v group (six operations, two invariant strains), so its forces are
    nonzero and its stress is not hydrostatic."""
    from mandacaru.algorithms.crystal_forces import (_operations,
                                                     invariant_strains)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1] += 0.04 * np.ones(3)
    kpts = {"size": (2, 2, 2), "gamma": True}
    h = float(np.linalg.norm(atoms.cell[0])) / 10
    full, _ = _solve(atoms, h, kpts=kpts)
    reduced, _ = _solve(atoms, h, kpts=kpts, grid=full.crystal.grid,
                        symmetry=True)
    assert len(_operations(reduced.crystal)) == 6
    assert len(invariant_strains(reduced.crystal)) == 2
    assert len(reduced.crystal.kpoints) < len(full.crystal.kpoints)
    f_full = crystal_gradient(full).forces
    f_reduced = crystal_gradient(reduced).forces
    assert np.max(np.abs(f_full)) > 0.1
    assert f_reduced == pytest.approx(f_full, abs=1e-6)
    s_full = crystal_stress(full, atoms, OPTIONS, "paw-lcao", kpts)
    s_reduced = crystal_stress(reduced, atoms, OPTIONS, "paw-lcao", kpts)
    assert abs(s_full[0, 1]) > 1e-6                  # not hydrostatic
    assert s_reduced == pytest.approx(s_full, abs=1e-8)

