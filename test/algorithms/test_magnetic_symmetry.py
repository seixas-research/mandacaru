"""Collinear magnetic crystals on a symmetry-reduced k mesh, through ``Mandacaru(method="dft")``.

A collinear moment pattern keeps the operations that map every atom onto an atom of the same element and the same
initial moment (the moment-preserving subgroup of the magnetic group: an operation that sends a moment to its opposite
would need a spin flip of the channels and is not kept). The reference of every number is the same crystal with
``symmetry=False``, which keeps the whole mesh for a magnetic crystal. The energies, the occupation matrices and the
magnetization of the cubic cells are gated in Rust (``periodic_magnetic_symmetry.rs``); here are the group sizes,
the total magnetic moment, the forces and the stress. Needs ``MANDACARU_PAW_PATH``.
"""

import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

pytestmark = pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set")

SZ = {"name": "PAW-LCAO", "size": "SZ"}
SCF = {"max_iterations": 300, "tolerance": 1e-10, "density_tolerance": 5e-10, "mixing": 0.25, "history": 7,
       "width_ha": 0.05 / HARTREE_TO_EV}
SMEARING = {"method": "fermi-dirac", "width": 0.05}


def calculator(symmetry, mesh, grid_spacing, **options):
    return Mandacaru(method="dft", xc="lda", basis=SZ, trace=False, grid_spacing=grid_spacing, smearing=SMEARING,
                     kpts={"size": mesh, "gamma": True}, symmetry=symmetry, periodic_scf_options=SCF, **options)


def triangle(symmetry, moments=(0.5, 0.5, 0.5), mesh=(2, 2, 1)):
    """Three Ni around a threefold axis (D3h), plain DFT: the quadrature is exactly invariant under its operations."""
    a, c, x = 5.0, 3.9, 0.28
    atoms = Atoms("Ni3", cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]], pbc=True,
                  scaled_positions=[[x, 0, 0], [0, x, 0], [-x, -x, 0]])
    atoms.set_initial_magnetic_moments(list(moments))
    atoms.calc = calculator(symmetry, mesh, 0.42)
    return atoms


def operations(atoms):
    group = atoms.calc.periodic_problem.symmetry
    return 0 if group is None else group.n_operations


def kpoints(atoms):
    return len(atoms.calc.periodic_problem.kpoints)


def nickel_oxide(symmetry, moments=(2.0, -2.0, 0.0, 0.0), shift=0.15, grid_spacing=0.37):
    """Rocksalt NiO in the four-atom rhombohedral cell of the type-II antiferromagnet, Dudarev U = 6 eV, one oxygen moved
    by ``shift`` A along [111] (the threefold axis stays: a force along it, 6 operations with the moments)."""
    a = 4.17
    cell = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * a / 2
    positions = np.array([[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, 0], [a / 2, a / 2, a / 2]], dtype=float)
    positions[2] += shift * np.ones(3) / np.sqrt(3)
    atoms = Atoms("Ni2O2", cell=cell, positions=positions, pbc=True)
    atoms.set_initial_magnetic_moments(list(moments))
    atoms.calc = calculator(symmetry, (2, 2, 2), grid_spacing, hubbard_u={"Ni": {"l": "d", "U": 6.0}})
    return atoms


def ferromagnetic_nickel(symmetry):
    """fcc Ni, 0.6 mu_B per atom (ASE's default), Dudarev U = 4 eV, 2x2x2: 8 points whole, 3 on the wedge."""
    atoms = bulk("Ni", "fcc", a=3.52)
    atoms.set_initial_magnetic_moments([0.6])
    atoms.calc = calculator(symmetry, (2, 2, 2), 0.4, hubbard_u={"Ni": {"l": "d", "U": 4.0}})
    return atoms


def three_fold_average_forces(forces):
    """The force of an atom on the threefold axis [111] (or mapped onto itself by it): along [111]."""
    return np.array([[np.mean(f)] * 3 for f in forces])


# --------------------------------------------------------------------------- #
# The negative control: the pattern selects the group.
# --------------------------------------------------------------------------- #


def test_a_moment_pattern_keeps_only_the_operations_that_preserve_it():
    """12 operations for the uniform pattern and for none, 4 once one site is flipped (the mirror through it and the plane of
    the atoms): the group follows the moments, not the geometry."""
    counts = {}
    for name, moments in (("ferro", (0.5, 0.5, 0.5)), ("flipped", (0.5, 0.5, -0.5)), ("none", (0.0, 0.0, 0.0))):
        atoms = triangle(True, moments)
        atoms.get_potential_energy()
        counts[name] = operations(atoms)
    assert counts == {"ferro": 12, "flipped": 4, "none": 12}


def test_a_magnetic_crystal_with_symmetry_off_keeps_the_whole_mesh():
    whole = triangle(False)
    whole.get_potential_energy()
    assert operations(whole) == 0 and kpoints(whole) == 4


# --------------------------------------------------------------------------- #
# (c) Plain DFT, hexagonal: the discretization carries the operations, the gates are tight.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def triangle_whole():
    atoms = triangle(False)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def triangle_reduced():
    atoms = triangle(True)
    atoms.get_potential_energy()
    return atoms


@pytest.mark.slow
def test_a_ferromagnetic_triangle_gives_the_whole_meshs_energy_and_moment(triangle_whole, triangle_reduced):
    assert operations(triangle_reduced) == 12 and kpoints(triangle_reduced) < kpoints(triangle_whole) == 4
    assert triangle_reduced.get_potential_energy() == pytest.approx(triangle_whole.get_potential_energy(), abs=1e-8)
    assert triangle_whole.get_magnetic_moment() > 0.1
    assert triangle_reduced.get_magnetic_moment() == pytest.approx(triangle_whole.get_magnetic_moment(), abs=1e-8)


@pytest.mark.slow
def test_a_ferromagnetic_triangle_gives_the_whole_meshs_force_and_stress(triangle_whole, triangle_reduced):
    assert np.abs(triangle_whole.get_forces()).max() > 1.0
    assert np.abs(triangle_reduced.get_forces() - triangle_whole.get_forces()).max() < 1e-6
    assert np.abs(triangle_reduced.get_stress() - triangle_whole.get_stress()).max() < 1e-6


# --------------------------------------------------------------------------- #
# (b) Ferromagnetic fcc Ni with +U, cubic: the whole mesh breaks the cubic symmetry at the floor of the sphere quadrature.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_ferromagnetic_nickel_stress_on_the_wedge_is_the_whole_meshs_to_the_cubic_floor():
    """The floor of the cubic quadrature in the stress is 4e-4 eV/A^3 for fcc Ni without moments and without +U (measured on
    2x2x2): the gate is 1e-3, the wedge is isotropic to rounding, and the magnetization is the whole mesh's."""
    reduced, whole = ferromagnetic_nickel(True), ferromagnetic_nickel(False)
    stress_reduced, stress_whole = reduced.get_stress(), whole.get_stress()
    assert kpoints(reduced) == 3 and kpoints(whole) == 8
    assert reduced.get_magnetic_moment() == pytest.approx(whole.get_magnetic_moment(), abs=1e-6)
    assert np.ptp(stress_reduced[:3]) < 1e-12 and np.abs(stress_reduced[3:]).max() < 1e-12
    assert np.abs(stress_reduced - stress_whole).max() < 1e-3


# --------------------------------------------------------------------------- #
# (a) Antiferromagnetic NiO with +U, cubic: forces and stress against the threefold average of the whole mesh.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def oxide_whole():
    atoms = nickel_oxide(False)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def oxide_reduced():
    atoms = nickel_oxide(True)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def oxide_force_whole(oxide_whole):
    return oxide_whole.get_forces()


@pytest.fixture(scope="module")
def oxide_force_reduced(oxide_reduced):
    return oxide_reduced.get_forces()


@pytest.mark.slow
def test_antiferromagnetic_nickel_oxide_energy_and_moments_on_the_wedge(oxide_whole, oxide_reduced):
    """The moments of the two Ni sites stay +-: 4 points on the wedge against 8, 6 operations (the moments forbid the swap of the
    two Ni). The energy differs by the floor of the cubic quadrature (3e-5 eV, plain DFT included: see
    ``periodic_hubbard_symmetry.rs``)."""
    assert operations(oxide_reduced) == 6 and kpoints(oxide_reduced) == 4 and kpoints(oxide_whole) == 8
    assert oxide_reduced.get_potential_energy() == pytest.approx(oxide_whole.get_potential_energy(), abs=1e-4)
    assert abs(oxide_whole.get_magnetic_moment()) < 1e-3 and abs(oxide_reduced.get_magnetic_moment()) < 1e-3
    for shell_reduced, shell_whole in zip(oxide_reduced.calc.result.hubbard, oxide_whole.calc.result.hubbard):
        moment = shell_whole["trace"][0] - shell_whole["trace"][1]
        assert abs(moment) > 0.5
        assert shell_reduced["trace"][0] - shell_reduced["trace"][1] == pytest.approx(moment, abs=1e-5)


@pytest.mark.slow
def test_antiferromagnetic_nickel_oxide_force_on_the_wedge_is_the_whole_meshs_symmetrized(oxide_force_whole,
                                                                                         oxide_force_reduced):
    """The wedge force is along [111] exactly; the whole mesh's is along it only to 8e-4 eV/A (the floor of the cubic
    quadrature: 8e-4 on the same cell without moments too), and its threefold average is the wedge force to 1.3e-5
    (1.7e-5 without moments). The stress of this cell (144 s on the wedge, 252 s on the whole mesh) is too slow for a test; its
    measurement is in the log."""
    assert np.abs(oxide_force_whole).max() > 0.1
    assert np.abs(oxide_force_reduced - three_fold_average_forces(oxide_force_reduced)).max() < 1e-12
    assert np.abs(oxide_force_reduced - three_fold_average_forces(oxide_force_whole)).max() < 5e-5
    assert np.abs(oxide_force_reduced - oxide_force_whole).max() < 2e-3
