"""Targeted verification tests for forces (molecules, crystals) and stresses (crystals)
under DFT, VQE, and ADAPT-VQE across unpolarized, spin-polarized, and spin-orbit regimes.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from mandacaru import Mandacaru


# ============================================================================ #
# 1. DFT Method: Molecules (Forces)
# ============================================================================ #

def test_dft_molecule_unpolarized_analytic_forces():
    """Unpolarized H2 with DFT (PBE): analytic forces match numerical finite difference."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[6.0] * 3)
    calc = Mandacaru(method="dft", xc="pbe", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    atoms.calc = calc
    analytic_forces = atoms.get_forces()
    assert "analytic" in calc.result.force_details["scheme"].lower()
    assert analytic_forces.shape == (2, 3)

    # Central difference of energy on frozen grid
    h = 0.002
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.calc = Mandacaru(method="dft", xc="pbe", basis="HAO", grid=grid, grid_spacing=0.5, resources={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(0.74 + h) - energy_at(0.74 - h)) / (2 * h)
    assert analytic_forces[1, 2] == pytest.approx(num_force_z, abs=0.05)


def test_dft_molecule_spin_polarized_analytic_forces():
    """Spin-polarized OH radical with DFT (PBE, spin=True): analytic forces match finite difference."""
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], cell=[6.0] * 3)
    atoms.set_initial_magnetic_moments([1.0, 0.0])
    calc = Mandacaru(method="dft", xc="pbe", spin=True, basis="HAO", grid_spacing=0.5, project_translation=False, resources={"threads": 1})
    atoms.calc = calc
    analytic_forces = atoms.get_forces()
    assert "analytic" in calc.result.force_details["scheme"].lower()
    assert analytic_forces.shape == (2, 3)

    h = 0.002
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.set_initial_magnetic_moments([1.0, 0.0])
        m.calc = Mandacaru(method="dft", xc="pbe", spin=True, basis="HAO", grid=grid, grid_spacing=0.5, project_translation=False, resources={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(1.0 + h) - energy_at(1.0 - h)) / (2 * h)
    assert analytic_forces[1, 2] == pytest.approx(num_force_z, abs=0.05)


def test_dft_molecule_soc_analytic_forces():
    """Molecular spin-orbit PAW datasets (lda-fr/pbe-fr) compute analytic forces in Rust."""
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], cell=[6.0, 6.0, 7.0])
    atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.4, paw_dataset="lda-fr",
                           basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200},
                           symmetry=False, trace=False,
                           periodic_scf_options={"max_iterations": 200, "tolerance": 1e-10})
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert abs(forces[0, 2] + forces[1, 2]) < 0.05
    with pytest.raises(PropertyNotImplementedError, match="periodic structures"):
        atoms.get_stress()


def test_dft_molecule_stress_raises_property_not_implemented():
    """Molecules have no periodic boundary conditions, so stress is not implemented."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[6.0] * 3)
    atoms.calc = Mandacaru(method="dft", xc="pbe", basis="HAO", grid_spacing=0.5)
    with pytest.raises(PropertyNotImplementedError, match="periodic structures"):
        atoms.get_stress()


# ============================================================================ #
# 2. DFT Method: Crystals (Forces and Stresses)
# ============================================================================ #

def test_dft_crystal_unpolarized_analytic_forces_and_stress():
    """Unpolarized LiH crystal: analytic forces and stress are both supported."""
    atoms = Atoms("LiH", cell=[[6.0, 0, 0], [0.4, 5.8, 0], [0.3, 0.2, 5.6]], pbc=True,
                  positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])
    calc = Mandacaru(
        method="dft",
        basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200},
        grid_spacing=0.45,
        kpts={"size": (1, 1, 1), "gamma": True},
        symmetry=False,
        trace=False,
        periodic_scf_options={"max_iterations": 200, "tolerance": 1e-10},
    )
    atoms.calc = calc
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)


def test_dft_crystal_spin_polarized_forces_and_stress_supported():
    """Spin-polarized crystal: analytic forces and stress are both supported."""
    atoms = Atoms("HH", cell=[6.2] * 3, pbc=True, positions=[[3.1, 3.3, 2.0], [3.62, 3.0, 4.0]])
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    atoms.calc = Mandacaru(
        method="dft",
        basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200},
        grid_spacing=0.45,
        spin=True,
        kpts={"size": (1, 1, 1), "gamma": True},
        symmetry=False,
        trace=False,
        periodic_scf_options={"max_iterations": 200, "tolerance": 1e-10},
    )
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    stress = atoms.get_stress()
    assert stress.shape == (6,)


def test_dft_crystal_spin_orbit_coupling_forces_and_stress():
    """Spinor (SOC) crystal: analytic forces and stress are both supported."""
    from ase.build import bulk
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(
        method="dft",
        xc="lda",
        paw_dataset="lda-fr",
        basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200},
        grid_spacing=0.34,
        kpts={"size": (1, 1, 1), "gamma": True},
        symmetry=False,
        trace=False,
        periodic_scf_options={"max_iterations": 200, "tolerance": 1e-10},
    )
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert forces.shape == (2, 3)
    assert np.allclose(forces, 0.0, atol=1e-5)
    assert stress.shape == (6,)


# ============================================================================ #
# 3. VQE and ADAPT-VQE Methods: Molecules (Forces)
# ============================================================================ #

def test_vqe_molecule_unpolarized_variational_forces():
    """Unpolarized H2 with VQE: variational finite difference forces match numerical difference."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[6.0] * 3)
    calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    atoms.calc = calc
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert calc.result.force_details["scheme"] == "Central-reduced-Hamiltonian"

    h = 0.001
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.calc = Mandacaru(method="vqe", basis="HAO", grid=grid, grid_spacing=0.5, resources={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(0.74 + h) - energy_at(0.74 - h)) / (2 * h)
    assert forces[1, 2] == pytest.approx(num_force_z, abs=1e-3)


def test_adapt_vqe_molecule_spin_polarized_variational_forces():
    """Spin-polarized H2 triplet with ADAPT-VQE: variational forces match numerical difference."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], cell=[6.0] * 3)
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    atoms.calc = calc
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert calc.native_reference.num_particles == [2, 0]

    h = 0.001
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.set_initial_magnetic_moments([1.0, 1.0])
        m.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid=grid, grid_spacing=0.5, resources={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(1.2 + h) - energy_at(1.2 - h)) / (2 * h)
    assert forces[1, 2] == pytest.approx(num_force_z, abs=1e-3)


def test_adapt_vqe_molecule_spin_orbit_pool_variational_forces():
    """ADAPT-VQE with pool='spin-orbit': variational forces match numerical difference."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[6.0] * 3)
    calc = Mandacaru(method="adapt-vqe", pool="spin-orbit", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    atoms.calc = calc
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)

    h = 0.001
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.calc = Mandacaru(method="adapt-vqe", pool="spin-orbit", basis="HAO", grid=grid, grid_spacing=0.5, resources={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(0.74 + h) - energy_at(0.74 - h)) / (2 * h)
    assert forces[1, 2] == pytest.approx(num_force_z, abs=1e-3)


# ============================================================================ #
# 4. VQE / ADAPT-VQE Crystals: Periodic Forces and Stresses (All Spin Cases)
# ============================================================================ #

def test_vqe_crystal_unpolarized_forces_and_stress():
    """Unpolarized crystal with VQE: forces and stress are computed in Rust."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[4.0, 4.0, 4.0], pbc=True)
    atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)


def test_adapt_vqe_crystal_spin_polarized_forces_and_stress():
    """Collinear spin-polarized crystal with ADAPT-VQE: forces and stress are computed in Rust."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], cell=[4.0, 4.0, 4.0], pbc=True)
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)


def test_adapt_vqe_crystal_spin_orbit_forces_and_stress():
    """Spin-orbit coupling crystal with ADAPT-VQE: forces and stress are computed in Rust."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[4.0, 4.0, 4.0], pbc=True)
    atoms.calc = Mandacaru(method="adapt-vqe", pool="spin-orbit", basis="HAO", grid_spacing=0.5, resources={"threads": 1})
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)
