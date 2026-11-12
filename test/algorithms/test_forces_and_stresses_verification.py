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
    calc = Mandacaru(method="dft", xc="pbe", basis="HAO", grid_spacing=0.5, parallelization={"threads": 1})
    atoms.calc = calc
    analytic_forces = atoms.get_forces()
    assert "analytic" in calc.result.force_details["scheme"].lower()
    assert analytic_forces.shape == (2, 3)

    # Central difference of energy on frozen grid
    h = 0.002
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.calc = Mandacaru(method="dft", xc="pbe", basis="HAO", grid=grid, grid_spacing=0.5, parallelization={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(0.74 + h) - energy_at(0.74 - h)) / (2 * h)
    assert analytic_forces[1, 2] == pytest.approx(num_force_z, abs=0.05)


def test_dft_molecule_spin_polarized_analytic_forces():
    """Spin-polarized OH radical with DFT (PBE, spin=True): analytic forces match finite difference."""
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], cell=[6.0] * 3)
    atoms.set_initial_magnetic_moments([1.0, 0.0])
    calc = Mandacaru(method="dft", xc="pbe", spin=True, basis="HAO", grid_spacing=0.5, project_translation=False, parallelization={"threads": 1})
    atoms.calc = calc
    analytic_forces = atoms.get_forces()
    assert "analytic" in calc.result.force_details["scheme"].lower()
    assert analytic_forces.shape == (2, 3)

    h = 0.002
    grid = calc.integrals.grid
    def energy_at(z):
        m = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[6.0] * 3)
        m.set_initial_magnetic_moments([1.0, 0.0])
        m.calc = Mandacaru(method="dft", xc="pbe", spin=True, basis="HAO", grid=grid, grid_spacing=0.5, project_translation=False, parallelization={"threads": 1})
        return m.get_potential_energy()

    num_force_z = -(energy_at(1.0 + h) - energy_at(1.0 - h)) / (2 * h)
    assert analytic_forces[1, 2] == pytest.approx(num_force_z, abs=0.05)


def test_dft_molecule_soc_analytic_forces():
    """Molecular spin-orbit PAW datasets (lda-fr/pbe-fr) compute analytic forces in Rust
    (the molecular spinor Kohn-Sham state)."""
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], cell=[6.0, 6.0, 7.0])
    atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.4, paw_dataset="lda-fr",
                           basis={"name": "PAW-LCAO", "size": "SZ"}, trace=False)
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

def _paw_h2(method, z, grid=None, **options):
    """H2 on a coarse PAW-LCAO grid in its two lowest orbitals (the analytic variational force)."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, z]], cell=[5.0] * 3)
    atoms.calc = Mandacaru(method=method, basis="PAW-LCAO", grid=grid, grid_spacing=0.4,
                           active_space={"orbitals": 2}, parallelization={"threads": 1}, **options)
    return atoms


def _check_bond_force(method, **options):
    atoms = _paw_h2(method, 0.74, **options)
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert atoms.calc.result.force_details["scheme"] == "analytic"
    h = 0.001
    grid = atoms.calc.integrals.grid

    def energy_at(z):
        return _paw_h2(method, z, grid, **options).get_potential_energy()

    num_force_z = -(energy_at(0.74 + h) - energy_at(0.74 - h)) / (2 * h)
    assert forces[1, 2] == pytest.approx(num_force_z, abs=1e-3)


def test_vqe_molecule_unpolarized_variational_forces():
    """Unpolarized H2 with VQE: the analytic variational force matches the energy difference."""
    _check_bond_force("vqe")


def test_adapt_vqe_molecule_spin_polarized_variational_forces():
    """Spin-polarized H2 triplet with ADAPT-VQE on an unrestricted reference: the analytic
    variational force matches the energy difference."""
    def triplet(z, grid=None):
        atoms = _paw_h2("adapt-vqe", z, grid)
        atoms.set_initial_magnetic_moments([1.0, 1.0])
        return atoms

    atoms = triplet(1.2)
    forces = atoms.get_forces()
    assert atoms.calc.native_reference.num_particles == [2, 0]
    assert atoms.calc.result.force_details["scheme"] == "analytic"
    h = 0.001
    grid = atoms.calc.integrals.grid
    num_force_z = -(triplet(1.2 + h, grid).get_potential_energy()
                    - triplet(1.2 - h, grid).get_potential_energy()) / (2 * h)
    assert forces[1, 2] == pytest.approx(num_force_z, abs=1e-3)


def test_adapt_vqe_molecule_spin_orbit_pool_variational_forces():
    """ADAPT-VQE with pool='spin-orbit': the analytic variational force matches the energy difference."""
    _check_bond_force("adapt-vqe", pool="spin-orbit")


# ============================================================================ #
# 4. Quantum methods on a crystal: finite-difference forces, stress refused
# ============================================================================ #

CRYSTAL_METHODS = [("vqe", {}), ("adapt-vqe", {}), ("vasqa", {"seed": 0}), ("valsqa", {"seed": 0})]


def _h2_chain(method, options):
    """H2 chain, a = 3 A, bond tilted off the axis, 2 k-points (4 qubits)."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.05, 0.0]],
                  cell=np.diag([3.0, 6.0, 6.0]), pbc=[True, False, False])
    atoms.calc = Mandacaru(method=method, crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.3, kpts={"size": (2, 1, 1), "gamma": True},
                           optimizer={"method": "SLSQP", "maxiter": 1000, "tol": 1e-12},
                           parallelization={"threads": 1}, trace=False, **options)
    return atoms


@pytest.mark.parametrize("method, options", CRYSTAL_METHODS)
def test_a_quantum_crystal_force_sums_to_zero(method, options):
    """The supercell energy is invariant under a rigid translation of the crystal (band-limited
    orbitals, G-space ion potential), so the finite-difference forces cancel. Measured
    2026-11-10: net force 1e-10 eV/A; all four methods give F_x = -6.353022 eV/A on atom 0 to
    1e-5 (each reaches the exact energy at 4 qubits). Before the band limit the forces were grid
    artifacts summing to -15.5 eV/A."""
    atoms = _h2_chain(method, options)
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert np.abs(forces.sum(axis=0)).max() < 1e-7
    assert forces[0, 0] == pytest.approx(-6.353022, abs=5e-5)
    assert atoms.calc.result.force_details["scheme"] == "Central-finite-difference"


def test_a_quantum_crystal_force_is_the_derivative_of_the_reported_energy():
    """Independent route: central difference of ``get_potential_energy`` itself."""
    atoms = _h2_chain("vqe", {})
    forces = atoms.get_forces()
    step = 2e-3
    energies = []
    for sign in (1.0, -1.0):
        moved = _h2_chain("vqe", {})
        moved.positions[1, 1] += sign * step
        energies.append(moved.get_potential_energy())
    assert forces[1, 1] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)


def test_a_search_force_on_a_crystal_needs_a_seed():
    atoms = _h2_chain("vasqa", {})
    with pytest.raises(ValueError, match="seed"):
        atoms.get_forces()


@pytest.mark.parametrize("method, options", CRYSTAL_METHODS)
def test_a_quantum_crystal_refuses_stress(method, options):
    """A strain moves the grid's wavevectors against the band limit: E(strain) jumps."""
    atoms = _h2_chain(method, options)
    with pytest.raises(NotImplementedError, match="stress"):
        atoms.get_stress()
