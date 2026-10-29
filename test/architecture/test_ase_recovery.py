"""Public ASE workflows that must survive the native implementation change."""

from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
import numpy as np
import pytest

from mandacaru import Mandacaru


def h2():
    return Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3)


def test_primary_calculator_is_ase_and_preserves_cache_and_native_ownership():
    atoms = h2()
    with Mandacaru(
        method="dft", xc="pbe", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        assert isinstance(calc, Calculator) and atoms.calc is calc
        energy = atoms.get_potential_energy()
        force = atoms.get_forces()
        assert not calc.calculation_required(atoms, ["energy", "forces"])
        assert calc.atoms is not atoms
        assert calc.num_particles == (1, 1)
        assert calc.n_qubits == 4
        assert not calc.native_reference.has_model
        assert not calc.integrals.has_eri
        result = calc.result
        force[:] = 0
        assert np.linalg.norm(atoms.get_forces()) > 0
        assert calc.result is result
        atoms.positions[1, 2] += 0.01
        assert calc.calculation_required(atoms, ["energy"])
        assert atoms.get_potential_energy() != pytest.approx(energy, abs=1e-8)
        assert calc.result is not result
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_stress()


def test_failed_geometry_does_not_return_previous_energy():
    atoms = h2()
    with Mandacaru(
        method="rhf", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        old = atoms.get_potential_energy()
        atoms.positions[1] = atoms.positions[0]
        for _ in range(2):
            with pytest.raises(ValueError):
                atoms.get_potential_energy()
            assert calc.result is None and "energy" not in calc.results
        atoms.positions[1, 2] = 2.87
        assert atoms.get_potential_energy() == pytest.approx(old, abs=1e-10)


@pytest.mark.parametrize("extension", ["json", "parquet"])
def test_ase_hamiltonian_export_reload_and_direct_quantum_handoff(tmp_path, extension):
    atoms = h2()
    path = tmp_path / ("h2." + extension)
    with Mandacaru(
        method="rhf", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        reference = atoms.get_potential_energy()
        assert calc.result.scf.n_occupied == 1
        assert calc.result.scf.mo_energies.shape == (2,)
        assert calc.result.scf.n_iterations > 0
        calc.result.save_hamiltonian(path)
        problem = calc.result.as_quantum_problem()
    with Mandacaru(
        method="adapt-vqe", load_hamiltonian=path, resources={"threads": 1}
    ) as cached:
        result = cached.run()
        assert result.success and result.optimal_energy <= reference
        energy = result.optimal_energy
    with Mandacaru(method="adapt-vqe", **problem, resources={"threads": 1}) as direct:
        assert direct.n_qubits == 4 and direct.num_particles == (1, 1)
        assert direct.run().optimal_energy == pytest.approx(energy, abs=1e-10)


@pytest.mark.parametrize("pool", ["fermionic", "qeb", "qubit", "ceo", "ceo-ovp"])
def test_quantum_state_solver_view_and_checkpoint_have_the_same_physical_state(pool):
    atoms = h2()
    with Mandacaru(
        method="adapt-vqe",
        pool=pool,
        atoms=atoms,
        grid_spacing=0.5,
        resources={"threads": 1},
    ) as calc:
        atoms.get_potential_energy()
        result = calc.result
        state = result.state()
        assert state.shape == (2**calc.n_qubits,)
        assert np.vdot(
            state, result.qubit_hamiltonian().matrix() @ state
        ).real == pytest.approx(result.energy_hartree, abs=1e-11)
        assert np.allclose(calc.solver.checkpoint.state_vector(), state, atol=1e-12)
        assert np.allclose(
            calc.solver.ansatz.state(result.optimal_parameters), state, atol=1e-12
        )
        compact = result.snapshot()["state"]
        compact_rdm = calc.problem.rdms(compact, two_body=False)[0]
        full_rdm = calc.problem.rdms(state, two_body=False)[0]
        assert np.allclose(compact_rdm, full_rdm, atol=1e-12)


def test_molecular_ase_spectrum_and_native_dos_conserve_the_state_count():
    atoms = h2()
    with Mandacaru(
        method="rhf", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        atoms.get_potential_energy()
        assert calc.get_number_of_spins() == 1
        assert calc.get_number_of_bands() == 2
        assert calc.get_occupation_numbers().sum() == 2.0
        levels = calc.get_eigenvalues()
        assert calc.get_fermi_level() == levels[0]
        axis, dos = calc.dos(width=0.2, npoints=5001)
        assert np.trapezoid(dos, axis) == pytest.approx(4.0, abs=1e-10)


def test_ghf_ase_levels_are_singly_occupied_spinors():
    atoms = h2()
    with Mandacaru(
        method="ghf", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        atoms.get_potential_energy()
        levels = calc.get_eigenvalues()
        assert len(levels) == 4
        assert np.array_equal(calc.get_occupation_numbers(), [1., 1., 0., 0.])
        assert calc.result.scf.n_occupied == 2
        assert calc.get_fermi_level() == levels[1]
        axis, dos = calc.dos(width=0.2, npoints=5001)
        assert np.trapezoid(dos, axis) == pytest.approx(4., abs=1e-10)


@pytest.mark.parametrize(
    "example",
    [
        "basis_gaussian/01_h2_gaussian_bases.py",
        "basis_nao/01_nao_multizeta_h2.py",
    ],
)
def test_existing_ase_basis_examples_run_without_modifying_their_model(
    example, tmp_path
):
    import subprocess
    import sys

    path = Path(__file__).resolve().parents[2] / "examples" / example
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "eV" in completed.stdout
