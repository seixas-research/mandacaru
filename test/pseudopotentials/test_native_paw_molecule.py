"""Public native PAW recovery, checked against the existing molecular references."""

from ase import Atoms
import numpy as np
import pytest

from mandacaru import Mandacaru, _native
from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.environment import LibraryPathError
from mandacaru.units import HARTREE_TO_EV


@pytest.fixture(scope="module", autouse=True)
def datasets():
    try:
        for symbol in ("H", "Li"):
            get_paw(symbol)
    except (FileNotFoundError, LibraryPathError) as error:
        pytest.skip(f"external PAW dataset library is required: {error}")


@pytest.mark.parametrize(
    "method,expected", [("rhf", -0.726037), ("adapt-vqe", -0.738683)]
)
def test_lithium_closed_1s_exchange_preserves_the_existing_energy_reference(
    method, expected
):
    # Existing PAW_DEFAULT values and geometry in test_paw.py; not remeasured.
    atoms = Atoms("LiH", positions=[[4.0, 4.0, 3.2], [4.0, 4.0, 4.8]], cell=[8.0] * 3)
    with Mandacaru(
        method=method,
        basis="PAW-LCAO",
        atoms=atoms,
        grid_spacing=0.30,
        resources={"threads": 1},
    ) as calc:
        energy = atoms.get_potential_energy() / HARTREE_TO_EV
        assert energy == pytest.approx(expected, abs=3e-6)
        assert isinstance(calc.integrals, _native.MolecularIntegrals)
        assert calc.num_particles == (1, 1) and calc.n_qubits == 4
        snapshot = calc.integrals.snapshot()
        assert snapshot["representation"] == "cholesky+augmentation"
        assert snapshot["augmentation_rank"] > 0
        assert np.isfinite(calc.integrals.resolution_ratios).all()
        if method == "rhf":
            assert not calc.native_reference.has_model
        else:
            import json

            active = calc.native_reference.active_model(json.dumps({"frozen": "auto"}))
            assert tuple(active.num_particles) == (1, 1)
            assert active.n_spatial_orbitals == 2
            state = calc.result.state()
            operator = calc.result.qubit_hamiltonian().matrix()
            assert np.vdot(state, operator @ state).real == pytest.approx(
                energy, abs=1e-10
            )


def test_projector_coordinate_changes_preserve_energy_and_augmented_overlap():
    atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]], cell=[8.0] * 3)
    energies = []
    for basis in ("raw", "dual"):
        with Mandacaru(
            method="rhf",
            atoms=atoms,
            basis={"name": "PAW-LCAO", "size": "DZP", "projector_basis": basis},
            grid_spacing=0.25,
            resources={"threads": 1},
        ) as calc:
            energies.append(atoms.get_potential_energy())
            natural = calc.natural_orbitals()
            overlap = calc.integrals.snapshot()["overlap"]
            assert np.allclose(
                natural.coefficients.conj().T @ overlap @ natural.coefficients,
                np.eye(10),
                atol=1e-10,
            )
            field = calc.volumetric_field()
            assert field.integral + field.augmentation_charge == pytest.approx(
                2.0, abs=1e-10
            )
            assert not calc.native_reference.has_model
    assert energies[0] == pytest.approx(energies[1], abs=2e-5)


def test_retained_paw_result_keeps_its_basis_and_correct_nuclear_charges(tmp_path):
    atoms = Atoms("LiH", positions=[[4.0, 4.0, 3.2], [4.0, 4.0, 4.8]], cell=[8.0] * 3)
    with Mandacaru(
        method="rhf",
        basis="PAW-LCAO",
        atoms=atoms,
        grid_spacing=0.30,
        resources={"threads": 1},
    ) as calc:
        atoms.get_potential_energy()
        result = calc.result
    field = result.volumetric_field()
    assert np.array_equal(field.numbers, [3, 1])
    assert np.array_equal(field.charges, [1.0, 1.0])
    assert field.integral + field.augmentation_charge == pytest.approx(2.0, abs=1e-10)
    path = tmp_path / "lih.cube"
    field.write(path)
    lines = path.read_text().splitlines()
    assert "augmentation=" in lines[0]
    assert int(lines[6].split()[0]) == 3
    assert float(lines[6].split()[1]) == 1.0
