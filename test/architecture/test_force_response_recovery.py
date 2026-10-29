"""PAW ownership and variational forces checked by independent total energies."""
from ase import Atoms
import numpy as np
import pytest

from mandacaru import Geometry, Mandacaru


def lih():
    return Atoms("LiH", positions=[[4., 4., 3.2], [4., 4., 4.8]], cell=[8.] * 3)


def test_owned_paw_setup_rebuilds_after_the_creating_calculator_is_closed():
    atoms = lih()
    with Mandacaru(method="rhf", atoms=atoms, basis="PAW-LCAO", grid_spacing=.35,
                   resources={"threads": 1}) as calc:
        atoms.get_potential_energy()
        owned = calc.integrals
    atoms.positions[1, 2] += .01
    geometry = Geometry(tuple(atoms.numbers), atoms.positions, atoms.cell.array)
    rebuilt = owned.at_geometry(geometry.native())
    rebuilt_reference = rebuilt.mean_field("rhf", (1, 1))
    with Mandacaru(method="rhf", atoms=atoms, basis="PAW-LCAO", grid=owned.grid,
                   grid_spacing=.35, resources={"threads": 1}) as fresh:
        fresh.run()
        assert rebuilt_reference.snapshot()["electronic_energy"] == pytest.approx(
            fresh.native_reference.snapshot()["electronic_energy"], abs=1e-11)
        assert np.allclose(rebuilt.snapshot()["overlap"], fresh.integrals.snapshot()["overlap"], atol=1e-12)


def test_paw_variational_force_uses_native_response_and_matches_reoptimized_bond_energies(monkeypatch):
    import scipy.optimize

    atoms = lih()
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="PAW-LCAO", grid_spacing=.35,
                   resources={"threads": 1}) as calc:
        atoms.get_potential_energy()
        result = calc.result
        grid = calc.integrals.grid

        def forbidden(*args, **kwargs):
            pytest.fail("force response must remain inside Rust")

        monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
        monkeypatch.setattr(np.linalg, "eigh", forbidden)
        force = atoms.get_forces()
        assert np.max(np.abs(force.sum(axis=0))) < 1e-12
        assert result.force_details["orbital_response"] == "displaced-SCF-and-selector"
        assert result.force_details["scheme"] == "Richardson-reduced-Hamiltonian"
        assert result.force_details["directions"] == 1
        assert result.force_details["energy_ha"] == pytest.approx(result.energy_hartree, abs=1e-10)
    monkeypatch.undo()
    step = 1e-4
    energies = []
    for offset in (step, -step):
        moved = lih()
        moved.positions[0, 2] -= offset / 2
        moved.positions[1, 2] += offset / 2
        with Mandacaru(method="adapt-vqe", atoms=moved, basis="PAW-LCAO", grid=grid,
                       grid_spacing=.35, resources={"threads": 1}) as displaced:
            energies.append(moved.get_potential_energy())
    assert force[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)


def test_variational_cartesian_force_retains_the_unprojected_discrete_energy_gradient():
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="HAO", grid_spacing=.5,
                   project_translation=False, resources={"threads": 1}) as calc:
        force = atoms.get_forces()
        grid = calc.integrals.grid
        assert calc.result.force_details["directions"] == 6
    energies = []
    step = 1e-4
    for offset in (step, -step):
        moved = atoms.copy()
        moved.positions[1, 2] += offset
        with Mandacaru(method="adapt-vqe", atoms=moved, basis="HAO", grid=grid,
                       grid_spacing=.5, project_translation=False,
                       resources={"threads": 1}) as calc:
            energies.append(moved.get_potential_energy())
    assert force[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)
