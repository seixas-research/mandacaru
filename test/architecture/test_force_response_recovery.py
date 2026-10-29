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
        assert result.force_details["scheme"] == "Central-reduced-Hamiltonian"
        assert result.force_details["directions"] == 1
        assert result.force_details["energy_ha"] == pytest.approx(result.energy_hartree, abs=1e-10)
        richardson = result.forces(internal=True, step=1e-3, extrapolate=True)
        assert result.force_details["scheme"] == "Richardson-reduced-Hamiltonian"
        result.force_details["gradient"].fill(123.)
        assert np.allclose(result.forces(internal=True, step=1e-3, extrapolate=True), richardson)
        central = result.forces(internal=True, step=1e-4, extrapolate=False)
        assert result.force_details["scheme"] == "Central-reduced-Hamiltonian"
        assert result.force_details["step_error_estimate"] is None
        assert np.allclose(central, richardson, atol=1e-4)
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
    assert central[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)
    assert richardson[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)


def test_variational_cartesian_force_retains_the_unprojected_discrete_energy_gradient():
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="HAO", grid_spacing=.5,
                   project_translation=False, resources={"threads": 1}) as calc:
        force = atoms.get_forces()
        central = calc.result.forces(step=1e-4, extrapolate=False)
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
    assert central[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)


def test_central_active_space_response_matches_independent_exact_water_energies():
    import json
    from ase.build import molecule
    from mandacaru.units import HARTREE_TO_EV, BOHR_TO_ANGSTROM

    atoms = molecule("H2O")
    atoms.center(vacuum=3.)
    directions = np.zeros((3, 3))
    for hydrogen in (1, 2):
        bond = atoms.positions[hydrogen] - atoms.positions[0]
        directions[hydrogen] = bond / np.linalg.norm(bond)
    directions -= directions.mean(axis=0)
    directions /= np.linalg.norm(directions)
    assert np.linalg.norm(np.cross(atoms.positions - atoms.positions.mean(axis=0), directions).sum(axis=0)) < 1e-12
    spec = json.dumps({"orbitals": {"occupied": 2, "virtual": 2}, "method": "mp2"})

    def exact(frame, grid=None, *, response=False):
        with Mandacaru(method="rhf", atoms=frame, basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=.35, grid=grid, resources={"threads": 1}) as calc:
            calc.run()
            model = calc.native_reference.active_model(spec)
            problem = model.problem(mapping="jordan_wigner", sector=True)
            full = model.operator().matrix()
            indices = problem.basis_indices
            energies, vectors = np.linalg.eigh(full[np.ix_(indices, indices)])
            derivative = None
            if response:
                raw = calc.native_reference.quantum_gradient(
                    problem, model, np.ascontiguousarray(vectors[:, 0]), active_spec=spec,
                    internal=True, step=1e-4, extrapolate=False)
                assert raw["energy_ha"] == pytest.approx(energies[0], abs=1e-10)
                derivative = np.sum(raw["gradient"] * directions) * HARTREE_TO_EV / BOHR_TO_ANGSTROM
            return float(energies[0]) * HARTREE_TO_EV, calc.integrals.grid, derivative

    _, grid, derivative = exact(atoms, response=True)
    step = 1e-4
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions += step * directions
    minus.positions -= step * directions
    high = exact(plus, grid)[0]
    low = exact(minus, grid)[0]
    assert derivative == pytest.approx((high - low) / (2 * step), abs=1e-4)


def test_relaxation_log_records_the_used_forces_and_ase_completion_verdict(tmp_path):
    from ase.optimize import BFGS
    from mandacaru.utils.logging import parse_output

    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    path = tmp_path / "relaxation.txt"
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="HAO", grid_spacing=.5,
                   txt=path, resources={"threads": 1}) as calc:
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=.02, steps=2)
        final_force = atoms.get_forces()
        assert calc.write_optimization_summary(optimizer=optimizer, force=True)
        assert not calc.write_optimization_summary(optimizer=optimizer, force=True)
        text = path.read_text()
        assert text.count("[FORCES]") == optimizer.nsteps + 1
        assert text.count("[RELAXATION COMPLETE]") == 1
        parsed = parse_output(path)
        assert float(parsed["forces"]["max_force"]) == pytest.approx(
            np.linalg.norm(final_force, axis=1).max(), abs=1e-8)
        footer = text.split("[RELAXATION COMPLETE]")[1]
        assert ("NOT converged" not in footer) == bool(optimizer.converged())
