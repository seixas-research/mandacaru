"""PAW ownership and variational forces checked by independent total energies."""
from contextlib import nullcontext

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
        assert result.force_details["orbital_response"] == "CPHF-and-natural-orbital-Lagrangian"
        assert result.force_details["scheme"] == "analytic"
        assert result.force_details["directions"] == 1
        assert result.force_details["energy_ha"] == pytest.approx(result.energy_hartree, abs=1e-10)
        internal = result.forces(internal=True)
        result.force_details["gradient"].fill(123.)
        assert np.allclose(result.forces(internal=True), internal)
    monkeypatch.undo()
    step = 1e-4
    energies = []
    for offset in (step, -step):
        moved = lih()
        moved.positions[0, 2] -= offset / 2
        moved.positions[1, 2] += offset / 2
        with Mandacaru(method="adapt-vqe", atoms=moved, basis="PAW-LCAO", grid=grid,
                       grid_spacing=.35, resources={"threads": 1}):
            energies.append(moved.get_potential_energy())
    assert force[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)
    assert internal[1, 2] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=1e-4)


def test_an_all_electron_variational_force_raises_by_name():
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="HAO", grid_spacing=.5,
                   project_translation=False, resources={"threads": 1}):
        with pytest.raises(NotImplementedError, match="PAW-LCAO only"):
            atoms.get_forces()


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
                    problem, model, np.ascontiguousarray(vectors[:, 0]), internal=True)
                assert raw["energy_ha"] == pytest.approx(energies[0], abs=1e-10)
                assert raw["stationarity"] == "assumed"
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
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="PAW-LCAO", grid_spacing=.4,
                   active_space={"orbitals": 2}, txt=path, resources={"threads": 1}) as calc:
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


def _bare_gradient(calc):
    """The state's gradient with no ansatz: its density matrices contracted as an exact optimum's."""
    result = calc.result
    raw = calc.native_reference.quantum_gradient(
        result._problem, result._model, np.ascontiguousarray(result._raw["state"]), internal=True)
    assert raw["stationarity"] == "assumed"
    return raw["gradient"]


def _h2_force(optimizer, grid=None, method="vqe"):
    from mandacaru.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    with Mandacaru(method=method, atoms=atoms, basis="PAW-LCAO", grid=grid, grid_spacing=.4,
                   active_space={"orbitals": 2}, optimizer=optimizer, resources={"threads": 1}) as calc:
        calc.run()
        result = calc.result
        force = result.forces(internal=True)
        details = dict(result.force_details)
        bare = -_bare_gradient(calc) * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        return force[1, 2], bare[1, 2], details, calc.integrals.grid


def test_lagrangian_correction_makes_a_loose_vqe_force_second_order_in_the_residual_gradient():
    tight_force, tight_bare, tight, grid = _h2_force({"method": "BFGS", "maxiter": 500, "tol": 1e-12})
    # A converged state is stationary: nothing is solved and the force is the bare one.
    assert tight["stationarity"] == "stationary"
    assert tight["parameter_gradient_norm"] < 1e-10
    assert abs(tight_force - tight_bare) < 1e-9
    rows = []
    for maxiter in (1, 2):
        force, bare, details, _ = _h2_force({"method": "BFGS", "maxiter": maxiter, "tol": 1e-3}, grid)
        assert details["stationarity"] == "lagrangian" and details["z_vector_dropped"] == 0
        assert details["hessian_lowest_curvature"] > details["hessian_cutoff"] > 0
        assert details["hessian_parameters"] == 3 and details["z_vector_gradient_evaluations"] == 6
        assert details["hessian_exact"] is False and details["hessian_hamiltonian_actions"] == 6
        assert details["z_vector_neglected_scale"] == 0
        assert abs(force - bare) > 0
        rows.append((details["parameter_gradient_norm"], abs(bare - tight_force), abs(force - tight_force)))
    (g_loose, plain_loose, corrected_loose), (g_near, plain_near, corrected_near) = rows
    assert g_loose > 100 * g_near
    # The bare contraction misses by O(|g|), the corrected one by O(|g|^2).
    assert plain_near == pytest.approx(plain_loose * g_near / g_loose, rel=0.3)
    assert corrected_near == pytest.approx(corrected_loose * (g_near / g_loose) ** 2, rel=1.)
    assert corrected_near < plain_near / 1000 and corrected_loose < plain_loose / 20


def test_lagrangian_correction_of_a_loose_adapt_lih_state_matches_the_converged_force():
    from mandacaru.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

    def run(optimizer, grid=None):
        atoms = lih()
        with Mandacaru(method="adapt-vqe", atoms=atoms, basis="PAW-LCAO", grid=grid, grid_spacing=.35,
                       optimizer=optimizer, max_iterations=3, convergence={"gradient": 1e-9},
                       resources={"threads": 1}) as calc:
            with pytest.warns(RuntimeWarning) if optimizer["tol"] > 1e-8 else nullcontext():
                calc.run()
            force = calc.result.forces(internal=True)
            details = dict(calc.result.force_details)
            bare = -_bare_gradient(calc) * HARTREE_TO_EV / BOHR_TO_ANGSTROM
            return force[1, 2], bare[1, 2], details, calc.integrals.grid

    tight_force, _, tight, grid = run({"method": "BFGS", "maxiter": 500, "tol": 1e-12})
    assert tight["stationarity"] == "stationary"
    force, bare, details, _ = run({"method": "L-BFGS", "maxiter": 3, "tol": 1e-4}, grid)
    assert details["stationarity"] == "lagrangian"
    assert details["parameter_gradient_norm"] > 1e-3
    assert abs(bare - tight_force) > 1e-3
    assert abs(force - tight_force) < abs(bare - tight_force) / 50
    # Second order in the distance to the optimum: |z| is the Newton step, |delta|.
    assert abs(force - tight_force) < 5 * details["z_vector_norm"] ** 2


def test_search_methods_report_their_stationarity_and_subspace_forces_are_refused_by_name():
    _, _, details, _ = _h2_force({"method": "BFGS", "maxiter": 200, "tol": 1e-10}, method="vasqa")
    assert details["stationarity"] in {"stationary", "lagrangian"}
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    with Mandacaru(method="subspace-vqe", atoms=atoms, basis="HAO", grid_spacing=.5,
                   resources={"threads": 1}) as calc:
        calc.run()
        with pytest.raises(ValueError, match="subspace method"):
            calc.result.forces()

