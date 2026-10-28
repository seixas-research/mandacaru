"""Phase 3 comparisons and invariants for geometry-owned native calculations."""
import gc
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru, _native
from mandacaru.integrations.ase import geometry_from_atoms
from mandacaru.specifications import basis_specification
from mandacaru.basis import BasisSet
from mandacaru.core import MolecularIntegrals
from mandacaru.integrals import Grid
from mandacaru.integrals.poisson import PoissonFFTSolver, PeriodicPoissonSolver
from mandacaru.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV

HERE = Path(__file__).parent
CASES = json.loads((HERE / "cases.json").read_text())["cases"]
FROZEN = json.loads((HERE / "reference/numpy.json").read_text())


def h2(distance=.74):
    return Atoms("H2", positions=[[2.5, 2.5, 2.5-distance/2], [2.5, 2.5, 2.5+distance/2]], cell=[5]*3)


@pytest.mark.parametrize("case_id", ["h2_rhf", "h_uhf", "h2_ghf", "h2o_rhf_sto3g"])
def test_geometry_mean_field_matches_frozen_phase0(case_id):
    case = next(c for c in CASES if c["id"] == case_id)
    atoms = Atoms(case["symbols"], positions=case["positions"], cell=case["cell"])
    atoms.calc = Mandacaru(**case["options"],  trace=False)
    energy = atoms.get_potential_energy()
    expected = FROZEN["cases"][case_id]["samples"][0]["values"]["energy_ev"]
    assert energy == pytest.approx(expected, abs=1e-7)
    result = atoms.calc.result
    assert result.success
    assert result.energy_unit == "eV" and result.in_units("Ha") == pytest.approx(energy / HARTREE_TO_EV)
    assert result.snapshot()["n_iterations"] > 0
    reference = result.reference_state()
    assert np.vdot(reference, reference).real == pytest.approx(1.)
    if case_id != "h2o_rhf_sto3g":
        state = result.scf_state()
        op = result.qubit_hamiltonian().matrix()
        assert np.vdot(state, op @ state).real * HARTREE_TO_EV == pytest.approx(energy, abs=1e-8)


@pytest.mark.parametrize("basis", ["HAO", {"name": "HAO", "virtual_orbitals": 1},
                                   "STO-3G", "6-31G", "cc-pVDZ", "def2-SVP"])
@pytest.mark.parametrize("kinetic", ["spectral", "fd"])
def test_native_grid_integrals_and_lowdin_match_python(basis, kinetic):
    atoms = h2()
    geometry = geometry_from_atoms(atoms)
    ctx = _native.Context(threads=1)
    native = ctx.molecular_integrals(geometry, basis_spec=json.dumps(basis_specification(basis)),
                                    spacing_bohr=.45 * ANGSTROM_TO_BOHR, kinetic=kinetic)
    snapshot = native.snapshot(include_samples=True,include_hamiltonian=True)
    if isinstance(basis, dict):
        options = dict(basis)
        family = BasisSet.build(options.pop("name"), **options)
    else:
        family = BasisSet.build(basis)
    orbitals = [fn for atom in atoms for fn in family.atom(atom.symbol, center=atom.position)]
    grid = Grid(center=atoms.positions.mean(axis=0), cell=atoms.cell.array, grid_spacing=.45)
    py = MolecularIntegrals([(atom.number, atom.position) for atom in atoms], orbitals, grid,
                            softening=.5 * min(grid.dx, grid.dy, grid.dz), kinetic=kinetic)
    assert native.grid.shape == list(grid.shape)
    assert np.allclose(native.grid.step_bohr, grid.step, atol=1e-14)
    assert np.allclose(native.grid.origin_bohr, [grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0]], atol=1e-14)
    assert np.allclose(snapshot["samples"], py._engine._psi, atol=1e-13)
    assert np.allclose(snapshot["overlap"], py.overlap(), atol=2e-12)
    assert np.allclose(snapshot["lowdin"], py._lowdin_x(), atol=2e-10)
    t, v = py.kinetic_and_local()
    assert np.allclose(snapshot["kinetic"], t, atol=2e-11)
    assert np.allclose(snapshot["potential"], v, atol=2e-11)
    assert np.allclose(snapshot["one_body"], py.one_body(), atol=2e-10, rtol=1e-10)
    assert np.allclose(snapshot["two_body"], np.asarray(py.two_body()), atol=3e-10, rtol=1e-10)
    psi = snapshot["orthonormal_samples"]
    assert np.allclose(psi @ psi.T * native.grid.dv_bohr3, np.eye(len(orbitals)), atol=1e-10)
    eri = snapshot["two_body"]
    assert np.allclose(eri, eri.transpose(2, 1, 0, 3), atol=1e-13)
    assert np.allclose(eri, eri.transpose(1, 0, 3, 2), atol=1e-13)


@pytest.mark.parametrize("z,basis", [(1, "HAO"), (8, "HAO"), (26, "HAO"), (118, "HAO"),
                                      (8, "STO-6G"), (8, "6-31G(d)"), (8, "cc-pVTZ"),
                                      (11, "aug-cc-pVQZ"), (8, "def2-TZVP")])
def test_atomic_basis_sampling_matches_generated_python_families(z, basis):
    atoms = Atoms(numbers=[z], positions=[[.1, -.2, .3]])
    rng = np.random.default_rng(90)
    points = np.ascontiguousarray(rng.normal(size=(91, 3)) * 2)
    points[0] = atoms.positions[0] * ANGSTROM_TO_BOHR
    native = _native.Context().sample_basis(geometry_from_atoms(atoms), points,
                                          basis_spec=json.dumps(basis_specification(basis)))
    functions = BasisSet.build(basis).atom(z, center=atoms.positions[0])
    expected = np.stack([f.evaluate(*points.T) for f in functions])
    assert np.allclose(native, expected, atol=3e-12, rtol=1e-12)


def test_per_element_basis_matches_each_selected_family():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    spec = {"O": "STO-3G", "*": "HAO"}
    points = np.ascontiguousarray(np.random.default_rng(9).normal(size=(41, 3)))
    native = _native.Context().sample_basis(geometry_from_atoms(atoms), points,
                                          basis_spec=json.dumps(basis_specification(spec)))
    expected = np.stack([f.evaluate(*points.T) for atom in atoms
                         for f in BasisSet.build(spec).atom(atom.symbol, atom.position)])
    assert np.allclose(native, expected, atol=1e-13)


@pytest.mark.parametrize("shape", [(3, 4, 5), (6, 5, 4)])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("boundary", ["isolated", "periodic", "slab"])
def test_fft_poisson_matches_scipy_for_complex_densities(shape, skew, boundary):
    step = np.diag([.4, .5, .6])
    if skew:
        step[0, 1] = .15
        if boundary != "slab":
            step[0, 2] = -.12
    grid = _native.SpatialGrid(shape, [0., 0., 0.], step, periodic=boundary != "isolated")
    rho = np.random.default_rng(5).normal(size=np.prod(shape)) + .3j * np.random.default_rng(6).normal(size=np.prod(shape))
    actual = _native.Context(threads=1).poisson(grid, rho, periodic=boundary != "isolated",
                                               slab_axis=2 if boundary == "slab" else None)
    py = (PoissonFFTSolver(shape, step=step, workers=1) if boundary == "isolated" else
          PeriodicPoissonSolver(shape, step=step, workers=1,
                                truncation="slab" if boundary == "slab" else "none", axis=2 if boundary == "slab" else None))
    assert np.allclose(actual, py.solve(rho), atol=2e-12, rtol=2e-12)


def test_native_triclinic_grid_and_integrals_use_the_explicit_metric():
    atoms = h2()
    cell = np.array([[4., 0., 0.], [.7, 4., 0.], [.3, .2, 4.]])
    atoms.set_cell(cell)
    py_grid = Grid(center=atoms.positions.mean(axis=0), cell=cell, skew=True, grid_spacing=.55)
    native_grid = _native.SpatialGrid.from_geometry(geometry_from_atoms(atoms), [.55 * ANGSTROM_TO_BOHR]*3, skew=True)
    assert native_grid.shape == list(py_grid.shape)
    assert np.allclose(native_grid.step_bohr, py_grid.step, atol=2e-14)
    calc = Mandacaru(method="rhf",  grid=native_grid, trace=False)
    atoms.calc = calc
    actual = atoms.get_potential_energy()
    py = atoms.copy()
    py.calc = Mandacaru(method="rhf", grid=py_grid, trace=False)
    assert actual == pytest.approx(py.get_potential_energy(), abs=2e-8)


@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"])
def test_geometry_quantum_solve_and_native_reference_reuse(method, mapping):
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf",  grid_spacing=.35, trace=False)
    atoms.get_potential_energy()
    reference = atoms.calc.result
    # Uses native MO integrals without rebuilding a geometry or mapping in Python.
    calc = Mandacaru(method=method, mapping=mapping, trace=False, native_model=reference.hamiltonian)
    result = calc.run()
    assert (result.success if method == "vqe" else result.converged)
    assert result.optimal_energy == pytest.approx(-25.871895455446612, abs=1e-9)
    work = h2()
    work.calc = Mandacaru(method=method,  mapping=mapping, grid_spacing=.35, trace=False)
    assert work.get_potential_energy() == pytest.approx(result.optimal_energy, abs=1e-9)


def test_uhf_stretched_bond_breaks_spin_symmetry_and_exports_the_actual_determinant():
    atoms=h2(2.5)
    atoms.calc=Mandacaru(method="uhf",grid_spacing=.35)
    atoms.get_potential_energy()
    native=atoms.calc.result
    occupations=np.asarray(native.snapshot()["natural_occupations"])
    assert np.any((occupations>.01)&(occupations<1.99))
    state = native.scf_state()
    assert np.vdot(state, state).real == pytest.approx(1., abs=1e-10)
    energy = np.vdot(state, native.qubit_hamiltonian().matrix() @ state).real * HARTREE_TO_EV
    assert energy == pytest.approx(native.optimal_energy, abs=2e-8)
    assert native.reference_energy >= native.optimal_energy - 1e-8


def test_packed_cholesky_and_frozen_active_models_preserve_reference_energy():
    ctx = _native.Context(threads=1)
    geometry = geometry_from_atoms(h2())
    packed = ctx.molecular_integrals(geometry, spacing_bohr=.35 * ANGSTROM_TO_BOHR)
    chol = ctx.molecular_integrals(geometry, spacing_bohr=.35 * ANGSTROM_TO_BOHR, cholesky_tolerance=1e-11)
    assert np.allclose(chol.snapshot(include_hamiltonian=True)["two_body"], packed.snapshot(include_hamiltonian=True)["two_body"], atol=1e-11)
    assert chol.snapshot()["cholesky_rank"] <= 3
    case = next(c for c in CASES if c["id"] == "h2o_rhf_sto3g")
    atoms = Atoms(case["symbols"], positions=case["positions"], cell=case["cell"])
    atoms.calc = Mandacaru(**case["options"],  active_space={"frozen": "auto", "orbitals": 4}, trace=False)
    energy = atoms.get_potential_energy()
    result = atoms.calc.result
    model=result.hamiltonian
    assert model.n_spatial_orbitals == 4 and tuple(model.num_particles) == (4, 4)
    problem = model.problem()
    assert problem.dimension == 1
    assert problem.evaluate([], indices=[])["energy_ha"] * HARTREE_TO_EV == pytest.approx(energy, abs=1e-8)


@pytest.mark.parametrize("method", ["rhf", "uhf", "ghf"])
def test_cholesky_fock_and_mo_transforms_match_packed_water(method):
    case = next(c for c in CASES if c["id"] == "h2o_rhf_sto3g")
    atoms = Atoms(case["symbols"], positions=case["positions"], cell=case["cell"])
    ctx = _native.Context(threads=1)
    options = dict(basis_spec=json.dumps(basis_specification("STO-3G")), spacing_bohr=.4 * ANGSTROM_TO_BOHR)
    packed = ctx.molecular_integrals(geometry_from_atoms(atoms), **options)
    chol = ctx.molecular_integrals(geometry_from_atoms(atoms), cholesky_tolerance=1e-11, **options)
    expected = packed.mean_field(method, [5, 5]).snapshot(include_hamiltonian=True)
    actual = chol.mean_field(method, [5, 5]).snapshot(include_hamiltonian=True)
    assert actual["converged"] and actual["electronic_energy"] == pytest.approx(expected["electronic_energy"], abs=1e-8)
    c = actual["model_orbitals"]
    eri = packed.snapshot(include_hamiltonian=True)["two_body"]
    if method == "ghf":
        from mandacaru.algorithms.hartree_fock import spinor_integrals
        h = packed.snapshot()["one_body"]
        spin_h = np.kron(np.eye(2), h)
        h_mo, eri_mo = spinor_integrals(spin_h, eri, c)
    else:
        h_mo = c.conj().T @ packed.snapshot()["one_body"] @ c
        eri_mo = np.einsum("ap,bq,cr,ds,abcd->pqrs", c.conj(), c.conj(), c, c, eri, optimize=True)
    assert np.allclose(actual["h_mo"], h_mo, atol=2e-10, rtol=1e-11)
    assert np.allclose(actual["eri_mo"], eri_mo, atol=2e-9, rtol=1e-10)


def test_native_json_export_reloads_in_phase2_and_geometry_roundtrip_is_fresh(tmp_path):
    path = tmp_path / "model.json"
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf",  grid_spacing=.35, trace=False, save_hamiltonian=path)
    initial = atoms.get_potential_energy()
    calc = Mandacaru(method="vqe",  load_hamiltonian=path, trace=False)
    assert calc.run().optimal_energy == pytest.approx(-25.871895455446612, abs=1e-9)
    atoms.positions[1, 2] += .2
    changed = atoms.get_potential_energy()
    assert abs(changed - initial) > .01
    atoms.positions[1, 2] -= .2
    assert atoms.get_potential_energy() == pytest.approx(initial, abs=1e-10)


def test_native_pipeline_avoids_python_scientific_implementations(monkeypatch):
    from mandacaru.algorithms import _hamiltonian_from_atoms
    from mandacaru.algorithms.hartree_fock import RHF, UHF, GHF
    from mandacaru.core.mapping import Fermion
    from mandacaru.integrals.engine import IntegralEngine
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical implementation entered the native pipeline")
    monkeypatch.setattr(_hamiltonian_from_atoms, "build_basis_hamiltonian", forbidden)
    monkeypatch.setattr(BasisSet, "build", forbidden)
    monkeypatch.setattr(MolecularIntegrals, "__init__", forbidden)
    monkeypatch.setattr(IntegralEngine, "one_body", forbidden)
    monkeypatch.setattr(IntegralEngine, "two_body", forbidden)
    monkeypatch.setattr(RHF, "run", forbidden)
    monkeypatch.setattr(UHF, "solve", forbidden)
    monkeypatch.setattr(GHF, "solve", forbidden)
    monkeypatch.setattr(Fermion, "map_to_qubits", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    for method in ("rhf", "uhf", "ghf", "vqe", "adapt-vqe"):
        atoms = h2()
        atoms.calc = Mandacaru(method=method,  grid_spacing=.35, trace=False)
        assert atoms.get_potential_energy() < -25.


def test_failed_scf_invalid_inputs_and_budget_cleanup_are_explicit():
    ctx = _native.Context(memory_limit_bytes=1024)
    with pytest.raises(MemoryError):
        ctx.molecular_integrals(geometry_from_atoms(h2()))
    assert ctx.memory_used_bytes == 0
    ctx = _native.Context(threads=1)
    integrals = ctx.molecular_integrals(geometry_from_atoms(h2()), spacing_bohr=.35 * ANGSTROM_TO_BOHR)
    live = ctx.memory_used_bytes
    with pytest.raises(ValueError):
        integrals.mean_field("rhf", [1, 0])
    assert ctx.memory_used_bytes == live
    failed = integrals.mean_field("rhf", [1, 1], max_iterations=1).snapshot()
    assert not failed["converged"] and failed["n_iterations"] == 1
    reference = integrals.mean_field("uhf", [1, 1])
    model = reference.model
    with pytest.raises(ValueError):
        model.reduce([1], [0])
    copy = integrals.snapshot()["one_body"]
    copy[:] = 0
    assert np.linalg.norm(integrals.snapshot()["one_body"]) > .1
    del reference, model, integrals
    gc.collect()
    assert ctx.memory_used_bytes == 0
    work = h2()
    work.calc = Mandacaru(method="vqe",  grid_spacing=.35, trace=False,
                         scf_options={"max_iterations": 1})
    with pytest.raises(RuntimeError, match="did not converge"):
        work.get_potential_energy()
    for options in ({"kinetic": "other"}, {"scf_options": {"tolerance": -1}}, {"active_space": {"method": "mp2", "orbitals": 2}}):
        with pytest.raises((ValueError, NotImplementedError)):
            calc=Mandacaru(method="rhf", **options)
            reference=calc.run(h2())
            if "active_space" in options:reference.hamiltonian
    atoms = h2()
    atoms.calc = Mandacaru(method="ghf",  trace=False)
    with pytest.raises(NotImplementedError, match="forces"):
        atoms.get_forces()
