"""Phase 4 scientific comparisons: native XC, radial atoms, DFT and gradients."""
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
from mandacaru.integrals import Grid
from mandacaru.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV


def h2():
    return Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3)


def native_grid(grid):
    return _native.SpatialGrid(grid.shape, [grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0]], grid.step)


@pytest.mark.parametrize("functional", ["lda", "pbe"])
@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("spacing", [.35, .4])
def test_native_xc_matches_python_and_is_the_discrete_energy_derivative(functional, spin, spacing):
    from mandacaru.integrals.exchange_correlation import evaluate, evaluate_spin
    grid = Grid(center=[0, 0, 0], cell=np.diag([2, 2.4, 2.8]), grid_spacing=spacing)
    rng = np.random.default_rng(55)
    rho = rng.uniform(.005, .2, size=(2, grid.size))
    if not spin:
        rho[1] = rho[0]
    context = _native.Context(threads=1)
    result = context.exchange_correlation(native_grid(grid), rho, functional=functional)
    expected = (evaluate_spin(grid, *rho, functional) if spin else
                evaluate(grid, rho.sum(axis=0), functional))
    assert result["energy"] == pytest.approx(expected.energy, abs=1e-11)
    potential = (np.stack([expected.potential_up, expected.potential_dn]) if spin else
                 np.stack([expected.potential] * 2))
    assert np.allclose(result["potentials"], potential, atol=2e-10, rtol=1e-10)
    delta = rng.normal(size=rho.shape)
    step = 1e-6
    plus = context.exchange_correlation(native_grid(grid), rho + step * delta, functional=functional)["energy"]
    minus = context.exchange_correlation(native_grid(grid), rho - step * delta, functional=functional)["energy"]
    assert (plus - minus) / (2 * step) == pytest.approx(np.sum(result["potentials"] * delta) * grid.dV, abs=2e-8)
    assert context.memory_used_bytes == 0


@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_empty_and_fully_polarized_xc_are_finite(functional):
    from mandacaru.integrals.exchange_correlation import evaluate_spin
    grid = Grid(center=[0, 0, 0], cell=np.eye(3)*2, grid_spacing=.4)
    rho = np.zeros((2, grid.size))
    ctx = _native.Context()
    zero = ctx.exchange_correlation(native_grid(grid), rho, functional=functional)
    assert zero["energy"] == 0 and np.all(zero["potentials"] == 0)
    rho[0] = np.linspace(1e-14, .5, grid.size)
    actual = ctx.exchange_correlation(native_grid(grid), rho, functional=functional)
    expected = evaluate_spin(grid, *rho, functional)
    assert actual["energy"] == pytest.approx(expected.energy, abs=1e-11)
    assert np.allclose(actual["potentials"], [expected.potential_up, expected.potential_dn], atol=3e-9)


def test_dft_lda_energy_and_forces_match_frozen_phase0_and_round_trip():
    path = Path(__file__).parent / "reference/numpy.json"
    reference = json.loads(path.read_text())["cases"]["h2_dft_lda_forces"]["samples"][0]["values"]
    atoms = h2()
    atoms.calc = Mandacaru(method="dft",  xc="lda", grid_spacing=.35,
                           project_translation=False, trace=False)
    assert atoms.get_potential_energy() == pytest.approx(reference["energy_ev"], abs=1e-8)
    solver = atoms.calc.native_reference
    forces = atoms.get_forces()
    assert atoms.calc.native_reference is solver  # the energy's owned SCF is reused
    assert np.allclose(forces, reference["forces_ev_per_angstrom"], atol=2e-6)
    gradient=solver.gradient()
    assert np.allclose(gradient["gradient"],gradient["hellmann_feynman"]+gradient["pulay"])
    atoms.positions[1, 2] += .02
    assert atoms.get_potential_energy() != pytest.approx(reference["energy_ev"], abs=1e-4)
    atoms.positions[1, 2] -= .02
    assert atoms.get_potential_energy() == pytest.approx(reference["energy_ev"], abs=1e-8)


@pytest.mark.parametrize("method,xc,unrestricted", [("rhf", None, False), ("uhf", None, False),
                          ("dft", "lda", False), ("dft", "pbe", False), ("dft", "pbe", True)])
@pytest.mark.parametrize("basis", ["HAO", "STO-3G", {"name": "NAO", "size": "DZ", "r_c": 6., "n_grid": 400},
                                   {"name": "NAO-AE", "tier": 1, "onset": 1.5, "width": .8, "points": 500, "r_max": 12.}])
def test_stationary_forces_match_independent_fixed_grid_scf_differences(method, xc, unrestricted, basis):
    atoms = h2()
    # Asymmetric position on a common grid exercises transverse derivatives too.
    atoms.positions[0, 0] += .07
    grid = Grid(center=[2.5] * 3, cell=np.eye(3)*5, grid_spacing=.4)
    options = dict(method=method,  basis=basis, grid=grid, grid_spacing=.4,
                   project_translation=False, trace=False, scf_options={"tolerance": 1e-11})
    if xc:
        options.update(xc=xc, unrestricted=unrestricted)
    atoms.calc = Mandacaru(**options)
    forces = atoms.get_forces()
    for atom, axis in [(0, 0), (1, 2)]:
        energies = []
        step = 2e-5
        for sign in [-1, 1]:
            moved = atoms.copy()
            moved.positions[atom, axis] += sign * step
            moved.calc = Mandacaru(**options)
            energies.append(moved.get_potential_energy())
        fd = -(energies[1] - energies[0]) / (2 * step)
        assert forces[atom, axis] == pytest.approx(fd, abs=2e-5)


@pytest.mark.parametrize("symbols,method,xc,unrestricted", [
    ("H2", "rhf", None, False), ("H2", "uhf", None, False), ("H2", "dft", "lda", False),
    ("H2", "dft", "pbe", False), ("H2", "dft", "pbe", True),
    # Lithium carries a closed 1s core: its frozen-core exchange is part of the force.
    ("LiH", "rhf", None, False), ("LiH", "dft", "pbe", False)])
def test_paw_stationary_forces_match_independent_fixed_grid_scf_differences(symbols, method, xc, unrestricted):
    """PAW-LCAO mean-field forces in eV/Angstrom against converged energies in eV.

    The native gradient is a fixed-density Richardson difference in Hartree per
    Bohr; the conversion to the ASE unit happens once, in ``Result.forces``, so
    agreement with a finite difference of ``get_potential_energy()`` over
    Angstrom displacements checks the unit as well as the derivative.
    """
    atoms = h2() if symbols == "H2" else Atoms("LiH", positions=[[2.5, 2.5, 1.85], [2.6, 2.5, 3.45]], cell=[5] * 3)
    atoms.positions[0, 0] += .07  # off the bond axis, so a transverse force exists
    grid = Grid(center=[2.5] * 3, cell=np.eye(3) * 5, grid_spacing=.4)
    options = dict(method=method, basis={"name": "PAW-LCAO", "size": "SZ"}, grid=grid, grid_spacing=.4,
                   project_translation=False, trace=False, scf_options={"tolerance": 1e-11})
    if xc:
        options.update(xc=xc, unrestricted=unrestricted)
    atoms.calc = Mandacaru(**options)
    assert "forces" in atoms.calc.implemented_properties
    forces = atoms.get_forces()
    details = atoms.calc.result.force_details
    assert details["scheme"] == "Richardson-fixed-density-PAW"
    assert details["step_bohr"] == 1e-3 and details["step_error_estimate"] < 1e-5
    for atom, axis in [(0, 0), (1, 2)]:
        energies = []
        step = 2e-5
        for sign in [-1, 1]:
            moved = atoms.copy()
            moved.positions[atom, axis] += sign * step
            moved.calc = Mandacaru(**options)
            energies.append(moved.get_potential_energy())
        fd = -(energies[1] - energies[0]) / (2 * step)
        assert forces[atom, axis] == pytest.approx(fd, abs=2e-5)


@pytest.mark.parametrize("functional", ["lda", "pbe"])
@pytest.mark.parametrize("system", ["h2", "h", "water"])
def test_kohn_sham_energy_density_and_spectrum_match_python(functional, system):
    atoms = h2() if system == "h2" else (Atoms("H", positions=[[2.43, 2.51, 2.52]], cell=[5]*3) if system == "h" else
            Atoms("OH2", positions=[[2.5, 2.5, 2.5], [3.25, 2.5, 3.1], [1.75, 2.5, 3.1]], cell=[5]*3))
    if system == "h":
        atoms.set_initial_magnetic_moments([1.])
    options = dict(method="dft", xc=functional, basis="STO-3G", grid_spacing=.4,
                   trace=False)
    py = atoms.copy()
    py.calc = Mandacaru(**options)
    expected = py.get_potential_energy()
    atoms.calc = Mandacaru(**options)
    actual = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success and result.method == "dft"
    assert actual == pytest.approx(expected, abs=2e-7)
    assert result.snapshot()["hartree_energy"] == pytest.approx(py.calc.result.snapshot()["hartree_energy"], abs=2e-7)
    assert result.snapshot()["xc_energy"] == pytest.approx(py.calc.result.snapshot()["xc_energy"], abs=2e-7)
    assert result.snapshot()["functional"] == functional and not atoms.calc.native_reference.has_model
    rdms = atoms.calc.native_reference.rdm()
    assert np.trace(rdms).real == pytest.approx(len(atoms) if system != "water" else 10.)
    if system != "h":
        assert np.allclose(np.asarray(result.snapshot()["energies"]), np.asarray(py.calc.result.snapshot()["energies"]), atol=2e-7)
    if system == "h2":
        state = result.scf_state()
        expectation = np.vdot(state, result.qubit_hamiltonian().matrix() @ state).real * HARTREE_TO_EV
        assert np.isfinite(expectation)
        assert abs(expectation - actual) > .01
        quantum = Mandacaru(method="vqe", native_model=result.hamiltonian, trace=False).run()
        assert quantum.success and quantum.optimal_energy <= expectation + 1e-8


@pytest.mark.parametrize("size,scheme", [("SZ", None), ("DZ", None), ("DZP", None),
                  ("TZ2P", "last_zeta"), ("QZ", [.2, .1, .05])])
@pytest.mark.parametrize("z", [1, 8, 26])
def test_numerical_atomic_orbitals_match_python_splines_and_shell_order(z, size, scheme):
    atoms = Atoms(numbers=[z], positions=[[0, 0, 0]])
    options = dict(size=size, r_c=7., n_grid=300, zeta_split=scheme)
    spec = dict(name="NAO", **options)
    points = np.ascontiguousarray(np.random.default_rng(55).normal(size=(90, 3))*1.5)
    points[:4] = [[0, 0, 0], [0, 0, 1], [0, 0, -2], [0, 0, 8]]
    # Three more points 1e-4 Bohr from the nucleus along each axis: the
    # direction is defined there, so the near-nucleus behavior of every shell
    # is compared at the full tolerance.
    points = np.ascontiguousarray(np.vstack([points, 1e-4*np.eye(3)]))
    actual = _native.Context(threads=1).sample_basis(geometry_from_atoms(atoms), points,
                           basis_spec=json.dumps(basis_specification(spec)))
    functions = BasisSet.build("NAO", **options).atom(z)
    expected = np.stack([f.evaluate(*points.T) for f in functions])
    assert actual.shape == expected.shape
    # Point 0 is the nucleus, where the direction of a shell with l > 0 is
    # undefined.  The tabulated radial function starts at r = 0 with a small
    # nonzero R(0) for those shells (1.6e-4 for the hydrogen polarization p,
    # 2e-2 for the oxygen p), so the value there is R(0) times a harmonic of an
    # arbitrary direction: the Python sampler used theta = pi/2, phi = 0, the
    # native one the zero vector (p and m != 0 shells give 0, d_z2 keeps its
    # -0.315 R(0)).  Only l = 0 is well defined there and is compared at the
    # full tolerance; for l > 0 the sample must stay within the harmonic's
    # bound times R(0).
    for i, f in enumerate(functions):
        if f.l > 0:
            bound = abs(float(f.radial(np.zeros(1))[0])) * np.sqrt(2*(2*f.l + 1)/(4*np.pi))
            assert abs(actual[i, 0]) <= bound + 1e-15
            expected[i, 0] = actual[i, 0]
    assert np.allclose(actual, expected, atol=2e-10, rtol=2e-9)
    assert np.all(actual[:, 3] == 0.)


@pytest.mark.parametrize("n,l,z", [(1, 0, 1.), (2, 0, 2.), (2, 1, 1.), (3, 2, 4.)])
def test_native_confined_radial_equation_matches_python_and_hydrogen_limit(n, l, z):
    from mandacaru.basis.nao import solve_confined_radial
    expected_r, expected, energy = solve_confined_radial(n, l, z, 25., 800)
    context = _native.Context(threads=1)
    actual = context.confined_radial(n, l, z, 25., intervals=800)
    assert np.allclose(actual["r"], expected_r, atol=1e-14)
    assert np.allclose(actual["radial"], expected, atol=2e-11)
    assert actual["energy"] == pytest.approx(energy, abs=1e-11)
    if n == 1:
        assert actual["energy"] == pytest.approx(-z*z/(2*n*n), abs=.0002)
    assert context.memory_used_bytes == 0


@pytest.mark.parametrize("z", [1, 2, 8, 18])
def test_native_spherical_lda_atom_matches_python_fixed_point(z):
    from mandacaru.basis.atomic_solver import solve_atom
    options = dict(points=600, r_max=18., tolerance=1e-9)
    expected = solve_atom(z, **options)
    actual = solve_atom(z,  resources={"threads": 1}, **options)
    assert actual.converged and actual.details["engine"] == "rust"
    assert actual.total_energy == pytest.approx(expected.total_energy, abs=2e-7)
    assert np.allclose(actual.density, expected.density, atol=1e-8, rtol=1e-7)
    charge = np.trapezoid(4*np.pi*actual.r**2*actual.density, actual.r)
    assert charge == pytest.approx(z, abs=1e-7)
    for key in actual.occupations:
        assert actual.eigenvalues[key] == pytest.approx(expected.eigenvalues[key], abs=2e-8)
        assert np.trapezoid(actual.orbitals[key]**2, actual.r) == pytest.approx(1.)


def test_native_radial_atom_confinement_and_nonconvergence_are_honest():
    from mandacaru.basis.atomic_solver import solve_atom
    r = np.arange(1, 401)*12/401
    options = dict(points=400, r_max=12., confinement=np.maximum(r-5, 0.)**2)
    expected = solve_atom(2, **options)
    actual = solve_atom(2,  **options)
    assert actual.total_energy == pytest.approx(expected.total_energy, abs=1e-6)
    failed = solve_atom(8,  max_iterations=1, points=100)
    assert not failed.converged and failed.iterations == 1
    # Scalar-relativistic and PBE atoms are solved natively since Wave 1a
    # (test/basis/test_atomic_solver.py holds their frozen parity); what is not
    # supported raises instead of selecting another calculation.
    for options, error in (({"xc": "tpss"}, ValueError),
                           ({"relativity": "zora"}, ValueError),
                           ({"grid": "log"}, TypeError)):
        with pytest.raises(error):
            solve_atom(1,  **options)


def test_phase4_pipeline_does_not_call_python_numerical_implementations(monkeypatch):
    from mandacaru.basis import atomic_solver, nao, nao_ae
    from mandacaru.integrals import exchange_correlation
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical callback entered native Phase 4")
    # The Python Kohn-Sham loop, Poisson solver, force module and Hamiltonian
    # builder are deleted (mandacaru.algorithms, integrals.poisson no longer
    # exist), so absence is their guard; every Python numerical entry point
    # that survives is poisoned, and a missing name fails the patch loudly.
    for cls, name in [(BasisSet, "build"), (np.linalg, "eigh"), (np.linalg, "solve"),
                      (atomic_solver, "solve_radial"), (nao, "solve_confined_radial"), (nao_ae, "build_species"),
                      (exchange_correlation, "evaluate"), (exchange_correlation, "evaluate_spin")]:
        monkeypatch.setattr(cls, name, forbidden)
    for functional in ["lda", "pbe"]:
        atoms = h2()
        atoms.calc = Mandacaru(method="dft",  xc=functional, trace=False,
                               basis={"name": "NAO", "size": "DZ", "r_c": 6., "n_grid": 400}, grid_spacing=.35)
        assert np.isfinite(atoms.get_potential_energy()) and np.all(np.isfinite(atoms.get_forces()))
    atoms = h2()
    atoms.calc = Mandacaru(method="dft",  xc="pbe", trace=False,
                           basis={"name": "NAO-AE", "tier": 1, "onset": 1.5, "width": .8,
                                  "points": 500, "r_max": 12.}, grid_spacing=.35)
    assert np.all(np.isfinite(atoms.get_forces()))
    assert atomic_solver.solve_atom(2,  points=300).converged


def test_dft_force_failures_and_budget_lifetimes_are_explicit():
    atoms = h2()
    ctx = _native.Context(threads=1)
    integrals = ctx.molecular_integrals(geometry_from_atoms(atoms), spacing_bohr=.35*ANGSTROM_TO_BOHR)
    live = ctx.memory_used_bytes
    with pytest.raises(ValueError, match="occupation"):
        integrals.kohn_sham("pbe", [1, 0])
    with pytest.raises(ValueError, match="XC"):
        integrals.kohn_sham("unsupported-xc", [1, 1])
    assert ctx.memory_used_bytes == live
    failed = integrals.kohn_sham("lda", [1, 1], max_iterations=1)
    with pytest.raises(ValueError, match="converged"):
        failed.gradient()
    del failed
    reference = integrals.kohn_sham("pbe", [1, 1])
    owned = ctx.memory_used_bytes
    reference.gradient()
    assert ctx.memory_used_bytes == owned
    snapshot = reference.snapshot()
    snapshot["coefficients"][0][:] = 0.
    assert np.linalg.norm(reference.snapshot()["coefficients"][0]) > 1
    del reference, integrals
    gc.collect()
    assert ctx.memory_used_bytes == 0
    tiny = _native.Context(memory_limit_bytes=100)
    with pytest.raises(MemoryError):
        tiny.solve_atom(1)
    assert tiny.memory_used_bytes == 0
    # "vqe" left this list: variational molecular forces are now supported
    # (specifications.CAPABILITIES["vqe"].forces); subspace-vqe has none.
    for options in ({"method": "ghf"}, {"method": "rhf", "kinetic": "fd"},
                    {"method": "subspace-vqe"}):
        atoms = h2()
        atoms.calc = Mandacaru( grid_spacing=.35, trace=False, **options)
        with pytest.raises(NotImplementedError):
            atoms.get_forces()


@pytest.mark.parametrize("tier", [0, 1, 2])
@pytest.mark.parametrize("z", [1, 8])
def test_all_electron_nao_tiers_match_python_species_and_irregular_wall_spline(tier, z):
    options = dict(tier=tier, onset=1.5, width=.8, points=500, r_max=12.)
    spec = dict(name="NAO-AE", **options)
    atom = Atoms(numbers=[z], positions=[[0., 0., 0.]])
    wall = (options["onset"] + options["width"]) * ANGSTROM_TO_BOHR
    points = np.ascontiguousarray(np.random.default_rng(27).normal(size=(91, 3)))
    points[:5] = [[0, 0, 0], [0, 0, .7], [0, 0, -1.3], [0, 0, wall-.003], [0, 0, wall+.001]]
    actual = _native.Context(threads=1).sample_basis(geometry_from_atoms(atom), points,
                           basis_spec=json.dumps(basis_specification(spec)))
    expected = np.stack([f.evaluate(*points.T) for f in BasisSet.build("NAO-AE", **options).atom(z)])
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=8e-7, rtol=3e-6)
    assert np.all(actual[:, 4] == 0.)


def test_all_electron_nao_custom_functions_and_failed_species_are_explicit():
    options = dict(tier=0, onset=1.5, width=.8, points=400, r_max=12.,
                   extra=[(2, 1, 1.2), (2, 0, .6)], tail_norm=None)
    atom = Atoms("He", positions=[[0., 0., 0.]])
    points = np.ascontiguousarray(np.random.default_rng(28).normal(size=(41, 3)))
    spec = basis_specification(dict(name="NAO-AE", **options))
    actual = _native.Context().sample_basis(geometry_from_atoms(atom), points, basis_spec=json.dumps(spec))
    expected = np.stack([f.evaluate(*points.T) for f in BasisSet.build("NAO-AE", **options).atom(2)])
    assert np.allclose(actual, expected, atol=1e-6, rtol=3e-6)
    ctx = _native.Context()
    for invalid in (dict(spec, tier=3), dict(spec, r_max=1.), dict(spec, width=0.)):
        with pytest.raises(ValueError):
            ctx.sample_basis(geometry_from_atoms(atom), points, basis_spec=json.dumps(invalid))
        assert ctx.memory_used_bytes == 0
