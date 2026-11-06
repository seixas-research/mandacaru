"""Native Phase 5 thermodynamics, crystal SCF, electrostatics and analysis."""
import copy
import gc
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru, _native
from mandacaru.units import HARTREE_TO_EV

HERE=Path(__file__).parent
SI=HERE/'phase5/si_crystal.json'


def complex_values(a):
    a=np.asarray(a,complex).ravel()
    return np.column_stack([a.real,a.imag]).tolist()


def synthetic(functional='lda',spins=1):
    shape=(4,4,4)
    n=np.prod(shape)
    x=np.indices(shape)[0].ravel()
    psi=np.stack([np.ones(n),np.exp(2j*np.pi*x/shape[0])])/np.sqrt(n)
    # ngrid=volume, so the samples are exactly orthonormal.
    freq=np.stack(np.meshgrid(*[np.fft.fftfreq(i)*2*np.pi for i in shape],indexing='ij'))
    g2=np.sum(freq*freq,axis=0).ravel()
    kernel=np.zeros(n)
    kernel[g2>0]=4*np.pi/g2[g2>0]
    xc_wave=freq.reshape(3,-1).copy()
    # Derivatives of real fields have zero Nyquist multiplier.
    for a in range(3):
        nyq=np.indices(shape)[a].ravel()==shape[a]//2
        xc_wave[a,nyq]=0.
    return dict(schema_version=1,representation='periodic-paw-snapshot',energy_unit='Ha',length_unit='bohr',
                grid=dict(shape=shape,origin=[0.]*3,step=np.eye(3).ravel().tolist()),
                m=2,p=0,channels=0,channel_labels=[],
                kpoints=[dict(k=[0.]*3,fractional=[0.]*3,weight=1.,psi=complex_values(psi),
                              overlap=complex_values(np.eye(2)),fixed=complex_values(np.diag([-.5,.3])),projections=[])],
                blocks=[],core=np.zeros(n).tolist(),initial=np.full(n,2/n).tolist(),initial_m=np.zeros(n).tolist(),initial_q=[],
                kernel=kernel.tolist(),xc_wave=xc_wave.ravel().tolist(),phase=complex_values(np.ones(n)),kerker=np.ones(n).tolist(),
                ion=complex_values(np.zeros(n)),hats=[],u=[],uion=[],linear=[],eion=0.,ion_constant=0.,constant=.13,
                reference=.02,electrons=2.,spins=spins,functional=functional,relativistic=False,permutations=[],qsym=[])


def smearing_occupation(x,method):
    """Occupation per spin state at x=(e-mu)/width, from the textbook closed forms."""
    from scipy.special import erfc,expit
    x=np.asarray(x,float)
    if method=='fermi-dirac':
        return expit(-x)
    if method=='gaussian':
        return .5*erfc(x)
    # Methfessel-Paxton, first order: 1/2 erfc(x) + A1 H1(x) exp(-x^2), A1=-1/(4 sqrt(pi)), H1=2x.
    return .5*erfc(x)-x*np.exp(-x*x)/(2*np.sqrt(np.pi))


def smearing_entropy(x,method):
    """Entropy term S(x) with F = E - width*sum(S): -f ln f - (1-f) ln(1-f) for
    Fermi-Dirac, (1/2) A_N H_2N(x) exp(-x^2) for the Hermite expansions."""
    from scipy.special import xlogy
    x=np.asarray(x,float)
    if method=='fermi-dirac':
        f=smearing_occupation(x,method)
        return -xlogy(f,f)-xlogy(1-f,1-f)
    if method=='gaussian':
        return np.exp(-x*x)/(2*np.sqrt(np.pi))
    return .5*(-1/(4*np.sqrt(np.pi)))*(4*x*x-2)*np.exp(-x*x)


def reference_fermi_level(levels,weights,electrons,method,width,degeneracy):
    """Root of the charge neutrality condition by Brent's method."""
    from scipy.optimize import brentq
    def excess(mu):
        return degeneracy*np.sum(smearing_occupation((levels-mu)/width,method)*weights[:,None])-electrons
    return brentq(excess,levels.min()-60*width,levels.max()+60*width,xtol=1e-15,rtol=1e-15,maxiter=500)


def _active_paw_snapshot(active, electrons, functional="lda", **options):
    from mandacaru.integrals.exchange_correlation import core_correction_offset
    from mandacaru.integrations.paw import crystal_input
    from mandacaru.models.datasets import _datasets_relativistic

    return crystal_input(active, electrons, functional,
        constant=sum(d.one_center_energy + core_correction_offset(d) for d in active.datasets),
        relativistic=_datasets_relativistic(active.datasets), **options)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_borrowed_native_kinetic_matches_numpy_for_skew_grid(dtype):
    from mandacaru.integrals import Grid, reciprocal as rc
    from mandacaru.pseudopotentials.periodic_paw import PeriodicPAW

    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, 1., 10.]])
    grid = Grid(center=[5., 4., 6.], box_size=0., grid_spacing=1., units="bohr",
                cell=cell, periodic=True)
    crystal = object.__new__(PeriodicPAW)
    crystal.grid = grid
    crystal.volume = rc.cell_volume(grid)
    rng = np.random.default_rng(792)
    psi = np.asarray(rng.normal(size=(4, grid.size)) + 1j * rng.normal(size=(4, grid.size)),
                     dtype=dtype, order="C")
    k = [0.13, -0.07, 0.11]
    phase = np.exp(-1j * (k[0] * grid.X + k[1] * grid.Y + k[2] * grid.Z)).ravel()
    transform = np.fft.fftn((psi * phase[None, :]).reshape(len(psi), *grid.shape),
                           axes=(1, 2, 3)).reshape(len(psi), -1) * grid.dV
    _, squares = rc.bloch_aliases(grid, k)
    expected = (transform.conj() * (0.5 * squares.ravel() / crystal.volume)) @ transform.T
    native_grid = _native.SpatialGrid(grid.shape, rc.grid_origin(grid),
                                     np.ascontiguousarray(grid.step), periodic=True)
    ctx = _native.Context(threads=1)
    method = ctx.periodic_kinetic_f32 if dtype == np.complex64 else ctx.periodic_kinetic
    actual = method(native_grid, k, psi)
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-10)
    crystal._native_context = ctx
    assert np.allclose(crystal._kinetic(psi, k), expected, rtol=2e-12, atol=2e-10)
    with pytest.raises(ValueError, match="C-contiguous"):
        method(native_grid, k, psi[:, ::-1])


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("k", [[0., 0., 0.], [0.13, -0.07, 0.11]])
def test_native_bloch_gradients_match_numpy_on_skew_grid(dtype, k):
    from mandacaru.integrals import Grid, reciprocal as rc
    from mandacaru.pseudopotentials.periodic_paw import PeriodicPAW

    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, 1., 10.]])
    grid = Grid(center=[5., 4., 6.], box_size=0., grid_spacing=1., units="bohr",
                cell=cell, periodic=True)
    crystal = object.__new__(PeriodicPAW)
    crystal.grid = grid
    crystal._native_context = _native.Context(threads=1)
    rng = np.random.default_rng(918)
    psi = np.asarray(rng.normal(size=(4, grid.size)) + 1j * rng.normal(size=(4, grid.size)),
                     dtype=dtype, order="C")
    phase = np.exp(1j * (k[0] * grid.X + k[1] * grid.Y + k[2] * grid.Z)).ravel()
    transform = np.fft.fftn((psi * phase.conj()[None, :]).reshape(len(psi), *grid.shape),
                            axes=(1, 2, 3))
    vectors, _ = rc.bloch_aliases(grid, k)
    expected = np.stack([np.fft.ifftn(transform * (1j * vectors[axis])[None],
                                       axes=(1, 2, 3)).reshape(len(psi), -1) * phase[None, :]
                         for axis in range(3)])
    actual = crystal.bloch_gradients(psi, k)
    assert actual.shape == (3, len(psi), grid.size)
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12)
    from types import SimpleNamespace
    crystal.M, crystal.symmetry = len(psi), None
    crystal.kpoint_data = [SimpleNamespace(psi=psi, k=k, weight=1.)]
    a = rng.normal(size=(len(psi), len(psi))) + 1j * rng.normal(size=(len(psi), len(psi)))
    density = a @ a.conj().T
    expected_tau = sum(0.5 * np.real(np.sum(g * (density @ g.conj()), axis=0)) for g in expected)
    assert np.allclose(crystal.kinetic_energy_density([density]), expected_tau, rtol=2e-12, atol=2e-12)


def test_orbitals_above_float64_precision_are_not_silently_narrowed():
    from mandacaru.pseudopotentials.periodic_paw import PeriodicPAW

    crystal = object.__new__(PeriodicPAW)
    if np.dtype(np.clongdouble).itemsize <= np.dtype(np.complex128).itemsize:
        pytest.skip("platform has no wider complex dtype")
    with pytest.raises(ValueError, match="above float64 precision"):
        crystal._orbital_samples(np.ones((1, 2), dtype=np.clongdouble))


def test_native_grid_overlap_matches_numpy_without_copying_samples():
    rng = np.random.default_rng(314)
    psi = np.ascontiguousarray(rng.normal(size=(5, 431)) + 1j * rng.normal(size=(5, 431)))
    ctx = _native.Context(threads=1)
    expected = (psi.conj() @ psi.T) * 0.17
    assert np.allclose(ctx.periodic_overlap(psi, 0.17), expected, rtol=2e-13, atol=2e-12)
    with pytest.raises(ValueError, match="C-contiguous"):
        ctx.periodic_overlap(psi[:, ::-1], 0.17)
    with pytest.raises(ValueError, match="volume"):
        ctx.periodic_overlap(psi, -1.)


@pytest.mark.parametrize("symbol", ["Si", "Fe"])
@pytest.mark.parametrize("center", [[0., 0., 0.], [2.2, 3.1, 4.4]])
def test_native_valence_field_matches_full_scipy_spline_and_images(symbol, center):
    import os
    from scipy.interpolate import CubicSpline
    from mandacaru.integrals import Grid, reciprocal as rc
    from mandacaru.pseudopotentials.periodic_paw import PeriodicPAW
    from mandacaru.pseudopotentials.record import read_dataset

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    dataset = read_dataset(Path(root) / "lda-sr" / f"{symbol}.parquet", warn=False)
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, 1., 10.]])
    grid = Grid(center=[5., 4., 6.], box_size=0., grid_spacing=1., units="bohr",
                cell=cell, periodic=True)
    crystal = object.__new__(PeriodicPAW)
    crystal.grid, crystal.lattice = grid, rc.lattice_vectors(grid)
    crystal._native_context = _native.Context(threads=1, memory_limit_bytes=1 << 28)
    center = np.asarray(center)
    values, radius = np.asarray(dataset.valence_density), np.asarray(dataset.r)
    support = radius[np.flatnonzero(np.abs(values) > 1e-12 * np.max(np.abs(values)))[-1]]
    spline = CubicSpline(radius, values)
    points = np.stack(crystal._grid_points(), axis=1)
    midpoint, bound = crystal._cell_region()
    expected = np.zeros(grid.size)
    for translation in rc.lattice_translations(crystal.lattice, support + 2. * bound):
        position = center + translation
        if np.linalg.norm(position - midpoint) <= support + bound:
            distance = np.linalg.norm(points - position, axis=1)
            inside = distance <= support
            expected[inside] += spline(distance[inside])
    actual = crystal._superposed_valence(dataset, center)
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-13)


def test_native_initial_density_and_magnetization_assemble_each_atom():
    import os
    from mandacaru.integrals import Grid, reciprocal as rc
    from mandacaru.pseudopotentials.periodic_paw import PeriodicPAW
    from mandacaru.pseudopotentials.record import read_dataset

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    folder = Path(root) / "lda-sr"
    datasets = [read_dataset(folder / f"{s}.parquet", warn=False) for s in ("Li", "H")]
    grid = Grid(center=[4., 4.5, 5.], box_size=0., grid_spacing=1., units="bohr",
                cell=np.diag([8., 9., 10.]), periodic=True)
    crystal = object.__new__(PeriodicPAW)
    crystal.grid, crystal.lattice = grid, rc.lattice_vectors(grid)
    crystal._native_context = _native.Context(threads=1, memory_limit_bytes=1 << 28)
    crystal.datasets, crystal.centers = datasets, [np.array([1., 2., 3.]), np.array([4., 5., 6.])]
    crystal.channels = [(0, 0, 0), (0, 1, -1), (1, 0, 0)]
    parts = [crystal._superposed_valence(d, c) for d, c in zip(datasets, crystal.centers)]
    rho, charges = crystal.initial_density()
    assert np.allclose(rho, parts[0] + parts[1], rtol=2e-13, atol=2e-14)
    assert charges[(0, 0, 0)] == pytest.approx(datasets[0].compensation_charge / np.sqrt(4. * np.pi))
    assert charges[(0, 1, -1)] == 0.
    assert charges[(1, 0, 0)] == pytest.approx(datasets[1].compensation_charge / np.sqrt(4. * np.pi))
    moments = [0.4, -100.]
    expected = (np.clip(moments[0] / datasets[0].valence_charge, -1., 1.) * parts[0]
                - parts[1])
    assert np.allclose(crystal.initial_magnetization(moments), expected, rtol=2e-13, atol=2e-14)
    assert not np.any(crystal.initial_magnetization(None))
    pieces = np.stack(parts)
    total = pieces.sum(axis=0)
    shares = np.divide(pieces, total, out=np.zeros_like(pieces), where=total > 0.)
    assert np.allclose(crystal.valence_shares(), shares, rtol=2e-13, atol=2e-14)
    with pytest.raises(ValueError, match="one charge and one moment"):
        crystal.initial_magnetization([1.])


@pytest.mark.parametrize("q_complex,d_complex", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("m,p", [(4, 0), (4, 5), (24, 72)])
def test_native_paw_matrix_products_match_complex_reference(q_complex, d_complex, m, p):
    rng = np.random.default_rng(319)
    def complex_matrix(shape):
        return np.ascontiguousarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    c = complex_matrix((m, p))
    q = complex_matrix((p, p)) if q_complex else np.ascontiguousarray(rng.normal(size=(p, p)))
    d = complex_matrix((p, p)) if d_complex else np.ascontiguousarray(rng.normal(size=(p, p)))
    s0, kinetic, local = (complex_matrix((m, m)) for _ in range(3))
    ctx = _native.Context(threads=1)
    s, h = ctx.periodic_paw_matrices(c, q, d, s0, kinetic, local)
    expected_s = s0 + c @ q @ c.conj().T
    expected_h = kinetic + local + c @ d @ c.conj().T
    assert np.allclose(s, 0.5 * (expected_s + expected_s.conj().T), rtol=2e-13, atol=2e-12)
    assert np.allclose(h, 0.5 * (expected_h + expected_h.conj().T), rtol=2e-13, atol=2e-12)


def test_native_paw_matrix_products_reject_invalid_shapes_and_layout():
    ctx = _native.Context(threads=1)
    c = np.ones((3, 4), dtype=np.complex128)
    operator = np.eye(4)
    base = np.eye(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="operator shape"):
        ctx.periodic_paw_matrices(c, np.eye(3), operator, base, base, base)
    with pytest.raises(ValueError, match="C-contiguous"):
        ctx.periodic_paw_matrices(c[:, ::-1], operator, operator, base, base, base)
    with pytest.raises(ValueError, match="functions, functions"):
        ctx.periodic_paw_matrices(c, operator, operator, base[:2, :2].copy(), base, base)
    invalid = operator.copy()
    invalid[0, 1] = np.nan
    with pytest.raises(ValueError, match="operators must be finite"):
        ctx.periodic_paw_matrices(c, invalid, operator, base, base, base)
    invalid_c = c.copy()
    invalid_c[0, 1] = np.nan
    with pytest.raises(ValueError, match="projections must be finite"):
        ctx.periodic_paw_matrices(invalid_c, operator, operator, base, base, base)


@pytest.mark.parametrize("option,value", [
    ("charge", 1.0),
    ("energy_shift", 0.2), ("size", "DZP"), ("functional", "hse06"),
])
def test_native_comparison_refuses_unimplemented_options(option, value):
    request = {"lattice": [8., 0., 0., 0., 8., 0., 0., 0., 8.],
               "grid_shape": [8, 8, 8],
               "atoms": [{"species": 0, "symbol": "H", "fractional": [0., 0., 0.]}]}
    request[option] = value
    ctx = _native.Context(threads=1, memory_limit_bytes=64 << 20)
    with pytest.raises(NotImplementedError):
        ctx._periodic_problem_from_request(json.dumps(request), [])


def _assert_raw_setup_arrays(active, metadata, arrays, native):
    expected_meta = json.loads(metadata)
    actual_meta = json.loads(native._setup_metadata())
    for key in ("constant", "eion", "ion_constant", "reference", "electrons"):
        assert actual_meta[key] == pytest.approx(expected_meta[key], abs=1e-10), key
    for key in ("relativistic", "time_reversal", "spins", "spinor"):
        assert actual_meta[key] == expected_meta[key], key
    actual = native._setup_arrays()
    points = [f"k{i}.{field}" for i in range(len(active.weights))
              for field in ("psi", "overlap", "fixed", "projections")]
    if native.spinor:
        points.extend(f"k{i}.spin_orbit" for i in range(len(active.weights)))
    for key in (*points, "blocks", "core", "initial", "initial_q", "kernel",
                "ion", "hats", "u", "uion", "linear"):
        expected = np.asarray(arrays[key]).reshape(-1)
        value = np.asarray(actual[key]).reshape(-1)
        np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-9, err_msg=key)


@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_raw_dzp_hydrogen_setup_matches_active_crystal(functional):
    """Compare the complete small setup before routing production through Rust."""
    import os
    from ase import Atoms
    from mandacaru.integrals import Grid
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    from mandacaru.units import BOHR_TO_ANGSTROM

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    cell = np.eye(3) * (8.0 * BOHR_TO_ANGSTROM)
    atoms = Atoms("H", positions=[[0., 0., 0.]], cell=cell, pbc=True)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                grid_spacing=BOHR_TO_ANGSTROM, units="angstrom", cell=cell, periodic=True)
    ctx = _native.Context(threads=1, memory_limit_bytes=1 << 30)
    active, _ = build_crystal(atoms, BOHR_TO_ANGSTROM,
        {"size": "DZP", "paw_dataset": "lda-sr"}, kpts=(1, 1, 1),
        grid=grid, symmetry=False, native_context=ctx)
    metadata, arrays = _active_paw_snapshot(active, 1.0, functional)
    request = {
        "lattice": (np.eye(3) * 8.0).reshape(-1).tolist(),
        "grid_shape": list(grid.shape),
        "atoms": [{"species": 0, "symbol": "H", "fractional": [0., 0., 0.]}],
        "basis_size": "DZP", "symmetry": False, "functional": functional,
    }
    native = ctx._periodic_problem_from_request(
        json.dumps(request), [str(Path(root) / "lda-sr" / "H.parquet")])
    assert native.n_orbitals == active.M == 5
    _assert_raw_setup_arrays(active, metadata, arrays, native)


def test_collinear_magnetic_hydrogen_setup_matches_active_crystal():
    import os
    from ase import Atoms
    from mandacaru.integrals import Grid
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    from mandacaru.units import BOHR_TO_ANGSTROM

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    cell = np.eye(3) * (8.0 * BOHR_TO_ANGSTROM)
    atoms = Atoms("H", positions=[[0., 0., 0.]], cell=cell, pbc=True)
    atoms.set_initial_magnetic_moments([0.7])
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                grid_spacing=BOHR_TO_ANGSTROM, units="angstrom", cell=cell, periodic=True)
    ctx = _native.Context(threads=1, memory_limit_bytes=1 << 30)
    active, _ = build_crystal(atoms, BOHR_TO_ANGSTROM,
        {"size": "DZP", "paw_dataset": "lda-sr"}, kpts=(1, 1, 1),
        grid=grid, symmetry=False, full_mesh=True, native_context=ctx)
    metadata, arrays = _active_paw_snapshot(active, 1.0,
        magnetic_moments=np.array([0.7]), unrestricted=True)
    request = {
        "lattice": (np.eye(3) * 8.0).reshape(-1).tolist(),
        "grid_shape": list(grid.shape),
        "atoms": [{"species": 0, "symbol": "H", "fractional": [0., 0., 0.], "moment": 0.7}],
        "basis_size": "DZP", "symmetry": True,
    }
    native = ctx._periodic_problem_from_request(
        json.dumps(request), [str(Path(root) / "lda-sr" / "H.parquet")])
    assert native.spins == 2
    _assert_raw_setup_arrays(active, metadata, arrays, native)
    expected_result = ctx.periodic_problem(metadata, arrays).solve()
    actual_result = native.solve()
    assert expected_result.converged and actual_result.converged
    assert actual_result.snapshot(include_matrices=False)["energy"] == pytest.approx(
        expected_result.snapshot(include_matrices=False)["energy"], abs=1e-8)


@pytest.mark.parametrize("magnetic", [False, True])
def test_spinor_silicon_setup_matches_active_crystal(magnetic, tmp_path):
    import os
    from ase import Atoms
    from mandacaru.integrals import Grid
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    from mandacaru.units import BOHR_TO_ANGSTROM

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    cell = np.eye(3) * (8.0 * BOHR_TO_ANGSTROM)
    atoms = Atoms("Si", positions=[[0., 0., 0.]], cell=cell, pbc=True)
    moment = 0.4 if magnetic else 0.0
    atoms.set_initial_magnetic_moments([moment])
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.,
                grid_spacing=BOHR_TO_ANGSTROM, units="angstrom", cell=cell, periodic=True)
    ctx = _native.Context(threads=1, memory_limit_bytes=1 << 30)
    active, setup = build_crystal(atoms, BOHR_TO_ANGSTROM,
        {"size": "SZ", "paw_dataset": "lda-fr"}, kpts=(2, 1, 1),
        grid=grid, symmetry=False, spin_orbit=True, full_mesh=magnetic, native_context=ctx)
    metadata, arrays = _active_paw_snapshot(active, 4.0, magnetic_moments=np.array([moment]),
        time_reversal=not magnetic, unrestricted=magnetic)
    request = {
        "lattice": (np.eye(3) * 8.0).reshape(-1).tolist(), "grid_shape": list(grid.shape),
        "atoms": [{"species": 0, "symbol": "Si", "fractional": [0., 0., 0.], "moment": moment}],
        "basis_size": "SZ", "spinor": True, "symmetry": False,
        "kpoints": [{"fractional": k.tolist(), "weight": float(w)}
                    for k, w in zip(setup["kpoints_fractional"], active.weights)],
    }
    native = ctx._periodic_problem_from_request(
        json.dumps(request), [str(Path(root) / "lda-fr" / "Si.parquet")])
    assert native.spinor and native.time_reversal == (not magnetic)
    _assert_raw_setup_arrays(active, metadata, arrays, native)
    expected_result = ctx.periodic_problem(metadata, arrays).solve(
        max_iterations=1000, tolerance=1e-10, density_tolerance=1e-7)
    actual_result = native.solve(max_iterations=1000, tolerance=1e-10, density_tolerance=1e-7)
    assert expected_result.converged and actual_result.converged
    assert actual_result.snapshot(include_matrices=False)["energy"] == pytest.approx(
        expected_result.snapshot(include_matrices=False)["energy"], abs=1e-8)
    snapshot = tmp_path / "spinor-comparison.json"
    native.save(str(snapshot))
    loaded = ctx.load_crystal(str(snapshot))
    _assert_raw_setup_arrays(active, metadata, arrays, loaded)
    loaded_result = loaded.solve(max_iterations=1000, tolerance=1e-10, density_tolerance=1e-7)
    assert loaded_result.converged
    assert loaded_result.snapshot(include_matrices=False)["energy"] == pytest.approx(
        actual_result.snapshot(include_matrices=False)["energy"], abs=1e-8)
    with pytest.raises(NotImplementedError, match="parity"):
        loaded_result.forces()


def test_native_hubbard_fe_setup_matches_active_crystal(tmp_path):
    import os
    from ase import Atoms
    from mandacaru import hubbard
    from mandacaru.integrals import Grid
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    from mandacaru.units import BOHR_TO_ANGSTROM

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    cell = np.eye(3) * (8. * BOHR_TO_ANGSTROM)
    atoms = Atoms("Fe", positions=[[0., 0., 0.]], cell=cell, pbc=True)
    moments = np.array([2.])
    atoms.set_initial_magnetic_moments(moments)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.,
                grid_spacing=BOHR_TO_ANGSTROM, units="angstrom", cell=cell, periodic=True)
    ctx = _native.Context(threads=1, memory_limit_bytes=1 << 30)
    active, setup = build_crystal(atoms, BOHR_TO_ANGSTROM,
        {"size": "SZ", "paw_dataset": "lda-sr"}, kpts=(1, 1, 1),
        grid=grid, symmetry=False, full_mesh=True, native_context=ctx)
    option = {"Fe": {"l": "d", "U": 4.0}}
    spec = hubbard.validate(option, xc="lda", precision="float64")
    table = hubbard.table(spec, ctx, active, moments)
    metadata, arrays = _active_paw_snapshot(active, active.datasets[0].valence_charge,
        magnetic_moments=moments, unrestricted=True, hubbard=table)
    request = {"lattice": (np.eye(3) * 8.).reshape(-1).tolist(), "grid_shape": list(grid.shape),
        "atoms": [{"species": 0, "symbol": "Fe", "fractional": [0., 0., 0.], "moment": 2.}],
        "basis_size": "SZ", "symmetry": False, "hubbard_u": option,
        "kpoints": [{"fractional": k.tolist(), "weight": float(w)}
                    for k, w in zip(setup["kpoints_fractional"], active.weights)]}
    native = ctx._periodic_problem_from_request(json.dumps(request),
        [str(Path(root) / "lda-sr" / "Fe.parquet")])
    _assert_raw_setup_arrays(active, metadata, arrays, native)
    expected = ctx.periodic_problem(metadata, arrays).solve(max_iterations=1000)
    result = native.solve(max_iterations=1000)
    assert expected.converged and result.converged
    assert result.snapshot(include_matrices=False)["energy"] == pytest.approx(
        expected.snapshot(include_matrices=False)["energy"], abs=1e-8)
    snapshot = tmp_path / "native-hubbard.json"
    native.save(str(snapshot))
    loaded = ctx.load_crystal(str(snapshot))
    loaded.check_hubbard_u(spec)


@pytest.mark.parametrize("functional", ["lda", "pbe"])
@pytest.mark.parametrize("magnetic", [False, True])
def test_raw_lih_skew_multik_setup_matches_active_crystal(functional, magnetic, tmp_path):
    import hashlib
    import os
    from ase import Atoms
    from mandacaru.integrals import Grid
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    from mandacaru.units import BOHR_TO_ANGSTROM

    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        pytest.skip("MANDACARU_PAW_PATH is not set")
    lattice = np.array([[8., 0., 0.], [1., 9., 0.], [.5, 1., 10.]])
    cell = lattice * BOHR_TO_ANGSTROM
    fractional = np.array([[.1, .2, .3], [.55, .6, .7]])
    atoms = Atoms("LiH", scaled_positions=fractional, cell=cell, pbc=True)
    moments = np.array([0.7, -0.2]) if magnetic else np.zeros(2)
    atoms.set_initial_magnetic_moments(moments)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.,
                grid_spacing=BOHR_TO_ANGSTROM, units="angstrom", cell=cell, periodic=True)
    ctx = _native.Context(threads=1, memory_limit_bytes=1 << 30)
    active, setup = build_crystal(atoms, BOHR_TO_ANGSTROM,
        {"size": "SZ", "paw_dataset": "lda-sr"}, kpts=(2, 1, 1),
        grid=grid, symmetry=False, full_mesh=True, native_context=ctx)
    metadata, arrays = _active_paw_snapshot(active, 2., functional,
        magnetic_moments=moments, unrestricted=magnetic)
    request = {
        "lattice": lattice.T.reshape(-1).tolist(), "grid_shape": list(grid.shape),
        "atoms": [{"species": 1, "symbol": "Li", "fractional": fractional[0].tolist(), "moment": float(moments[0])},
                  {"species": 0, "symbol": "H", "fractional": fractional[1].tolist(), "moment": float(moments[1])}],
        "basis_size": "SZ", "symmetry": False, "functional": functional,
        "kpoints": [{"fractional": k.tolist(), "weight": float(w)}
                    for k, w in zip(setup["kpoints_fractional"], active.weights)],
    }
    folder = Path(root) / "lda-sr"
    native = ctx._periodic_problem_from_request(
        json.dumps(request), [str(folder / "H.parquet"), str(folder / "Li.parquet")])
    assert native.n_orbitals == active.M == 2
    provenance = json.loads(native._setup_metadata())["datasets"]
    for (filename, digest), symbol in zip(provenance, ("Li", "H")):
        path = folder / f"{symbol}.parquet"
        assert Path(filename) == path
        with path.open("rb") as stream:
            assert digest == hashlib.file_digest(stream, "sha256").hexdigest()
    _assert_raw_setup_arrays(active, metadata, arrays, native)
    expected_result = ctx.periodic_problem(metadata, arrays).solve()
    actual_result = native.solve()
    assert expected_result.converged and actual_result.converged
    expected_energy = expected_result.snapshot(include_matrices=False)
    actual_energy = actual_result.snapshot(include_matrices=False)
    for key in ("energy", "free_energy", "extrapolated_energy"):
        assert actual_energy[key] == pytest.approx(expected_energy[key], abs=1e-8), key
    with pytest.raises(NotImplementedError, match="parity"):
        actual_result.forces()
    snapshot = tmp_path / "comparison-crystal.json"
    native.save(str(snapshot))
    loaded = ctx.load_crystal(str(snapshot))
    _assert_raw_setup_arrays(active, metadata, arrays, loaded)
    loaded_result = loaded.solve()
    assert loaded_result.converged
    assert loaded_result.snapshot(include_matrices=False)["energy"] == pytest.approx(
        actual_energy["energy"], abs=1e-8)
    with pytest.raises(NotImplementedError, match="parity"):
        loaded_result.forces()


@pytest.mark.parametrize('method',['fermi-dirac','gaussian','methfessel-paxton'])
@pytest.mark.parametrize('degeneracy',[1.,2.])
def test_metallic_fermi_occupations_entropy_and_variational_free_energy(method,degeneracy):
    # The deleted Python smearing code is replaced by the closed forms above
    # and a Brent root of the charge neutrality condition; same tolerances.
    levels=np.array([[-.3,-.01,.12,.4],[-.2,.03,.17,.5]])
    weights=np.array([.25,.75])
    width=.06
    electrons=degeneracy*1.7
    ctx=_native.Context()
    raw=ctx.fermi_distribution(levels.tolist(),weights.tolist(),electrons,method=method,width_ha=width,degeneracy=degeneracy)
    mu=reference_fermi_level(levels,weights,electrons,method,width,degeneracy)
    assert raw['fermi_level']==pytest.approx(mu,abs=1e-12)
    f=degeneracy*smearing_occupation((levels-mu)/width,method)
    assert np.allclose(raw['occupations'],f,atol=2e-12)
    assert np.sum(f*weights[:,None])==pytest.approx(electrons,abs=2e-11)
    expected_entropy=degeneracy*np.sum(smearing_entropy((levels-mu)/width,method)*weights[:,None])
    assert raw['entropy']==pytest.approx(expected_entropy,abs=1e-11)
    def free(w):
        r=ctx.fermi_distribution(levels.tolist(),weights.tolist(),electrons,method=method,width_ha=w,degeneracy=degeneracy)
        return np.sum(np.array(r['occupations'])*levels*weights[:,None])-w*r['entropy']
    assert (free(width+1e-5)-free(width-1e-5))/2e-5==pytest.approx(-raw['entropy'],abs=1e-7)


@pytest.mark.parametrize('size',[(1,1,1),(2,3,4),(3,3,3)])
@pytest.mark.parametrize('gamma',[False,True])
def test_native_kmesh_matches_ase_and_time_reversal_orbits(size,gamma):
    from ase.calculators.calculator import kpts2ndarray
    from ase.dft.kpoints import monkhorst_pack
    from mandacaru.pseudopotentials.periodic_paw import time_reversal_reduce
    # ASE's own mesh helpers replace the deleted Python wrapper around them:
    # the plain Monkhorst-Pack mesh, or the Gamma-centered one (ASE spells
    # that {"size": ..., "gamma": True}).
    full=kpts2ndarray(dict(size=size,gamma=True)) if gamma else monkhorst_pack(size)
    points,weights=time_reversal_reduce(full)
    result=_native.Context().kmesh(size,gamma=gamma)
    assert np.allclose(result['full'],full,atol=1e-14)
    assert np.allclose(result['points'],points,atol=1e-14)
    assert np.allclose(result['weights'],weights,atol=1e-14)
    mapping=np.array(result['mapping'])
    assert np.allclose(np.bincount(mapping)/len(full),result['weights'])


def test_space_group_reduction_checks_mesh_preservation():
    ctx=_native.Context()
    rotation=[0,1,0,1,0,0,0,0,1]
    a=ctx.kmesh([3,3,1],gamma=True,time_reversal=False)
    b=ctx.kmesh([3,3,1],gamma=True,rotations=[rotation],time_reversal=False)
    assert len(b['points'])<len(a['points']) and sum(b['weights'])==pytest.approx(1.)
    with pytest.raises(ValueError,match='preserve'):
        ctx.kmesh([2,3,1],rotations=[rotation])
    with pytest.raises(ValueError,match='unimodular'):
        ctx.kmesh([2,2,2],rotations=[[0]*9])


@pytest.mark.parametrize('skew',[False,True])
@pytest.mark.parametrize('eta',[None,.18,.45])
def test_ewald_matches_python_for_energy_forces_stress_and_potential(skew,eta):
    from mandacaru.core.ewald import ewald_energy,ewald_forces,ewald_stress,ewald_potential
    cell=np.diag([6.,7.,8.])
    if skew:
        cell+=np.array([[0.,.3,.1],[.2,0.,.4],[.1,.2,0.]])
    positions=np.array([[.7,1.,1.2],[2.1,3.,3.1]])
    charges=np.array([1.,-2.])
    points=np.array([[.3,2.,1.1],[3.3,1.,.9]])
    ctx=_native.Context(threads=1)
    raw=ctx.ewald(positions,charges,cell,eta=eta,points_bohr=points,softening_bohr=.2)
    assert raw['energy']==pytest.approx(ewald_energy(positions,charges,cell,eta=eta),abs=3e-12)
    assert np.allclose(raw['forces'],ewald_forces(positions,charges,cell,eta=eta),atol=3e-12)
    assert np.allclose(raw['stress'],ewald_stress(positions,charges,cell,eta=eta),atol=3e-12)
    assert np.allclose(raw['potential'],ewald_potential(positions,charges,cell,points,eta=eta,softening=.2),atol=3e-12)
    assert np.linalg.norm(raw['forces'].sum(axis=0))<1e-13
    moved=positions.copy();moved[0]+=cell[0]-2*cell[1]
    assert ctx.ewald(moved,charges,cell,eta=eta)['energy']==pytest.approx(raw['energy'],abs=3e-12)
    assert ctx.memory_used_bytes==0


def test_ewald_strain_and_displacement_derivatives_and_madelung_limit():
    ctx=_native.Context(threads=1)
    r=np.array([[.7,1.,1.2],[2.1,3.,3.1]])
    q=np.array([1.,-2.]);cell=np.array([[6.,.3,.1],[.2,7.,.4],[.1,.2,8.]])
    raw=ctx.ewald(r,q,cell)
    eps=1e-5
    plus=r.copy();minus=r.copy();plus[0,1]+=eps;minus[0,1]-=eps
    fd=-(ctx.ewald(plus,q,cell)['energy']-ctx.ewald(minus,q,cell)['energy'])/(2*eps)
    assert raw['forces'][0,1]==pytest.approx(fd,abs=1e-9)
    for a,b in [(0,0),(0,2),(2,1)]:
        strain=np.zeros((3,3));strain[a,b]=eps
        eplus=ctx.ewald(np.ascontiguousarray(r@(np.eye(3)+strain).T),q,np.ascontiguousarray(cell@(np.eye(3)+strain).T))['energy']
        eminus=ctx.ewald(np.ascontiguousarray(r@(np.eye(3)-strain).T),q,np.ascontiguousarray(cell@(np.eye(3)-strain).T))['energy']
        assert raw['stress'][a,b]==pytest.approx((eplus-eminus)/(2*eps*np.linalg.det(cell)),abs=2e-10)
    unit=ctx.ewald(np.zeros((1,3)),np.ones(1),np.eye(3)*5)
    assert 2*unit['energy']*5==pytest.approx(-2.83729747948062,abs=1e-10)


@pytest.mark.parametrize('functional',['lda','pbe'])
@pytest.mark.parametrize('spins',[1,2])
def test_uniform_periodic_gas_matches_independent_grid_functional(functional,spins):
    from mandacaru.integrals import Grid
    from mandacaru.integrals.exchange_correlation import evaluate
    payload=synthetic(functional,spins)
    ctx=_native.Context(threads=1)
    problem=ctx.periodic_problem(json.dumps(payload), {})
    raw=problem.solve(width_ha=.06).snapshot()
    assert raw['converged']
    n=64
    grid=Grid(center=[2/1.8897261246257702]*3,cell=np.eye(3)*4/1.8897261246257702,
              grid_spacing=1/1.8897261246257702,periodic=True)
    rho=np.full(n,2/n)
    xc=evaluate(grid,rho,functional)
    band=sum(np.sum(raw['occupations'][s,0]*np.array([-.5,.3])) for s in range(spins))
    assert raw['energy']==pytest.approx(band+xc.energy+.13,abs=1e-10)
    assert np.sum(raw['density'])==pytest.approx(2.,abs=1e-10)
    assert np.allclose(raw['eigenvalues'][0,0],np.array([-.5,.3])+xc.potential[0]+.02,atol=1e-10)
    for d in raw['density_matrices'].reshape(-1,2,2):
        assert np.allclose(d,d.conj().T)
    del problem
    gc.collect()
    assert ctx.memory_used_bytes==0


@pytest.fixture(scope='module')
def si_native():
    ctx=_native.Context(threads=1)
    problem=ctx.load_crystal(str(SI))
    result=problem.solve(width_ha=.01/HARTREE_TO_EV)
    return ctx,problem,result


def test_cached_silicon_matches_frozen_phase0_with_explicit_fermi_plateau(si_native):
    _,_,result=si_native
    raw=result.snapshot()
    frozen=json.loads((HERE/'reference/numpy.json').read_text())['cases']['si_periodic_lda']
    expected=frozen['samples'][0]['values']
    assert raw['converged']
    assert raw['extrapolated_energy']*HARTREE_TO_EV==pytest.approx(expected['energy_ev'],abs=1e-6)
    assert np.allclose(raw['eigenvalues'][0]*HARTREE_TO_EV,expected['eigenvalues_ev'],atol=1e-6)
    assert np.allclose(raw['weights'],expected['kpoint_weights'],atol=1e-14)
    eps=raw['eigenvalues'][0]*HARTREE_TO_EV
    valence,conduction=eps[:,3].max(),eps[:,4].min()
    # At 0.01 eV the count saturates over an interval in this gap. The
    # chemical potential is nonunique at machine precision; both values must
    # be in the gap and the native occupations must give the exact count.
    assert valence<raw['fermi_level']*HARTREE_TO_EV<conduction
    assert valence<expected['fermi_level_ev']<conduction
    assert np.sum(raw['occupations'][0]*np.array(raw['weights'])[:,None])==pytest.approx(8.,abs=1e-10)
    assert raw['free_energy']==pytest.approx(raw['energy'],abs=1e-12)
    payload=json.loads(SI.read_text())
    assert payload['provenance']['dataset_sources'][0]['sha256']==frozen['dataset_inputs'][0]['sha256']


def test_cached_crystal_runs_without_python_scf_fft_linalg_or_xc_callbacks(monkeypatch):
    from mandacaru.integrals import exchange_correlation
    def forbidden(*args,**kwargs):
        raise AssertionError('Python numerical callback entered the native periodic solve')
    # The Python Kohn-Sham loop, Pulay mixer and CPU eigensolver lived in the
    # deleted mandacaru.algorithms package, so their absence is the guard; the
    # NumPy and Python XC entry points that remain are poisoned.
    for obj,name in [(np.linalg,'eigh'),(np.linalg,'solve'),
                     (np.fft,'fftn'),(np.fft,'ifftn'),(exchange_correlation,'evaluate')]:
        monkeypatch.setattr(obj,name,forbidden)
    result=Mandacaru(method='dft',load_crystal=SI,
                     smearing={'width':.01},trace=False).run()
    assert result.success and result.periodic is not None
    assert result.optimal_energy==pytest.approx(-211.60193975895757,abs=1e-6)


def test_frozen_bands_dos_projection_and_trial_subspaces_are_consistent(si_native):
    _,_,result=si_native
    raw=result.snapshot();payload=json.loads(SI.read_text())
    bands=result.bands(json.dumps(payload['kpoints']), {})
    assert np.allclose(bands['eigenvalues'],raw['eigenvalues'],atol=1e-12)
    assert np.allclose(bands['projections'].sum(axis=2),1.,atol=1e-10)
    energies=np.linspace(-2.,2.,1601)
    total=result.dos(energies,width_ha=.03)[:,0]
    projected=result.dos(energies,width_ha=.03,projected=True)
    assert np.allclose(projected.sum(axis=1),total,atol=1e-12)
    assert np.trapezoid(total,energies)==pytest.approx(16.,abs=1e-9)
    full=result.projectability(np.eye(8,dtype=complex))
    assert np.allclose(full['weights'],1.,atol=1e-10)
    assert np.allclose(full['hamiltonians'],full['hamiltonians'].swapaxes(-1,-2).conj(),atol=1e-10)
    assert np.allclose(np.linalg.eigvalsh(full['hamiltonians']),raw['eigenvalues'],atol=1e-10)
    trials=np.eye(8,dtype=complex)[:,:2].copy()
    small=result.projectability(trials)
    assert np.all(small['weights']>=-1e-12) and np.all(small['weights']<=1+1e-10)
    assert np.allclose(small['weights'].sum(axis=-1),2.,atol=1e-10)
    with pytest.raises(ValueError,match='dependent'):
        result.projectability(np.ones((8,2),dtype=complex))


@pytest.mark.parametrize('hopping,expected',[(.5,np.pi),(1.5,0.)])
def test_wilson_loop_ssh_phase_is_invariant_under_local_gauges(hopping,expected):
    k=np.linspace(0.,2*np.pi,301,endpoint=False)
    phase=np.angle(hopping+np.exp(1j*k))
    occupied=np.stack([-np.exp(1j*phase),np.ones(len(k))],axis=1)/np.sqrt(2)
    links=np.array([np.vdot(occupied[i],occupied[(i+1)%len(k)]) for i in range(len(k))]).reshape(-1,1,1)
    ctx=_native.Context()
    berry,singular=ctx.wilson_loop(links)
    assert abs(np.angle(np.exp(1j*(berry-expected))))<1e-10
    gauge=np.exp(1j*np.random.default_rng(27).normal(size=len(k)))
    links[:,0,0]*=np.conj(gauge)*np.roll(gauge,-1)
    shifted,_=ctx.wilson_loop(links)
    assert abs(np.angle(np.exp(1j*(berry-shifted))))<1e-10
    assert singular>.99
    links[0]=0.
    with pytest.raises(ValueError,match='singular'):
        ctx.wilson_loop(links)


def test_cached_validation_failure_budget_and_result_ownership(tmp_path):
    payload=synthetic()
    for mutate in [lambda p:p.update(energy_unit='eV'),lambda p:p['kpoints'][0].update(weight=.5),
                   lambda p:p['kpoints'][0].update(overlap=complex_values(np.diag([1.,-1.]))),
                   lambda p:p.update(kernel=[1.]*64),lambda p:p.update(permutations=[[0]*64])]:
        bad=copy.deepcopy(payload);mutate(bad)
        ctx=_native.Context()
        with pytest.raises(ValueError):
            ctx.periodic_problem(json.dumps(bad), {})
        assert ctx.memory_used_bytes==0
    ctx=_native.Context(memory_limit_bytes=1024)
    with pytest.raises(MemoryError):
        ctx.periodic_problem(json.dumps(payload), {})
    assert ctx.memory_used_bytes==0
    ctx=_native.Context()
    problem=ctx.periodic_problem(json.dumps(payload), {})
    base=ctx.memory_used_bytes
    failed=problem.solve(max_iterations=1)
    assert not failed.converged
    del failed
    assert ctx.memory_used_bytes==base
    result=problem.solve()
    owned=ctx.memory_used_bytes
    result.dos(np.linspace(-1,1,20),width_ha=.1)
    assert ctx.memory_used_bytes==owned
    snap=result.snapshot();snap['eigenvalues'][:]=0
    assert np.linalg.norm(result.snapshot()['eigenvalues'])>.1
    path=tmp_path/'crystal.json';problem.save(str(path))
    saved=ctx.load_crystal(str(path)).solve().snapshot()
    assert np.allclose(saved['eigenvalues'],result.snapshot()['eigenvalues'])
    del result,problem
    gc.collect()
    assert ctx.memory_used_bytes==0


def test_ase_silicon_adapter_getters_and_native_off_mesh_spectra():
    from mandacaru.pseudopotentials.paw import get_paw
    try:
        get_paw('Si',paw_dataset='lda-sr')
    except FileNotFoundError:
        pytest.skip('external lda-sr/Si.parquet dataset unavailable; the portable native crystal tests still run')
    atoms=bulk('Si','diamond',a=5.43)
    from mandacaru.integrations.ase import ASECalculator
    atoms.calc=ASECalculator(method='dft',basis={'name':'PAW-LCAO','size':'SZ','filter_cutoff':200},
                        grid_spacing=.4,kpts={'size':(2,2,2),'gamma':True},smearing=.01,trace=False)
    assert atoms.get_potential_energy()==pytest.approx(-211.60193975895757,abs=1e-6)
    assert atoms.calc.calculation.result.success
    assert atoms.calc.get_number_of_spins()==1
    assert atoms.calc.get_eigenvalues().shape==(8,)
    assert len(atoms.calc.get_ibz_k_points())==3
    assert sum(atoms.calc.get_k_point_weights())==pytest.approx(1.)
    assert atoms.calc.get_bz_k_points().shape==(8,3)
    band=atoms.calc.band_structure(path='GX',npoints=9)
    assert band.energies.shape==(1,9,8)
    axis,dos=atoms.calc.dos(npoints=801)
    _,pdos=atoms.calc.pdos(energies=axis)
    assert np.allclose(sum(pdos.values()),dos,atol=1e-10)
    # Through the ASE adapter a crystal's forces and stress are central finite
    # differences of the energy taken by ASE on the attached Atoms: an INTERIM Python
    # route that the PAW plan replaces with native derivatives (CF-7). This pins what
    # it returns today, not a final contract. Measured on this cell: every force
    # component is below 2e-10 eV/A and their sum below 3e-10 (diamond Si is a stationary
    # point and the translation sum rule holds); 1e-6 leaves three decades of margin without
    # hiding a real violation, which would be of order 1e-2 eV/A.
    forces=atoms.get_forces()
    assert forces.shape==(2,3)
    assert np.abs(forces.sum(axis=0)).max()<1e-6
    assert atoms.get_stress().shape==(6,)


@pytest.mark.parametrize('spin',[False,True])
def test_periodic_pbe_and_collinear_spin_match_python_on_skew_cells(spin):
    from ase import Atoms
    from mandacaru.pseudopotentials.paw import get_paw
    xc='lda' if spin else 'pbe'
    try:
        get_paw('H',paw_dataset=f'{xc}-sr')
    except FileNotFoundError:
        pytest.skip(f'external {xc}-sr/H.parquet dataset unavailable')
    cell=np.array([[4.,.4,0.],[0.,4.,.3],[.1,0.,4.5]])
    atoms=Atoms('H' if spin else 'H2',positions=[[1.2,1.6,1.9]] if spin else
                [[1.2,1.6,1.9],[2.,2.3,2.1]],cell=cell,pbc=True)
    if spin:
        atoms.set_initial_magnetic_moments([1.])
    opts=dict(method='dft',xc=xc,spin=spin,basis={'name':'PAW-LCAO','size':'SZ','filter_cutoff':200},
              grid_spacing=.7,kpts={'size':(2,1,1),'gamma':True},smearing=.3,trace=False)
    py=atoms.copy();py.calc=Mandacaru(**opts)
    expected=py.get_potential_energy()
    atoms.calc=Mandacaru(**opts)
    actual=atoms.get_potential_energy()
    assert atoms.calc.result.success and py.calc.result.success
    assert actual==pytest.approx(expected,abs=3e-6)
    assert atoms.calc.result.snapshot()["magnetic_moment"]==pytest.approx(py.calc.result.snapshot()["magnetic_moment"],abs=1e-5)
    assert atoms.calc.result.snapshot()["n_spins"]==(2 if spin else 1)
    if spin:
        # Magnetic initial states must not be projected onto a nonmagnetic
        # spatial symmetry group during density assembly.
        assert atoms.calc._crystal_adapter.symmetry is None
    assert atoms.calc.results['free_energy']==pytest.approx(py.calc.results['free_energy'],abs=3e-6)
    expected_levels=np.asarray(py.calc.result.snapshot()["eigenvalues"])
    actual_levels=np.asarray(atoms.calc.result.snapshot()["eigenvalues"])
    assert np.allclose(actual_levels,expected_levels,atol=2e-6)
