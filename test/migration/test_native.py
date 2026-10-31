"""Scientific and ownership contracts of the Phase 1 checked bindings."""

import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal
from scipy.special import spherical_jn

from mandacaru import _native
from mandacaru.integrations.ase import geometry_from_atoms
from mandacaru.integrals import _backend
from mandacaru.units import ANGSTROM_TO_BOHR


def test_ase_snapshot_is_owned_and_converts_units_once():
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5] * 3)
    geometry = geometry_from_atoms(atoms)
    expected = atoms.positions * ANGSTROM_TO_BOHR
    atoms.positions[:] = 0
    assert np.allclose(geometry.positions_bohr, expected, atol=1e-14)
    copy = geometry.positions_bohr
    copy[:] = -1
    assert np.allclose(geometry.positions_bohr, expected)
    assert geometry.numbers == [1, 1]
    assert np.allclose(geometry.cell_bohr, np.eye(3) * 5 * ANGSTROM_TO_BOHR)


def test_geometry_rejects_nonfinite_and_degenerate_periodic_cells():
    for positions, cell, pbc in [(np.full((1, 3), np.nan), np.eye(3), [False]*3),
                                 (np.zeros((1, 3)), np.zeros((3, 3)), [True]*3)]:
        with pytest.raises(ValueError):
            _native.Geometry([1], positions, cell, pbc)
    geometry = _native.Geometry([1], np.zeros((1, 3)), np.diag([1., 1., 0.]), [True, True, False])
    assert geometry.pbc == [True, True, False] or geometry.pbc == (True, True, False)


def test_live_workspace_budget_and_independent_numpy_lifetimes():
    ctx = _native.Context(threads=1, memory_limit_bytes=80)
    workspace = ctx.workspace(10)
    workspace.fill(3.0)
    assert ctx.memory_used_bytes == 80
    with pytest.raises(MemoryError, match="budget"):
        ctx.workspace(1)
    snapshot = workspace.to_numpy()
    snapshot[:] = 7.0
    assert np.all(workspace.to_numpy() == 3.0)
    del workspace
    gc.collect()
    assert ctx.memory_used_bytes == 0
    del ctx
    assert np.all(snapshot == 7.0)


def test_workspace_outlives_its_context():
    ctx = _native.Context(threads=1, memory_limit_bytes=80)
    workspace = ctx.workspace(10)
    del ctx
    gc.collect()
    workspace.fill(2.0)
    assert np.all(workspace.to_numpy() == 2.0)


@pytest.mark.parametrize("threads,budget", [(0, 1024), (1025, 1024), (1, 0)])
def test_context_refuses_invalid_resource_limits(threads, budget):
    with pytest.raises(ValueError):
        _native.Context(threads=threads, memory_limit_bytes=budget)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_laplacian_matches_the_zero_boundary_stencil(dtype):
    ctx = _native.Context(threads=2)
    grid = _native.Grid.orthogonal([3, 4, 5], [0.5, 0.75, 1.0])
    rng = np.random.default_rng(4)
    field = rng.normal(size=(2, 3, 4, 5)).astype(dtype)
    padded = np.pad(field, ((0,0),(1,1),(1,1),(1,1)))
    expected = np.zeros_like(field, dtype=np.float64)
    for axis, spacing in enumerate([0.5, 0.75, 1.0], 1):
        lo = [slice(None), slice(1,-1), slice(1,-1), slice(1,-1)]
        hi = lo.copy(); lo[axis] = slice(0,-2); hi[axis] = slice(2,None)
        expected += (padded[tuple(lo)].astype(float) + padded[tuple(hi)] - 2.0*field) / spacing**2
    result = (ctx.laplacian if dtype == np.float64 else ctx.laplacian_f32)(grid, field)
    assert np.allclose(result, expected, atol=1e-5 if dtype == np.float32 else 1e-12)
    assert result.dtype == dtype
    assert 0 < ctx.worker_peak <= 2
    assert ctx.memory_used_bytes == 0


def test_one_body_and_complex_projectors_match_numpy_contractions():
    ctx = _native.Context(threads=2)
    grid = _native.Grid.orthogonal([3,4,5], [0.7,0.8,0.9])
    rng = np.random.default_rng(10)
    psi = rng.normal(size=(3,60)) + 1j*rng.normal(size=(3,60))
    chi = rng.normal(size=(2,60)) + 1j*rng.normal(size=(2,60))
    potential = rng.normal(size=60)
    t, v = ctx.one_body(grid, psi, potential)
    lap = ctx.laplacian(grid, psi.real.copy().reshape(3,3,4,5)) + 1j*ctx.laplacian(grid, psi.imag.copy().reshape(3,3,4,5))
    assert np.allclose(t, -0.5 * psi.conj() @ lap.reshape(3,60).T * grid.dv_bohr3, atol=1e-12)
    assert np.allclose(v, psi.conj() @ (psi*potential).T * grid.dv_bohr3, atol=1e-12)
    assert np.allclose(ctx.project(grid.dv_bohr3, psi, chi), psi.conj() @ chi.T * grid.dv_bohr3, atol=1e-12)


@pytest.mark.parametrize("l", [0, 1, 3, 6, 10])
def test_radial_transforms_match_scipy(l):
    ctx = _native.Context(threads=2)
    r = np.linspace(0, 3.0, 40)
    k = np.linspace(0, 5.0, 50)
    weights = np.exp(-r) * r**2
    expected = spherical_jn(l, r[:,None]*k[None,:])
    assert np.allclose(ctx.bessel(l, r, k), expected, atol=2e-12)
    assert np.allclose(ctx.fourier_bessel(l, r, weights, k), weights @ expected, atol=2e-12)


@pytest.mark.parametrize("index", [0, 2, 5])
def test_eigenpair_matches_scipy_without_fixing_vector_phase(index):
    ctx = _native.Context()
    diagonal = np.arange(6, dtype=float) + 2
    off = np.full(5, -0.3)
    expected, vectors = eigh_tridiagonal(diagonal, off)
    value, vector = ctx.eigenpair(diagonal, off, index)
    assert value == pytest.approx(expected[index], abs=1e-12)
    assert np.allclose(np.outer(vector,vector), np.outer(vectors[:,index],vectors[:,index]), atol=1e-11)


def test_numerov_preserves_a_linear_solution_without_mutating_inputs():
    ctx = _native.Context()
    zero = np.zeros(10)
    outward = zero.copy(); outward[1] = 0.1
    result = ctx.numerov(zero, zero, outward, 0.01, 0)
    assert np.allclose(result, np.arange(10)*0.1, atol=1e-14)
    assert np.count_nonzero(outward) == 1
    inward = zero.copy(); inward[-2:] = [0.8, 0.9]
    assert np.allclose(ctx.numerov(zero, zero, inward, 0.01, 0, inward=True), result, atol=1e-14)
    with pytest.raises(ValueError, match="zero source"):
        ctx.numerov(zero, np.ones(10), inward, 0.01, 0, inward=True)


def test_real_shell_normalization_and_checked_angular_bounds():
    ctx = _native.Context(threads=2)
    xyz = np.array([[0.0, 0.2, 2.0], [0.0]*3, [0.0]*3])
    args = (xyz, [0.0]*3, 1.5, np.array([0.0,1.5]), np.array([0.0,0.0,0.0,1.0]), 1.5, 0)
    result = ctx.real_shell(*args, np.array([0], dtype=np.int32))
    assert np.allclose(result, [[1/np.sqrt(4*np.pi),1/np.sqrt(4*np.pi),0.0]], atol=1e-14)
    with pytest.raises(ValueError):
        ctx.real_shell(*args, np.array([-2**31], dtype=np.int32))


def test_bad_shapes_layouts_types_and_nonfinite_values_are_rejected():
    ctx = _native.Context()
    grid = _native.Grid.orthogonal([2]*3, [1.0]*3)
    for field in (np.zeros((1,2,2,3)), np.full((1,2,2,2), np.nan),
                  np.zeros((1,2,2,4))[:,:,:,::2], np.asfortranarray(np.zeros((1,2,2,2)))):
        with pytest.raises(ValueError):
            ctx.laplacian(grid, field)
    with pytest.raises(TypeError):
        ctx.laplacian(grid, np.zeros((1,2,2,2), dtype=np.float32))
    with pytest.raises(TypeError):
        ctx.laplacian(grid, np.ones((1,2,2,2), dtype=">f8"))
    unaligned = np.ndarray((1,2,2,2), dtype=float, buffer=bytearray(65), offset=1)
    with pytest.raises((ValueError, TypeError)):
        ctx.laplacian(grid, unaligned)
    with pytest.raises(ValueError):
        ctx.one_body(grid, np.zeros((1,7), complex), np.zeros(8))
    with pytest.raises((ValueError, OverflowError)):
        _native.Grid.orthogonal([2**31]*3, [1.0]*3)
    assert ctx.memory_used_bytes == 0


def test_budget_rejection_happens_before_copying_large_inputs():
    ctx = _native.Context(memory_limit_bytes=8)
    grid = _native.Grid.orthogonal([2]*3,[1.0]*3)
    with pytest.raises(MemoryError):
        ctx.laplacian(grid, np.zeros((1,2,2,2)))
    assert ctx.memory_used_bytes == 0


def test_parallel_contexts_keep_distinct_thread_limits(monkeypatch):
    monkeypatch.setenv("MANDACARU_NUM_THREADS", "7")
    first, second = _native.Context(threads=1), _native.Context(threads=2)
    r, k = np.linspace(0,2,500), np.linspace(0,3,600)
    weights = np.exp(-r)
    with ThreadPoolExecutor(2) as executor:
        a = executor.submit(first.fourier_bessel, 0, r, weights, k)
        b = executor.submit(second.fourier_bessel, 0, r, weights, k)
        assert np.allclose(a.result(), b.result(), atol=1e-12)
    assert first.workers_launched == 1 and first.worker_peak == 1
    assert second.workers_launched == 2 and second.worker_peak <= 2
    assert _native.Context().threads == 7


def test_simultaneous_calls_on_one_context_share_the_worker_limit():
    ctx = _native.Context(threads=2)
    r, k = np.linspace(0, 2, 500), np.linspace(0, 3, 600)
    weights = np.exp(-r)
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(ctx.fourier_bessel, 0, r, weights, k) for _ in range(2)]
        assert np.allclose(futures[0].result(), futures[1].result(), atol=1e-12)
    assert ctx.workers_launched == 4 and ctx.worker_peak <= 2
    assert ctx.memory_used_bytes == 0


def test_packaged_extension_also_exports_the_legacy_abi():
    assert _native.Context.__module__ == "mandacaru._native"
    lib = ctypes.CDLL(_native.__file__)
    assert lib.mandacaru_backend_abi_version() == _native.ABI_VERSION == 1
    assert lib.mandacaru_thread_policy_version() == _native.THREAD_POLICY_VERSION == 1
    _backend._bind(lib)
    assert ctypes.sizeof(lib.mandacaru_bloch_shell.argtypes[3]) == 8
