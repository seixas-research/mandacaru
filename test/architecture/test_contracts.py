"""Behavioral checks of the supported calculation boundary and ownership."""

import gc
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mandacaru import Geometry, Mandacaru, _native

REFERENCE = Path(__file__).parents[1] / "migration/reference/h2_hamiltonian.json"


def molecule():
    return Geometry(
        (1, 1), np.array([[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]]), np.eye(3) * 5
    )


def calculator(**options):
    return Mandacaru(
        method="dft", xc="pbe", grid_spacing=0.5, resources={"threads": 1}, **options
    )


def test_no_execution_selector_or_mutable_scientific_configuration():
    assert "engine" not in inspect.signature(Mandacaru).parameters
    for value in ("python", "rust"):
        with pytest.raises(TypeError):
            Mandacaru(engine=value)
    with pytest.raises(TypeError):
        Mandacaru(engine_options={})
    with calculator() as calc:
        with pytest.raises(AttributeError):
            calc.xc = "lda"


@pytest.mark.parametrize("functional", _native.FUNCTIONALS)
def test_scf_retains_no_dense_hamiltonian_until_explicit_request(functional):
    with Mandacaru(
        method="dft", xc=functional, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        result = calc.run(molecule())
        assert result.success
        assert not result.scf.has_model
        assert not calc.integrals.has_eri
        assert "eri_mo" not in result.snapshot()
        assert "two_body" not in calc.integrals.snapshot()
        result.forces()
        assert not result.scf.has_model
        assert not calc.integrals.has_eri
        before = calc.context.memory_used_bytes
        model = result.hamiltonian
        assert result.scf.has_model and calc.integrals.has_eri
        assert result.hamiltonian is model
        used = calc.context.memory_used_bytes
        assert used > before
        result.hamiltonian
        assert calc.context.memory_used_bytes == used
        snapshot = calc.integrals.snapshot()
        assert snapshot["cholesky_rank"] <= 3
        assert snapshot["cholesky_residual"] <= 1.1e-10


def test_calculator_resources_and_results_have_independent_lifetimes(tmp_path):
    options = {"method": "rhf", "grid_spacing": 0.5}
    first = Mandacaru(**options, resources={"threads": 1, "scratch_dir": tmp_path})
    second = Mandacaru(**options, resources={"threads": 2, "scratch_dir": tmp_path})
    try:
        one, two = first.resources.scratch, second.resources.scratch
        assert one != two
        result = first.run(molecule())
        energy = result.optimal_energy
        other = second.run(molecule())
        assert other.optimal_energy == pytest.approx(energy, abs=1e-10)
        assert first.context.threads == 1 and second.context.threads == 2
        first.close()
        assert not one.exists() and two.exists()
        assert result.optimal_energy == energy
        assert np.isfinite(result.forces()).all()
        with pytest.raises(RuntimeError, match="closed"):
            first.run(molecule())
        assert second.run(molecule()).success
    finally:
        first.close()
        second.close()
    assert not two.exists()


def test_geometry_and_snapshots_do_not_alias_inputs():
    positions = molecule().positions.copy()
    geometry = Geometry((1, 1), positions, np.eye(3) * 5)
    positions[:] = 0
    assert geometry.positions[0, 0] == 2.5
    with calculator() as calc:
        result = calc.run(geometry)
        snapshot = result.snapshot()
        snapshot["coefficients"][0][:] = 0
        assert np.linalg.norm(result.snapshot()["coefficients"][0]) > 1
        forces = calc.get_forces()
        forces[:] = 0
        assert np.linalg.norm(calc.get_forces()) > 1


@pytest.mark.parametrize(
    "method", ["vqe", "adapt-vqe", "subspace-vqe", "subspace-adapt-vqe"]
)
def test_native_quantum_methods_against_frozen_hamiltonian(method):
    with Mandacaru(
        method=method, load_hamiltonian=REFERENCE, resources={"threads": 1}
    ) as calc:
        result = calc.run()
        assert result.success
        assert result.optimal_energy == pytest.approx(-25.871895455446616, abs=1e-8)
        assert np.linalg.norm(
            result.state()[0] if result.state().ndim == 2 else result.state()
        ) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="mapping"):
            result.qubit_hamiltonian("parity")


def test_checkpoint_resume_uses_owned_native_snapshot(tmp_path):
    path = tmp_path / "state.json"
    with Mandacaru(
        method="adapt-vqe", load_hamiltonian=REFERENCE, checkpoint=path
    ) as calc:
        initial = calc.run().energy_hartree
    assert path.exists()
    with Mandacaru(method="adapt-vqe", load_hamiltonian=REFERENCE, resume=path) as calc:
        assert calc.run().energy_hartree == pytest.approx(initial, abs=1e-10)


def test_native_calculation_avoids_unrelated_integration_imports_and_scratch(tmp_path):
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','jax','qiskit','braket','cirq','matplotlib','pandas','threadpoolctl'}:
            raise AssertionError('optional import: '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from mandacaru import Mandacaru, Geometry
g=Geometry((1,1),np.array([[2.5,2.5,2.13],[2.5,2.5,2.87]]),np.eye(3)*5)
for basis in ['HAO','STO-3G']:
    with Mandacaru(method='dft',xc='pbe',basis=basis,grid_spacing=.5,resources={'threads':1}) as c:
        r=c.run(g)
        assert r.success and not r.scf.has_model
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / ".mandacaru").exists()


def typed_crystal():
    """Uniform gas in two orthonormal orbitals; no PAW setup dependency."""
    n = 64
    indices = np.indices((4, 4, 4))
    psi = np.stack([np.ones(n), np.exp(2j * np.pi * indices[0].ravel() / 4)]) / np.sqrt(
        n
    )
    wave = np.stack(np.meshgrid(*[np.fft.fftfreq(4) * 2 * np.pi] * 3, indexing="ij"))
    g2 = np.sum(wave * wave, axis=0).ravel()
    kernel = np.zeros(n)
    kernel[g2 > 0] = 4 * np.pi / g2[g2 > 0]
    for axis in range(3):
        wave[axis][indices[axis] == 2] = 0
    arrays = dict(
        core=np.zeros(n),
        initial=np.full(n, 2 / n),
        initial_m=np.zeros(n),
        initial_q=np.array([], complex),
        kernel=kernel,
        xc_wave=wave.ravel(),
        phase=np.ones(n, complex),
        kerker=np.ones(n),
        ion=np.zeros(n, complex),
        psi=psi.ravel(),
        overlap=np.eye(2, dtype=complex).ravel(),
        fixed=np.diag([-0.5, 0.3]).astype(complex).ravel(),
        projections=np.array([], complex),
    )
    for key in ("blocks", "hats", "u", "uion", "linear", "qsym"):
        arrays[key] = np.array([], complex)
    metadata = dict(
        schema_version=1,
        representation="periodic-paw-snapshot",
        energy_unit="Ha",
        length_unit="bohr",
        grid=dict(shape=[4] * 3, origin=[0.0] * 3, step=np.eye(3).ravel().tolist()),
        m=2,
        p=0,
        channels=0,
        channel_labels=[],
        kpoints=[
            dict(
                k=[0.0] * 3,
                fractional=[0.0] * 3,
                weight=1.0,
                **{key: key for key in ("psi", "overlap", "fixed", "projections")},
            )
        ],
        **{
            key: key
            for key in arrays
            if key not in ("psi", "overlap", "fixed", "projections")
        },
        eion=0.0,
        ion_constant=0.0,
        constant=0.13,
        reference=0.02,
        electrons=2.0,
        spins=1,
        functional="lda",
        relativistic=False,
        permutations=[],
    )
    return json.dumps(metadata), arrays


def test_periodic_typed_transfer_owns_buffers_and_roundtrips(tmp_path):
    metadata, arrays = typed_crystal()
    ctx = _native.Context(threads=1)
    problem = ctx.periodic_problem(metadata, arrays)
    arrays["fixed"][:] = 0
    arrays["initial"][:] = 0
    solved = problem.solve(width_ha=0.06)
    expected = solved.snapshot()
    assert expected["converged"]
    path = tmp_path / "crystal.json"
    problem.save(str(path))
    loaded = ctx.load_crystal(str(path)).solve(width_ha=0.06)
    assert loaded.snapshot()["energy"] == pytest.approx(expected["energy"], abs=1e-12)
    with Mandacaru(method="dft", xc="lda", load_crystal=path) as calc:
        result = calc.run()
        assert result.periodic is not None and result.n_spatial_orbitals == 2
        assert sum(result.num_particles) == pytest.approx(2.0)
    del problem, solved, loaded
    gc.collect()
    assert ctx.memory_used_bytes == 0


@pytest.mark.parametrize(
    "defect", ["dtype", "nan", "size", "missing", "extra", "stride"]
)
def test_periodic_array_validation_releases_all_allocations(defect):
    metadata, arrays = typed_crystal()
    if defect == "dtype":
        arrays["initial"] = arrays["initial"].astype(np.float32)
    if defect == "nan":
        arrays["initial"][0] = np.nan
    if defect == "size":
        arrays["initial"] = arrays["initial"][:-1]
    if defect == "missing":
        del arrays["initial"]
    if defect == "extra":
        arrays["unused"] = np.zeros(4)
    if defect == "stride":
        arrays["initial"] = np.zeros(128)[::2]
    ctx = _native.Context(threads=1)
    with pytest.raises(ValueError):
        ctx.periodic_problem(metadata, arrays)
    assert ctx.memory_used_bytes == 0


def test_periodic_transfer_honors_native_memory_budget():
    metadata, arrays = typed_crystal()
    ctx = _native.Context(memory_limit_bytes=32)
    with pytest.raises(MemoryError):
        ctx.periodic_problem(metadata, arrays)
    assert ctx.memory_used_bytes == 0
