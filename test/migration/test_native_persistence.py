"""Phase 7 schema compatibility, native restarts, streams and SDK boundaries."""
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru import Mandacaru, _native
from mandacaru.core.checkpoint import WavefunctionCheckpoint, prepare_state
from mandacaru.core.mapping import PauliSum
from mandacaru.core.serialization import load_hamiltonian, save_hamiltonian
from mandacaru.backends.measurement import (qwc_partition, qubit_wise_commuting_groups,
                                          energy_from_group_counts)
from mandacaru.backends.providers import build_provider
from mandacaru.utils.cube import write_volumetric
from mandacaru.utils.bxsf import write_bxsf, read_bxsf

H2 = Path(__file__).parent / "reference/h2_hamiltonian.json"
TAG = "mandacaru-wavefunction-checkpoint"


def record(preparation="product"):
    return WavefunctionCheckpoint(
        2, [0], [PauliSum({"XY": .3j, "YX": -.2j}, num_qubits=2),
                  PauliSum({"ZI": .7j, "IX": .2j}, num_qubits=2)],
        np.array([.31, -.47]), preparation=preparation,
        hamiltonian=PauliSum({"ZZ": .4, "XI": .2, "II": -.8}, num_qubits=2),
        metadata={"notes": "Unicode mandacaru 🌵", "nested": {"x": [1, None]}},
        status={"complete": False, "iteration": 2, "history": [-.3, -.4]})


@pytest.mark.parametrize("preparation", ["sum", "product"])
@pytest.mark.parametrize("writer", ["python", "rust"])
def test_checkpoint_interoperability_and_exact_state(tmp_path, preparation, writer):
    original = record(preparation)
    path = tmp_path / "checkpoints/state.json"
    original.save(path, engine=writer)
    for reader in ["python", "rust"]:
        loaded = WavefunctionCheckpoint.load(path, engine=reader)
        assert loaded.metadata == original.metadata
        assert loaded.status == original.status
        assert loaded.preparation == preparation
        seed = np.array([0, 0, 1, 0], dtype=complex)
        matrices = [g.to_matrix() for g in original.generators]
        if preparation == "sum":
            expected = expm(sum(t*g for t, g in zip(original.parameters, matrices))) @ seed
        else:
            expected = seed
            for t, g in zip(original.parameters, matrices):
                expected = expm(t*g) @ expected
        actual = loaded.state_vector(engine="rust")
        assert np.allclose(actual, expected, atol=2e-13, rtol=0)
        assert loaded.expectation(engine="rust") == pytest.approx(
            np.vdot(expected, loaded.hamiltonian.to_matrix() @ expected).real, abs=1e-13)


def test_old_checkpoint_version_and_native_qpe_preparation(tmp_path, monkeypatch):
    payload = record().to_payload()
    payload["version"] = 1
    payload.pop("preparation")
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload))
    loaded = WavefunctionCheckpoint.load(path, engine="rust")
    assert loaded.preparation == "product"
    expected = loaded.state_vector()
    def forbidden(*args, **kwargs):
        raise AssertionError("Python sparse preparation was used")
    monkeypatch.setattr(PauliSum, "to_sparse_matrix", forbidden)
    actual = loaded.state_vector(engine="rust")
    assert np.allclose(actual, expected)
    from mandacaru.algorithms.qpe import QuantumPhaseEstimation
    qpe = QuantumPhaseEstimation(loaded.hamiltonian, 4, engine="rust", verbose=False)
    result = qpe.run((loaded.n_qubits, loaded.reference_qubits,
                      loaded.generators, loaded.parameters))
    assert result.probabilities.sum() == pytest.approx(1.)


@pytest.mark.parametrize("field,value", [("version", 3), ("energy_unit", "eV"),
                                        ("n_qubits", 51), ("reference_qubits", [2]),
                                        ("preparation", "unknown"), ("metadata", [])])
def test_schema_rejections_preserve_previous_snapshot(tmp_path, field, value):
    ctx = _native.Context()
    path = tmp_path / "state.json"
    good = record().to_payload()
    ctx.parse_snapshot(json.dumps(good), TAG).save(str(path))
    previous = path.read_bytes()
    broken = dict(good, **{field: value})
    with pytest.raises(ValueError):
        ctx.parse_snapshot(json.dumps(broken), TAG).save(str(path))
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
    assert ctx.memory_used_bytes == 0


@pytest.mark.parametrize("format", ["json", "parquet"])
@pytest.mark.parametrize("writer", ["python", "rust"])
def test_hamiltonian_cross_readers_and_native_solver(tmp_path, format, writer):
    original = load_hamiltonian(H2)
    path = tmp_path / f"h2.{format}"
    metadata = {"note": "migrated 🌵", "extra": {"values": [1, 2]}}
    save_hamiltonian(path, original.hamiltonian, mapping=original.mapping,
                     num_particles=original.num_particles,
                     n_spatial_orbitals=original.n_spatial_orbitals,
                     metadata=metadata, engine="rust" if writer == "rust" else "auto")
    for reader in ["auto", "rust"]:
        loaded = load_hamiltonian(path, engine=reader)
        assert loaded.hamiltonian.terms == original.hamiltonian.terms
        assert loaded.metadata == {"energy_unit": "Ha", **metadata}
    result = Mandacaru(method="vqe", engine="rust", load_hamiltonian=str(path),
                       sparse=True, trace=False, references=False).run()
    assert result.success
    assert result.optimal_energy == pytest.approx(-25.8718954554466, abs=1e-8)
    assert result.timings["cache_engine"] == (
        "native-json" if format == "json" else "python-parquet-adapter")


@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
def test_native_public_checkpoint_export_and_warm_restart(tmp_path, method):
    path = tmp_path / "saved.json"
    options = dict(method=method, engine="rust", load_hamiltonian=str(H2),
                   trace=False, references=False)
    calc = Mandacaru(**options, checkpoint=path)
    result = calc.run()
    assert path.exists()
    checkpoint = calc.solver.checkpoint
    assert checkpoint.method == method
    assert checkpoint.expectation(engine="rust") * 27.211386245988 == pytest.approx(result.optimal_energy)
    resumed = Mandacaru(**options, resume=path).run()
    assert resumed.optimal_energy == pytest.approx(result.optimal_energy, abs=1e-10)
    assert resumed.converged if method == "adapt-vqe" else resumed.success
    # Existing Python consumers can read and prepare the native file.
    legacy = WavefunctionCheckpoint.load(path)
    assert np.allclose(legacy.state_vector(), checkpoint.state_vector(engine="rust"), atol=1e-12)
    legacy_options = {**options, "engine": "python", "resume": path}
    if method == "adapt-vqe":
        legacy_options["pool"] = "fermionic"
    legacy_result = Mandacaru(**legacy_options).run()
    assert legacy_result.optimal_energy == pytest.approx(result.optimal_energy, abs=1e-8)
    # A Python-written copy is also a valid native warm start.
    legacy.save(path)
    assert Mandacaru(**options, resume=path).run().optimal_energy == pytest.approx(result.optimal_energy, abs=1e-10)


@pytest.mark.parametrize("mutation", ["hamiltonian", "mapping", "preparation", "reference_qubits"])
def test_native_restart_refuses_other_problem_or_state(tmp_path, mutation):
    path = tmp_path / "checkpoint.json"
    options = dict(method="vqe", engine="rust", load_hamiltonian=str(H2),
                   trace=False, references=False)
    Mandacaru(**options, checkpoint=path).run()
    p = json.loads(path.read_text())
    if mutation == "hamiltonian":
        p["hamiltonian"]["terms"][0][1] += .01
    elif mutation == "mapping":
        p["mapping"] = "parity"
    elif mutation == "preparation":
        p["preparation"] = "product"
    else:
        p["reference_qubits"] = []
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        Mandacaru(**options, resume=path).run()


@pytest.mark.parametrize("width", [1, 4, 65, 137])
def test_native_qwc_matches_greedy_policy_at_arbitrary_width(width, monkeypatch):
    rng = np.random.default_rng(width)
    labels = ["".join(row) for row in rng.choice(list("IXYZ"), size=(70, width))]
    weights = rng.integers(0, 5, size=len(labels)).astype(float)
    expected = qwc_partition(labels, weights)
    import mandacaru.core.pauli_algebra as algebra
    def forbidden(*args, **kwargs):
        raise AssertionError("Python grouping was used")
    monkeypatch.setattr(algebra, "pack_labels", forbidden)
    actual = qwc_partition(labels, weights, engine="rust")
    assert actual == expected
    assert sorted(i for basis, members in actual for i in members) == list(range(len(labels)))


def test_counts_and_sdk_gate_streams_use_native_core(monkeypatch):
    h = PauliSum({"IX": .3, "ZX": -.4, "II": -.7, "YI": .2}, num_qubits=2)
    groups, identity = qubit_wise_commuting_groups(h, engine="rust")
    counts = [{"00": 27, "01": 13, "10": 7, "11": 3} for _ in groups]
    expected = energy_from_group_counts(groups, identity, counts)
    actual = energy_from_group_counts(groups, identity, counts, engine="rust")
    assert actual == pytest.approx(expected, abs=1e-14)
    generators = [PauliSum({"XY": .5j, "YX": -.5j}, num_qubits=2)]
    expected = list(build_provider("qiskit")._emit(2, [0], generators, [.37]))
    import mandacaru.backends.providers as providers
    def forbidden(*args, **kwargs):
        raise AssertionError("Python generator compilation was used")
    monkeypatch.setattr(providers, "pauli_rotations", forbidden)
    for name in ["qiskit", "braket", "cirq"]:
        native = build_provider(name, engine="rust", engine_options={"threads": 1})
        assert list(native._emit(2, [0], generators, [.37])) == expected
        state = native.statevector(2, [0], generators, [.37])
        exact = prepare_state(2, [0], generators, [.37], engine="rust")
        assert np.allclose(state, exact, atol=1e-12)


def test_invalid_circuit_and_counts_rejected():
    ctx = _native.Context()
    assert ctx.qwc_groups(["XX", "ZZ"], [-0.0, 0.0]) == qwc_partition(["XX", "ZZ"], [-0.0, 0.0])
    for generator in [[("X", 0., .3), ("Z", 0., .2)], [("Y", .3, 0.)]]:
        with pytest.raises(ValueError):
            ctx.compile_circuit(1, [], [generator], [.3])
    for counts in [[("0", -1.)], [("2", 1.)], [("00", 1.)], [("0", float("nan"))]]:
        with pytest.raises(ValueError):
            ctx.count_expectations(["Z"], counts)
    with pytest.raises(ValueError):
        ctx.qwc_groups(["X", "ZZ"])
    assert ctx.memory_used_bytes == 0


def test_ledger_atomic_budget_failures_and_concurrent_completions():
    ctx = _native.Context(threads=1)
    ledger = ctx.run_ledger(jobs=8, total_shots=800)
    tickets = ledger.reserve(4, 400)
    before = ledger.usage
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve(5, 500)
    assert ledger.usage == before
    more = ledger.reserve(4, 400)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda ticket: ledger.finish(ticket, ticket % 2 == 0), tickets+more))
    assert ledger.usage == dict(attempted_jobs=8, completed_jobs=4, failed_jobs=4, total_shots=800)
    with pytest.raises(ValueError):
        ledger.finish(0, True)
    del ledger
    gc.collect()
    assert ctx.memory_used_bytes == 0


@pytest.mark.parametrize("format", ["cube", "xsf"])
@pytest.mark.parametrize("periodic", [False, True])
def test_streamed_scalar_field_with_independent_ase_reader(tmp_path, format, periodic):
    data = np.arange(24).reshape(2, 3, 4) / 7.
    step = np.array([[.2, .01, .03], [0., .3, .02], [0., 0., .4]])
    origin = np.array([-.1, .5, .2])
    cell = (step @ np.diag(data.shape)).T
    path = tmp_path / f"field.{format}"
    write_volumetric(path, data, origin, step, [1], [[.2, .3, .4]],
                     cell=cell if periodic else None, periodic=periodic, engine="rust")
    if format == "cube":
        from ase.io.cube import read_cube
        with path.open() as handle:
            read = read_cube(handle)
        assert np.allclose(read["data"], data, atol=6e-6)
        assert np.allclose(read["origin"], origin * .529177210903, atol=1e-7)
    else:
        from ase.io.xsf import read_xsf
        with path.open() as handle:
            loaded, grid_origin, spans, atoms = read_xsf(handle, read_data=True)
        expected = np.pad(data, ((0, 1),)*3, mode="wrap") if periodic else data
        assert np.allclose(loaded, expected, atol=6e-7)


def test_band_stream_closing_planes_and_small_memory(tmp_path):
    bands = np.arange(48, dtype=float).reshape(2, 2, 3, 4) / 11
    cell = np.array([[1., .2, .1], [0., 1.1, .3], [0., 0., .9]])
    path = tmp_path / "bands.bxsf"
    write_bxsf(path, bands, cell, .73, engine="rust", engine_options={"memory_limit_bytes": 16384})
    loaded = read_bxsf(path)
    assert np.allclose(loaded.energies, bands, atol=6e-7)
    assert np.allclose(loaded.reciprocal_cell, cell)
    assert loaded.fermi_level == pytest.approx(.73)
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        write_bxsf(path, bands * np.nan, cell, .73, engine="rust")
    assert path.read_bytes() == previous


def test_document_memory_and_failed_atomic_replace(tmp_path):
    ctx = _native.Context(memory_limit_bytes=1 << 20)
    doc = ctx.parse_snapshot(json.dumps(record().to_payload()), TAG)
    used = ctx.memory_used_bytes
    with pytest.raises(MemoryError):
        ctx.parse_snapshot(" "*(1 << 20), TAG)
    assert ctx.memory_used_bytes == used
    destination = tmp_path / "directory"
    destination.mkdir()
    with pytest.raises(ValueError):
        doc.save(str(destination))
    assert destination.is_dir()
    assert not list(tmp_path.glob("*.tmp"))
    vector = doc.state()
    del doc
    gc.collect()
    assert ctx.memory_used_bytes == 0
    assert np.linalg.norm(vector) == pytest.approx(1.)
