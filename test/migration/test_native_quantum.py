"""Phase 2 scientific comparisons, including independent dense exponentials."""
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru import Mandacaru, _native
from mandacaru.core.mapping import Fermion, PauliSum, reference_qubit_bits
from mandacaru.core.sector import ParticleSector
from mandacaru.core.serialization import save_hamiltonian
from mandacaru.units import HARTREE_TO_EV

FIXTURE = Path(__file__).parent / "reference/h2_hamiltonian.json"
MAPPINGS = ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"]


def triples(operator):
    return [(p, c.real, c.imag) for p, c in operator.terms.items()]


def py_operator(native):
    return PauliSum({p: complex(re, im) for p, re, im in native.terms}, num_qubits=native.num_qubits)


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_complex_fermion_algebra_and_encoding(mapping):
    ctx = _native.Context()
    terms = {((0, True), (1, False)): .2 + .7j,
             ((1, True), (0, False)): .2 - .7j,
             ((4, True), (5, False)): -.3j,
             ((5, True), (4, False)): .3j,
             ((0, True), (3, True), (4, False), (1, False)): .31}
    f = Fermion(terms, n_modes=6)
    native = ctx.fermion_operator(6, [1, 1], [(list(t), c.real, c.imag) for t, c in terms.items()], mapping=mapping)
    expected = f.map_to_qubits(mapping, n_modes=6, num_particles=(1, 1))
    assert np.allclose(native.matrix(), expected.to_matrix(), atol=1e-13)
    assert np.allclose(native.dagger().matrix(), expected.to_matrix().conj().T, atol=1e-13)
    jw = f.map_to_qubits("jordan_wigner", n_modes=6)
    encoded = ctx.pauli_operator(6, triples(jw)).map_to(mapping, [1, 1])
    assert np.allclose(encoded.matrix(), expected.to_matrix(), atol=1e-13)


def test_pauli_products_commutators_and_matrix_free_action():
    rng = np.random.default_rng(1729)
    ctx = _native.Context()
    a = PauliSum({"YZX": .7 + .2j, "XIY": -.3j, "III": .2}, num_qubits=3)
    b = PauliSum({"XYZ": -.3 + .4j, "ZIZ": .9}, num_qubits=3)
    native_a = ctx.pauli_operator(3, triples(a))
    native_b = ctx.pauli_operator(3, triples(b))
    aa, bb = a.to_matrix(), b.to_matrix()
    assert np.allclose(native_a.compose(native_b).matrix(), aa @ bb)
    assert np.allclose(native_a.commutator(native_b).matrix(), aa @ bb - bb @ aa)
    assert np.allclose(native_a.add(native_b, .2, -.8).matrix(), aa + (.2 - .8j) * bb)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    assert np.allclose(native_a.apply(state), aa @ state)
    with pytest.raises(ValueError, match="contiguous"):
        native_a.apply(np.zeros(16, dtype=complex)[::2])
    with pytest.raises(ValueError, match="state"):
        native_a.apply(np.zeros(7, dtype=complex))


@pytest.mark.parametrize("mapping", MAPPINGS[:3])
def test_fermion_anticommutation(mapping):
    ctx = _native.Context()
    ops = [ctx.fermion_operator(4, [1, 1], [([(p, False)], 1., 0.)], mapping=mapping) for p in range(4)]
    for p in range(4):
        for q in range(4):
            anti = ops[p].compose(ops[q].dagger()).add(ops[q].dagger().compose(ops[p]))
            assert np.allclose(anti.matrix(), np.eye(16) * (p == q), atol=1e-13)


def cache(tmp_path, mapping, modes=6):
    """Complex, spin-conserving Hamiltonian, independent of the H₂ fixture."""
    terms = {((j, True), (j, False)): float(j) * .13 for j in range(modes)}
    terms.update({((0, True), (1, False)): .17 + .21j,
                  ((1, True), (0, False)): .17 - .21j,
                  ((modes//2, True), (modes//2+1, False)): .11,
                  ((modes//2+1, True), (modes//2, False)): .11})
    f = Fermion(terms, n_modes=modes)
    h = f.map_to_qubits(mapping, n_modes=modes, num_particles=(1, 1))
    path = tmp_path / f"{mapping}.json"
    save_hamiltonian(path, h, mapping=mapping, num_particles=(1, 1), n_spatial_orbitals=modes//2)
    return path, h


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("sector", [False, True])
@pytest.mark.parametrize("trotter", [False, True])
def test_states_and_exact_gradients_against_dense_exponential(tmp_path, mapping, sector, trotter):
    path, h = cache(tmp_path, mapping)
    ctx = _native.Context()
    problem = ctx.load_hamiltonian(str(path), sector=sector)
    if sector:
        expected_indices = ParticleSector(h.num_qubits, (1, 1), mapping=mapping).indices
        assert np.array_equal(problem.basis_indices, expected_indices)
    matrices = [PauliSum({p: complex(re, im) for p, re, im in terms}, num_qubits=h.num_qubits).to_matrix()
                for terms in problem.pool_terms]
    ix = np.asarray(problem.basis_indices)
    matrices = [g[np.ix_(ix, ix)] for g in matrices]
    # Four singles and four doubles have nonzero, noncommuting amplitudes.
    theta = np.random.default_rng(56).normal(scale=.35, size=len(matrices))
    from mandacaru.core.mapping import reference_qubit_bits
    bits=reference_qubit_bits(mapping,6,[0,3])
    encoded=sum(int(b)<<(len(bits)-1-i) for i,b in enumerate(bits))
    hf=np.zeros(problem.dimension,dtype=complex)
    hf[np.flatnonzero(ix==encoded)[0]]=1.
    if trotter:
        expected = hf.copy()
        for t, g in zip(theta, matrices):
            expected = expm(t * g) @ expected
    else:
        expected = expm(sum(t * g for t, g in zip(theta, matrices))) @ hf
    value = problem.evaluate(theta, trotter=trotter)
    assert np.allclose(value["state"], expected, atol=3e-13)
    hh = h.to_matrix()[np.ix_(ix, ix)]
    assert value["energy_ha"] == pytest.approx(np.vdot(expected, hh @ expected).real, abs=2e-13)
    for k in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += 1e-5
        minus[k] -= 1e-5
        derivative = (problem.evaluate(plus, trotter=trotter)["energy_ha"]
                      - problem.evaluate(minus, trotter=trotter)["energy_ha"]) / 2e-5
        assert value["gradient"][k] == pytest.approx(derivative, abs=1e-9)
    # Full and reduced calculations must give the same RDMs in occupation space.
    one, two = problem.rdms(value["state"])
    assert np.trace(one).real == pytest.approx(2., abs=2e-12)
    assert np.allclose(one, one.conj().T, atol=2e-13)
    assert np.allclose(two, -two.swapaxes(0, 1), atol=2e-13)
    assert np.allclose(np.einsum("pqrq->pr", two), one, atol=2e-13)
    if mapping != "parity_reduced":
        from mandacaru.algorithms.rdm import one_rdm, two_rdm
        py_sector = ParticleSector(h.num_qubits, (1, 1), mapping=mapping) if sector else None
        assert np.allclose(one, one_rdm(expected, 6, mapping, sector=py_sector), atol=2e-13)
        assert np.allclose(two, two_rdm(expected, 6, mapping, sector=py_sector), atol=2e-13)


@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
@pytest.mark.parametrize("sector", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("optimizer", ["BFGS", "L-BFGS"])
def test_public_h2_solve_matches_phase0(method, sector, sparse, optimizer):
    calc = Mandacaru(method=method,  load_hamiltonian=FIXTURE,
                    sector=sector, sparse=sparse, trace=False,
                    resources={"threads": 1}, optimizer={"method": optimizer, "tol": 1e-12})
    result = calc.run()
    assert (result.success if method == "vqe" else result.converged)
    assert result.optimal_energy == pytest.approx(-25.871895455446612, abs=1e-9)
    assert result.reference_energy == pytest.approx(-25.5004433585557, abs=1e-9)
    assert calc.problem.num_qubits == 4 and result.num_particles == (1, 1)
    state = result.state()
    assert np.vdot(state, state).real == pytest.approx(1., abs=1e-13)
    key = "h2_cached_adapt" if sector else "h2_cached_vqe"
    frozen = json.loads((FIXTURE.parent / "numpy.json").read_text())["cases"][key]["samples"][0]["values"]["state_projector"]
    assert np.allclose(np.outer(state, state.conj()), np.asarray(frozen["real"]) + 1j * np.asarray(frozen["imag"]), atol=1e-8)
    if method == "adapt-vqe":
        assert result.operators == ["D(0,2->1,3)"]
        assert result.final_max_gradient < 1e-10
    one, two = calc.problem.rdms(state)
    assert np.trace(one).real == pytest.approx(2.)
    assert result.optimizer_steps > 0 and result.num_evaluations > result.optimizer_steps


def test_entire_solve_has_no_python_scientific_callbacks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Python scientific computation entered a native solve")
    from mandacaru.optimizers.optim import Optimizer
    from mandacaru.core import serialization
    from mandacaru.circuits.ansatz import UCCSD
    monkeypatch.setattr(Optimizer, "minimize", forbidden)
    monkeypatch.setattr(serialization, "load_hamiltonian", forbidden)
    monkeypatch.setattr(UCCSD, "state", forbidden)
    monkeypatch.setattr(UCCSD, "__init__", forbidden)
    for method in ("vqe", "adapt-vqe"):
        calc = Mandacaru(method=method,  load_hamiltonian=FIXTURE, trace=False)
        assert calc.run().optimal_energy < -25.8


def test_failure_status_units_and_log(tmp_path):
    calc = Mandacaru(method="vqe",  load_hamiltonian=FIXTURE,
                    atomic_units=True, trace=False, txt=tmp_path / "run.txt",
                    optimizer={"method": "BFGS", "maxiter": 0})
    result = calc.run()
    assert not result.success and result.optimizer_steps == 0
    assert result.num_evaluations == 1 and result.energy_unit == "Ha"
    assert result.in_units("eV") == pytest.approx(result.optimal_energy * HARTREE_TO_EV)
    assert "iteration budget exhausted" in (tmp_path / "run.txt").read_text()
    problem = _native.Context().load_hamiltonian(str(FIXTURE))
    exhausted = problem.solve(max_evaluations=1)
    assert not exhausted["success"] and exhausted["message"] == "evaluation budget exhausted"
    assert not problem.solve(adapt=True, max_growth=0)["success"]
    failed = problem.solve(adapt=True, max_iterations=0)
    assert not failed["success"] and not failed["iterations"][0][5]
    assert failed["final_max_gradient"] > 1e-3
    with pytest.raises(ValueError, match="parameter"):
        problem.solve(initial_parameters=[0.])
    with pytest.raises(ValueError, match="finite"):
        problem.solve(initial_parameters=[float("nan")] * 3)


def test_budgets_ownership_and_concurrent_contexts():
    small = _native.Context(memory_limit_bytes=1024)
    with pytest.raises(MemoryError):
        small.load_hamiltonian(str(FIXTURE))
    assert small.memory_used_bytes == 0
    ctx = _native.Context(threads=1)
    problem = ctx.load_hamiltonian(str(FIXTURE))
    live = ctx.memory_used_bytes
    with pytest.raises(MemoryError):
        problem.solve(max_evaluations=10**9)
    assert ctx.memory_used_bytes == live
    def run(_):
        return problem.solve()["state"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(run, range(8)))
    assert ctx.memory_used_bytes == live
    del problem
    gc.collect()
    assert ctx.memory_used_bytes == 0
    assert all(np.allclose(s, states[0]) for s in states)
    states[0][:] = 0
    assert np.linalg.norm(states[1]) == pytest.approx(1.)
    with pytest.raises(OverflowError):
        ctx.pauli_operator(60, [("I" * 60, 1., 0.)]).matrix()
    with pytest.raises(MemoryError):
        ctx.pauli_operator(40, [("I" * 40, 1., 0.)]).matrix()


@pytest.mark.parametrize("field,value", [("version", 2), ("num_qubits", 5), ("n_spatial_orbitals", 3),
                                         ("mapping", "other"), ("num_particles", [4, 1]),
                                         ("terms", [["X", 1., 0.]]), ("terms", [["IIII", 0., 1.]]),
                                         ("metadata", {"energy_unit": "eV"})])
def test_bad_cache_rejected_without_leaking_budget(tmp_path, field, value):
    payload = json.loads(FIXTURE.read_text())
    payload[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    ctx = _native.Context()
    with pytest.raises(ValueError):
        ctx.load_hamiltonian(str(path))
    assert ctx.memory_used_bytes == 0


def test_nonconserving_hamiltonian_and_unsupported_options_fail_explicitly(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    payload["terms"] = [["XIII", 1., 0.]]
    path = tmp_path / "leak.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sector"):
        _native.Context().load_hamiltonian(str(path))
    assert _native.Context().load_hamiltonian(str(path), sector=False).dimension == 16
    for options in ({"optimizer": "SLSQP"}, {"ansatz": "hva"}):
        with pytest.raises((NotImplementedError,ValueError)):
            Mandacaru(method="vqe",  load_hamiltonian=FIXTURE, **options)
    Mandacaru(method="dft",  xc="r2scan")
    with pytest.raises(TypeError, match="tetris"):
        Mandacaru(method="adapt-vqe",  load_hamiltonian=FIXTURE, tetris=True)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("sector", [False, True])
def test_rdms_of_complex_states_are_encoding_independent(tmp_path, mapping, sector):
    from mandacaru.algorithms.rdm import one_rdm, two_rdm
    path, _ = cache(tmp_path, mapping)
    problem = _native.Context().load_hamiltonian(str(path), sector=sector)
    jw_sector = ParticleSector(6, (1, 1))
    amplitudes = np.random.default_rng(99).normal(size=(2, jw_sector.dim))
    amplitudes = amplitudes[0] + 1j * amplitudes[1]
    amplitudes /= np.linalg.norm(amplitudes)
    encoded = np.zeros(problem.dimension, dtype=complex)
    indices = np.asarray(problem.basis_indices)
    for occupation, amplitude in zip(jw_sector.indices, amplitudes):
        occupied = [j for j in range(6) if int(occupation) & (1 << (5 - j))]
        bits = reference_qubit_bits(mapping, 6, occupied)
        index = sum(int(b) << (len(bits) - 1 - j) for j, b in enumerate(bits))
        encoded[np.searchsorted(indices, index)] = amplitude
    one, two = problem.rdms(encoded)
    assert np.max(np.abs(one.imag)) > .01
    assert np.allclose(one, one_rdm(amplitudes, 6, sector=jw_sector), atol=1e-13)
    assert np.allclose(two, two_rdm(amplitudes, 6, sector=jw_sector), atol=1e-13)
