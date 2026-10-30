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


def annihilate(mode, vector):
    """Apply a_mode to {occupied-mode tuple: amplitude}; Jordan-Wigner sign of ascending order."""
    out = {}
    for occupied, amplitude in vector.items():
        if mode in occupied:
            sign = -1. if occupied.index(mode) % 2 else 1.
            key = tuple(m for m in occupied if m != mode)
            out[key] = out.get(key, 0.) + sign * amplitude
    return out


def overlap(left, right):
    return sum(np.conj(a) * right[k] for k, a in left.items() if k in right)


def determinant_rdms(amplitudes, n_modes):
    """Reference RDMs from fermionic ladder algebra on Slater determinants.

    ``amplitudes`` maps a tuple of occupied spin-orbitals (ascending) to its
    amplitude.  gamma_pq = <a_p psi|a_q psi> and Gamma_pqrs = <a_q a_p psi|a_s a_r psi>
    are evaluated without any qubit encoding, so they are independent of the
    mappings, of the native RDM code and of the removed Python RDM module.
    """
    psi = dict(amplitudes)
    single = [annihilate(p, psi) for p in range(n_modes)]
    one = np.array([[overlap(single[p], single[q]) for q in range(n_modes)] for p in range(n_modes)])
    pair = {(p, q): annihilate(q, single[p]) for p in range(n_modes) for q in range(n_modes)}
    two = np.array([[[[overlap(pair[p, q], pair[r, s]) for s in range(n_modes)]
                      for r in range(n_modes)] for q in range(n_modes)] for p in range(n_modes)])
    return one, two


def encoded_index(mapping, occupied, n_modes=6):
    bits = reference_qubit_bits(mapping, n_modes, list(occupied))
    return sum(int(b) << (len(bits) - 1 - i) for i, b in enumerate(bits))


def sector_determinants():
    """Determinants of the (1, 1) sector, in the ParticleSector(6, (1, 1)) order."""
    sector = ParticleSector(6, (1, 1))
    return [tuple(j for j in range(6) if int(occupation) & (1 << (5 - j))) for occupation in sector.indices]


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
    encoded = encoded_index(mapping, [0, 3])
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
    # Reference RDMs: decode the encoded state into determinant amplitudes and
    # apply the ladder algebra directly (no encoding, no native RDM code).
    determinants = sector_determinants()
    decoded = {}
    for det in determinants:
        decoded[det] = expected[np.flatnonzero(ix == encoded_index(mapping, det))[0]]
    assert sum(abs(a)**2 for a in decoded.values()) == pytest.approx(1., abs=2e-12)
    reference_one, reference_two = determinant_rdms(decoded, 6)
    assert np.allclose(one, reference_one, atol=2e-13)
    assert np.allclose(two, reference_two, atol=2e-13)


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
    # state() is the full register; a sector solve lives on basis_indices only.
    full = result.state()
    basis = np.asarray(calc.problem.basis_indices, dtype=int)
    state = full[basis]
    assert np.vdot(full, full).real == pytest.approx(1., abs=1e-13)
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
    import scipy.optimize
    from mandacaru.optimizers.optim import Optimizer
    from mandacaru.core import serialization
    monkeypatch.setattr(Optimizer, "minimize", forbidden)
    monkeypatch.setattr(serialization, "load_hamiltonian", forbidden)
    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    monkeypatch.setattr(Fermion, "map_to_qubits", forbidden)
    monkeypatch.setattr(PauliSum, "to_matrix", forbidden)
    monkeypatch.setattr(PauliSum, "to_sparse_matrix", forbidden)
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
    # The run log owns the converged flag and the effort; the termination
    # reason is carried by the result.
    assert result.timings["termination"] == "iteration budget exhausted"
    log = (tmp_path / "run.txt").read_text()
    assert "converged: False" in log and "optimizer_steps: 0" in log
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
    # SLSQP is a native optimizer (the public default), so it is no longer an
    # unsupported option; Adam and L-BFGS-B were removed from the public set.
    for name in ("Adam", "L-BFGS-B"):
        with pytest.raises(ValueError, match="native optimizer"):
            Mandacaru(method="vqe",  load_hamiltonian=FIXTURE, optimizer=name)
    # HVA needs the physical decomposition of a molecular Hamiltonian, which a
    # cached Hamiltonian does not have: the solve raises, it never substitutes.
    with pytest.raises(ValueError, match="HVA"):
        Mandacaru(method="vqe",  load_hamiltonian=FIXTURE, ansatz="hva", trace=False).run()
    Mandacaru(method="dft",  xc="r2scan")
    with pytest.raises(TypeError, match="tetris"):
        Mandacaru(method="adapt-vqe",  load_hamiltonian=FIXTURE, tetris=True)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("sector", [False, True])
def test_rdms_of_complex_states_are_encoding_independent(tmp_path, mapping, sector):
    path, _ = cache(tmp_path, mapping)
    problem = _native.Context().load_hamiltonian(str(path), sector=sector)
    jw_sector = ParticleSector(6, (1, 1))
    determinants = sector_determinants()
    amplitudes = np.random.default_rng(99).normal(size=(2, jw_sector.dim))
    amplitudes = amplitudes[0] + 1j * amplitudes[1]
    amplitudes /= np.linalg.norm(amplitudes)
    encoded = np.zeros(problem.dimension, dtype=complex)
    indices = np.asarray(problem.basis_indices)
    for occupied, amplitude in zip(determinants, amplitudes):
        index = encoded_index(mapping, occupied)
        encoded[np.searchsorted(indices, index)] = amplitude
    one, two = problem.rdms(encoded)
    reference_one, reference_two = determinant_rdms(dict(zip(determinants, amplitudes)), 6)
    assert np.max(np.abs(one.imag)) > .01
    assert np.allclose(one, reference_one, atol=1e-13)
    assert np.allclose(two, reference_two, atol=1e-13)


# -- complex64 states (precision="float32") --------------------------------------------------
# Hamiltonian coefficients, parameters and every reduction stay float64; only the stored state is
# complex64. The pairs below use ONE Hamiltonian in both precisions, so they isolate the state.

def hubbard_problem(ctx, n, precision, *, sector=True):
    """Half-filled Hubbard chain (t=1, U=2), 2n qubits, fermionic pool."""
    terms = []
    for spin in range(2):
        for i in range(n - 1):
            a, b = i + spin * n, i + 1 + spin * n
            terms += [([(a, True), (b, False)], -1., 0.), ([(b, True), (a, False)], -1., 0.)]
    terms += [([(i, True), (i, False), (i + n, True), (i + n, False)], 2., 0.) for i in range(n)]
    operator = ctx.fermion_operator(2 * n, [n // 2, n // 2], terms, mapping="jordan_wigner")
    return ctx.quantum_problem(operator, n, [n // 2, n // 2], mapping="jordan_wigner", sector=sector,
                               include_singles=True, precision=precision)


def test_float32_problem_stores_complex64_and_refuses_the_other_dtype():
    ctx = _native.Context()
    low, high = hubbard_problem(ctx, 4, "float32"), hubbard_problem(ctx, 4, "float64")
    assert (low.precision, high.precision) == ("float32", "float64")
    theta = np.random.default_rng(3).normal(scale=.3, size=6)
    a, b = high.evaluate(theta, indices=list(range(6))), low.evaluate(theta, indices=list(range(6)))
    assert a["state"].dtype == np.complex128 and b["state"].dtype == np.complex64
    assert b["energy_ha"] == pytest.approx(a["energy_ha"], abs=2e-6)
    assert np.allclose(b["gradient"], a["gradient"], atol=2e-6)
    assert np.allclose(b["state"], a["state"], atol=1e-6)
    # The full-register state keeps its dtype through expand_state and state().
    assert low.state(theta, indices=list(range(6))).dtype == np.complex64
    assert low.expand_state(b["state"]).dtype == np.complex64
    # RDMs are float64 results; their input must be the problem's dtype, never cast silently.
    one, two = low.rdms(b["state"])
    reference_one, reference_two = high.rdms(a["state"])
    assert one.dtype == np.complex128 and np.allclose(one, reference_one, atol=2e-6)
    assert np.allclose(two, reference_two, atol=2e-6)
    with pytest.raises(TypeError, match="complex128.*precision='float32'.*complex64"):
        low.rdms(a["state"])
    with pytest.raises(TypeError, match="complex64.*precision='float64'.*complex128"):
        high.rdms(b["state"])
    with pytest.raises(TypeError, match="precision='float32'"):
        low.expand_state(a["state"])
    with pytest.raises(ValueError, match="precision"):
        hubbard_problem(ctx, 4, "single")


@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
def test_public_h2_float32_solve_matches_the_float64_energy(method):
    energies = {}
    for precision in ("float64", "float32"):
        calc = Mandacaru(method=method, load_hamiltonian=FIXTURE, precision=precision, trace=False,
                         resources={"threads": 1}, optimizer={"method": "BFGS", "tol": 1e-6},
                         convergence={"gradient": 1e-5, "energy": 1e-4})
        result = calc.run()
        assert (result.success if method == "vqe" else result.converged)
        energies[precision] = result.optimal_energy
        assert result.state().dtype == (np.complex64 if precision == "float32" else np.complex128)
    assert energies["float32"] == pytest.approx(energies["float64"], abs=5e-4)
    assert energies["float32"] == pytest.approx(-25.871895455446612, abs=1e-6)


def test_float32_refuses_settings_its_state_cannot_resolve():
    ctx = _native.Context()
    low, high = hubbard_problem(ctx, 3, "float32"), hubbard_problem(ctx, 3, "float64")
    floor = _native.FLOAT32_ENERGY_FLOOR_HA
    with pytest.raises(ValueError, match=r"float32 state floor"):
        low.solve(tolerance=floor / 10)
    with pytest.raises(ValueError, match="float32 state floor"):
        low.solve(tolerance=None)
    with pytest.raises(ValueError, match="gradient threshold.*float32 state floor"):
        low.solve(adapt=True, gradient_tolerance=_native.FLOAT32_GRADIENT_FLOOR / 10, tolerance=floor)
    with pytest.raises(ValueError, match="energy threshold.*float32 state floor"):
        low.solve(adapt=True, gradient_tolerance=1e-3, energy_tolerance=floor / 10, tolerance=floor)
    with pytest.raises(ValueError, match="float32 state floor"):
        low.energy_levels(tolerance=floor / 10)
    with pytest.raises(ValueError, match="float32 state floor"):
        low.solve_subspace([2., 1.], tolerance=floor / 10)
    # The same settings are legal in float64, and at the floor in float32.
    assert high.solve(tolerance=floor / 10)["success"] in (True, False)
    assert low.solve(tolerance=floor, max_iterations=5)["state"].dtype == np.complex64


def test_float32_adapt_and_subspace_reach_the_float64_energy_on_one_hamiltonian():
    ctx = _native.Context()
    options = dict(adapt=True, trotter=True, max_growth=6, gradient_tolerance=1e-4, tolerance=1e-6)
    low, high = hubbard_problem(ctx, 4, "float32"), hubbard_problem(ctx, 4, "float64")
    a, b = high.solve(**options), low.solve(**options)
    # The chain's pool has exactly degenerate gradients: rounding may pick another member of a
    # tie, so the energy and the length of the ansatz are compared, not its labels.
    assert b["state"].dtype == np.complex64 and len(b["indices"]) == len(a["indices"])
    assert b["energy_ha"] == pytest.approx(a["energy_ha"], abs=5e-6)
    assert b["final_max_gradient"] == pytest.approx(a["final_max_gradient"], rel=1e-4)
    small_low, small_high = hubbard_problem(ctx, 3, "float32"), hubbard_problem(ctx, 3, "float64")
    sub = dict(weights=[2., 1.], adapt=False, trotter=True, tolerance=1e-6, max_iterations=300)
    s64, s32 = small_high.solve_subspace(**sub), small_low.solve_subspace(**sub)
    assert s32["states"].dtype == np.complex64
    assert np.allclose(s32["energies_ha"], s64["energies_ha"], atol=5e-5)
    levels = small_low.energy_levels(num_states=2, restarts=1, tolerance=1e-6, optimizer="BFGS")
    assert levels["states"].dtype == np.complex64
    reference = small_high.energy_levels(num_states=2, restarts=1)
    assert levels["energies_ha"][0] == pytest.approx(reference["energies_ha"][0], abs=2e-4)


def test_propagator_and_qpe_follow_their_declared_precision():
    ctx = _native.Context()
    terms = [("XXI", .7, 0.), ("IYY", -.4, 0.), ("ZIZ", .3, 0.), ("III", .2, 0.)]
    state = np.zeros(8, dtype=complex)
    state[1], state[6] = .6, .8j
    exact = ctx.propagator(3, terms)
    low = ctx.propagator(3, terms, precision="float32")
    assert (exact.precision, low.precision) == ("float64", "float32")
    out = low.evolve(state.astype(np.complex64), .9, steps=6)
    assert out.dtype == np.complex64
    assert np.allclose(out, exact.evolve(state, .9, steps=6), atol=5e-6)
    assert low.apply(state.astype(np.complex64)).dtype == np.complex64
    with pytest.raises(TypeError, match="precision='float32'"):
        low.evolve(state, .9)
    operator = ctx.pauli_operator(3, terms)
    reference = operator.qpe(state, 4, [-1.5, 1.5])
    result = operator.qpe(state.astype(np.complex64), 4, [-1.5, 1.5], precision="float32")
    assert result["amplitudes"].dtype == np.complex64 and reference["amplitudes"].dtype == np.complex128
    assert np.allclose(result["probabilities"], reference["probabilities"], atol=5e-6)
    with pytest.raises(TypeError, match="precision='float32'"):
        operator.qpe(state, 4, [-1.5, 1.5], precision="float32")


def test_float32_state_budget_succeeds_where_float64_runs_out_of_memory():
    """With the limit at 0.75 of the float64 peak, float64 raises MemoryError and float32 matches its energy."""
    megabyte = 1024 ** 2

    def evaluate(precision, limit):
        ctx = _native.Context(threads=1, memory_limit_bytes=limit)
        problem = hubbard_problem(ctx, 8, precision, sector=False)     # 16 qubits, full register
        value = problem.evaluate([.1] * 4, trotter=False, indices=[0, 1, 2, 3])
        return value["energy_ha"], ctx.memory_peak_bytes

    energy64, peak64 = evaluate("float64", 512 * megabyte)
    limit = int(.75 * peak64)
    with pytest.raises(MemoryError):
        evaluate("float64", limit)
    energy32, peak32 = evaluate("float32", limit)
    assert peak32 <= limit
    assert energy32 == pytest.approx(energy64, abs=5e-4)


FROZEN_QUANTUM = Path(__file__).parent / "reference/precision/quantum_energies.json"


@pytest.mark.parametrize("case_id", ["q/NAO/H2/DZ/adapt-vqe", "q/NAO/H2/DZ/vqe", "q/NAO/LiH/DZ/vqe"])
def test_float64_quantum_energies_equal_the_frozen_precision_baseline(case_id):
    from ase.build import molecule

    frozen = next(c for c in json.loads(FROZEN_QUANTUM.read_text())["cases"] if c["case"]["id"] == case_id)
    spec = frozen["case"]
    atoms = molecule(spec["mol"])
    atoms.center(vacuum=3.0)
    options = dict(method=spec["method"], basis={"name": spec["family"], "size": spec["size"]},
                   grid_spacing=spec["h"], optimizer={"method": "BFGS", "maxiter": 300, "tol": spec["opt_tol"]},
                   resources={"threads": 2}, trace=False)
    if spec["method"].startswith("adapt"):
        options.update(convergence={"gradient": spec["grad"], "energy": None}, max_iterations=spec["max_ops"])
    atoms.calc = Mandacaru(**options)
    atoms.get_potential_energy()
    assert atoms.calc.result.energy_hartree == pytest.approx(frozen["float64"]["E_Ha"], abs=1e-10)


def test_state_reductions_sum_in_float64_for_either_dtype():
    """The native reductions the workflows use: float64 sums, dtype-matched inputs."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=1 << 12) + 1j * rng.normal(size=1 << 12)
    b = rng.normal(size=1 << 12) + 1j * rng.normal(size=1 << 12)
    assert _native.state_norm_squared(a) == pytest.approx(np.vdot(a, a).real, rel=1e-13)
    assert complex(*_native.state_inner(a, b)) == pytest.approx(np.vdot(a, b), rel=1e-12)
    a32, b32 = a.astype(np.complex64), b.astype(np.complex64)
    exact = np.vdot(a32.astype(complex), b32.astype(complex))
    assert complex(*_native.state_inner(a32, b32)) == pytest.approx(exact, rel=1e-12)
    with pytest.raises(TypeError, match="same complex dtype"):
        _native.state_inner(a, b32)
    with pytest.raises(TypeError, match="complex64 or complex128"):
        _native.state_norm_squared(a.real)
    with pytest.raises(ValueError, match="same length"):
        _native.state_inner(a, b[:-1])


def test_normalization_tolerance_belongs_to_the_dtype():
    state = np.zeros(8, complex)
    state[0] = 1 + 2e-6
    _native.require_normalized_state(state.astype(np.complex64))
    with pytest.raises(ValueError, match="normalized"):
        _native.require_normalized_state(state)
