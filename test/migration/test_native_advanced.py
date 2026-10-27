"""Phase 6: independent dense dynamics, MP2 and shared-unitary comparisons."""
import gc
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru import Mandacaru, _native
from mandacaru.algorithms.mp2 import (mp2_natural_orbitals,
                                      mp2_natural_orbitals_from_block)
from mandacaru.algorithms.qpe import QuantumPhaseEstimation
from mandacaru.algorithms.time_evolution import SuzukiTrotter, time_evolve
from mandacaru.core.mapping import PauliSum
from mandacaru.units import HARTREE_TO_EV

H2 = Path(__file__).parent / "reference/h2_hamiltonian.json"


def state(n=3):
    rng = np.random.default_rng(628)
    s = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return s / np.linalg.norm(s)


def hamiltonian():
    # Deliberately differs from native canonical mask order.
    return PauliSum({"YIX": .31, "ZZI": -.42, "III": -1.3,
                     "XZY": .17, "IYI": -.28}, num_qubits=3)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
@pytest.mark.parametrize("time", [-1.7, 0., .61])
def test_ordered_propagation_and_exact_discrete_adjoint(order, time):
    h, psi = hamiltonian(), state()
    original = psi.copy()
    native = SuzukiTrotter(h, order=order, engine="rust")
    legacy = SuzukiTrotter(h, order=order)
    for inverse in [False, True]:
        actual = native.evolve(psi, time, steps=3, inverse=inverse)
        expected = legacy.evolve(psi, time, steps=3, inverse=inverse)
        assert np.allclose(actual, expected, atol=3e-13, rtol=0)
    evolved = native.evolve(psi, time, steps=3)
    recovered = native.evolve(evolved, time, steps=3, inverse=True)
    assert np.allclose(recovered, psi, atol=3e-13, rtol=0)
    assert np.array_equal(psi, original)
    assert np.allclose(native._apply_operator(psi), h.to_matrix() @ psi)
    # Correlation branches are not necessarily normalized.
    assert np.allclose(native._evolve(2.3 * psi, time, 3),
                       2.3 * evolved, atol=1e-12, rtol=0)


def test_suzuki_convergence_and_snapshot():
    h, psi = hamiltonian(), state()
    target = expm(-.9j * h.to_matrix()) @ psi
    errors = []
    for order in [2, 4, 6]:
        errors.append(np.linalg.norm(time_evolve(
            psi, h, .9, steps=4, order=order, engine="rust") - target))
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 20
    compiled = SuzukiTrotter(h, engine="rust")
    expected = compiled.evolve(psi, .4)
    h.terms["III"] = 99.
    assert np.allclose(compiled.evolve(psi, .4), expected)


def test_native_circuit_matches_propagation():
    from qiskit.quantum_info import Statevector
    compiled = SuzukiTrotter(hamiltonian(), order=4, engine="rust")
    psi = state()
    for inverse in [False, True]:
        circuit = compiled.circuit(.7, steps=2, inverse=inverse)
        expected = Statevector(psi).evolve(circuit).data
        assert np.allclose(compiled.evolve(psi, .7, steps=2, inverse=inverse), expected)


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("window", [None, (-3., 1.)])
def test_qpe_distribution_and_collapsed_states(keep, window):
    h, psi = hamiltonian(), state()
    options = dict(n_evaluation_qubits=5, energy_window=window,
                   atomic_units=True, verbose=False, keep_amplitudes=keep)
    expected = QuantumPhaseEstimation(h, **options).run(psi)
    actual = QuantumPhaseEstimation(h, engine="rust", **options).run(psi)
    assert np.allclose(actual.probabilities, expected.probabilities, atol=2e-13)
    assert np.allclose(actual.exact_energies, expected.exact_energies)
    # Eigenvectors inside a degenerate eigenspace can differ between solvers.
    for energy in np.unique(np.round(expected.exact_energies, 10)):
        indices = np.isclose(expected.exact_energies, energy, atol=1e-10, rtol=0)
        assert np.sum(actual.overlaps[indices]) == pytest.approx(
            np.sum(expected.overlaps[indices]), abs=1e-12)
    assert actual.ground_state_overlap == pytest.approx(expected.ground_state_overlap, abs=1e-12)
    assert np.allclose(actual.energies, expected.energies)
    assert actual.energy_window == expected.energy_window
    assert sum(actual.probabilities) == pytest.approx(1., abs=2e-13)
    if keep:
        for i in np.flatnonzero(actual.probabilities > .01):
            assert np.allclose(actual.collapsed_state(int(i)),
                               expected.collapsed_state(int(i)), atol=2e-12)
    else:
        assert actual._amplitudes is None


def test_qpe_matrix_free_branch_and_units(monkeypatch):
    # Above the dense threshold. Exact two-state spectral oracle, no dense matrix.
    n = 13
    h = PauliSum({"Z" + "I" * (n-1): .25, "I" * n: -.75}, num_qubits=n)
    psi = np.zeros(2**n, dtype=complex)
    psi[0], psi[2**(n-1)] = np.sqrt(.3), 1j*np.sqrt(.7)
    t = 4
    qpe = QuantumPhaseEstimation(h, t, engine="rust", verbose=False)
    lo, hi = qpe._window(h)
    k = np.arange(2**t)
    expected = sum(w * np.abs(np.fft.fft(np.exp(
        2j*np.pi*k*(e-lo)/(hi-lo))) / 2**t)**2
                   for w, e in [(.3, -.5), (.7, -1.)])
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical callback used")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.fft, "fft", forbidden)
    monkeypatch.setattr(PauliSum, "to_matrix", forbidden)
    monkeypatch.setattr(PauliSum, "to_sparse_matrix", forbidden)
    actual = qpe.run(psi)
    assert np.allclose(actual.probabilities, expected, atol=1e-12)
    assert actual.exact_energies is None
    assert np.allclose(actual.in_units("Ha").energies * HARTREE_TO_EV, actual.energies)


def integrals(m=5, occupied=2, rotate=False):
    rng = np.random.default_rng(662)
    # Positive chemists' pair tensor, transposed to physicists' notation.
    pairs = rng.normal(scale=.08, size=(7, m, m))
    pairs = (pairs + pairs.transpose(0, 2, 1)) / 2
    g = np.einsum("Lpr,Lqs->pqrs", pairs, pairs)
    eps = np.linspace(-1.5, 2., m)
    h = np.diag(eps) - sum(2*g[:, i, :, i] - g[:, i, i, :]
                          for i in range(occupied))
    if rotate:
        c = np.zeros((m, m))
        c[:occupied, :occupied] = np.linalg.qr(rng.normal(size=(occupied, occupied)))[0]
        c[occupied:, occupied:] = np.linalg.qr(rng.normal(size=(m-occupied, m-occupied)))[0]
        h = c.T @ h @ c
        g = np.einsum("pi,qj,rk,sl,pqrs->ijkl", c, c, c, c, g, optimize=True)
    return h, g


@pytest.mark.parametrize("m,o", [(2, 1), (5, 2), (6, 3)])
@pytest.mark.parametrize("rotate", [False, True])
def test_mp2_density_trace_and_semicanonical_covariance(m, o, rotate):
    h, g = integrals(m, o, rotate)
    expected = mp2_natural_orbitals(h, g, o)
    actual = mp2_natural_orbitals(h, g, o, engine="rust")
    assert actual.correlation_energy == pytest.approx(expected.correlation_energy, abs=1e-13)
    assert np.allclose(actual.occupations, expected.occupations, atol=1e-12)
    assert np.allclose(actual.occupied_density, expected.occupied_density, atol=1e-12)
    assert actual.occupations.sum() == pytest.approx(2*o, abs=2e-12)
    assert np.all(np.diff(actual.virtual_occupations) <= 0)
    assert np.allclose(actual.rotation.T @ actual.rotation, np.eye(m), atol=1e-12)
    assert np.allclose(actual.rotation[:o, o:], 0)
    # Compare physical density, allowing arbitrary eigenvector signs.
    for result in [expected, actual]:
        natural_density = np.zeros((m, m))
        natural_density[:o, :o] = result.occupied_density
        natural_density[o:, o:] = np.diag(result.virtual_occupations)
        density = result.rotation @ natural_density @ result.rotation.T
        if result is expected:
            expected_density = density
        else:
            assert np.allclose(density, expected_density, atol=2e-12)


def test_mp2_against_independent_spin_orbital_prefactors():
    m, o = 5, 2
    h, g = integrals(m, o)
    energies = np.tile(np.linspace(-1.5, 2., m), 2)
    occupied = [p+s*m for s in range(2) for p in range(o)]
    virtual = [p+s*m for s in range(2) for p in range(o, m)]
    anti = np.zeros((len(occupied), len(occupied), len(virtual), len(virtual)))
    t = np.zeros_like(anti)
    for i, I in enumerate(occupied):
        for j, J in enumerate(occupied):
            for a, A in enumerate(virtual):
                for b, B in enumerate(virtual):
                    value = (g[I % m, J % m, A % m, B % m]
                             if I//m == A//m and J//m == B//m else 0.)
                    value -= (g[I % m, J % m, B % m, A % m]
                              if I//m == B//m and J//m == A//m else 0.)
                    anti[i, j, a, b] = value
                    t[i, j, a, b] = value / (energies[I]+energies[J]-energies[A]-energies[B])
    expected_energy = .25 * np.sum(anti*t)
    dvv = .5 * np.einsum("ijac,ijbc->ab", t, t)
    doo = -.5 * np.einsum("ikab,jkab->ij", t, t)
    v = m-o
    spatial_vv = dvv[:v, :v] + dvv[v:, v:]
    spatial_oo = doo[:o, :o] + doo[o:, o:]
    result = mp2_natural_orbitals(h, g, o, engine="rust")
    assert result.correlation_energy == pytest.approx(expected_energy, abs=1e-13)
    assert np.allclose(result.occupied_density, 2*np.eye(o)+spatial_oo, atol=1e-12)
    assert np.allclose(result.virtual_occupations, np.linalg.eigvalsh(spatial_vv)[::-1], atol=1e-12)


def test_mp2_block_and_no_python_numerical_callbacks(monkeypatch):
    import mandacaru.algorithms.mp2 as module
    h, g = integrals()
    expected = mp2_natural_orbitals(h, g, 2)
    f = module.fock_matrix(h, g, 2)
    eps = np.diag(f)
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical callback used")
    for name in ["fock_matrix", "semicanonical_rotation", "mp2_energy", "mp2_density"]:
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    actual = mp2_natural_orbitals(h, g, 2, engine="rust")
    block = mp2_natural_orbitals_from_block(g[:2, :2, 2:, 2:], eps, 2,
                                           np.eye(5), engine="rust")
    assert actual.correlation_energy == pytest.approx(expected.correlation_energy)
    assert block.correlation_energy == pytest.approx(expected.correlation_energy)
    assert np.allclose(block.occupations, actual.occupations)


@pytest.mark.parametrize("method", ["subspace-vqe", "subspace-adapt-vqe"])
@pytest.mark.parametrize("sector", [False, True])
@pytest.mark.parametrize("trotter", [False, True])
def test_public_subspace_shared_unitary_and_sorted_states(method, sector, trotter):
    options = dict(method=method, engine="rust", load_hamiltonian=str(H2),
                   sector=sector, num_states=2, atomic_units=True,
                   trace=False, references=False, engine_options={"threads": 1})
    if method == "subspace-vqe":
        options["ansatz"] = {"name": "uccsd", "trotter": trotter}
    calc = Mandacaru(**options)
    result = calc.run()
    problem = calc.solver.problem
    basis = np.asarray(problem.basis_indices, dtype=int)
    dense = calc.solver.hamiltonian.to_matrix()[np.ix_(basis, basis)]
    states = np.asarray(result.states)
    assert np.allclose(states.conj() @ states.T, np.eye(2), atol=3e-12)
    assert np.allclose(np.einsum("ji,ik,jk->j", states.conj(), dense, states).real,
                       result.energies, atol=1e-12)
    assert result.optimal_energy == pytest.approx(np.linalg.eigvalsh(dense)[0], abs=1e-9)
    assert np.all(np.diff(result.energies) >= 0)
    assert (result.converged if method.endswith("adapt-vqe") else result.success)
    assert np.allclose(result.in_units("eV"), result.energies * HARTREE_TO_EV)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"])
@pytest.mark.parametrize("trotter", [False, True])
def test_shared_unitary_against_dense_reconstruction(tmp_path, mapping, trotter):
    from mandacaru.algorithms.subspace import subspace_determinants
    from mandacaru.core.mapping import Fermion, reference_qubit_bits
    from mandacaru.core.serialization import save_hamiltonian
    terms = {((j, True), (j, False)): .13*j for j in range(6)}
    terms.update({((0, True), (1, False)): .17,
                  ((1, True), (0, False)): .17,
                  ((3, True), (4, False)): .11,
                  ((4, True), (3, False)): .11})
    h = Fermion(terms, n_modes=6).map_to_qubits(mapping, n_modes=6, num_particles=(1, 1))
    path = tmp_path / "subspace.json"
    save_hamiltonian(path, h, mapping=mapping, num_particles=(1, 1), n_spatial_orbitals=3)
    calc = Mandacaru(method="subspace-vqe", engine="rust", load_hamiltonian=str(path),
                     ansatz={"name": "uccsd", "trotter": trotter}, num_states=3,
                     atomic_units=True, trace=False, references=False)
    result = calc.run()
    problem = calc.solver.problem
    basis = np.asarray(problem.basis_indices, dtype=int)
    matrices = [PauliSum({p: complex(r, i) for p, r, i in terms}, num_qubits=h.num_qubits)
                .to_matrix()[np.ix_(basis, basis)] for terms in problem.pool_terms]
    unitary = np.eye(len(basis), dtype=complex)
    if trotter:
        for theta, g in zip(result.optimal_parameters, matrices):
            unitary = expm(theta*g) @ unitary
    else:
        unitary = expm(sum(t*g for t, g in zip(result.optimal_parameters, matrices)))
    expected = []
    for det in subspace_determinants(6, [0, 3], 3):
        bits = reference_qubit_bits(mapping, 6, det)
        encoded = sum(int(b) << (len(bits)-1-i) for i, b in enumerate(bits))
        row = np.flatnonzero(basis == encoded)[0]
        expected.append(unitary[:, row])
    expected = np.asarray(expected)[calc.solver.state_reference_indices]
    assert np.allclose(expected, result.states, atol=2e-12, rtol=0)


def test_subspace_no_python_solver_and_inner_failure(monkeypatch):
    import mandacaru.algorithms.subspace as subspace
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical callback used")
    monkeypatch.setattr(subspace.SubspaceVQE, "_subspace_optimize", forbidden)
    monkeypatch.setattr(subspace.SubspaceADAPTVQE, "_subspace_optimize", forbidden)
    for method in ["subspace-vqe", "subspace-adapt-vqe"]:
        calc = Mandacaru(method=method, engine="rust", load_hamiltonian=str(H2),
                         optimizer={"maxfev": 1}, trace=False, references=False)
        result = calc.run()
        assert not (result.converged if method.endswith("adapt-vqe") else result.success)
        assert "evaluation budget" in result.timings["termination"]
        assert result.num_evaluations == 1


def test_native_rhf_mp2_and_geometry_subspace():
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    atoms.center(vacuum=2.)
    calc = Mandacaru(method="rhf", engine="rust", grid_spacing=.5,
                     trace=False, references=False)
    reference = calc.run(geometry=atoms)
    actual = reference.mp2_natural_orbitals()
    expected = mp2_natural_orbitals(reference.scf.h_mo, reference.scf.eri_mo, 1)
    assert actual.correlation_energy == pytest.approx(expected.correlation_energy, abs=1e-12)
    atoms.calc = Mandacaru(method="subspace-vqe", engine="rust", grid_spacing=.5,
                           trace=False, references=False)
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert energy == pytest.approx(atoms.calc.solver.result.in_units("eV")[0])


def test_rejections_memory_and_ownership():
    ctx = _native.Context(threads=1, memory_limit_bytes=200000)
    prop = ctx.propagator(1, [("X", .4, 0.)])
    used = ctx.memory_used_bytes
    for invalid in [np.array([2., 0.], dtype=complex),
                    np.array([np.nan, 0.], dtype=complex), np.ones(3, dtype=complex)]:
        with pytest.raises(ValueError):
            prop.evolve(invalid, .5)
        assert ctx.memory_used_bytes == used
    with pytest.raises(ValueError):
        prop.evolve(np.array([1., 0.], dtype=complex), .5, steps=0)
    with pytest.raises((ValueError, OverflowError)):
        ctx.propagator(1, [("X", .4, 0.)], order=100)
    op = ctx.pauli_operator(1, [("Z", 1., 0.)])
    with pytest.raises(MemoryError):
        op.qpe(np.array([1., 0.], dtype=complex), 20, (-2., 2.))
    h, g = integrals()
    with pytest.raises(ValueError, match="real"):
        mp2_natural_orbitals(h + 1j*np.eye(5), g, 2, engine="rust")
    with pytest.raises(ValueError, match="denominator"):
        mp2_natural_orbitals_from_block(np.ones((1, 1, 1, 1)),
                                        np.zeros(2), 1, np.eye(2), engine="rust")
    with pytest.raises(ValueError, match="strictly decreasing"):
        _native.Context().load_hamiltonian(str(H2)).solve_subspace([1., 1.])
    with pytest.raises(ValueError, match="sector"):
        _native.Context().load_hamiltonian(str(H2)).solve_subspace([5., 4., 3., 2., 1.])
    output = prop.evolve(np.array([1., 0.], dtype=complex), .5)
    del prop, op
    gc.collect()
    assert ctx.memory_used_bytes == 0
    assert np.linalg.norm(output) == pytest.approx(1.)
