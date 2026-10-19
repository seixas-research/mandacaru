# -*- coding: utf-8 -*-
# file: test/backends/test_gradient_measurement.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""ADAPT-VQE's pool gradients from Pauli expectation values.

A shot-based run measures ``g_k = <psi|[H, A_k]|psi>`` instead of contracting
a ``2^n`` state.  These tests pin the observables against the state-vector
formula ``2 Re<H psi|A_k psi>``, the exact classical gradients at a basis
state, the batching and the job packing, and the shot statistics.
"""

import numpy as np
import pytest

from mandacaru.backends.gradient_measurement import PoolGradients
from mandacaru.backends.measurement import (is_qubit_wise_commuting,
                                            qwc_groups, qwc_partition)
from mandacaru.circuits.pools import build_pool
from mandacaru.core.mapping import Fermion, PauliSum
from mandacaru.core.pauli_algebra import pack_labels

qiskit = pytest.importorskip("qiskit")


def molecular_fermion(n_orbitals, seed=3):
    """A random Hermitian two-body Hamiltonian with real-orbital symmetry."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n_orbitals,) * 2)
    h = np.diag(np.linspace(-3, 3, n_orbitals)) + 0.1 * (h + h.T)
    chemist = np.zeros((n_orbitals,) * 4)
    for _ in range(3):
        L = rng.standard_normal((n_orbitals,) * 2)
        L = (L + L.T) / 2
        chemist += np.einsum("pr,qs->prqs", L, L)
    return Fermion.from_spatial_integrals(
        h, 0.05 * chemist.transpose(0, 2, 1, 3))


def basis_state(n, occupied):
    index = sum(1 << (n - 1 - q) for q in occupied)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[index] = 1.0
    return psi


def state_vector_gradients(hamiltonian, operators, psi):
    h_psi = hamiltonian.to_sparse_matrix() @ psi
    return np.array([2.0 * np.real(np.vdot(
        h_psi, op.generator.to_sparse_matrix() @ psi)) for op in operators])


class ExactPauli:
    """A device double whose Pauli expectation values are exact: the
    executed circuit's state vector (QiskitProvider), contracted."""

    device_spec = "exact"

    def __init__(self, shots=0):
        self.shots = shots
        from mandacaru.backends.providers import QiskitProvider
        self.circuit = QiskitProvider()
        self.jobs = []

    def expectation_values(self, n, occupied, generators, thetas, labels,
                           max_bases_per_job=None):
        self.jobs.append(list(labels))
        psi = self.circuit.statevector(n, occupied, generators, thetas)
        values = {label: float(np.real(np.vdot(
            psi, PauliSum({label: 1.0}).to_sparse_matrix() @ psi)))
            for label in labels}
        return values, dict.fromkeys(labels, 0.0)


@pytest.fixture(scope="module")
def problem():
    fermion = molecular_fermion(4)
    return fermion.map_to_qubits("jordan_wigner", n_modes=8)


class TestReferenceGradients:
    """At a computational basis state every gradient is classical."""

    @pytest.mark.parametrize("pool_name", ["fermionic", "qeb", "qubit",
                                           "ceo", "ceo-ovp"])
    def test_they_equal_the_state_vector_gradients(self, problem, pool_name):
        pool = build_pool(pool_name, 4, (2, 1))
        ops = pool.operators()
        gradients = PoolGradients(problem, [op.generator for op in ops])
        psi = basis_state(8, pool.occupied_orbitals)
        exact = state_vector_gradients(problem, ops, psi)
        assert np.abs(exact).max() > 0.1
        assert gradients.reference_gradients(pool.occupied_orbitals) == \
            pytest.approx(exact, abs=1e-12)

    @pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
    def test_any_encoding_of_the_determinant(self, mapping):
        """The reference is a basis state of the *qubits*, not of the modes:
        the parity and Bravyi-Kitaev images of the determinant."""
        from mandacaru.core.mapping import reference_qubit_bits

        hamiltonian = molecular_fermion(3).map_to_qubits(mapping, n_modes=6)
        pool = build_pool("fermionic", 3, (1, 1), mapping=mapping)
        ops = pool.operators()
        bits = reference_qubit_bits(mapping, 6, pool.occupied_orbitals)
        qubits = [q for q, bit in enumerate(bits) if bit]
        exact = state_vector_gradients(hamiltonian, ops,
                                       basis_state(6, qubits))
        got = PoolGradients(hamiltonian, [op.generator for op in ops]
                            ).reference_gradients(qubits)
        assert got == pytest.approx(exact, abs=1e-12)


class TestMeasuredGradients:
    """Away from a basis state the gradients are measured."""

    @pytest.fixture(scope="class")
    def state(self, problem):
        pool = build_pool("qeb", 4, (2, 1))
        ops = pool.operators()
        generators = [ops[k].generator for k in (0, 4, 9)]
        thetas = np.array([0.31, -0.22, 0.47])
        occupied = list(pool.occupied_orbitals)
        psi = ExactPauli().circuit.statevector(8, occupied, generators,
                                               thetas)
        return pool, ops, (8, occupied, generators, thetas), psi

    def test_an_exact_device_gives_the_state_vector_gradients(
            self, problem, state):
        pool, ops, circuit, psi = state
        gradients = PoolGradients(problem, [op.generator for op in ops])
        measured, errors = gradients.measure(ExactPauli(), circuit,
                                             range(len(ops)))
        exact = state_vector_gradients(problem, ops, psi)
        assert np.abs(exact).max() > 0.05
        assert measured == pytest.approx(exact, abs=1e-10)
        assert not errors.any()

    def test_batches_and_jobs_do_not_change_the_answer(self, problem,
                                                       state):
        """Tiny batches (strings shared across batches are measured twice)
        and tiny jobs (a basis is never split) give the same gradients."""
        pool, ops, circuit, psi = state
        small = PoolGradients(problem, [op.generator for op in ops],
                              batch_strings=500)
        device = ExactPauli()
        measured, _ = small.measure(device, circuit, range(len(ops)),
                                    max_bases=7, max_strings=60)
        exact = state_vector_gradients(problem, ops, psi)
        assert measured == pytest.approx(exact, abs=1e-10)
        plan = small.plan(device, range(len(ops)), 7, 60)
        assert plan.batches > 1 and plan.jobs == len(device.jobs)
        assert plan.max_job_bases <= 7
        for labels in device.jobs:
            assert len(qwc_partition(labels)) <= 7
            # A job is whole bases: more than 60 strings only as one basis.
            if len(labels) > 60:
                assert len(qwc_partition(labels)) == 1

    def test_a_subset_is_measured_alone(self, problem, state):
        pool, ops, circuit, psi = state
        gradients = PoolGradients(problem, [op.generator for op in ops])
        subset = [2, 5, 11]
        measured, _ = gradients.measure(ExactPauli(), circuit, subset)
        exact = state_vector_gradients(problem, ops, psi)[subset]
        assert measured == pytest.approx(exact, abs=1e-10)

    def test_shot_noise_is_within_its_standard_error(self, problem, state):
        """The local estimator's sampled values: every gradient within five
        of its reported standard errors, and the errors under the plan's
        bound."""
        from mandacaru.backends.providers import QiskitProvider

        pool, ops, circuit, psi = state
        gradients = PoolGradients(problem, [op.generator for op in ops])
        device = QiskitProvider(shots=4000, seed=11)
        measured, errors = gradients.measure(device, circuit,
                                             range(len(ops)))
        exact = state_vector_gradients(problem, ops, psi)
        assert (errors > 0).all()
        assert np.all(np.abs(measured - exact) < 5 * errors + 1e-12)
        plan = gradients.plan(device, range(len(ops)))
        assert errors.max() <= plan.noise_bound_hartree * (1 + 1e-9)


class TestThePlan:
    def test_it_counts_what_is_measured(self, problem):
        pool = build_pool("fermionic", 4, (2, 2))
        ops = pool.operators()
        gradients = PoolGradients(problem, [op.generator for op in ops])
        plan = gradients.plan(ExactPauli(shots=1000), range(len(ops)))
        union = set()
        for k in range(len(ops)):
            union |= set(gradients.observable(k).labels())
        assert plan.strings == len(union)
        assert plan.bases == len(qwc_partition(sorted(union)))
        assert plan.operators == plan.pool_size == len(ops)
        # Merging pays: every operator's commutator alone needs more bases.
        alone = sum(len(qwc_partition(gradients.observable(k).labels()))
                    for k in range(len(ops)))
        assert plan.bases < alone / 3
        assert plan.total_shots == plan.bases * 1000

    def test_a_plan_belongs_to_its_provider(self, problem):
        """Cached per operators *and* provider: a second provider with shots
        does not read the first one's (shot-free) cost."""
        pool = build_pool("qeb", 4, (2, 1))
        gradients = PoolGradients(problem, [op.generator
                                            for op in pool.operators()])
        free = gradients.plan(ExactPauli(), range(len(gradients)))
        shot = gradients.plan(ExactPauli(shots=1000), range(len(gradients)))
        assert free.total_shots == 0 and free.noise_bound_hartree == 0.0
        assert shot.total_shots == shot.bases * 1000
        assert shot.noise_bound_hartree > 0

    def test_the_merged_bases_cost_a_few_energies(self, problem):
        pool = build_pool("fermionic", 4, (2, 2))
        gradients = PoolGradients(problem, [op.generator
                                            for op in pool.operators()])
        plan = gradients.plan(ExactPauli(), range(len(gradients)))
        ratio = plan.bases / gradients.energy_bases()
        assert 2 < ratio < 20


class TestTheGrouping:
    """qwc_partition runs on bit tables; it must stay the same partition."""

    @pytest.mark.parametrize("width", [7, 64, 70])
    def test_every_basis_is_qubit_wise_commuting(self, width):
        rng = np.random.default_rng(width)
        labels = sorted({"".join(rng.choice(list("IXYZ"), width,
                                            p=[0.7, 0.1, 0.1, 0.1]))
                         for _ in range(400)})
        partition = qwc_partition(labels)
        assert sorted(i for _b, members in partition for i in members) == \
            list(range(len(labels)))
        for basis, members in partition:
            for i in members:
                assert is_qubit_wise_commuting(basis, labels[i])
                assert all(b == letter for b, letter in zip(basis, labels[i])
                           if letter != "I")
        # The first-fit rule: a label opens a basis only when no earlier one
        # takes it.
        x, z = pack_labels(labels, width)
        _bx, _bz, members = qwc_groups(x, z, range(len(labels)))
        assert [m for _b, m in partition] == members


class StabilizerDevice:
    """A device double that simulates the circuit it is handed as a
    stabilizer state: exact and polynomial at any width, at Clifford angles
    (every Pauli rotation a multiple of pi/4).  The Hamiltonian's and the
    circuit's qubit order coincide (Mandacaru qubit k on wire n-1-k)."""

    shots = 0
    device_spec = "stabilizer"

    def __init__(self):
        from mandacaru.backends.providers import QiskitProvider
        self.circuit = QiskitProvider()

    def state(self, n, occupied, generators, thetas):
        from qiskit.quantum_info import Clifford, StabilizerState
        qc = self.circuit.build(n, occupied, generators, thetas)
        return StabilizerState(Clifford(qc))

    def expectation_values(self, n, occupied, generators, thetas, labels,
                           max_bases_per_job=None):
        from qiskit.quantum_info import Pauli
        state = self.state(n, occupied, generators, thetas)
        values = {label: float(state.expectation_value(Pauli(label)))
                  for label in labels}
        return values, dict.fromkeys(labels, 0.0)


def clifford_angles(generators):
    """Angles that make every rotation ``exp(i theta c P)`` a pi/4 one."""
    return np.array([np.pi / 4 / abs(next(iter(g.terms.values()))).real
                     for g in generators])


class TestAtFiftyQubits:
    """The circuit, the measurement and the contraction at 50 qubits, on a
    stabilizer simulation of the device: no state vector anywhere."""

    def test_the_stabilizer_device_is_exact_where_a_state_vector_is(
            self, problem):
        pool = build_pool("qeb", 4, (2, 1))
        ops = pool.operators()
        generators = [ops[k].generator for k in (3, 7)]
        thetas = clifford_angles(generators)
        occupied = list(pool.occupied_orbitals)
        circuit = (8, occupied, generators, thetas)
        gradients = PoolGradients(problem, [op.generator for op in ops])
        measured, _ = gradients.measure(StabilizerDevice(), circuit,
                                        range(len(ops)))
        psi = ExactPauli().circuit.statevector(*circuit)
        assert measured == pytest.approx(
            state_vector_gradients(problem, ops, psi), abs=1e-10)

    def test_merged_batches_equal_each_commutator_measured_alone(self):
        """Measured the shipped way (merged strings, bases, jobs, the
        sparse contraction) against each ``<psi|[H, A_k]|psi>`` read off the
        stabilizer state directly, at a non-trivial Clifford state."""
        from qiskit.quantum_info import SparsePauliOp

        terms = {}
        for spin in (0, 25):                       # a Hubbard chain, 50 modes
            for i in range(24):
                p, q = i + spin, i + 1 + spin
                terms[((p, True), (q, False))] = -1.0
                terms[((q, True), (p, False))] = -1.0
        for i in range(25):
            terms[((i, True), (i, False), (i + 25, True),
                   (i + 25, False))] = 4.0
        hamiltonian = Fermion(terms, n_modes=50).map_to_qubits(
            "jordan_wigner", n_modes=50)
        pool = build_pool("qeb", 25, (2, 2))
        ops = pool.operators()
        occupied = list(pool.occupied_orbitals)
        gradients = PoolGradients(hamiltonian,
                                  [op.generator for op in ops],
                                  batch_strings=400)
        # The state: the three single Pauli rotations the reference ranks
        # first, at pi/4 (a superposition of determinants, still a
        # stabilizer state); measured: the twenty pool operators the
        # reference ranks first and ten random others.
        strings = build_pool("qubit", 25, (2, 2)).operators()
        rotations = PoolGradients(hamiltonian, [op.generator
                                                for op in strings])
        first = np.argsort(-np.abs(rotations.reference_gradients(occupied)),
                           kind="stable")[:3]
        generators = [strings[k].generator for k in first]
        thetas = clifford_angles(generators)
        ranked = np.argsort(-np.abs(gradients.reference_gradients(occupied)),
                            kind="stable")
        rng = np.random.default_rng(4)
        chosen = sorted(set(ranked[:20].tolist())
                        | set(rng.choice(len(ops), 10).tolist()))
        device = StabilizerDevice()
        measured, _ = gradients.measure(device, (50, occupied, generators,
                                                 thetas), chosen,
                                        max_bases=5, max_strings=200)
        state = device.state(50, occupied, generators, thetas)
        alone = []
        for k in chosen:
            observable = hamiltonian.commutator(ops[k].generator)
            alone.append(float(np.real(state.expectation_value(
                SparsePauliOp(list(observable.terms),
                              list(observable.terms.values()))))))
        assert np.abs(alone).max() > 0.1
        assert measured == pytest.approx(alone, abs=1e-10)
