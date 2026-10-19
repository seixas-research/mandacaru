# -*- coding: utf-8 -*-
# file: test/circuits/test_ansatz.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ansatz protocol: anything that satisfies it drops in.

:class:`~mandacaru.circuits.Ansatz` is satisfied *structurally* --
``num_parameters`` / ``n_qubits`` / ``reference_state`` / ``state`` /
``evolve`` -- so a hand-written ansatz needs no base class and no driver
change.  :class:`~mandacaru.circuits.SerializableAnsatz` is the optional
extension a checkpoint needs.
"""

import numpy as np
import pytest

from mandacaru import Mandacaru
from mandacaru.circuits import UCCSD, Ansatz, SerializableAnsatz
from mandacaru.core.mapping import PauliSum
from mandacaru.optimizers import Optimizer


class Rotation:
    """The documented structural protocol and nothing more."""

    n_qubits = 1
    num_parameters = 1

    def reference_state(self):
        return np.array([1.0, 0.0])

    def state(self, theta):
        return np.array([np.cos(theta[0]), np.sin(theta[0])])

    def evolve(self, theta, references):
        c, s = np.cos(theta[0]), np.sin(theta[0])
        return np.array([[c, -s], [s, c]]) @ references


class TestCustomAnsatz:
    def test_the_protocols(self):
        assert isinstance(Rotation(), Ansatz)
        assert not isinstance(Rotation(), SerializableAnsatz)
        assert isinstance(UCCSD(2, (1, 1)), SerializableAnsatz)

    def test_it_runs_without_a_checkpoint(self):
        # `Rotation` starts at theta = 0, which is the *maximum* of cos(theta):
        # the gradient is exactly zero there, so a gradient method (the SLSQP
        # default included) correctly reports a stationary point and does not
        # move.  This test is about a custom Ansatz running through VQE, so it
        # names a direct search -- the family that can leave such a point.
        calc = Mandacaru(method="vqe", hamiltonian=PauliSum({"Z": 1.0}),
                         ansatz=Rotation(), trace=False, atomic_units=True,
                         optimizer=Optimizer(method="Nelder-Mead",
                                             maxiter=1000, tol=1e-12))
        result = calc.run()
        assert result.optimal_energy == pytest.approx(-1.0, abs=1e-4)
        assert calc.solver.checkpoint is None

    def test_asking_for_a_checkpoint_is_refused_before_the_optimization(
            self, tmp_path):
        evaluations = []

        class Counting(Rotation):
            def state(self, theta):
                evaluations.append(1)
                return super().state(theta)

        calc = Mandacaru(method="vqe", hamiltonian=PauliSum({"Z": 1.0}),
                         ansatz=Counting(), trace=False,
                         checkpoint=str(tmp_path / "state.json"))
        with pytest.raises(TypeError, match="SerializableAnsatz"):
            calc.run()
        assert not evaluations


class TestUCCSDInAParticleSector:
    """The sector representation: ``C(M, na) C(M, nb)`` amplitudes and sparse
    generators, the same state as the dense ``2^N`` form."""

    @pytest.mark.parametrize("trotter", [False, True])
    @pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                         "bravyi_kitaev", "parity_reduced"])
    def test_the_state_is_the_register_state_restricted(self, mapping,
                                                         trotter):
        from mandacaru.core.sector import ParticleSector

        dense = UCCSD(4, (2, 2), mapping, trotter=trotter)
        sector = ParticleSector(dense.n_qubits, (2, 2), mapping)
        sparse = UCCSD(4, (2, 2), mapping, trotter=trotter, sector=sector)
        theta = np.random.default_rng(3).uniform(-1, 1, dense.num_parameters)
        assert sparse.state(theta).size == sector.dim == 36
        assert np.allclose(sector.embed(sparse.state(theta)),
                           dense.state(theta), atol=1e-12)
        assert np.allclose(sector.embed(sparse.reference_state()),
                           dense.reference_state(), atol=1e-14)
        stack = np.stack([sparse.reference_state()] * 2, axis=1)
        assert sparse.evolve(theta, stack).shape == (36, 2)

    def test_no_register_sized_array_is_allocated(self):
        """12 qubits: one dense generator is 268 MB, the 92 of them 25 GB."""
        import tracemalloc

        from mandacaru.core.sector import ParticleSector

        ansatz = UCCSD(6, (2, 2))
        ansatz.use_sector(ParticleSector(12, (2, 2), "jordan_wigner"))
        theta = np.full(ansatz.num_parameters, 0.05)
        tracemalloc.start()
        try:
            state = ansatz.state(theta)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-12
        assert peak < 32 * 2 ** 20

    def test_the_adjoint_gradient_exists_only_for_the_ordered_product(self):
        from mandacaru.core.sector import ParticleSector

        sector = ParticleSector(8, (2, 2), "jordan_wigner")
        assert not hasattr(UCCSD(4, (2, 2), sector=sector), "energy_and_gradient")
        assert not hasattr(UCCSD(4, (2, 2), trotter=True),
                           "energy_and_gradient")
        ansatz = UCCSD(4, (2, 2), trotter=True, sector=sector)
        rng = np.random.default_rng(5)
        h = rng.standard_normal((36, 36)); h = h + h.T
        theta = rng.uniform(-0.5, 0.5, ansatz.num_parameters)
        energy, gradient = ansatz.energy_and_gradient(theta, lambda v: h @ v)
        assert energy == pytest.approx(
            np.real(np.vdot(ansatz.state(theta), h @ ansatz.state(theta))))
        step = 1e-6
        for k in (0, 7, ansatz.num_parameters - 1):
            plus, minus = theta.copy(), theta.copy()
            plus[k] += step
            minus[k] -= step
            numeric = (np.real(np.vdot(ansatz.state(plus),
                                       h @ ansatz.state(plus)))
                       - np.real(np.vdot(ansatz.state(minus),
                                         h @ ansatz.state(minus)))) / (2 * step)
            assert gradient[k] == pytest.approx(numeric, abs=1e-6)

    def test_a_sector_and_a_circuit_provider_exclude_each_other(self):
        from mandacaru.core.sector import ParticleSector

        class Provider:
            name = "stub"

        with pytest.raises(ValueError, match="sector"):
            UCCSD(2, (1, 1), trotter=True, provider=Provider(),
                  sector=ParticleSector(4, (1, 1), "jordan_wigner"))
        with pytest.raises(ValueError, match="spans"):
            UCCSD(3, (1, 1), sector=ParticleSector(4, (1, 1), "jordan_wigner"))
