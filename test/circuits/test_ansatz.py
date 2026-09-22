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
        assert calc.checkpoint is None

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
