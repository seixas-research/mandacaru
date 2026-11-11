# -*- coding: utf-8 -*-
# file: test/integrations/test_measured_rdms.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``Mandacaru.measured_rdms()``: the 1- and 2-RDMs of the optimized circuit, measured on the
circuit device (``integrations.circuits.measure_rdms``, ``quantum::rdm_measurement``).

The independent references: on an exact circuit device the RDMs contract with the model's
integrals to the run's energy; on a sampled one their traces are the electron and pair counts
within the propagated shot noise. Their equality to the state-vector RDMs in every mapping is
pinned in Rust (``crates/mandacaru-core/tests/rdm_measurement.rs``); the 50-qubit memory rule
in ``test/architecture/test_shot_path_memory.py``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

OPTIONS = dict(method="vqe", basis="HAO", grid_spacing=0.35, trace=False,
               optimizer={"method": "COBYLA", "maxiter": 40, "tol": 1e-6})


def h2(**options):
    atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]], cell=[6.0, 6.0, 6.0])
    atoms.calc = Mandacaru(**{**OPTIONS, **options})
    atoms.get_potential_energy()
    return atoms.calc


def contracted_energy(calc, one, two):
    """``c + sum h gamma + (1/2) sum g Gamma`` with the model's integrals, Hartree."""
    model = calc.result.hamiltonian
    h, g = model.one_body.real, model.two_body.real
    return model.constant_energy + np.sum(h * one) + 0.5 * np.sum(g * two)


def test_on_an_exact_circuit_device_the_rdms_give_the_run_energy():
    calc = h2(execute_circuits=True)
    rdms = calc.measured_rdms()
    one, two = rdms["one"], rdms["two"]
    assert one.shape == (2, 2) and two.shape == (2, 2, 2, 2)
    assert np.allclose(one, one.T, atol=1e-12)
    assert np.trace(one) == pytest.approx(2.0, abs=1e-10)
    assert contracted_energy(calc, one, two) == pytest.approx(calc.result.in_units("Ha"), abs=1e-8)


def test_on_a_sampled_device_the_reported_error_is_the_observed_scatter():
    """Local shots are sampled on noiseless Aer. Re-measuring one optimized circuit with 16
    seeds, the scatter of the excited orbital's occupation ``gamma_11`` matches the reported
    error, which includes the covariance of the alpha and beta occupations: they move together
    in the paired H2 state, so the independent-string error would be smaller by sqrt(2)."""
    shots = 2000
    calc = h2(shots=shots)
    occupation, reported, traces = [], [], []
    for seed in range(16):
        calc.provider.seed, calc.provider._estimator = 100 + seed, None
        rdms = calc.measured_rdms()
        occupation.append(rdms["one"][1, 1])
        reported.append(rdms["one_error"][1, 1])
        traces.append(np.trace(rdms["one"]))
    observed, error = np.std(occupation, ddof=1), np.mean(reported)
    assert error > 0
    # 16 samples estimate a standard deviation to about 18 percent; sqrt(2) is outside.
    assert 0.6 < observed / error < 1.5, (observed, error)
    # Every shot of a number-conserving circuit holds two electrons.
    assert np.allclose(traces, 2.0, atol=1e-9)


def test_a_state_vector_run_and_an_exceeded_budget_are_refused():
    with pytest.raises(ValueError, match="circuit device"):
        h2().measured_rdms()
    calc = h2(execute_circuits=True)
    calc._measurement_budget = {"observables": 3}
    with pytest.raises(ValueError, match="observables budget"):
        calc.measured_rdms()
    with pytest.raises(ValueError, match="finished variational run"):
        Mandacaru(**OPTIONS, execute_circuits=True).measured_rdms()
