# -*- coding: utf-8 -*-
# file: test_vasqe.py

"""VASQE: ADAPT-VQE with stochastic (softmax) operator selection.

Checks the selection probabilities and temperature schedules, that VASQE reduces
to ADAPT-VQE at low temperature (reaching FCI on H2), that selection is genuinely
stochastic at high temperature, and that the excited-state extensions (deflation
and subspace) inherit the stochastic selection.
"""

import numpy as np
import pytest

from carcara.algorithms import (
    SubspaceVASQE,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)
from carcara.algorithms.adapt_vqe import ADAPTVQEResult
from carcara.algorithms.vasqe import TEMPERATURE_SCHEDULES
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid
from carcara.optimizers import Optimizer


# --------------------------------------------------------------------------- #
# Shared H2 fixture (MO basis).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    mints = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
    return mints.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def h2_fci(h2_hamiltonian):
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


def _vasqe(h2_hamiltonian, **kwargs):
    kwargs.setdefault("optimizer", Optimizer("L-BFGS-B", maxiter=2000))
    return VASQE(h2_hamiltonian, "fermionic", num_particles=(1, 1),
                 n_spatial_orbitals=2, verbose=False, profile=False,
                 gradient_tolerance=1e-6, **kwargs)


# --------------------------------------------------------------------------- #
# Softmax selection probabilities.
# --------------------------------------------------------------------------- #

class TestSoftmax:
    def test_low_temperature_is_argmax(self):
        g = np.array([0.1, 0.9, 0.4])
        p = softmax_selection_probabilities(g, 1e-6)
        assert np.argmax(p) == 1                    # largest |g|
        assert p[1] == pytest.approx(1.0, abs=1e-9)

    def test_high_temperature_is_uniform(self):
        g = np.array([0.1, 0.9, 0.4])
        p = softmax_selection_probabilities(g, 1e6)
        np.testing.assert_allclose(p, np.full(3, 1 / 3), atol=1e-3)

    def test_probabilities_normalized(self):
        g = np.array([0.2, 0.5, 0.7, 0.01])
        p = softmax_selection_probabilities(g, 0.3)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= 0)

    def test_uses_absolute_gradients(self):
        # sign must not matter: |−0.9| beats |0.1|.
        p = softmax_selection_probabilities(np.array([-0.9, 0.1]), 0.05)
        assert np.argmax(p) == 0

    def test_favours_larger_gradient(self):
        p = softmax_selection_probabilities(np.array([0.3, 0.6]), 1.0)
        assert p[1] > p[0]                          # larger |g| -> larger P


# --------------------------------------------------------------------------- #
# Temperature schedules.
# --------------------------------------------------------------------------- #

class TestSchedules:
    def test_constant_ignores_final(self):
        for k in range(5):
            assert annealed_temperature("constant", 2.0, 0.01, k, 5) == 2.0

    @pytest.mark.parametrize("schedule", ["linear", "exponential", "logarithmic"])
    def test_annealing_endpoints(self, schedule):
        assert annealed_temperature(schedule, 2.0, 0.05, 0, 6) == pytest.approx(2.0)
        assert annealed_temperature(schedule, 2.0, 0.05, 5, 6) == pytest.approx(0.05)

    @pytest.mark.parametrize("schedule", ["linear", "exponential", "logarithmic"])
    def test_monotone_decreasing(self, schedule):
        temps = [annealed_temperature(schedule, 2.0, 0.05, k, 8) for k in range(8)]
        assert all(a >= b - 1e-12 for a, b in zip(temps, temps[1:]))

    def test_exponential_cools_faster_than_linear_early(self):
        lin = annealed_temperature("linear", 2.0, 0.02, 1, 8)
        exp = annealed_temperature("exponential", 2.0, 0.02, 1, 8)
        assert exp < lin                            # geometric drop is steeper early

    def test_step_clamped(self):
        # step beyond the horizon stays at the final temperature.
        assert annealed_temperature("linear", 2.0, 0.05, 99, 6) == pytest.approx(0.05)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            annealed_temperature("cubic", 1.0, 0.1, 0, 5)


# --------------------------------------------------------------------------- #
# VASQE driver.
# --------------------------------------------------------------------------- #

class TestVASQE:
    def test_low_temperature_reduces_to_adapt(self, h2_hamiltonian, h2_fci):
        # tau -> 0 always picks the largest-gradient operator: ADAPT-VQE / FCI.
        r = _vasqe(h2_hamiltonian, temperature=1e-4).run()
        assert isinstance(r, VASQEResult)
        assert r.optimal_energy == pytest.approx(h2_fci, abs=1e-6)

    def test_result_records_schedule(self, h2_hamiltonian):
        r = _vasqe(h2_hamiltonian, temperature=1.5, final_temperature=0.02,
                   schedule="linear", max_iterations=6).run()
        assert r.schedule == "linear"
        assert r.initial_temperature == 1.5
        assert r.final_temperature == 0.02
        assert len(r.temperatures) == r.num_operators   # one tau per grown op

    def test_result_is_adaptvqe_result_subclass(self, h2_hamiltonian):
        r = _vasqe(h2_hamiltonian, temperature=0.1).run()
        assert isinstance(r, ADAPTVQEResult)            # inherits ADAPT fields
        assert r.metrics is None or hasattr(r.metrics, "num_operators")

    def test_reproducible_with_seed(self, h2_hamiltonian):
        r1 = _vasqe(h2_hamiltonian, temperature=3.0, seed=7, max_iterations=5).run()
        r2 = _vasqe(h2_hamiltonian, temperature=3.0, seed=7, max_iterations=5).run()
        assert r1.operators == r2.operators             # same seed -> same picks

    def test_high_temperature_selection_is_stochastic(self, h2_hamiltonian):
        # With a hot temperature different seeds should not all agree on the first
        # picked operator (the pool has several operators to choose from).
        first_ops = set()
        for seed in range(12):
            r = _vasqe(h2_hamiltonian, temperature=50.0, seed=seed,
                       max_iterations=1).run()
            if r.operators:
                first_ops.add(r.operators[0])
        assert len(first_ops) >= 2                       # genuinely random choice

    def test_annealing_reaches_fci(self, h2_hamiltonian, h2_fci):
        r = _vasqe(h2_hamiltonian, temperature=2.0, final_temperature=1e-3,
                   schedule="exponential", max_iterations=10, seed=1).run()
        assert r.optimal_energy == pytest.approx(h2_fci, abs=1e-6)

    def test_invalid_schedule_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            _vasqe(h2_hamiltonian, schedule="parabolic")

    def test_nonpositive_temperature_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            _vasqe(h2_hamiltonian, temperature=0.0)

    def test_is_ase_calculator(self):
        from ase import Atoms
        atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
                      cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)
        atoms.calc = VASQE(basis="FAO", h=0.30, temperature=1e-3, verbose=False,
                           profile=False, gradient_tolerance=1e-4,
                           max_iterations=10)
        atoms.get_potential_energy()
        assert isinstance(atoms.calc.vasqe_result, VASQEResult)


# --------------------------------------------------------------------------- #
# Excited-state extensions (deflation + subspace) inherit stochastic selection.
# --------------------------------------------------------------------------- #

class TestVASQEExcitedStates:
    def test_energy_levels_deflation(self, h2_hamiltonian, h2_fci):
        levels = _vasqe(h2_hamiltonian, temperature=1e-3).energy_levels(2)
        m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
        spectrum = np.sort(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).real)
        assert levels.ground_state_energy == pytest.approx(h2_fci, abs=1e-6)
        for e in levels.energies:                        # true eigenvalues
            assert float(np.min(np.abs(spectrum - e))) < 1e-5

    def test_subspace_vasqe(self, h2_hamiltonian, h2_fci):
        sv = SubspaceVASQE(h2_hamiltonian, "fermionic", num_states=2,
                           num_particles=(1, 1), n_spatial_orbitals=2,
                           temperature=0.5, optimizer=Optimizer("L-BFGS-B",
                                                                maxiter=2000),
                           verbose=False, profile=False, gradient_tolerance=1e-6,
                           max_iterations=20, seed=1)
        result = sv.run()
        assert result.num_states == 2
        assert result.energies[0] == pytest.approx(h2_fci, abs=1e-4)
        assert result.energies[1] >= result.energies[0] - 1e-9

    def test_subspace_vasqe_uses_stochastic_selection(self, h2_hamiltonian):
        # SubspaceVASQE must route selection through VASQE._select_operator.
        sv = SubspaceVASQE(h2_hamiltonian, "fermionic", num_states=2,
                           num_particles=(1, 1), n_spatial_orbitals=2,
                           temperature=1.0, verbose=False, profile=False)
        assert sv._select_operator.__qualname__.startswith("VASQE")


# --------------------------------------------------------------------------- #
# Selection hook is behaviour-preserving for ADAPTVQE.
# --------------------------------------------------------------------------- #

def test_schedule_names_exported():
    assert TEMPERATURE_SCHEDULES == ("constant", "exponential", "linear",
                                     "logarithmic")
