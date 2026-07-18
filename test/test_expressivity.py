# -*- coding: utf-8 -*-
# file: test_expressivity.py

"""Expressibility metric: Haar distribution, KL divergence and ADAPT tracking."""

import numpy as np
import pytest

from carcara.algorithms import (ADAPTExpressivityTracker, AdaptVQE,
                                ExpressibilityResult, active_space_dimension,
                                calculate_haar_distribution,
                                calculate_kl_divergence, compute_expressibility,
                                estimate_effective_dimension,
                                sample_pqc_fidelities, track_adapt_expressivity)
from carcara.circuits import UCCSD
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid


@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.25)
    return MolecularIntegrals(nuclei, minimal_fao_basis(nuclei),
                               grid).molecular_hamiltonian(mo_basis=True,
                                                           n_electrons=2)


def _haar_states(n, d, rng):
    v = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Haar distribution.
# --------------------------------------------------------------------------- #

class TestHaarDistribution:
    @pytest.mark.parametrize("d", [2, 4, 16, 64])
    def test_bins_are_a_normalized_distribution(self, d):
        p = calculate_haar_distribution(d, 50)
        assert len(p) == 50
        assert np.all(p >= 0)
        assert p.sum() == pytest.approx(1.0)

    def test_dimension_below_two_rejected(self):
        with pytest.raises(ValueError):
            calculate_haar_distribution(1, 50)


# --------------------------------------------------------------------------- #
# KL divergence.
# --------------------------------------------------------------------------- #

class TestKLDivergence:
    def test_haar_random_states_score_near_zero(self):
        rng = np.random.default_rng(0)
        d = 8
        a, b = _haar_states(6000, d, rng), _haar_states(6000, d, rng)
        F = np.abs(np.sum(np.conj(a) * b, axis=1)) ** 2
        kl = calculate_kl_divergence(F, num_qubits=3, dim=d, num_bins=75)
        assert kl < 0.05                       # matches the Haar reference

    def test_inexpressive_circuit_scores_high(self):
        # A circuit whose states barely differ -> fidelities pile up near 1.
        F = np.full(5000, 1.0)
        kl = calculate_kl_divergence(F, num_qubits=3, dim=8, num_bins=75)
        assert kl > 1.0

    def test_is_non_negative(self):
        rng = np.random.default_rng(1)
        F = rng.uniform(0, 1, 2000)
        assert calculate_kl_divergence(F, num_qubits=2, num_bins=50) >= 0.0

    def test_default_dimension_is_full_space(self):
        # With no dim/particles, d defaults to 2**num_qubits: a delta at F=1 has a
        # larger gap from a bigger Haar space.
        F = np.full(3000, 1.0)
        e_small = calculate_kl_divergence(F, num_qubits=2, num_bins=50)   # d=4
        e_large = calculate_kl_divergence(F, num_qubits=4, num_bins=50)   # d=16
        assert e_large > e_small


# --------------------------------------------------------------------------- #
# Effective dimension.
# --------------------------------------------------------------------------- #

class TestEffectiveDimension:
    def test_active_space_dimension(self):
        assert active_space_dimension(4, (1, 1)) == 4          # H2
        assert active_space_dimension(8, (2, 2)) == 36         # H4
        assert active_space_dimension(4) == 16                 # full space

    def test_empirical_matches_uccsd_sector(self):
        # UCCSD conserves particle number and Sz: it spans exactly the 4-dim
        # (n_alpha, n_beta) = (1, 1) sector for H2.
        ansatz = UCCSD(2, (1, 1))
        d = estimate_effective_dimension(ansatz, num_probe=200,
                                         rng=np.random.default_rng(2))
        assert d == 4


# --------------------------------------------------------------------------- #
# Fidelity sampling and the high-level driver.
# --------------------------------------------------------------------------- #

class TestSamplingAndDriver:
    def test_fidelities_shape_and_range(self):
        ansatz = UCCSD(2, (1, 1))
        F = sample_pqc_fidelities(ansatz, num_samples=500,
                                  rng=np.random.default_rng(3))
        assert F.shape == (500,)
        assert np.all((F >= -1e-12) & (F <= 1.0 + 1e-12))

    def test_zero_parameter_ansatz_is_constant(self):
        # An ADAPT ansatz with no operators yields the fixed reference: F == 1.
        from carcara.algorithms.adapt_vqe import AdaptAnsatz
        ansatz = AdaptAnsatz(4, occupied=(0, 2))
        F = sample_pqc_fidelities(ansatz, num_samples=50,
                                  rng=np.random.default_rng(4))
        assert np.allclose(F, 1.0)

    def test_compute_expressibility_result(self):
        ansatz = UCCSD(2, (1, 1))
        res = compute_expressibility(ansatz, num_samples=1000, bins=50,
                                     num_particles=(1, 1),
                                     rng=np.random.default_rng(5))
        assert isinstance(res, ExpressibilityResult)
        assert res.dimension == 4                  # number-conserving sector
        assert res.num_qubits == 4                 # inferred from the ansatz
        assert res.kl_divergence >= 0.0
        assert res.fidelities.shape == (1000,)

    def test_num_qubits_inferred_from_generator(self):
        res = compute_expressibility(UCCSD(2, (1, 1)), num_samples=200,
                                     num_particles=(1, 1),
                                     rng=np.random.default_rng(6))
        assert res.num_qubits == 4


# --------------------------------------------------------------------------- #
# ADAPT-VQE tracking.
# --------------------------------------------------------------------------- #

class TestADAPTTracking:
    def test_tracker_records_one_step_per_operator(self, h2_hamiltonian):
        adapt = AdaptVQE(h2_hamiltonian, "fermionic", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False)
        tracker = ADAPTExpressivityTracker(4, num_particles=(1, 1),
                                           num_samples=200, bins=50,
                                           rng=np.random.default_rng(7))
        adapt.run(callback=tracker, max_iterations=3, gradient_tol=1e-8)
        assert len(tracker.history) >= 1
        # Operator counts increase by one each accepted step.
        assert tracker.num_operators == list(range(1, len(tracker.history) + 1))
        assert all(np.isfinite(e) for e in tracker.scores)
        assert tracker.dimension == 4

    def test_track_helper_returns_result_and_history(self, h2_hamiltonian):
        adapt = AdaptVQE(h2_hamiltonian, "fermionic", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False)
        result, history = track_adapt_expressivity(
            adapt, num_samples=200, rng=np.random.default_rng(8),
            max_iterations=2, gradient_tol=1e-8)
        assert result.num_operators == len(history)
        assert history[0].energy is not None
