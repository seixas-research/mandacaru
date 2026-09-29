# -*- coding: utf-8 -*-
# file: test_mcas.py

"""The Markov Chain Ansatz Search, checked without any Hamiltonian.

Everything here runs on pool *indices* and a synthetic cost, so the proposal
probabilities and the Metropolis-Hastings chain are verified exactly -- by
enumerating a small architecture space and building its transition matrix --
before VQE energies enter (``test_vasqa.py``).
"""

import itertools
import math

import numpy as np
import pytest

from mandacaru.algorithms.mcas import (MOVES, Action, ProposalKernel,
                                       TemperatureSchedule, log_acceptance,
                                       metropolis_accept)

#: Deliberately unequal, so a symmetric-proposal shortcut would show.
WEIGHTS = {"insert": 1.0, "delete": 2.0, "replace": 0.7, "swap": 1.3}


def _space(pool_size, max_length, min_length=0):
    return [c for length in range(min_length, max_length + 1)
            for c in itertools.product(range(pool_size), repeat=length)]


def _proposal_rows(kernel, space):
    """``q(.|C)`` for every ``C``, summed over the enumerated actions."""
    rows = {}
    for c in space:
        row: dict = {}
        for action, p in kernel.actions(c):
            d = action.apply(c)
            row[d] = row.get(d, 0.0) + p
        rows[c] = row
    return rows


def _transition_matrix(kernel, space, cost, beta):
    index = {c: i for i, c in enumerate(space)}
    rows = _proposal_rows(kernel, space)
    P = np.zeros((len(space), len(space)))
    for c, row in rows.items():
        for d, q in row.items():
            ell = log_acceptance(beta, cost[c], cost[d], kernel.log_q(d, c),
                                 kernel.log_q(c, d))
            P[index[c], index[d]] += q * math.exp(min(0.0, ell))
        P[index[c], index[c]] += 1.0 - P[index[c]].sum()
    return P


class TestActions:
    def test_insert_reaches_every_slot(self):
        c = (4, 5)
        got = [Action("insert", j, operator=9).apply(c) for j in range(3)]
        assert got == [(9, 4, 5), (4, 9, 5), (4, 5, 9)]

    def test_delete_replace_and_swap(self):
        c = (1, 2, 3)
        assert Action("delete", 1).apply(c) == (1, 3)
        assert Action("replace", 2, operator=7).apply(c) == (1, 2, 7)
        assert Action("swap", 0, other=2).apply(c) == (3, 2, 1)

    def test_insert_transfers_a_zero_angle_into_the_slot(self):
        theta = np.array([0.1, 0.2])
        out = Action("insert", 1, operator=0).transfer(theta)
        np.testing.assert_array_equal(out, [0.1, 0.0, 0.2])

    def test_delete_takes_its_angle_with_it(self):
        out = Action("delete", 0).transfer([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(out, [0.2, 0.3])

    def test_replace_resets_only_the_replaced_angle(self):
        out = Action("replace", 1, operator=4).transfer([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(out, [0.1, 0.0, 0.3])

    def test_swap_moves_each_angle_with_its_operator(self):
        out = Action("swap", 0, other=2).transfer([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(out, [0.3, 0.2, 0.1])

    def test_transfer_does_not_mutate_the_input(self):
        theta = np.array([0.1, 0.2])
        Action("swap", 0, other=1).transfer(theta)
        np.testing.assert_array_equal(theta, [0.1, 0.2])


class TestValidity:
    def test_the_empty_architecture_can_only_grow(self):
        kernel = ProposalKernel(3, WEIGHTS, 0, 4)
        p = kernel.move_probabilities(())
        assert p == {"insert": 1.0, "delete": 0.0, "replace": 0.0,
                     "swap": 0.0}

    def test_the_longest_architecture_cannot_grow(self):
        kernel = ProposalKernel(3, None, 0, 2)
        assert not kernel.valid_moves((0, 1))["insert"]
        assert kernel.move_probabilities((0, 1))["insert"] == 0.0

    def test_min_length_blocks_deletion(self):
        kernel = ProposalKernel(3, None, 1, 3)
        assert not kernel.valid_moves((2,))["delete"]

    def test_swap_needs_two_different_labels(self):
        kernel = ProposalKernel(3, None, 0, 4)
        assert not kernel.valid_moves((1, 1, 1))["swap"]
        assert kernel.valid_moves((1, 1, 2))["swap"]

    def test_replace_needs_a_second_operator(self):
        kernel = ProposalKernel(1, None, 0, 3)
        assert not kernel.valid_moves((0,))["replace"]

    @pytest.mark.parametrize("c", [(), (0,), (0, 0), (0, 1, 1), (2, 1, 0, 2)])
    def test_move_probabilities_sum_to_one(self, c):
        p = ProposalKernel(3, WEIGHTS, 0, 4).move_probabilities(c)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-15)
        assert set(p) == set(MOVES)

    @pytest.mark.parametrize("weights, match", [
        ({"insert": 0.0, "delete": 1.0}, "insert and delete"),
        ({"insert": 1.0}, "insert and delete"),
        ({"insert": 1.0, "delete": 1.0, "grow": 1.0}, "unknown move"),
        ({"insert": 1.0, "delete": -1.0}, "non-negative"),
    ])
    def test_bad_weights_are_refused(self, weights, match):
        with pytest.raises(ValueError, match=match):
            ProposalKernel(3, weights, 0, 4)

    def test_bad_lengths_and_empty_pool_are_refused(self):
        with pytest.raises(ValueError, match="min_length < max_length"):
            ProposalKernel(3, None, 3, 3)
        with pytest.raises(ValueError, match="pool is empty"):
            ProposalKernel(0, None, 0, 3)


class TestProposalProbabilities:
    @pytest.mark.parametrize("pool_size, max_length", [(1, 3), (2, 3), (3, 3)])
    def test_every_row_is_normalized(self, pool_size, max_length):
        kernel = ProposalKernel(pool_size, WEIGHTS, 0, max_length)
        for c, row in _proposal_rows(kernel,
                                     _space(pool_size, max_length)).items():
            assert sum(row.values()) == pytest.approx(1.0, abs=1e-14), c
            assert c not in row, "a proposal must change the architecture"

    @pytest.mark.parametrize("pool_size, max_length", [(2, 3), (3, 3)])
    def test_closed_form_matches_the_action_sum(self, pool_size, max_length):
        kernel = ProposalKernel(pool_size, WEIGHTS, 0, max_length)
        space = _space(pool_size, max_length)
        rows = _proposal_rows(kernel, space)
        for c in space:
            for d in space:
                q = rows[c].get(d, 0.0)
                log_q = kernel.log_q(d, c)
                if q == 0.0:
                    assert log_q == -math.inf, (c, d)
                else:
                    assert log_q == pytest.approx(math.log(q), abs=1e-12)

    def test_repeated_labels_sum_both_insertion_slots(self):
        """MCAS.md section 6.1: q((A,A)|(A)) = p_I/M, not p_I/(2M)."""
        kernel = ProposalKernel(3, None, 0, 4)
        p_insert = kernel.move_probabilities((0,))["insert"]
        assert math.exp(kernel.log_q((0, 0), (0,))) == pytest.approx(
            p_insert / 3)
        p_delete = kernel.move_probabilities((0, 0))["delete"]
        assert math.exp(kernel.log_q((0,), (0, 0))) == pytest.approx(p_delete)

    def test_every_proposal_has_reverse_support(self):
        kernel = ProposalKernel(3, WEIGHTS, 0, 3)
        for c, row in _proposal_rows(kernel, _space(3, 3)).items():
            for d in row:
                assert kernel.log_q(c, d) > -math.inf, (c, d)

    def test_sampler_draws_from_the_stated_proposal(self):
        kernel = ProposalKernel(2, WEIGHTS, 0, 3)
        c = (0, 1, 1)
        rng = np.random.default_rng(11)
        n = 40_000
        counts: dict = {}
        for _ in range(n):
            d = kernel.sample(c, rng).apply(c)
            counts[d] = counts.get(d, 0) + 1
        for d, hits in counts.items():
            q = math.exp(kernel.log_q(d, c))
            # Five binomial standard deviations.
            assert abs(hits / n - q) < 5 * math.sqrt(q * (1 - q) / n), d

    def test_a_long_walk_stays_inside_the_length_bounds(self):
        kernel = ProposalKernel(4, WEIGHTS, 1, 5)
        rng = np.random.default_rng(5)
        c = (0,)
        for _ in range(3000):
            action = kernel.sample(c, rng)
            c = action.apply(c)
            assert 1 <= len(c) <= 5
            assert all(0 <= mu < 4 for mu in c)


class TestDetailedBalance:
    """MCAS.md section 15, item 3: the exhaustive finite-space check."""

    @pytest.mark.parametrize("pool_size, max_length", [(2, 3), (3, 2)])
    @pytest.mark.parametrize("beta", [0.0, 1.7, 25.0])
    def test_the_target_is_stationary_and_balanced(self, pool_size,
                                                   max_length, beta):
        kernel = ProposalKernel(pool_size, WEIGHTS, 0, max_length)
        space = _space(pool_size, max_length)
        rng = np.random.default_rng(3)
        cost = dict(zip(space, rng.normal(size=len(space))))
        P = _transition_matrix(kernel, space, cost, beta)
        assert np.all(P >= -1e-15)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-14)
        pi = np.array([math.exp(-beta * cost[c]) for c in space])
        pi /= pi.sum()
        flow = pi[:, None] * P
        assert np.abs(flow - flow.T).max() < 1e-15
        assert np.abs(pi @ P - pi).max() < 1e-14

    def test_the_chain_is_irreducible(self):
        kernel = ProposalKernel(2, WEIGHTS, 0, 3)
        space = _space(2, 3)
        P = _transition_matrix(kernel, space, dict.fromkeys(space, 0.0), 1.0)
        reach = (P > 0).astype(int)
        closure = np.linalg.matrix_power(reach + np.eye(len(space), dtype=int),
                                         len(space))
        assert np.all(closure > 0)

    def test_long_run_occupation_matches_the_target(self):
        """Rejected steps count as repeated states; drop them and this fails."""
        kernel = ProposalKernel(2, WEIGHTS, 0, 2)
        space = _space(2, 2)
        cost = {c: 0.3 * len(c) - 0.5 * c.count(1) for c in space}
        beta = 2.0
        rng = np.random.default_rng(2026)
        c, n = (), 60_000
        visits = dict.fromkeys(space, 0)
        for _ in range(n):
            action = kernel.sample(c, rng)
            d = action.apply(c)
            ell = log_acceptance(beta, cost[c], cost[d], kernel.log_q(d, c),
                                 kernel.log_q(c, d))
            if metropolis_accept(ell, rng):
                c = d
            visits[c] += 1
        pi = np.array([math.exp(-beta * cost[s]) for s in space])
        pi /= pi.sum()
        freq = np.array([visits[s] / n for s in space])
        # Correlated samples: a loose absolute tolerance, far above the noise
        # of this chain and far below any proposal-ratio error.
        np.testing.assert_allclose(freq, pi, atol=0.015)


class TestAcceptance:
    def test_the_hastings_ratio_enters_in_log_space(self):
        ell = log_acceptance(2.0, 1.0, 1.5, math.log(0.2), math.log(0.1))
        assert ell == pytest.approx(-2.0 * 0.5 + math.log(0.1 / 0.2))

    def test_zero_reverse_support_is_a_certain_rejection(self):
        assert log_acceptance(1.0, 0.0, -5.0, math.log(0.5),
                              -math.inf) == -math.inf

    @pytest.mark.parametrize("bad", [math.inf, math.nan])
    def test_a_non_finite_cost_is_rejected(self, bad):
        assert log_acceptance(1.0, 0.0, bad, 0.0, 0.0) == -math.inf

    def test_zero_forward_support_is_an_error(self):
        with pytest.raises(ValueError, match="forward proposal"):
            log_acceptance(1.0, 0.0, 0.0, -math.inf, 0.0)

    def test_zero_temperature_limit(self):
        inf = math.inf
        assert log_acceptance(inf, 1.0, 0.5, 0.0, math.log(0.1)) == inf
        assert log_acceptance(inf, 1.0, 1.5, 0.0, 0.0) == -inf
        # A tie keeps the Hastings ratio.
        assert log_acceptance(inf, 1.0, 1.0, math.log(0.4),
                              math.log(0.1)) == pytest.approx(math.log(0.25))

    def test_infinite_temperature_keeps_the_hastings_ratio(self):
        ell = log_acceptance(0.0, 0.0, 99.0, math.log(0.5), math.log(0.25))
        assert ell == pytest.approx(math.log(0.5))

    def test_an_improvement_can_still_be_rejected(self):
        """MCAS.md section 7: a lower cost with an unfavorable ratio."""
        ell = log_acceptance(1.0, 0.0, -0.1, math.log(0.9), math.log(0.01))
        assert ell < 0.0

    def test_acceptance_frequency_is_exp_ell(self):
        rng = np.random.default_rng(0)
        ell = math.log(0.3)
        n = 20_000
        rate = sum(metropolis_accept(ell, rng) for _ in range(n)) / n
        assert rate == pytest.approx(0.3, abs=5 * math.sqrt(0.21 / n))

    def test_every_decision_draws_exactly_once(self):
        """The random stream must not depend on the outcome."""
        for ell in (math.inf, 0.5, -0.3, -math.inf):
            a, b = np.random.default_rng(9), np.random.default_rng(9)
            metropolis_accept(ell, a)
            b.random()
            assert a.random() == b.random()


class TestTemperatureSchedule:
    def test_a_fixed_temperature(self):
        s = TemperatureSchedule(0.5, 10)
        assert not s.annealed
        assert s.temperature(0) == s.temperature(9) == 0.5
        assert s.beta(3) == 2.0

    def test_zero_is_the_greedy_limit(self):
        assert TemperatureSchedule(0.0, 10).beta(0) == math.inf

    def test_the_geometric_anneal_hits_both_ends(self):
        s = TemperatureSchedule({"initial": 1.0, "final": 1e-3}, 4)
        assert s.temperature(0) == pytest.approx(1.0)
        assert s.temperature(1) == pytest.approx(0.1)
        assert s.temperature(3) == pytest.approx(1e-3)
        assert s.temperature(99) == pytest.approx(1e-3)

    @pytest.mark.parametrize("spec", [-1.0, math.nan, {"initial": 1.0},
                                      {"initial": 1.0, "final": 0.0},
                                      {"initial": 1.0, "final": 0.1, "k": 2}])
    def test_bad_temperatures_are_refused(self, spec):
        with pytest.raises(ValueError):
            TemperatureSchedule(spec, 10)
