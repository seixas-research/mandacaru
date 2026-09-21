# -*- coding: utf-8 -*-
# file: test/test_growth_strategies.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""TETRIS growth and ansatz pruning: two literature ADAPT-VQE refinements.

``tetris=True`` (Anastasiou *et al.*, Phys. Rev. Research **6**, 013254, 2024)
appends every operator whose support is disjoint from the ones already taken
this step, in descending gradient order, instead of only the largest.  The
extra operators cost no measurement -- their gradients were screened anyway --
and they occupy the same circuit layer, so the ansatz grows denser and
shallower.

``prune=True`` (J. Chem. Theory Comput. **21**, 8720, 2025) removes, after each
growth, the one operator that has become irrelevant: the highest
``f_i = exp(-alpha x_i) / theta_i^2``, and only when its amplitude is below a
threshold set by the most recently added operators.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.adapt_vqe import (PRUNE_ALPHA, PRUNE_FRACTION,
                                            PRUNE_MIN_OPERATORS, PRUNE_RECENT,
                                            PRUNE_STEP_BUDGET)
from mandacaru.optimizers import Optimizer

# The classical optimizers used below, with the iteration budget and the
# convergence tolerance written out rather than left to the library default.
LBFGSB = Optimizer(method="L-BFGS-B", maxiter=2000, tol=1e-12)
#: A deliberately loose one.  Pruning exists to remove operators an
#: *under-converged* inner optimization left behind with near-zero amplitudes;
#: against `LBFGSB` above there is nothing to remove, which is its own
#: measurement (see `TestPruning`).
LOOSE = Optimizer(method="L-BFGS-B", maxiter=2000, tol=1e-8)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def beh2():
    atoms = Atoms("BeH2", positions=[[0, 0, -1.3], [0, 0, 0], [0, 0, 1.3]],
                  cell=[9.0] * 3)
    atoms.center()
    return atoms


def h4():
    """A linear H4 chain: correlated enough that the ansatz keeps growing.

    BeH2 converges in nine operators once the inner optimizer is tight, which
    leaves the growth strategies nothing to improve; this one still has slack.
    """
    atoms = Atoms("H4", positions=[[0, 0, 1.2 * i] for i in range(4)],
                  cell=[8.0] * 3)
    atoms.center()
    return atoms


def run(atoms, optimizer=LBFGSB, max_iterations=40, **options):
    work = atoms.copy()
    work.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35, pool="qeb",
                          optimizer=optimizer, max_iterations=max_iterations,
                          gradient_tolerance=1e-5, profile=True, trace=False,
                          **options)
    energy = work.get_potential_energy()
    return energy, work.calc.result


def fci(hamiltonian):
    matrix = hamiltonian.to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min())


class TestOptions:
    @pytest.mark.parametrize("option", ["tetris", "prune"])
    def test_defaults_are_off(self, option):
        calc = Mandacaru(method="adapt-vqe", basis="FAO")
        assert getattr(calc.solver, option) is False

    @pytest.mark.parametrize("option", ["tetris", "prune"])
    def test_only_booleans(self, option):
        with pytest.raises(ValueError, match=option):
            Mandacaru(method="adapt-vqe", basis="FAO", **{option: "yes"})

    def test_the_setup_block_says_which_growth_ran(self, tmp_path):
        from mandacaru.utils.logging import parse_output

        for options, growth, pruning in ((dict(), "single", "off"),
                                         (dict(tetris=True), "tetris", "off"),
                                         (dict(prune=True), "single", "on")):
            out = str(tmp_path / f"{growth}_{pruning}.txt")
            atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                          cell=[6.0] * 3)
            atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                                   pool="fermionic", max_iterations=2,
                                   txt=out, trace=False, profile=False,
                                   **options)
            atoms.get_potential_energy()
            setup = parse_output(out)["setup"]
            assert setup["growth"].startswith(growth)
            assert setup["pruning"].startswith(pruning)


class TestTetris:
    """Disjoint supports: the selection rule, then what it buys."""

    def test_selected_operators_have_disjoint_supports(self):
        """The selection rule itself, on a gradient vector we control.

        At the Hartree-Fock reference most QEB gradients vanish by Brillouin's
        theorem, so a real first step often has only one live operator; the
        rule is exercised with every operator live instead.
        """
        atoms = beh2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="qeb", max_iterations=1, tetris=True,
                               trace=False, profile=False)
        atoms.get_potential_energy()
        solver = atoms.calc.solver

        # Strictly descending, all live: the rule must walk them in order.
        grads = np.array([1.0 / (1 + i) for i in range(len(solver._pool_ops))])
        picks = solver._select_disjoint(grads)
        assert len(picks) > 1                      # otherwise nothing is tested
        assert picks[0] == 0                       # the largest gradient first
        assert picks == sorted(picks)              # descending gradient order
        seen = set()
        for index in picks:
            support = set(solver._pool_ops[index].support)
            assert not (support & seen)
            seen |= support

    def test_it_falls_back_to_one_operator(self):
        """A pool whose gradients all vanish but one still grows by one."""
        atoms = beh2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="qeb", max_iterations=1, tetris=True,
                               trace=False, profile=False)
        atoms.get_potential_energy()
        solver = atoms.calc.solver
        grads = np.zeros(len(solver._pool_ops))
        grads[3] = 1e-3
        assert solver._select_disjoint(grads) == [3]
        assert solver._select_disjoint(np.zeros_like(grads)) == [0]

    def test_shallower_at_the_same_accuracy(self):
        """The point of the method: fewer layers, no accuracy given up.

        On H4, where the ansatz still has room to grow: depth 463 -> 399 at
        the same energy, in 8 growth steps instead of 17.  The CNOT *count*
        may rise -- packing a layer means appending more operators per step --
        which is the trade the method makes.
        """
        atoms = h4()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain_e, plain = run(atoms, max_iterations=25)
            tetris_e, tetris = run(atoms, max_iterations=25, tetris=True)
        assert tetris_e == pytest.approx(plain_e, abs=1e-5)
        assert tetris.metrics.depth < plain.metrics.depth
        # Denser: fewer growth steps than operators, unlike the plain loop.
        assert len(tetris.iterations) < tetris.num_operators
        assert len(plain.iterations) == plain.num_operators
        # And the optimizer is handed fewer, larger problems.
        assert tetris.num_evaluations < plain.num_evaluations

    def test_it_still_saves_growth_steps_when_the_circuit_cannot_shrink(self):
        """BeH2 converges to the same nine operators with or without TETRIS.

        There is nothing left to compact -- the saving is in the *screening*,
        which runs once per growth step rather than once per operator.
        """
        atoms = beh2()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain_e, plain = run(atoms)
            tetris_e, tetris = run(atoms, tetris=True)
        assert tetris_e == pytest.approx(plain_e, abs=1e-6)
        assert tetris.metrics.cnot_count == plain.metrics.cnot_count
        assert len(tetris.iterations) < len(plain.iterations)
        assert tetris.num_evaluations < plain.num_evaluations


class TestPruning:
    def test_the_decision_factor_prefers_early_small_parameters(self):
        """``f = exp(-alpha x) / theta^2``: small amplitude, early position."""
        atoms = beh2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="qeb", max_iterations=1, prune=True,
                               trace=False, profile=False)
        atoms.get_potential_energy()
        solver = atoms.calc.solver

        class Fake:
            def __init__(self, n):
                self.num_parameters = n
                self.removed = None

            def remove(self, index):
                self.removed = index

                class Op:
                    label = f"op{index}"
                return Op()

        # Operator 0 is tiny and early; 1 and 2 are large; the last three set
        # the threshold.  0 must be the one chosen.
        theta = np.array([1e-6, 0.4, 0.5, 0.2, 0.3, 0.25])
        ansatz = Fake(len(theta))
        params, label = solver._prune_ansatz(ansatz, theta)
        assert ansatz.removed == 0 and label == "op0"
        assert len(params) == len(theta) - 1

    def test_nothing_is_pruned_when_every_amplitude_matters(self):
        atoms = beh2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="qeb", max_iterations=1, prune=True,
                               trace=False, profile=False)
        solver = atoms.calc.solver
        atoms.get_potential_energy()

        class Fake:
            num_parameters = 5

            def remove(self, index):                    # pragma: no cover
                raise AssertionError("should not prune")

        theta = np.array([0.3, 0.31, 0.29, 0.30, 0.32])
        params, label = solver._prune_ansatz(Fake(), theta)
        assert label is None
        assert np.array_equal(params, theta)

    def test_a_short_ansatz_is_left_alone(self):
        atoms = beh2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="qeb", max_iterations=1, prune=True,
                               trace=False, profile=False)
        solver = atoms.calc.solver
        atoms.get_potential_energy()

        class Fake:
            num_parameters = PRUNE_MIN_OPERATORS

            def remove(self, index):                    # pragma: no cover
                raise AssertionError("should not prune")

        theta = np.zeros(PRUNE_MIN_OPERATORS)
        assert solver._prune_ansatz(Fake(), theta)[1] is None

    def test_the_newest_operator_is_never_pruned(self):
        """It was just selected on the largest gradient; removing it would make
        the loop cycle on the same choice."""
        atoms = beh2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="qeb", max_iterations=1, prune=True,
                               trace=False, profile=False)
        solver = atoms.calc.solver
        atoms.get_potential_energy()

        class Fake:
            def __init__(self):
                self.num_parameters = 4
                self.removed = None

            def remove(self, index):
                self.removed = index

                class Op:
                    label = "x"
                return Op()

        # The last parameter is by far the smallest, but it is off limits.
        ansatz = Fake()
        solver._prune_ansatz(ansatz, np.array([1e-5, 0.3, 0.3, 1e-9]))
        assert ansatz.removed == 0

    def test_it_compacts_an_ansatz_a_loose_optimizer_left_behind(self):
        """What pruning is for: operators with near-zero amplitudes.

        Those appear when the inner optimization stops short, so this names a
        loose optimizer.  BeH2 then grows 15 operators / 412 CNOTs plain and
        10 / 260 pruned, at the same energy.
        """
        atoms = beh2()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain_e, plain = run(atoms, optimizer=LOOSE)
            pruned_e, pruned = run(atoms, optimizer=LOOSE, prune=True)
        assert pruned.num_operators < plain.num_operators
        assert pruned.metrics.cnot_count < plain.metrics.cnot_count
        assert pruned_e == pytest.approx(plain_e, abs=1e-5)
        assert pruned.pruned_operators                 # it recorded what it cut
        assert all(isinstance(step, int) and isinstance(label, str)
                   for step, label in pruned.pruned_operators)

    def test_a_tightly_converged_ansatz_has_nothing_to_prune(self):
        """The measurement that bounds the method's usefulness.

        With the library's own tolerance the same BeH2 run reaches nine
        operators and every amplitude matters, so pruning changes nothing.  It
        was compensating for an under-converged inner optimizer, not for a
        redundancy of ADAPT itself.
        """
        atoms = beh2()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain_e, plain = run(atoms)
            pruned_e, pruned = run(atoms, prune=True)
        assert pruned.num_operators == plain.num_operators
        assert pruned.metrics.cnot_count == plain.metrics.cnot_count
        assert pruned_e == pytest.approx(plain_e, abs=1e-8)
        assert not pruned.pruned_operators

    def test_the_growth_steps_are_bounded(self):
        """Pruning can hand back the operator the next step re-selects; the
        loop counts operators, so the steps need their own budget."""
        assert PRUNE_STEP_BUDGET >= 1
        atoms = beh2()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _energy, result = run(atoms, prune=True)
        assert len(result.iterations) <= PRUNE_STEP_BUDGET * 40


def test_the_constants_match_the_paper():
    assert PRUNE_ALPHA == 10.0        # exp(-alpha x), alpha ~ 10
    assert PRUNE_FRACTION == 0.1      # 0.1 of the recent mean amplitude
    assert PRUNE_RECENT == 4          # averaged over the last N_L = 4
