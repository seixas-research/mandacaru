# -*- coding: utf-8 -*-
# file: test/algorithms/test_scf_stability.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Stability of a converged spin-unrestricted Kohn-Sham state, and the
``max_iter`` / ``stability`` options of ``method="dft"``."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms import scf_stability


class _Quadratic:
    """A stand-in for the rotations: the gradient of a fixed quadratic form,
    so the Hessian is known exactly."""

    def __init__(self, hessian):
        self.H = hessian
        self.dim = hessian.shape[0]

    def diagonal(self):
        return np.diag(self.H).copy()

    def gradient(self, x):
        return self.H @ x, 0.0


def _hessian(dim, lowest, seed=3):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    values = np.sort(rng.uniform(0.3, 2.0, size=dim))
    values[0] = lowest
    return Q @ np.diag(values) @ Q.T, Q[:, 0]


def test_lowest_mode_finds_a_negative_eigenvalue():
    H, vector = _hessian(40, -0.05)
    value, mode, products = scf_stability.lowest_mode(
        _Quadratic(H), np.zeros(40), max_products=60)
    assert value < scf_stability.STABILITY_THRESHOLD
    assert value >= -0.05 - 1e-6              # a Ritz value bounds from above
    assert products <= 60
    assert np.linalg.norm(mode) == pytest.approx(1.0)


def test_lowest_mode_of_a_minimum_stays_positive():
    H, _ = _hessian(40, 0.4)
    value, _mode, products = scf_stability.lowest_mode(
        _Quadratic(H), np.zeros(40), max_products=60)
    assert value > 0.0
    assert products < 60


@pytest.mark.parametrize("bad", (0, -3, 1.5, True, "10"))
def test_max_iter_is_validated_when_the_calculator_is_built(bad):
    with pytest.raises(ValueError, match="max_iter"):
        Mandacaru(method="dft", max_iter=bad)


def test_stability_option_is_validated():
    with pytest.raises(ValueError, match="stability"):
        Mandacaru(method="dft", stability="yes")


def _hydrogen_chain(displacement: float = 0.0, **options) -> Atoms:
    radius = 2.5
    atoms = Atoms("H3", positions=radius * np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]]),
        cell=[radius + 6, radius + 6, 6])
    atoms.center()
    atoms.positions[2, 0] += displacement
    atoms.set_initial_magnetic_moments([1.0, 0, 0])
    atoms.calc = Mandacaru(
        method="dft", xc="lda", grid_spacing=0.3, trace=False,
        paw_dataset="lda-sr", basis={"name": "PAW-LCAO", "size": "SZ"},
        **options)
    return atoms


def test_equilateral_h3_doublet_leaves_the_symmetric_ensemble():
    """K36: the exact equilateral H3 doublet used to stop at a symmetric
    stationary point (a half-filled degenerate pair, or its integer saddle
    650 meV above the doublet) and report it converged; the ensemble is
    probed for a lower integer state, which the stability check follows to
    the doublet."""
    exact = _hydrogen_chain(0.0)
    energy = exact.get_potential_energy()
    scf = exact.calc._solver._scf
    assert scf.converged
    assert scf.stability.outcome == "lowered"
    assert scf.stability.energy_lowering > 0.1 / 27.2114   # > 100 meV
    assert scf.occupations_alpha is None                    # integer filling
    bent = _hydrogen_chain(0.003)
    assert bent.get_potential_energy() == pytest.approx(energy, abs=0.02)
    plain = _hydrogen_chain(0.0, stability=False)
    assert plain.get_potential_energy() > energy + 0.1
    assert plain.calc._solver._scf.stability is None


def test_a_stalled_loop_continues_with_newton_steps(monkeypatch):
    """After ``SECOND_ORDER_AFTER`` plain iterations an integer-occupation
    loop takes Newton-CG steps and reaches the plain loop's energy."""
    from ase.build import molecule

    from mandacaru.algorithms import dft

    def methyl():
        atoms = molecule("CH3")
        atoms.center(vacuum=3.5)
        atoms.calc = Mandacaru(
            method="dft", xc="lda", grid_spacing=0.3, trace=False,
            stability=False, basis={"name": "PAW-LCAO", "size": "SZ"})
        return atoms

    plain = methyl()
    energy = plain.get_potential_energy()
    monkeypatch.setattr(dft, "SECOND_ORDER_AFTER", 4)
    rescued = methyl()
    assert rescued.get_potential_energy() == pytest.approx(energy, abs=1e-5)
    scf = rescued.calc._solver._scf
    assert scf.converged
    assert scf.history[3]["iter"] == 4        # the plain iterations first
    assert scf.n_iterations < plain.calc._solver._scf.n_iterations


@pytest.mark.parametrize("xc, integer", [("lda", False), ("pbe", True)])
def test_a_hydroxyl_ensemble_is_kept_only_without_a_lower_integer_state(
        xc, integer):
    """OH's beta pi hole: under LDA no integer-occupation state converges
    below the half-and-half ensemble, which stays; under PBE one lies 142 meV
    below it and the check takes it."""
    atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                  cell=[6, 6, 7])
    atoms.center()
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZ"})
    atoms.get_potential_energy()
    scf = atoms.calc.result.scf
    assert (scf.occupations_beta is None) == integer
    assert scf.stability.outcome == ("lowered" if integer else "ensemble")
    if integer:
        assert scf.stability.energy_lowering == pytest.approx(
            0.142 / 27.2114, abs=2e-3 / 27.2114)


def test_the_iteration_limit_is_the_option():
    atoms = _hydrogen_chain(0.003, max_iter=3)
    with pytest.raises(RuntimeError, match="did not converge"):
        atoms.get_potential_energy()
