# -*- coding: utf-8 -*-
# file: test_orbital_tracking.py

"""Molecular orbitals followed between geometries, so a transferred ansatz
means the same thing at the next geometry.

The matching and the renaming are checked on synthetic overlaps; the whole
path through ``Mandacaru(method="vasqa", transfer=True)`` is checked by
relabeling the orbitals of a geometry on purpose -- a sign flip and two
orbitals exchanged, as an eigensolver may do -- and requiring the
transferred ansatz to reproduce the previous energy before any
re-optimization.
"""

import numpy as np
import pytest
from ase import Atoms

import mandacaru.core.hamiltonian as hamiltonian_module
from mandacaru import Mandacaru
from mandacaru.algorithms.orbital_tracking import (OrbitalMatch,
                                                   OrbitalSnapshot,
                                                   match_orbitals,
                                                   occupation_blocks,
                                                   transfer_ansatz)
from mandacaru.units import HARTREE_TO_EV


def _match(permutation, signs):
    n = len(permutation)
    return OrbitalMatch(np.asarray(permutation), np.asarray(signs),
                        np.ones(n))


class TestMatching:
    def test_a_relabeled_set_is_recovered(self):
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        near = np.eye(5) + 0.05 * Q          # nearly the identity
        permutation, signs = np.array([1, 0, 2, 4, 3]), \
            np.array([1, -1, 1, 1, -1])
        O = np.zeros((5, 5))
        O[:, permutation] = near * signs
        match = match_orbitals(O, occupation_blocks(5, (2, 2)))
        np.testing.assert_array_equal(match.permutation, permutation)
        np.testing.assert_array_equal(match.signs, signs)
        assert match.confidence > 0.9 and not match.is_identity

    def test_an_occupied_orbital_is_never_matched_to_an_empty_one(self):
        """Crossing the Fermi level is a reorganization, not a relabeling:
        the matching stays in its block and the confidence shows it."""
        O = np.array([[0.1, 0.99], [0.99, 0.1]])
        match = match_orbitals(O, occupation_blocks(2, (1, 1)))
        np.testing.assert_array_equal(match.permutation, [0, 1])
        assert match.confidence == pytest.approx(0.1)

    def test_occupation_blocks(self):
        assert occupation_blocks(5, (2, 2)) == [range(0, 2), range(2, 5)]
        assert occupation_blocks(4, (2, 1)) == [range(0, 1), range(1, 2),
                                                range(2, 4)]


class TestRenaming:
    POOL = {"D(0,4->2,6)", "D(0,4->3,7)", "D(1,4->2,6)", "S(0->2)",
            "S(0->3)", "QD(0,1->2,3)", "QS(0->3)"}

    def test_the_identity_changes_nothing(self):
        ops, angles = transfer_ansatz(["iP[XYII]"], [0.3],
                                      _match([0, 1, 2, 3], [1] * 4), 4,
                                      ["iP[XYII]"])
        assert ops == ["iP[XYII]"] and angles[0] == 0.3

    def test_a_sign_flip_flips_every_excitation_odd_in_it(self):
        match = _match([0, 1, 2, 3], [1, 1, -1, 1])
        ops, angles = transfer_ansatz(["S(0->2)", "D(0,4->2,6)"],
                                      [0.1, 0.2], match, 4, self.POOL)
        # S(0->2) touches orbital 2 once; the double touches it twice.
        assert ops == ["S(0->2)", "D(0,4->2,6)"]
        np.testing.assert_allclose(angles, [-0.1, 0.2])

    def test_exchanged_orbitals_rename_the_operator(self):
        match = _match([0, 1, 3, 2], [1, 1, 1, 1])
        ops, angles = transfer_ansatz(["S(0->2)", "D(0,4->2,6)"],
                                      [0.1, 0.2], match, 4, self.POOL)
        assert ops == ["S(0->3)", "D(0,4->3,7)"]
        np.testing.assert_allclose(angles, [0.1, 0.2])

    def test_reordering_fermionic_indices_costs_a_sign_qubit_ones_not(self):
        match = _match([1, 0, 3, 2], [1, 1, 1, 1])
        pool = {"D(0,1->2,3)", "QD(0,1->2,3)"}
        ops, angles = transfer_ansatz(["D(0,1->2,3)", "QD(0,1->2,3)"],
                                      [0.1, 0.2], match, 4, pool)
        assert ops == ["D(0,1->2,3)", "QD(0,1->2,3)"]
        np.testing.assert_allclose(angles, [0.1, 0.2])   # two swaps: (+1)
        match = _match([1, 0, 2, 3], [1, 1, 1, 1])
        _, angles = transfer_ansatz(["D(0,1->2,3)", "QD(0,1->2,3)"],
                                    [0.1, 0.2], match, 4, pool)
        np.testing.assert_allclose(angles, [-0.1, 0.2])

    def test_an_operator_that_cannot_be_followed_is_named(self):
        reason = transfer_ansatz(["iP[XYII]"], [0.3],
                                 _match([1, 0, 2, 3], [1] * 4), 4,
                                 ["iP[XYII]"])
        assert isinstance(reason, str) and "iP[XYII]" in reason

    def test_without_orbitals_the_reason_is_given(self):
        assert "directly" in OrbitalSnapshot.from_integrals(None, 2)


def _h4(spacing=0.95):
    atoms = Atoms("H4", positions=[[0, 0, k * spacing] for k in range(4)])
    atoms.center(vacuum=2.5)
    return atoms


def _calc(**options):
    options = {"basis": "HAO", "h": 0.4, "pool": "fermionic",
               "max_steps": 20, "transfer_steps": 0, "max_length": 6,
               "seed": 1, "transfer": True, "temperature": 0.0,
               "profile": False, "trace": False, "record": False, **options}
    return Mandacaru(method="vasqa", **options)


@pytest.fixture
def relabel(monkeypatch):
    """Relabel the next geometry's orbitals: exchange 2 and 3, flip 1 and 2."""
    original = hamiltonian_module.molecular_orbital_integrals
    switch = {"on": False}

    def relabeled(*args, **kwargs):
        h, eri, orbitals = original(*args, **kwargs)
        if not switch["on"]:
            return h, eri, orbitals
        n = h.shape[0]
        p = np.arange(n)
        p[[2, 3]] = [3, 2]
        s = np.ones(n)
        s[[1, 2]] = -1.0
        return (h[np.ix_(p, p)] * np.outer(s, s),
                eri[np.ix_(p, p, p, p)]
                * np.einsum("p,q,r,t->pqrt", s, s, s, s),
                np.asarray(orbitals)[:, p] * s)

    monkeypatch.setattr(hamiltonian_module, "molecular_orbital_integrals",
                        relabeled)
    return switch


class TestTransferFollowsTheOrbitals:
    def _start_energy(self, calc):
        """The transferred ansatz's energy before any re-optimization."""
        solver = calc.solver
        architecture, x0, start = solver._transferred_start(
            [op.label for op in solver._pool_ops])
        energy = solver.ansatz_energy(solver._ansatz_for(architecture), x0)
        return energy * HARTREE_TO_EV, start

    def test_relabeled_orbitals_reproduce_the_previous_energy(self, relabel):
        calc = _calc()
        atoms = _h4()
        atoms.calc = calc
        before = atoms.get_potential_energy()
        relabel["on"] = True
        calc.reset()
        atoms.get_potential_energy()
        energy, start = self._start_energy(calc)
        assert "2 reordered, 2 sign(s) aligned" in start
        assert energy == pytest.approx(before, abs=1e-8)

    def test_a_poor_match_rebuilds_the_ansatz(self):
        calc = _calc(transfer_threshold=1.0)
        for spacing in (0.95, 1.05):
            atoms = _h4(spacing)
            atoms.calc = calc
            atoms.get_potential_energy()
        assert "transfer_threshold 1" in calc.result.start
        assert calc.result.start_operators == []

    def test_another_molecule_is_not_a_previous_geometry(self):
        calc = _calc()
        for atoms in (_h4(), Atoms("H2", positions=[[0, 0, 0],
                                                    [0, 0, 0.74]])):
            atoms.center(vacuum=2.5)
            atoms.calc = calc
            atoms.get_potential_energy()
        assert "another system or pool" in calc.result.start
