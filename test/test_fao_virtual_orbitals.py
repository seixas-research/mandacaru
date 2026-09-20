# -*- coding: utf-8 -*-
# file: test/test_fao_virtual_orbitals.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The FAO basis's ``virtual_orbitals`` option: unoccupied levels on demand.

``basis={"name": "FAO", "virtual_orbitals": k}`` appends the ``k`` lowest
unoccupied subshells of each atom, in aufbau order.  Pinned here: which levels
those are, that a level is a *whole* subshell (so the atom stays spherically
symmetric), that the occupied set and the default ``k = 0`` are untouched, that
the option reaches the driver through every spelling of ``basis``, and that the
extra levels do what they are for -- lower the variational energy.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.dry_run import estimate_qubits
from mandacaru.basis import BasisSet, FAOBasisSet
from mandacaru.basis._config import ground_state_config, unoccupied_subshells
from mandacaru.optimizers import Optimizer

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
LBFGSB = Optimizer(method="L-BFGS-B", maxiter=2000, tol=1e-12)


def occupied_levels(symbol):
    """The ``(n, l)`` subshells the neutral atom's ground state fills."""
    from ase.data import atomic_numbers
    return sorted(ground_state_config(atomic_numbers[symbol]))


def h2(distance=0.74, cell=8.0):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, distance]],
                  cell=[cell] * 3)
    atoms.center()
    return atoms


# --------------------------------------------------------------------------- #
# Which levels are virtual.
# --------------------------------------------------------------------------- #

class TestUnoccupiedSubshells:
    @pytest.mark.parametrize("symbol, Z, expected", [
        ("H", 1, [(2, 0), (2, 1), (3, 0)]),      # 1s^1 -> 2s, 2p, 3s
        ("Li", 3, [(2, 1), (3, 0), (3, 1)]),     # [He] 2s^1 -> 2p, 3s, 3p
        ("C", 6, [(3, 0), (3, 1), (4, 0)]),      # 2p^2 counts as occupied
        ("O", 8, [(3, 0), (3, 1), (4, 0)]),
        ("Fe", 26, [(4, 1), (5, 0), (4, 2)]),    # [Ar] 3d^6 4s^2 -> 4p
    ])
    def test_aufbau_order(self, symbol, Z, expected):
        assert unoccupied_subshells(Z, 3) == expected

    def test_a_partly_filled_subshell_is_occupied(self):
        """Carbon's half-empty 2p is a basis shell already, not a virtual one."""
        assert (2, 1) in ground_state_config(6)
        assert (2, 1) not in unoccupied_subshells(6, 5)

    def test_none_requested(self):
        assert unoccupied_subshells(1, 0) == []

    def test_negative(self):
        with pytest.raises(ValueError, match="count must be >= 0"):
            unoccupied_subshells(1, -1)

    def test_beyond_the_filling_table(self):
        """A heavy atom has few levels left above its occupied set."""
        assert len(unoccupied_subshells(92, 2)) == 2
        with pytest.raises(ValueError, match="only 2 unoccupied subshell"):
            unoccupied_subshells(92, 3)


# --------------------------------------------------------------------------- #
# The basis it builds.
# --------------------------------------------------------------------------- #

class TestBasisConstruction:
    def test_the_default_is_the_historical_minimal_basis(self):
        default, explicit = BasisSet.build("FAO"), BasisSet.build(
            "FAO", virtual_orbitals=0)
        assert default.virtual_orbitals == 0
        for symbol in ("H", "Li", "C", "O", "Fe"):
            states = [o.state for o in default.atom(symbol)]
            assert states == [o.state for o in explicit.atom(symbol)]
            # Exactly the occupied subshells, nothing else.
            assert sorted({(n, l) for (n, l, _m) in states}) == \
                occupied_levels(symbol)

    @pytest.mark.parametrize("symbol, counts", [
        ("H", {0: 1, 1: 2, 2: 5, 3: 6}),      # 1s | +2s | +2p | +3s
        ("Li", {0: 2, 1: 5, 2: 6}),           # 1s2s | +2p | +3s
        ("O", {0: 5, 1: 6, 2: 9}),            # 1s2s2p | +3s | +3p
    ])
    def test_function_counts(self, symbol, counts):
        for k, expected in counts.items():
            basis = BasisSet.build("FAO", virtual_orbitals=k)
            assert len(basis.atom(symbol)) == expected
            assert basis.function_count(symbol) == expected

    def test_a_level_is_a_whole_shell(self):
        """One virtual p level is three functions -- the count is of levels.

        Half a shell would break the atom's spherical symmetry and make the
        energy depend on the molecule's orientation, so every appended level
        carries all its ``m`` components.
        """
        basis = BasisSet.build("FAO", virtual_orbitals=2)      # H: +2s, +2p
        states = [o.state for o in basis.atom("H")]
        assert states == [(1, 0, 0), (2, 0, 0),
                          (2, 1, -1), (2, 1, 0), (2, 1, 1)]
        for (n, l) in basis.subshells("H"):
            present = {m for (nn, ll, m) in states if (nn, ll) == (n, l)}
            assert present == set(range(-l, l + 1))

    def test_the_occupied_orbitals_are_unchanged(self):
        """Adding virtuals appends; it never perturbs the occupied set."""
        minimal = BasisSet.build("FAO").atom("O", center=[0.1, 0.2, 0.3])
        extended = BasisSet.build("FAO", virtual_orbitals=2).atom(
            "O", center=[0.1, 0.2, 0.3])
        assert len(extended) == len(minimal) + 1 + 3
        for before, after in zip(minimal, extended):
            assert before.state == after.state
            assert before.Z == after.Z
            assert np.allclose(before.center, after.center)

    def test_virtual_orbitals_carry_the_bare_nuclear_charge(self):
        """Same convention as the occupied FAO orbitals: no Slater screening."""
        basis = BasisSet.build("FAO", virtual_orbitals=1)
        assert {o.Z for o in basis.atom("O")} == {8.0}

    def test_subshells_lists_occupied_then_virtual(self):
        basis = BasisSet.build("FAO", virtual_orbitals=2)
        assert basis.subshells("C") == [(1, 0), (2, 0), (2, 1),
                                        (3, 0), (3, 1)]

    def test_repr_shows_the_option(self):
        assert repr(BasisSet.build("FAO", virtual_orbitals=2)) == \
            "FAOBasisSet(virtual_orbitals=2)"

    @pytest.mark.parametrize("bad, error", [
        (-1, ValueError), (1.5, ValueError), (True, TypeError),
        ("two", TypeError),
    ])
    def test_invalid_counts_are_refused(self, bad, error):
        with pytest.raises(error, match="virtual_orbitals"):
            FAOBasisSet(virtual_orbitals=bad)

    def test_an_unknown_option_is_still_refused(self):
        with pytest.raises(TypeError, match="tier"):
            BasisSet.build("FAO", tier=1)


# --------------------------------------------------------------------------- #
# Through the basis argument.
# --------------------------------------------------------------------------- #

class TestBasisArgument:
    @pytest.mark.parametrize("spec, n_qubits, per_atom", [
        ("FAO", 4, [("H", 1), ("H", 1)]),
        ({"name": "FAO"}, 4, [("H", 1), ("H", 1)]),
        ({"name": "FAO", "virtual_orbitals": 0}, 4, [("H", 1), ("H", 1)]),
        ({"name": "FAO", "virtual_orbitals": 1}, 8, [("H", 2), ("H", 2)]),
        ({"name": "FAO", "virtual_orbitals": 2}, 20, [("H", 5), ("H", 5)]),
    ])
    def test_the_dry_run_counts_the_virtual_levels(self, spec, n_qubits,
                                                   per_atom):
        estimate = estimate_qubits(h2(), basis=spec)
        assert estimate.n_qubits == n_qubits
        assert estimate.per_atom == per_atom

    def test_the_dry_run_label_carries_the_option(self):
        estimate = estimate_qubits(h2(),
                                   basis={"name": "FAO", "virtual_orbitals": 1})
        assert "virtual_orbitals" in estimate.basis

    def test_per_element(self):
        """Only the oxygen gets a virtual level."""
        water = Atoms("OH2", positions=[[0, 0, 0], [0, 0.76, 0.59],
                                        [0, -0.76, 0.59]], cell=[10.0] * 3)
        water.center()
        estimate = estimate_qubits(
            water, basis={"O": {"name": "FAO", "virtual_orbitals": 1},
                          "H": "FAO"})
        assert estimate.per_atom == [("O", 6), ("H", 1), ("H", 1)]
        assert estimate.n_qubits == 16

    def test_the_electron_count_does_not_change(self):
        """Virtual levels widen the basis, never the number of electrons."""
        plain = estimate_qubits(h2(), basis="FAO")
        wide = estimate_qubits(h2(),
                               basis={"name": "FAO", "virtual_orbitals": 1})
        assert wide.n_electrons == plain.n_electrons == 2
        assert wide.num_particles == plain.num_particles == (1, 1)
        assert wide.n_spatial_orbitals == 4 and plain.n_spatial_orbitals == 2


# --------------------------------------------------------------------------- #
# What they are for.
# --------------------------------------------------------------------------- #

class TestVariationalPayoff:
    def test_a_virtual_level_lowers_the_h2_energy(self):
        """The point of the option: room above the occupied orbitals.

        The minimal FAO basis is 2 spatial orbitals with a single double
        excitation; the 2s level makes it 4, and the correlated energy must
        fall (the smaller basis is a strict subset of the larger).
        """
        energies = {}
        for k in (0, 1):
            atoms = h2(cell=8.0)
            atoms.calc = Mandacaru(method="adapt-vqe",
                                   basis={"name": "FAO", "virtual_orbitals": k},
                                   h=0.30,
                                   pool="fermionic",
                                   optimizer=LBFGSB,
                                   max_iterations=40,
                                   gradient_tolerance=1e-7,
                                   profile=False)
            energies[k] = atoms.get_potential_energy()
            assert atoms.calc.n_qubits == (4 if k == 0 else 8)
        assert energies[1] < energies[0] - 0.05          # eV; measured 0.139
