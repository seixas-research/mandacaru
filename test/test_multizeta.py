# -*- coding: utf-8 -*-
# file: test/test_multizeta.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Multiple-zeta and polarized numerical atomic orbitals."""

from __future__ import annotations

import numpy as np
import pytest
from ase.data import atomic_numbers

from carcara.basis import BasisSet
from carcara.basis._config import ground_state_config, valence_subshells
from carcara.basis.multizeta import (DEFAULT_SPLIT_NORM, ZETA_NAMES,
                                     RadialTable, TabulatedOrbital,
                                     resolve_zeta, split_radius,
                                     split_valence_tail, zeta_tables)
from carcara.basis.nao import solve_confined_radial
from carcara.core import MolecularIntegrals
from carcara.integrals import Grid


@pytest.fixture(scope="module")
def hydrogen_1s():
    """A confined hydrogen 1s: the reference first zeta."""
    r, radial, _energy = solve_confined_radial(1, 0, 1.0, 4.0, 800)
    return r, radial


class TestZetaNames:
    """Parsing the basis-size notation."""

    @pytest.mark.parametrize("name, expected", [
        ("SZ", (1, 0)), ("DZ", (2, 0)), ("DZP", (2, 1)),
        ("TZP", (3, 1)), ("TZ2P", (3, 2)), ("QZP", (4, 1)),
    ])
    def test_known_names(self, name, expected):
        assert resolve_zeta(name) == expected

    def test_case_and_separators_are_ignored(self):
        assert resolve_zeta("dz-p") == resolve_zeta(" DZP ") == (2, 1)

    def test_explicit_pair_passes_through(self):
        assert resolve_zeta((3, 2)) == (3, 2)

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown basis size"):
            resolve_zeta("PZ")

    def test_every_name_is_self_consistent(self):
        """The name encodes its own (n_zeta, n_polarization)."""
        digits = {"S": 1, "D": 2, "T": 3, "Q": 4}
        for name, (n_zeta, n_pol) in ZETA_NAMES.items():
            assert digits[name[0]] == n_zeta
            assert name.count("P") == min(n_pol, 1)


class TestSplitValence:
    """The split-valence construction of the extra zetas."""

    def test_split_radius_leaves_the_requested_tail_norm(self, hydrogen_1s):
        r, radial = hydrogen_1s
        for target in (0.05, 0.15, 0.30):
            r_s = split_radius(r, radial, target)
            density = radial ** 2 * r ** 2
            tail = np.trapezoid(density[r >= r_s], r[r >= r_s])
            assert tail == pytest.approx(target, abs=2e-3)

    def test_larger_split_norm_pushes_the_radius_inward(self, hydrogen_1s):
        r, radial = hydrogen_1s
        assert split_radius(r, radial, 0.30) < split_radius(r, radial, 0.05)

    def test_tail_vanishes_beyond_the_split_radius(self, hydrogen_1s):
        r, radial = hydrogen_1s
        r_s = split_radius(r, radial, DEFAULT_SPLIT_NORM)
        tail = split_valence_tail(r, radial, 0, r_s)
        assert np.all(tail[r >= r_s] == 0.0)
        assert np.any(np.abs(tail[r < r_s]) > 1e-6)

    def test_tail_is_continuous_at_the_split_radius(self, hydrogen_1s):
        """Value and slope are matched there, so no discontinuity is injected."""
        r, radial = hydrogen_1s
        r_s = split_radius(r, radial, DEFAULT_SPLIT_NORM)
        tail = split_valence_tail(r, radial, 0, r_s)
        inside = np.nonzero(r < r_s)[0]
        assert abs(tail[inside[-1]]) < 1e-3

    def test_zeta_hierarchy_is_strictly_shorter_ranged(self, hydrogen_1s):
        """Each added zeta lives inside the previous one and carries less norm."""
        r, radial = hydrogen_1s
        tables = zeta_tables(r, radial, 1, 0, 4)
        assert [t.zeta for t in tables] == [1, 2, 3, 4]

        ranges, norms = [], []
        for table in tables:
            support = np.nonzero(np.abs(table.values) > 1e-12)[0]
            ranges.append(r[support.max()])
            norms.append(np.trapezoid(table.values ** 2 * r ** 2, r))

        assert ranges == sorted(ranges, reverse=True)
        assert norms == sorted(norms, reverse=True)
        assert norms[0] == pytest.approx(1.0, abs=1e-3)

    def test_zeta_count_is_honored(self, hydrogen_1s):
        r, radial = hydrogen_1s
        for n_zeta in (1, 2, 3, 4):
            assert len(zeta_tables(r, radial, 1, 0, n_zeta)) == n_zeta


class TestTabulatedOrbital:
    """The basis function built on a radial table."""

    def test_confined_outside_the_cutoff(self, hydrogen_1s):
        r, radial = hydrogen_1s
        orbital = TabulatedOrbital(RadialTable(r, radial, 1, 0), 0)
        far = orbital.evaluate(np.array([50.0]), np.array([0.0]),
                               np.array([0.0]))
        assert far[0] == 0.0

    def test_m_is_validated(self, hydrogen_1s):
        r, radial = hydrogen_1s
        with pytest.raises(ValueError, match=r"\|m\| <= l"):
            TabulatedOrbital(RadialTable(r, radial, 1, 0), 1)

    def test_normalized_on_a_grid(self, hydrogen_1s):
        r, radial = hydrogen_1s
        orbital = TabulatedOrbital(RadialTable(r, radial, 1, 0), 0)
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.10)
        values = orbital.sample(grid)
        norm = float(np.sum(np.abs(values) ** 2) * grid.dV)
        assert norm == pytest.approx(1.0, abs=2e-2)


class TestBasisConstruction:
    """What ``BasisSet.build("NAO", size=...)`` actually produces."""

    @pytest.mark.parametrize("symbol, size, n_functions", [
        # H: 1 s shell; polarization adds a p shell (3 functions).
        ("H", "SZ", 1), ("H", "DZ", 2), ("H", "TZ", 3), ("H", "QZ", 4),
        ("H", "DZP", 2 + 3), ("H", "TZP", 3 + 3),
        # O: s + p = 4 functions per zeta; polarization adds d (5 functions).
        ("O", "SZ", 4), ("O", "DZ", 8),
        ("O", "DZP", 8 + 5), ("O", "TZP", 12 + 5),
    ])
    def test_function_counts(self, symbol, size, n_functions):
        assert len(BasisSet.build("NAO", size=size).atom(symbol)) == n_functions

    def test_polarization_shell_is_present_and_flagged(self):
        """The regression this guards: polarization silently absent."""
        for symbol, l_pol in (("H", 1), ("O", 2)):
            functions = BasisSet.build("NAO", size="DZP").atom(symbol)
            polarizing = [f for f in functions if f.polarization]
            assert len(polarizing) == 2 * l_pol + 1
            assert {f.l for f in polarizing} == {l_pol}

    def test_polarization_is_one_l_above_the_valence(self):
        """It must add angular freedom the occupied shells do not have."""
        for symbol in ("H", "C", "O", "Si"):
            Z = atomic_numbers[symbol]
            l_max = max(l for (_n, l) in valence_subshells(Z))
            functions = BasisSet.build("NAO", size="DZP").atom(symbol)
            assert {f.l for f in functions if f.polarization} == {l_max + 1}

    def test_second_polarization_shell(self):
        functions = BasisSet.build("NAO", size="TZ2P").atom("H")
        assert sorted({f.l for f in functions if f.polarization}) == [1, 2]

    def test_single_zeta_is_unchanged_by_the_new_machinery(self):
        """SZ must still take the original NumericalAtomicOrbital path."""
        from carcara.basis.nao import NumericalAtomicOrbital
        functions = BasisSet.build("NAO", size="SZ").atom("O")
        assert all(isinstance(f, NumericalAtomicOrbital) for f in functions)

    def test_default_size_is_dzp(self):
        """DZP is the default: 8 (double-zeta s+p) + 5 (d polarization)."""
        from carcara.basis.multizeta import DEFAULT_NAO_SIZE
        assert DEFAULT_NAO_SIZE == "DZP"
        assert len(BasisSet.build("NAO").atom("O")) == 13
        assert len(BasisSet.build("NAO").atom("O")) == \
            len(BasisSet.build("NAO", size="DZP").atom("O"))

    def test_centering_is_applied(self):
        center = [0.5, 0.0, 0.0]
        for function in BasisSet.build("NAO", size="DZP").atom("H",
                                                               center=center):
            assert function.center[0] > 0.0


class TestVariationalQuality:
    """A bigger basis must lower the energy -- the only test that matters."""

    @staticmethod
    def _rhf(size):
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.20)
        basis = BasisSet.build("NAO", size=size)
        functions, nuclei = [], []
        for position in ([0, 0, -0.37], [0, 0, 0.37]):
            functions += basis.atom("H", center=position, units="angstrom")
            nuclei.append((1.0, np.asarray(position)))
        integrals = MolecularIntegrals(
            nuclei, functions, grid,
            softening=0.5 * min(grid.dx, grid.dy, grid.dz))
        result = integrals.hartree_fock(2)
        return result.electronic_energy + integrals.nuclear_repulsion

    def test_double_zeta_lowers_the_energy(self):
        """The variational principle: more freedom cannot raise the minimum."""
        single, double = self._rhf("SZ"), self._rhf("DZ")
        assert double < single
        assert single - double > 0.01          # and by a chemically real amount

    def test_polarization_lowers_the_energy_further(self):
        assert self._rhf("DZP") < self._rhf("DZ")

    def test_overlap_stays_invertible(self):
        """Multiple zeta risks linear dependence; the basis must stay usable."""
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.20)
        basis = BasisSet.build("NAO", size="DZP")
        functions, nuclei = [], []
        for position in ([0, 0, -0.37], [0, 0, 0.37]):
            functions += basis.atom("H", center=position, units="angstrom")
            nuclei.append((1.0, np.asarray(position)))
        overlap = np.real(MolecularIntegrals(nuclei, functions, grid).overlap())
        eigenvalues = np.linalg.eigvalsh(overlap)
        assert eigenvalues.min() > 1e-4
        assert eigenvalues.max() / eigenvalues.min() < 1e4


class TestValenceSubshells:
    """Semicore d and f shells belong in the valence."""

    @pytest.mark.parametrize("symbol, expected", [
        ("H", [(1, 0)]),
        ("C", [(2, 0), (2, 1)]),
        ("Si", [(3, 0), (3, 1)]),
        ("Ca", [(4, 0)]),
        ("Fe", [(3, 2), (4, 0)]),          # 3d must not be dropped
        ("Zn", [(3, 2), (4, 0)]),
        ("Ga", [(3, 2), (4, 0), (4, 1)]),
    ])
    def test_valence(self, symbol, expected):
        assert valence_subshells(atomic_numbers[symbol]) == expected

    def test_iron_is_not_a_two_electron_atom(self):
        """The regression: taking only the highest n gave Fe a 4s^2 valence."""
        config = ground_state_config(atomic_numbers["Fe"])
        electrons = sum(config[s]
                        for s in valence_subshells(atomic_numbers["Fe"]))
        assert electrons == 8

    def test_f_block_keeps_its_f_shell(self):
        assert (4, 3) in valence_subshells(atomic_numbers["W"])
