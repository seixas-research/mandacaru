# -*- coding: utf-8 -*-
# file: test_pople.py

"""Native Pople split-valence basis (6-31G / 6-31G(d)) generated from scratch."""

import numpy as np
import pytest

from carcara.basis import BasisSet, GaussianOrbital, pople_631g_shells
from carcara.basis.pople import (CORE_PRIMITIVES, VALENCE_INNER,
                                  _split_valence_shell)


class TestShellStructure:
    def test_hydrogen_is_split_valence_31(self):
        # H has only the 1s valence shell -> split into inner(3) + outer(1).
        shells = pople_631g_shells(1, polarization=True)   # (d) adds nothing on H
        assert len(shells) == 2
        (l0, e0, _), (l1, e1, c1) = shells
        assert l0 == 0 and l1 == 0
        assert len(e0) == VALENCE_INNER          # contracted inner
        assert len(e1) == 1                      # uncontracted outer
        assert np.allclose(c1, 1.0)

    def test_lithium_core_valence_and_polarization(self):
        # Li: 1s core (6-prim contraction) + 2s valence (3+1) + one d shell.
        no_pol = pople_631g_shells(3, polarization=False)
        ls = [l for (l, _e, _c) in no_pol]
        assert ls.count(0) == 3                  # 1s(core) + 2s inner + 2s outer
        core = no_pol[0]
        assert len(core[1]) == CORE_PRIMITIVES   # 6-primitive core

        with_pol = pople_631g_shells(3, polarization=True)
        assert any(l == 2 for (l, _e, _c) in with_pol)   # a d polarization shell

    def test_valence_split_exponents_are_tight_then_diffuse(self):
        inner, outer = _split_valence_shell(3, 2, 0)
        # Inner primitives are tighter (larger exponents) than the outer one.
        assert inner[1].min() > outer[1].max()


class TestFactory:
    def test_631g_counts(self):
        b = BasisSet.build("6-31G")
        assert len(b.atom("H")) == 2             # 1s split 3+1
        assert len(b.atom("Li")) == 3            # 1s + 2s(3+1)

    def test_631gd_adds_d_on_heavy_atoms_only(self):
        b = BasisSet.build("6-31G(d)")
        assert len(b.atom("H")) == 2             # no polarization on H
        # Li: 3 s-type + 5 d components (m = -2..2).
        assert len(b.atom("Li")) == 8
        assert sum(1 for o in b.atom("Li") if o.l == 2) == 5

    def test_builds_gaussian_orbitals(self):
        for o in BasisSet.build("6-31G(d)").atom("Li"):
            assert isinstance(o, GaussianOrbital)

    def test_names(self):
        assert BasisSet.build("6-31G").name == "6-31G"
        assert BasisSet.build("6-31G(d)").name == "6-31G(d)"

    def test_hydrogenic_factory(self):
        b = BasisSet.build("hydrogenic")
        assert len(b.atom("H")) == 1             # 1s
        assert len(b.atom("Li")) == 2            # 1s + 2s

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            BasisSet.build("cc-pVQZ")


class TestNormalization:
    def test_gaussian_orbitals_are_normalized_on_grid(self):
        # A contracted 6-31G(d) orbital integrates to ~1 over a fine radial grid.
        from carcara.integrals import Grid
        b = BasisSet.build("6-31G(d)")
        orb = b.atom("H", center=[0.0, 0.0, 0.0])[0]
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=8.0, h=0.12)
        psi = orb.evaluate(grid.X, grid.Y, grid.Z)
        norm = float(np.sum(np.abs(psi) ** 2) * grid.dV)
        assert norm == pytest.approx(1.0, abs=0.02)
