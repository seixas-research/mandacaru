# -*- coding: utf-8 -*-
# file: test_basisset.py

import numpy as np
import pytest

from carcara.basis import (BasisFunction, BasisSet, GTOBasisSet, NAOBasisSet,
                           available_bases, register)
from carcara.integrals import Grid


class TestBuild:
    def test_build_nao(self):
        nao = BasisSet.build(method="NAO", energy_shift=0.03)
        assert isinstance(nao, NAOBasisSet)

    def test_build_gto(self):
        gto = BasisSet.build(method="GTO", name="6-31G(d)")
        assert isinstance(gto, GTOBasisSet)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            BasisSet.build(method="STO")


class TestNAOFactory:
    def test_hydrogen_valence(self):
        nao = BasisSet.build(method="NAO", energy_shift=0.5)
        orbs = nao.atom("H", center=[0, 0, 0], units="bohr")
        assert len(orbs) == 1                       # 1s only
        assert all(isinstance(o, BasisFunction) for o in orbs)

    def test_carbon_valence_counts(self):
        # Carbon valence = 2s + 2p -> 1 + 3 = 4 orbitals.
        nao = BasisSet.build(method="NAO", energy_shift=1.0)
        orbs = nao.atom(6, center=[0, 0, 0], units="bohr")
        assert len(orbs) == 4


class TestGTOFactory:
    def test_families_are_available(self):
        avail = available_bases()
        for fam in ("sto-3g", "6-31g(d)", "6-311g(d,p)", "cc-pvdz", "cc-pvtz",
                    "def2-svp", "def2-tzvp"):
            assert fam in avail

    @pytest.mark.parametrize("name, n_h", [
        ("STO-3G", 1),        # 1s
        ("6-31G(d)", 2),      # 2s, no polarization on H
        ("6-311G(d,p)", 6),   # 3s + 3p
        ("cc-pVDZ", 5),       # 2s + 3p
        ("cc-pVTZ", 14),      # 3s + 6p + 5d
        ("def2-SVP", 5),      # 2s + 3p
        ("def2-TZVP", 6),     # 3s + 3p
    ])
    def test_hydrogen_counts_per_family(self, name, n_h):
        assert len(BasisSet.build(method="GTO", name=name).atom("H")) == n_h

    def test_carbon_counts(self):
        # 6-31G(d) and cc-pVDZ carbon: 3s + 6p + 5d = 14 spherical functions.
        assert len(BasisSet.build(method="GTO", name="6-31G(d)").atom("C")) == 14
        assert len(BasisSet.build(method="GTO", name="cc-pVDZ").atom("C")) == 14
        # cc-pVTZ carbon adds f functions: 4s + 6p + 10d + 7f = 30.
        assert len(BasisSet.build(method="GTO", name="cc-pVTZ").atom("C")) == 30

    def test_second_row_element(self):
        # Argon is now covered (H..Ar); STO-3G Ar = 1s2s2p3s3p = 9.
        assert len(BasisSet.build(method="GTO", name="STO-3G").atom("Ar")) == 9

    def test_missing_element_raises(self):
        # Iron (Z=26) is beyond the embedded H..Ar range.
        gto = BasisSet.build(method="GTO", name="cc-pVDZ")
        with pytest.raises(ValueError):
            gto.atom("Fe")

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            BasisSet.build(method="GTO", name="not-a-basis")

    def test_register_custom_basis(self):
        register("mini-h", "H S\n 1.0 1.0\n")
        orbs = BasisSet.build(method="GTO", name="mini-h").atom("H")
        assert len(orbs) == 1


class TestMoleculeAndEngine:
    def test_molecule_concatenates(self):
        gto = BasisSet.build(method="GTO", name="6-31G(d)")
        basis = gto.molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        assert len(basis) == 4                      # 2 per H

    def test_overlap_diagonal_is_normalized(self):
        # Each basis function is 3D-normalized -> S_aa ~ 1 on a fine grid.
        gto = BasisSet.build(method="GTO", name="6-31G(d)")
        basis = gto.atom("H", center=[0, 0, 0], units="bohr")
        grid = Grid(center=[0, 0, 0], box_size=10.0, h=0.25, units="bohr")
        psi = np.stack([b.evaluate(grid.X, grid.Y, grid.Z).ravel() for b in basis])
        S = (np.conj(psi) @ psi.T) * grid.dV
        assert np.allclose(np.diag(S).real, 1.0, atol=0.02)
