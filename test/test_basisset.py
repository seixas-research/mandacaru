# -*- coding: utf-8 -*-
# file: test_basisset.py

import numpy as np
import pytest

from carcara.basis import BasisFunction, BasisSet, GTOBasisSet, NAOBasisSet
from carcara.integrals import Grid


class TestBuild:
    def test_build_nao(self):
        nao = BasisSet.build(method="NAO", energy_shift=0.03)
        assert isinstance(nao, NAOBasisSet)

    def test_build_gto(self):
        gto = BasisSet.build(method="GTO", n_gaussians=3)
        assert isinstance(gto, GTOBasisSet)
        assert gto.name == "STO-3G"

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
    # STO-nG is a minimal basis: one contracted function per occupied subshell
    # (core + valence), so the *count* is fixed and only the contraction length
    # depends on n_gaussians.
    @pytest.mark.parametrize("element, n_funcs", [
        ("H", 1),     # 1s
        ("He", 1),    # 1s
        ("C", 5),     # 1s 2s 2p     -> 1 + 1 + 3
        ("O", 5),     # 1s 2s 2p
        ("Ne", 5),    # 1s 2s 2p
        ("Ar", 9),    # 1s 2s 2p 3s 3p -> 1+1+3+1+3
    ])
    def test_minimal_basis_counts(self, element, n_funcs):
        assert len(BasisSet.build(method="GTO").atom(element)) == n_funcs

    @pytest.mark.parametrize("n_gaussians", [2, 3, 6])
    def test_contraction_length_tracks_n_gaussians(self, n_gaussians):
        orb = BasisSet.build(method="GTO", n_gaussians=n_gaussians).atom("H")[0]
        assert orb.n_primitives == n_gaussians

    def test_transition_metal_supported(self):
        # Fe (Z=26): 1s2s2p3s3p4s3d -> all subshells have n <= 4 (Slater n* known).
        # 1+1+3+1+3+1+5 = 15 functions; generator handles it with no data files.
        assert len(BasisSet.build(method="GTO").atom("Fe")) == 15

    def test_beyond_slater_table_raises(self):
        # Francium (Z=87) occupies a 7s subshell; no Slater n* for n=7.
        with pytest.raises(ValueError):
            BasisSet.build(method="GTO").atom("Fr")


class TestMoleculeAndEngine:
    def test_molecule_concatenates(self):
        gto = BasisSet.build(method="GTO")
        basis = gto.molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        assert len(basis) == 2                      # 1 per H (minimal)

    def test_all_shells_radially_normalized(self):
        # Every contracted shell is analytically normalized (grid-independent,
        # so the tight core 1s isn't a resolution artifact).
        basis = BasisSet.build(method="GTO").atom("C", units="bohr")
        r = np.linspace(0.0, 40.0, 400000)
        for orb in basis:
            norm = float(np.trapezoid(orb.radial(r) ** 2 * r * r, r))
            assert np.isclose(norm, 1.0, atol=1e-4)

    def test_diffuse_orbital_grid_normalized(self):
        # The diffuse hydrogen 1s is resolved on a modest grid -> S_aa ~ 1.
        orb = BasisSet.build(method="GTO").atom("H", units="bohr")[0]
        grid = Grid(center=[0, 0, 0], box_size=12.0, h=0.25, units="bohr")
        psi = orb.evaluate(grid.X, grid.Y, grid.Z)
        assert abs(float(np.sum(np.abs(psi) ** 2) * grid.dV) - 1.0) < 0.02
