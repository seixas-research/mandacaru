# -*- coding: utf-8 -*-
# file: test/test_gaussian_families.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Named Gaussian basis-set families (STO-nG, Pople, Dunning, Karlsruhe).

Every named set is generated natively from its *structure* (parsed from the
name) and the Slater-orbital fits, so the checks here are structural: each
name loads, its parsed recipe is right, each atom gets the published shell
notation and function count, the shells are sane (normalized, tight-to-diffuse
ordering, diffuse below valence) and a molecular calculation initializes with a
representative set from every family.
"""

import numpy as np
import pytest
from ase import Atoms
from scipy.special import gamma

from carcara.basis import (NAMED_BASIS_SETS, BasisSet, GaussianBasisSet,
                           GaussianOrbital, GaussianRecipe, count_functions,
                           gaussian_shells, parse_basis_name, pople_631g_shells,
                           shell_notation)
from carcara.basis.gaussian_families import DIFFUSE_RATIO

REQUESTED = {
    "sto": ["STO-3G", "STO-4G", "STO-5G", "STO-6G"],
    "pople": ["3-21G", "3-21G*", "3-21G**", "3-21+G", "3-21++G", "3-21+G*",
              "3-21+G**", "4-21G", "4-31G", "6-21G", "6-31G", "6-31G*",
              "6-31+G*", "6-31G(3df,3pd)", "6-311G", "6-311G*", "6-311+G*",
              "6-311+G(2df,2p)"],
    "dunning": ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z", "aug-cc-pVDZ",
                "cc-pCVDZ"],
    "karlsruhe": ["def2-SV(P)", "def2-SVP", "def2-SVPD", "def2-TZDP",
                  "def2-TZVPD", "def2-TZVPP", "def2-TZVPPD", "def2-QZVP",
                  "def2-QZVPD", "def2-QZVPP", "def2-QZVPPD"],
}
ALL_NAMES = [name for names in REQUESTED.values() for name in names]

#: Published contracted-shell notation, carbon and hydrogen, for the sets
#: whose structure is standard; the native generation must reproduce it.
NOTATION = {
    "STO-3G": ("[2s1p]", "[1s]"),
    "3-21G": ("[3s2p]", "[2s]"),
    "3-21G*": ("[3s2p]", "[2s]"),          # * is second-row only
    "3-21+G": ("[4s3p]", "[2s]"),
    "3-21++G": ("[4s3p]", "[3s]"),
    "4-31G": ("[3s2p]", "[2s]"),
    "6-31G": ("[3s2p]", "[2s]"),
    "6-31G*": ("[3s2p1d]", "[2s]"),
    "6-31+G*": ("[4s3p1d]", "[2s]"),
    "6-31G(3df,3pd)": ("[3s2p3d1f]", "[2s3p1d]"),
    "6-311G": ("[4s3p]", "[3s]"),
    "6-311G*": ("[4s3p1d]", "[3s]"),
    "6-311+G*": ("[5s4p1d]", "[3s]"),
    "6-311+G(2df,2p)": ("[5s4p2d1f]", "[3s2p]"),
    "cc-pVDZ": ("[3s2p1d]", "[2s1p]"),
    "cc-pVTZ": ("[4s3p2d1f]", "[3s2p1d]"),
    "cc-pVQZ": ("[5s4p3d2f1g]", "[4s3p2d1f]"),
    "cc-pV5Z": ("[6s5p4d3f2g1h]", "[5s4p3d2f1g]"),
    "aug-cc-pVDZ": ("[4s3p2d]", "[3s2p]"),
    "cc-pCVDZ": ("[4s3p1d]", "[2s1p]"),
    "def2-SV(P)": ("[3s2p1d]", "[2s]"),
    "def2-SVP": ("[3s2p1d]", "[2s1p]"),
    "def2-TZVP": ("[5s3p2d1f]", "[3s1p]"),
    "def2-TZVPP": ("[5s3p2d1f]", "[3s2p1d]"),
    "def2-QZVP": ("[7s4p3d2f1g]", "[4s3p2d]"),
    "def2-QZVPP": ("[7s4p3d2f1g]", "[4s3p2d1f]"),
}

#: Function counts (spherical shells) implied by the notation above.
COUNTS = {
    "STO-3G": (1, 5), "3-21G": (2, 9), "6-31G*": (2, 14), "6-31+G*": (2, 18),
    "6-31G(3df,3pd)": (16, 31), "6-311+G(2df,2p)": (9, 34),
    "cc-pVDZ": (5, 14), "cc-pVTZ": (14, 30), "cc-pVQZ": (30, 55),
    "cc-pV5Z": (55, 91), "aug-cc-pVDZ": (9, 23), "cc-pCVDZ": (5, 18),
    "def2-SVP": (5, 14), "def2-TZVP": (6, 31), "def2-QZVPP": (30, 57),
}


def _radial_norm(orbital: GaussianOrbital) -> float:
    """``int |R|^2 r^2 dr`` in closed form from the stored contraction."""
    a = orbital.exponents[:, None] + orbital.exponents[None, :]
    integ = gamma(orbital.l + 1.5) / (2.0 * a ** (orbital.l + 1.5))
    return float(orbital._d @ integ @ orbital._d)


# --------------------------------------------------------------------------- #
# Loading and parsing.
# --------------------------------------------------------------------------- #

class TestLoading:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_every_requested_name_loads(self, name):
        bset = BasisSet.build(name)
        orbitals = bset.atom("O")
        assert orbitals and all(isinstance(o, GaussianOrbital) for o in orbitals)
        assert bset.atom("H")
        # "6-31G*" is served by the dedicated Pople class, spelled "6-31G(d)".
        assert bset.name.lower().replace("(d)", "*") == name.lower()

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_parse_round_trips(self, name):
        recipe = parse_basis_name(name)
        assert isinstance(recipe, GaussianRecipe)
        assert recipe.name.lower() == name.lower()
        assert recipe.family in REQUESTED
        assert name in REQUESTED[recipe.family]
        assert recipe.summary()

    def test_validated_list_matches_the_request(self):
        assert set(ALL_NAMES) <= set(NAMED_BASIS_SETS)
        assert "def2-TZVP" in NAMED_BASIS_SETS

    @pytest.mark.parametrize("spelling", ["cc-pvdz", "CC-PVDZ", "Def2-svp",
                                          "6-31+g*", "sto-4g", "STO4G",
                                          "aug-CC-pVDZ"])
    def test_case_insensitive(self, spelling):
        assert BasisSet.build(spelling).atom("H")

    @pytest.mark.parametrize("bad", ["cc-pVXZ", "def2-XYZ", "6-31G+", "6-31G(x)",
                                     "STO-G", "gaussian", "cc-pVDZ*"])
    def test_unknown_names_are_rejected(self, bad):
        with pytest.raises(ValueError):
            BasisSet.build(bad)

    def test_gaussian_basis_set_from_recipe_or_name(self):
        a = GaussianBasisSet("cc-pVDZ")
        b = GaussianBasisSet(parse_basis_name("cc-pVDZ"))
        assert a.name == b.name == "cc-pVDZ" and a.family == "dunning"
        assert a.notation("C") == "[3s2p1d]"
        assert "cc-pVDZ" in repr(a)
        with pytest.raises(TypeError):
            GaussianBasisSet(42)


class TestParser:
    def test_pople_grammar(self):
        r = parse_basis_name("6-311+G(2df,2p)")
        assert r.core == (6,) and r.valence == (3, 1, 1)
        assert r.polarization == ((2, 2), (3, 1))
        assert r.polarization_h == ((1, 2),)
        assert r.diffuse == (0, 1) and r.diffuse_h == ()
        r = parse_basis_name("3-21++G**")
        assert r.valence == (2, 1) and r.diffuse_h == (0,)
        assert r.polarization == ((2, 1),) and r.polarization_h == ((1, 1),)
        assert r.polarization_min_z == 11          # 3-21G*: second row only
        assert parse_basis_name("6-31G*").polarization_min_z == 3
        assert parse_basis_name("6-31G(3df,3pd)").polarization_h == ((1, 3), (2, 1))

    def test_dunning_grammar(self):
        r = parse_basis_name("aug-cc-pVTZ")
        assert r.zeta == 3 and r.diffuse_present and not r.tight
        assert r.polarization == ((2, 2), (3, 1))
        assert r.polarization_h == ((1, 2), (2, 1))
        r = parse_basis_name("cc-pCVDZ")
        assert r.tight == ((0, 1), (1, 1)) and not r.diffuse_present

    def test_karlsruhe_grammar(self):
        assert parse_basis_name("def2-SV(P)").polarization_h == ()
        assert parse_basis_name("def2-SVP").polarization_h == ((1, 1),)
        assert parse_basis_name("def2-TZVPD").diffuse_present
        assert parse_basis_name("def2-TZVPD").diffuse_l_max == 2
        assert parse_basis_name("def2-QZVP").zeta == 4
        tzdp = parse_basis_name("def2-TZDP")
        assert tzdp.valence == parse_basis_name("def2-TZVP").valence
        assert "TZVP" in tzdp.description

    def test_sto_grammar(self):
        r = parse_basis_name("STO-5G")
        assert r.family == "sto" and r.core == (5,) and r.valence == (5,)


# --------------------------------------------------------------------------- #
# Provenance: these are native recipes, and every label says so.
# --------------------------------------------------------------------------- #

class TestProvenance:
    def test_the_name_carries_a_visible_qualifier(self):
        basis = BasisSet.build("cc-pVTZ")
        assert basis.provenance == "native:cc-pVTZ-recipe"
        assert basis.name == "cc-pVTZ"              # still the selector
        assert "native:cc-pVTZ-recipe" in repr(basis)

    def test_the_native_namespace_is_an_explicit_spelling_of_the_default(self):
        for name in ("native:def2-SVP", "native:def2-SVP-recipe"):
            assert (parse_basis_name(name).summary()
                    == parse_basis_name("def2-SVP").summary())

    def test_the_published_namespace_resolves_to_nothing(self):
        """Reserved for real tabulated data, which Carcará does not ship."""
        with pytest.raises(ValueError, match="no published basis-set tables"):
            BasisSet.build("published:cc-pVTZ")
        with pytest.raises(ValueError, match="unknown basis-set namespace"):
            BasisSet.build("elsewhere:cc-pVTZ")

    def test_the_dry_run_reports_the_provenance(self):
        from carcara.algorithms.dry_run import estimate_qubits
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[6.0] * 3)
        assert estimate_qubits(atoms, basis="cc-pVDZ").basis == \
            "native:cc-pVDZ-recipe"

    def test_the_numbers_serialize_with_their_convention(self):
        """Exponents, coefficients and the angular convention, as plain data."""
        import json
        basis = BasisSet.build("6-31G*")         # canonically 6-31G(d)
        record = basis.to_dict(["H", 8])
        assert record["provenance"] == "native:6-31G(d)-recipe"
        assert record["published_data"] is False
        assert "spherical" in record["convention"]["functions_per_shell"]
        assert "Bohr^-2" in record["convention"]["units"]
        oxygen = record["elements"]["O"]
        assert oxygen["atomic_number"] == 8
        assert oxygen["n_functions"] == sum(2 * s["l"] + 1
                                            for s in oxygen["shells"])
        # The serialized numbers are the ones the orbitals are built from.
        shells = basis.shells("O")
        assert len(shells) == len(oxygen["shells"])
        for (l, exps, coeffs), stored in zip(shells, oxygen["shells"]):
            assert stored["l"] == l
            assert np.allclose(stored["exponents"], exps)
            assert np.allclose(stored["coefficients"], coeffs)
        json.dumps(record)                      # a record a reader can keep


# --------------------------------------------------------------------------- #
# Structure: shell notation and counts.
# --------------------------------------------------------------------------- #

class TestStructure:
    @pytest.mark.parametrize("name, expected", sorted(NOTATION.items()))
    def test_published_shell_notation(self, name, expected):
        recipe = parse_basis_name(name)
        assert shell_notation(6, recipe) == expected[0]
        assert shell_notation(1, recipe) == expected[1]

    @pytest.mark.parametrize("name, expected", sorted(COUNTS.items()))
    def test_function_counts(self, name, expected):
        bset = BasisSet.build(name)
        assert (len(bset.atom("H")), len(bset.atom("C"))) == expected
        recipe = parse_basis_name(name)
        assert (count_functions(1, recipe), count_functions(6, recipe)) == expected

    def test_second_row_polarization_rule(self):
        # 3-21G*: d functions on Na-Ar, none on C; 6-31G*: on both.
        assert shell_notation(14, parse_basis_name("3-21G*")) == "[4s3p1d]"
        assert shell_notation(6, parse_basis_name("3-21G*")) == "[3s2p]"
        assert shell_notation(6, parse_basis_name("6-31G*")) == "[3s2p1d]"

    def test_sto_ng_contraction_length(self):
        for n in (3, 4, 5, 6):
            (orb,) = BasisSet.build(f"STO-{n}G").atom("H")
            assert orb.n_primitives == n

    def test_631g_star_matches_the_native_pople_builder(self):
        """The generic engine reproduces the dedicated 6-31G(d) construction."""
        for Z in (1, 3, 6, 8):
            generic = gaussian_shells(Z, parse_basis_name("6-31G*"))
            dedicated = pople_631g_shells(Z, polarization=True)
            assert len(generic) == len(dedicated)
            for (l1, e1, c1), (l2, e2, c2) in zip(generic, dedicated):
                assert l1 == l2
                assert np.allclose(e1, e2) and np.allclose(c1, c2)

    def test_transition_metal(self):
        # Fe: semicore 3d treated as core, 4s valence; d polarization present.
        notation = shell_notation(26, parse_basis_name("def2-SVP"))
        assert notation.startswith("[") and "d" in notation


# --------------------------------------------------------------------------- #
# Physics sanity of the generated shells.
# --------------------------------------------------------------------------- #

class TestShells:
    @pytest.mark.parametrize("name", ["6-311+G(2df,2p)", "aug-cc-pVTZ",
                                      "def2-TZVPPD", "cc-pCVDZ"])
    @pytest.mark.parametrize("element", ["H", "C"])
    def test_orbitals_are_radially_normalized(self, name, element):
        for orbital in BasisSet.build(name).atom(element):
            assert _radial_norm(orbital) == pytest.approx(1.0, rel=1e-10)

    def test_exponents_are_finite_and_positive(self):
        for name in ("cc-pV5Z", "def2-QZVPPD", "6-31G(3df,3pd)"):
            for (l, exps, coeffs) in gaussian_shells(8, parse_basis_name(name)):
                assert np.all(np.isfinite(exps)) and np.all(exps > 0)
                assert np.all(np.isfinite(coeffs))

    def test_split_valence_is_tight_then_diffuse(self):
        shells = gaussian_shells(6, parse_basis_name("6-311G"))
        s_valence = [e for (l, e, _c) in shells if l == 0][1:]      # after the core
        assert len(s_valence) == 3
        assert s_valence[0].min() > s_valence[1].max() > s_valence[2].max()

    def test_diffuse_functions_are_the_most_diffuse(self):
        plain = gaussian_shells(6, parse_basis_name("6-31G*"))
        aug = gaussian_shells(6, parse_basis_name("6-31+G*"))
        assert len(aug) == len(plain) + 2                      # diffuse s and p
        for l in (0, 1):
            most_diffuse_plain = min(e.min() for (ll, e, _c) in plain if ll == l)
            diffuse = aug[-2 + l][1][0]
            assert diffuse == pytest.approx(most_diffuse_plain / DIFFUSE_RATIO)

    def test_tight_functions_are_tighter_than_the_valence(self):
        recipe = parse_basis_name("cc-pCVDZ")
        shells = gaussian_shells(6, recipe)
        plain = gaussian_shells(6, parse_basis_name("cc-pVDZ"))
        assert len(shells) == len(plain) + 2
        for l in (0, 1):
            valence_max = max(e.max() for (ll, e, _c) in plain[1:] if ll == l)
            tight = [e[0] for (ll, e, c) in shells if ll == l and e.size == 1
                     and e[0] > valence_max]
            assert tight

    def test_polarization_exponents_spread_geometrically(self):
        shells = gaussian_shells(6, parse_basis_name("6-31G(3df,3pd)"))
        d = sorted((e[0] for (l, e, _c) in shells if l == 2), reverse=True)
        assert len(d) == 3
        assert d[0] / d[1] == pytest.approx(d[1] / d[2])
        assert all(e[0] > 0 for (l, e, _c) in shells if l == 3)


# --------------------------------------------------------------------------- #
# Integration: a molecular calculation initializes with each family.
# --------------------------------------------------------------------------- #

REPRESENTATIVE = {"sto": "STO-4G", "pople": "6-31+G*", "dunning": "cc-pVDZ",
                  "karlsruhe": "def2-SVP"}


@pytest.fixture(scope="module")
def h2():
    return Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]], cell=[6.0] * 3)


class TestIntegration:
    @pytest.mark.parametrize("family, name", sorted(REPRESENTATIVE.items()))
    def test_dry_run_counts_the_family(self, h2, family, name):
        from carcara.algorithms import estimate_qubits
        est = estimate_qubits(h2, basis=name)
        expected = len(BasisSet.build(name).atom("H"))
        assert est.per_atom == [("H", expected), ("H", expected)]
        assert est.n_qubits == 4 * expected

    @pytest.mark.parametrize("family, name", sorted(REPRESENTATIVE.items()))
    def test_hamiltonian_builds_and_hartree_fock_converges(self, h2, family,
                                                            name):
        from carcara.core import MolecularIntegrals
        from carcara.integrals import Grid
        bset = BasisSet.build(name)
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.30)
        functions, nuclei = [], []
        for position in h2.positions:
            functions += bset.atom("H", center=position, units="angstrom")
            nuclei.append((1.0, np.asarray(position)))
        integrals = MolecularIntegrals(
            nuclei, functions, grid,
            softening=0.5 * min(grid.dx, grid.dy, grid.dz))
        rhf = integrals.hartree_fock(2)
        energy = rhf.electronic_energy + integrals.nuclear_repulsion
        assert np.isfinite(energy) and -1.25 < energy < -0.90
        hamiltonian = integrals.molecular_hamiltonian(mo_basis=True,
                                                      n_electrons=2)
        assert hamiltonian.n_modes() == 2 * len(functions)

    def test_split_valence_beats_minimal_at_hartree_fock(self, h2):
        """More flexibility cannot raise the variational minimum."""
        from carcara.core import MolecularIntegrals
        from carcara.integrals import Grid
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.25)

        def rhf(name):
            bset = BasisSet.build(name)
            functions, nuclei = [], []
            for position in h2.positions:
                functions += bset.atom("H", center=position, units="angstrom")
                nuclei.append((1.0, np.asarray(position)))
            integrals = MolecularIntegrals(
                nuclei, functions, grid,
                softening=0.5 * min(grid.dx, grid.dy, grid.dz))
            return (integrals.hartree_fock(2).electronic_energy
                    + integrals.nuclear_repulsion)

        assert rhf("cc-pVDZ") < rhf("3-21G") < rhf("STO-3G") + 1e-6

    def test_adapt_vqe_reaches_fci_with_a_named_set(self, h2):
        """A full run through the calculator with an STO-nG name."""
        from carcara.algorithms import Carcara
        atoms = h2.copy()
        atoms.calc = Carcara(method="adapt-vqe", basis="STO-4G",
                                       pool="qeb", h=0.35, verbose=False,
                                       profile=False, optimizer="L-BFGS-B",
                                       gradient_tolerance=1e-5)
        atoms.get_potential_energy()
        calc = atoms.calc
        h = calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        assert calc.n_qubits == 4
        assert calc.result.in_units("Ha") == pytest.approx(exact, abs=1e-5)
