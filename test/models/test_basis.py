# -*- coding: utf-8 -*-
# file: test/models/test_basis.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Named Gaussian basis sets (STO-nG, Pople, Dunning, Karlsruhe): the name grammar.

:mod:`mandacaru.models.basis` parses a name into the *structure* of the set; the Rust engine builds
the shells from it, so the checks are on the grammar, on the native function counts the recipes
imply and on a calculation that initializes with a representative set of every family. The shells'
numbers are pinned by ``test/migration/test_native_basis_samples.py``.
"""

import json
from dataclasses import asdict

import pytest
from ase import Atoms

from mandacaru import _native, estimate_qubits
from mandacaru.integrations.ase import geometry_from_atoms
from mandacaru.models.basis import NAMED_BASIS_SETS, GaussianRecipe, parse_basis_name
from mandacaru.specifications import basis_specification

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

#: Function counts (spherical shells) of hydrogen and carbon.
COUNTS = {
    "STO-3G": (1, 5), "3-21G": (2, 9), "6-31G*": (2, 14), "6-31+G*": (2, 18),
    "6-31G(3df,3pd)": (16, 31), "6-311+G(2df,2p)": (9, 34),
    "cc-pVDZ": (5, 14), "cc-pVTZ": (14, 30), "cc-pVQZ": (30, 55),
    "cc-pV5Z": (55, 91), "aug-cc-pVDZ": (9, 23), "cc-pCVDZ": (5, 18),
    "def2-SVP": (5, 14), "def2-TZVP": (6, 31), "def2-QZVPP": (30, 57),
}


def native_count(symbol, basis):
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    return _native.Context(threads=1).basis_size(
        geometry_from_atoms(atoms), basis_spec=json.dumps(basis_specification(basis)))


class TestLoading:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_every_requested_name_loads(self, name):
        assert native_count("O", name) > 0 and native_count("H", name) > 0

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
        assert native_count("H", spelling) > 0

    @pytest.mark.parametrize("bad", ["cc-pVXZ", "def2-XYZ", "6-31G+", "6-31G(x)",
                                     "STO-G", "cc-pVDZ*"])
    def test_unknown_names_are_rejected(self, bad):
        with pytest.raises(ValueError):
            basis_specification(bad)


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


class TestProvenance:
    def test_the_name_carries_a_visible_qualifier(self):
        recipe = parse_basis_name("cc-pVTZ")
        assert recipe.provenance == "native:cc-pVTZ-recipe"
        assert recipe.name == "cc-pVTZ"              # still the selector

    def test_the_native_namespace_is_an_explicit_spelling_of_the_default(self):
        for name in ("native:def2-SVP", "native:def2-SVP-recipe"):
            assert (parse_basis_name(name).summary()
                    == parse_basis_name("def2-SVP").summary())

    def test_the_published_namespace_resolves_to_nothing(self):
        """Reserved for real tabulated data, which Mandacaru does not ship."""
        with pytest.raises(ValueError, match="no published basis-set tables"):
            basis_specification("published:cc-pVTZ")
        with pytest.raises(ValueError, match="unknown basis-set namespace"):
            basis_specification("elsewhere:cc-pVTZ")

    def test_the_specification_carries_the_recipe_to_the_engine(self):
        spec = basis_specification("6-31G*")
        assert spec["name"] == "gaussian"
        assert spec["recipe"] == asdict(parse_basis_name("6-31G*"))


class TestStructure:
    @pytest.mark.parametrize("name, expected", sorted(COUNTS.items()))
    def test_function_counts(self, name, expected):
        assert (native_count("H", name), native_count("C", name)) == expected

    def test_the_dry_run_counts_the_family(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
        estimate = estimate_qubits(atoms, basis="cc-pVDZ")
        assert estimate.n_spatial_orbitals == 2 * COUNTS["cc-pVDZ"][0]
        assert estimate.n_qubits == 2 * estimate.n_spatial_orbitals

    def test_second_row_polarization_rule(self):
        # 3-21G*: d functions on Na-Ar, none on C; 6-31G*: on both.
        assert native_count("Si", "3-21G*") - native_count("Si", "3-21G") == 5
        assert native_count("C", "3-21G*") == native_count("C", "3-21G")
        assert native_count("C", "6-31G*") - native_count("C", "6-31G") == 5


REPRESENTATIVE = {"sto": "STO-4G", "pople": "6-31+G*", "dunning": "cc-pVDZ",
                  "karlsruhe": "def2-SVP"}


@pytest.fixture(scope="module")
def h2():
    return Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]], cell=[6.0] * 3)


def hartree_fock(atoms, basis, spacing):
    from mandacaru import Mandacaru
    from mandacaru.units import HARTREE_TO_EV

    atoms = atoms.copy()
    atoms.calc = Mandacaru(method="rhf", basis=basis, grid_spacing=spacing, trace=False)
    return atoms.get_potential_energy() / HARTREE_TO_EV, atoms.calc


class TestIntegration:
    @pytest.mark.parametrize("family, name", sorted(REPRESENTATIVE.items()))
    def test_hartree_fock_converges_with_a_set_of_each_family(self, h2, family, name):
        energy, calc = hartree_fock(h2, name, 0.30)
        assert -1.25 < energy < -0.90
        assert calc.result.success

    def test_split_valence_beats_minimal_at_hartree_fock(self, h2):
        """More flexibility cannot raise the variational minimum."""
        energies = {name: hartree_fock(h2, name, 0.25)[0] for name in ("cc-pVDZ", "3-21G", "STO-3G")}
        assert energies["cc-pVDZ"] < energies["3-21G"] < energies["STO-3G"] + 1e-6
