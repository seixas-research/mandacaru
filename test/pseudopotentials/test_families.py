# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_families.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The family registry as the basis argument sees it
(:mod:`mandacaru.pseudopotentials.families`)."""

import pytest


class TestBasisArgumentIsHonored:
    """The family is the basis name; its options are the size hierarchy."""

    def test_size_is_forwarded(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        family, options = resolve_pseudo_basis(
            *resolve_basis({"name": "PAW-LCAO", "size": "DZP"}), ["O"])
        assert family.label == "PAW-LCAO" and options["size"] == "DZP"

    @pytest.mark.parametrize("name", ["UPAW-LCAO", "upaw-lcao",
                                      "unitary-paw-lcao", "Unitary-PAW-LCAO"])
    def test_the_family_is_accepted_by_every_alias(self, name):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        family, options = resolve_pseudo_basis(
            *resolve_basis({"name": name, "size": "DZ"}), ["O"])
        assert family.name == "upaw-lcao" and options == {"size": "DZ"}

    def test_all_electron_names_stay_all_electron(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        assert resolve_pseudo_basis(*resolve_basis("HAO"), ["O"]) == (None, {})

    @pytest.mark.parametrize("basis", ["PP", "PSEUDO", {"name": "pp",
                                                        "size": "DZ"}])
    def test_the_retired_names_are_refused(self, basis):
        """``basis="PP"`` was the old spelling; it must not alias silently."""
        from mandacaru.algorithms._hamiltonian_from_atoms import resolve_basis

        with pytest.raises(ValueError, match="no longer a basis name"):
            resolve_basis(basis)



class TestTheDiffuseOption:
    """``diffuse`` is a PAW-LCAO option, so UPAW-LCAO has it too, and the
    per-element form writes it per element (validated like the single
    dict)."""

    @pytest.mark.parametrize("name", ["paw-lcao", "upaw-lcao"])
    def test_both_families_accept_it(self, name):
        from mandacaru.pseudopotentials.families import resolve_family

        family = resolve_family(name)
        assert "diffuse" in family.options
        # Off unless written: no family default.
        assert "diffuse" not in family.default_options

    def test_the_per_element_form_passes_it(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        _family, options = resolve_pseudo_basis(
            *resolve_basis({"O": {"name": "PAW-LCAO", "size": "DZP",
                                  "diffuse": {"shells": "p"}},
                            "H": {"name": "PAW-LCAO", "diffuse": 1}}),
            ["O", "H", "H"])
        assert options["diffuse"] == {"O": {"shells": "p"}, "H": 1}
        assert options["size"] == {"O": "DZP", "H": "SZ"}

    def test_an_element_left_out_has_none(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)
        from mandacaru.pseudopotentials.diffuse import diffuse_of

        _family, options = resolve_pseudo_basis(
            *resolve_basis({"O": {"name": "PAW-LCAO", "diffuse": True},
                            "H": "PAW-LCAO"}), ["O", "H", "H"])
        assert diffuse_of(options["diffuse"], "H") is None
        assert diffuse_of(options["diffuse"], "O") is not None

    def test_a_bad_per_element_value_is_refused(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        with pytest.raises(ValueError, match="rms_ratio"):
            resolve_pseudo_basis(
                *resolve_basis({"O": {"name": "PAW-LCAO",
                                      "diffuse": {"rms_ratio": 0.5}},
                                "H": "PAW-LCAO"}), ["O", "H"])
