# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_families.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The family registry as the basis argument sees it
(:mod:`mandacaru.pseudopotentials.families`)."""

import pytest

from mandacaru import Mandacaru
from mandacaru.pseudopotentials.families import (
    COMMON_OPTIONS, OPTION_VALIDATORS, PSEUDO_FAMILIES, FamilySpec,
    lookup_family, resolve_family)


def _construct(basis):
    return Mandacaru(method="adapt-vqe", basis=basis)


class TestBasisArgumentIsHonored:
    """The family is the basis name; its options are the size hierarchy."""

    def test_size_is_forwarded(self):
        calc = _construct({"name": "PAW-LCAO", "size": "DZP"})
        assert calc._basis["size"] == "DZP"
        assert calc._basis_spec is None

    @pytest.mark.parametrize("name", ["UPAW-LCAO", "upaw-lcao",
                                      "unitary-paw-lcao", "Unitary-PAW-LCAO"])
    def test_the_family_is_accepted_by_every_alias(self, name):
        assert lookup_family(name).name == "upaw-lcao"
        _construct({"name": name, "size": "DZ"})

    def test_all_electron_names_stay_all_electron(self):
        assert lookup_family("HAO") is None
        assert _construct("HAO")._basis_spec is not None

    @pytest.mark.parametrize("basis", ["PP", "PSEUDO", {"name": "pp",
                                                        "size": "DZ"}])
    def test_the_retired_names_are_refused(self, basis):
        """``basis="PP"`` was the old spelling; it must not alias silently."""
        with pytest.raises((ValueError, TypeError)):
            _construct(basis)


class TestOptionsAreValidatedAtConstruction:
    """One validation path: ``FamilySpec.validate_options``, called by the
    ``Mandacaru`` constructor, so a typo never reaches a run."""

    @pytest.mark.parametrize("family", ["paw-lcao", "upaw-lcao"])
    def test_every_declared_option_has_a_validator(self, family):
        assert set(resolve_family(family).options) <= set(OPTION_VALIDATORS)

    @pytest.mark.parametrize("name", ["PAW-LCAO", "UPAW-LCAO"])
    def test_an_unknown_key_names_itself_and_lists_the_valid_ones(self, name):
        with pytest.raises(ValueError) as caught:
            _construct({"name": name, "sise": "DZ"})
        message = str(caught.value)
        assert "unknown option" in message and "'sise'" in message
        for option in resolve_family(name).options:
            assert option in message

    def test_the_valid_set_is_the_family_declaration(self):
        family = resolve_family("paw-lcao")
        assert set(COMMON_OPTIONS) <= set(family.options)
        family.validate_options({})

    @pytest.mark.parametrize("option, bad, match", [
        ("size", "XYZ", "basis size"),
        ("size", {"O": "DZP", "H": "XYZ"}, "basis size"),
        ("zeta_split", "middle_zeta", "zeta_split"),
        ("filter_cutoff", "yes", "filter_cutoff"),
        ("filter_cutoff", -1.0, "filter_cutoff"),
        ("energy_shift", -3.0, "energy_shift"),
        ("confinement_potential", (12.0,), "confinement_potential"),
        ("polarization_method", "cubic", "polarization_method"),
        ("diffuse", {"rms_ratio": 0.5}, "rms_ratio"),
        ("projector_basis", "primal", "projector_basis"),
        ("paw_dataset", 3, "paw_dataset"),
    ])
    def test_a_bad_value_is_refused(self, option, bad, match):
        with pytest.raises(ValueError, match=match):
            _construct({"name": "PAW-LCAO", option: bad})

    def test_a_full_valid_dict_is_accepted(self):
        _construct({"name": "PAW-LCAO", "size": {"O": "DZP", "*": "SZ"},
                    "zeta_split": "last_zeta", "paw_dataset": "lda-sr",
                    "filter_cutoff": 500.0, "projector_basis": "dual",
                    "energy_shift": 0.1, "confinement_potential": (12.0, 0.6),
                    "polarization_method": "gaussian", "diffuse": {"shells": "p"}})

    def test_the_calculator_dataset_is_validated_like_the_basis_one(self):
        with pytest.raises(ValueError, match="paw_dataset"):
            Mandacaru(method="adapt-vqe", basis="PAW-LCAO", paw_dataset=3)

    def test_a_family_without_a_validator_for_an_option_refuses_it(self):
        spec = FamilySpec(name="bogus", description="", generate=None, get=None,
                          options=("size", "mystery"))
        spec.validate_options({"size": "DZ"})
        with pytest.raises(ValueError, match="no validator"):
            spec.validate_options({"mystery": 1})


class TestTheDiffuseOption:
    """``diffuse`` is a PAW-LCAO option, so UPAW-LCAO has it too."""

    @pytest.mark.parametrize("name", ["paw-lcao", "upaw-lcao"])
    def test_both_families_accept_it(self, name):
        family = PSEUDO_FAMILIES[name] if name in PSEUDO_FAMILIES \
            else resolve_family(name)
        assert "diffuse" in family.options
        # Off unless written: no family default.
        assert "diffuse" not in family.default_options

    def test_a_bad_value_is_refused(self):
        with pytest.raises(ValueError, match="rms_ratio"):
            _construct({"name": "PAW-LCAO", "diffuse": {"rms_ratio": 0.5}})
