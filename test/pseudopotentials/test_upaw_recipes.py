# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_upaw_recipes.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The UPAW-LCAO-only generator recipes (:data:`DEFAULT_UPAW_OPTIONS`).

A dataset at ``norm_deficit = 0`` needs different spheres, channels and
reference configurations from its PAW-LCAO sibling (an unprojected d or f
channel, a semicore p shell, a 4f valence), so the table that holds them is
read by ``generate_upaw`` alone.  These tests pin that boundary: a recipe never
reaches ``generate_paw``, never reaches another functional, and the caller's own
options win over it.
"""

import inspect
import warnings

import pytest
from ase.data import atomic_numbers

from mandacaru.pseudopotentials import paw
from mandacaru.pseudopotentials.paw import (DEFAULT_UPAW_OPTIONS, generate_paw,
                                            generate_upaw)


def _channels(dataset):
    return sorted(dataset.channels)


@pytest.fixture
def hydrogen_recipe(monkeypatch):
    """A recipe for hydrogen (a fraction of a second to generate) that adds a
    d channel above its p."""
    monkeypatch.setitem(DEFAULT_UPAW_OPTIONS, ("H", "lda"), {"extra_l": 2})


class TestTable:
    def test_every_key_is_an_element_and_a_known_functional(self):
        for symbol, xc in DEFAULT_UPAW_OPTIONS:
            assert symbol in atomic_numbers
            assert xc in ("lda", "pbe")

    def test_every_option_is_a_generator_option(self):
        known = set(inspect.signature(generate_paw).parameters)
        for key, recipe in DEFAULT_UPAW_OPTIONS.items():
            unknown = set(recipe) - known
            assert not unknown, f"{key}: {sorted(unknown)} is not a generate_paw option"

    def test_no_recipe_sets_what_defines_the_family(self):
        for key, recipe in DEFAULT_UPAW_OPTIONS.items():
            assert "norm_deficit" not in recipe, key
            assert "ghosts" not in recipe, key

    def test_the_recipes_are_lda_only_for_now(self):
        """The PBE and the Dirac sets are separate scans: nothing here is
        borrowed for them."""
        assert {xc for _s, xc in DEFAULT_UPAW_OPTIONS} <= {"lda"}


class TestBoundary:
    def test_upaw_reads_the_recipe(self, hydrogen_recipe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _channels(generate_upaw("H", ghosts="keep")) == [0, 1, 2]

    def test_paw_does_not(self, hydrogen_recipe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _channels(generate_paw("H", ghosts="keep")) == [0, 1]

    def test_the_callers_options_win(self, hydrogen_recipe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = generate_upaw("H", ghosts="keep", extra_l=0)
        assert _channels(dataset) == [0, 1]

    def test_another_functional_is_left_alone(self, hydrogen_recipe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _channels(generate_upaw("H", xc="pbe", ghosts="keep")) == [0, 1]

    def test_the_family_is_still_unitary_and_named(self, hydrogen_recipe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = generate_upaw("H", ghosts="keep")
        assert dataset.family == paw.UPAW_FAMILY
        assert dataset.norm_deficit == 0.0


class TestReferenceConfiguration:
    def test_it_replaces_the_tabulated_one(self, monkeypatch):
        """``reference_configuration`` is the per-call form of
        ``DEFAULT_REFERENCE_CONFIGURATIONS``: lithium with 2p occupied binds a
        p reference of its own."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = generate_paw("Li", ghosts="keep")
            excited = generate_paw("Li", ghosts="keep",
                                   reference_configuration={(2, 0): 0, (2, 1): 1})
        assert plain.valence_charge == excited.valence_charge == 1.0
        assert excited.energies["band"] != plain.energies["band"]
