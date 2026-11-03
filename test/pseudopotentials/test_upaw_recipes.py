# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_upaw_recipes.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The UPAW-LCAO-only generator recipes.

A dataset at ``norm_deficit = 0`` needs different spheres, channels and
reference configurations from its PAW-LCAO sibling (an unprojected d or f
channel, a semicore p shell, a 4f valence), so the table that holds them
(``mandacaru-core``: ``paw::generate::tables_upaw``, pinned entry by entry against
the frozen resolution by ``crates/mandacaru-core/tests/paw_recipe.rs``) is read by
the UPAW-LCAO recipe alone.  These tests pin that boundary through the Python
API with silicon, whose recipe adds an empty d channel: a recipe never reaches
``generate_paw``, never reaches another functional, and the caller's own
options win over it.
"""

import warnings

import pytest

from mandacaru.pseudopotentials.generation import generate_paw, generate_upaw


def _channels(dataset):
    return sorted(dataset.channels)


def _quiet(call, *args, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **options)


class TestBoundary:
    def test_upaw_reads_the_recipe(self):
        assert _channels(_quiet(generate_upaw, "Si", ghosts="keep")) == [0, 1, 2]

    def test_paw_does_not(self):
        assert _channels(_quiet(generate_paw, "Si", ghosts="keep")) == [0, 1]

    def test_the_callers_options_win(self):
        assert _channels(_quiet(generate_upaw, "Si", ghosts="keep", extra_l=0)) == [0, 1]

    def test_another_functional_is_left_alone(self):
        assert _channels(_quiet(generate_upaw, "Si", xc="pbe", ghosts="keep")) == [0, 1]

    def test_the_family_is_still_unitary_and_named(self):
        dataset = _quiet(generate_upaw, "Si", ghosts="keep")
        assert dataset.family == "upaw-lcao"
        assert dataset.norm_deficit == 0.0


class TestReferenceConfiguration:
    def test_it_replaces_the_tabulated_one(self):
        """``reference_configuration`` is the per-call form of the table of reference
        configurations: beryllium with one 2s electron promoted to 2p."""
        plain = _quiet(generate_paw, "Be", ghosts="keep")
        excited = _quiet(generate_paw, "Be", ghosts="keep",
                         reference_configuration={(2, 0): 1, (2, 1): 1})
        assert plain.valence_charge == excited.valence_charge == 2.0
        assert excited.energies["band"] != plain.energies["band"]

    @pytest.mark.xfail(strict=True, raises=ValueError, reason=(
        "lithium with its 2s emptied and the 2p occupied has no feasible smooth wave in "
        "the production (variational) atom: 'constrained quadratic failed its "
        "feasibility checks'; the Python generator it replaced built it on the "
        "atom-solver atom (PAW_MIGRATION_LOG.md, A7 / S2, finding for the owner)"))
    def test_lithium_with_an_empty_2s_is_built(self):
        dataset = _quiet(generate_paw, "Li", ghosts="keep",
                         reference_configuration={(2, 0): 0, (2, 1): 1})
        assert dataset.valence_charge == 1.0
