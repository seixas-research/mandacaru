# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_generation.py

# This code is part of Mandacaru.
# MIT License

"""Generating a dataset through the native engine (:mod:`mandacaru.pseudopotentials.generation`):
the options that are validated and worded in Python, the ones that raise because the engine
cannot express them, the search modes of ``ghosts=`` and the exceptions they map to."""

from __future__ import annotations

import pytest

from mandacaru.pseudopotentials.defects import GhostStateError
from mandacaru.pseudopotentials.generation import generate_paw, generate_upaw


class TestOptions:
    def test_a_pre_solved_atom_cannot_cross_the_boundary(self):
        with pytest.raises(NotImplementedError, match="cannot be handed to the generator"):
            generate_paw("H", atom=object())

    def test_free_waves_are_not_available(self):
        with pytest.raises(NotImplementedError, match="free smooth waves"):
            generate_paw("H", norm_deficit=None)

    def test_an_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="ghosts must be one of"):
            generate_paw("H", ghosts="ignore")

    @pytest.mark.parametrize("option", ["frozen_subshells", "semicore_subshells"])
    def test_subshells_must_be_integer_pairs(self, option):
        with pytest.raises(TypeError, match=f"{option} must be a list of"):
            generate_paw("H", **{option: 3})
        with pytest.raises(ValueError, match=f"{option} must contain"):
            generate_paw("H", **{option: [(1, 0, 0)]})
        with pytest.raises(ValueError, match=f"{option} must contain"):
            generate_paw("H", **{option: [[1, 0]]})
        with pytest.raises(ValueError, match=f"{option} must contain"):
            generate_paw("H", **{option: [(1.0, 0)]})

    def test_an_unknown_functional_or_treatment_raises(self):
        with pytest.raises(ValueError):
            generate_paw("H", xc="hse06")
        with pytest.raises(ValueError, match="unknown relativistic treatment"):
            generate_paw("H", relativity="warp")

    def test_a_subshell_the_atom_lacks_is_refused_by_the_engine(self):
        with pytest.raises(ValueError, match="not an occupied"):
            generate_paw("H", ghosts="keep", frozen_subshells=[(7, 3)])

    def test_the_unitary_family_takes_no_norm_deficit(self):
        with pytest.raises(TypeError, match="UPAW-LCAO is defined by norm_deficit = 0"):
            generate_upaw("H", norm_deficit=0.0)
        with pytest.raises(TypeError, match="unexpected keyword"):
            generate_upaw("H", surprise=1)

    def test_an_unknown_native_option_raises(self):
        from mandacaru import _native
        from mandacaru.pseudopotentials.native_context import context

        with pytest.raises(TypeError, match="unknown generation option"):
            _native.paw_generate(context(), "H", {"family": "paw-lcao", "xc": "lda",
                                                   "relativity": "scalar", "ghosts": "keep",
                                                   "surprise": 1})
        with pytest.raises(TypeError, match="generation option family is missing"):
            _native.paw_generate(context(), "H", {})


class TestHydrogen:
    @pytest.fixture(scope="class")
    def hydrogen(self):
        return generate_paw("H")

    def test_the_first_construction_is_clean_and_kept(self, hydrogen):
        search = hydrogen.search
        assert (search.index, search.label, search.tried, search.flagged) == (
            0, "first construction", (), False)
        assert search.built == 1 and search.missing == {}
        assert hydrogen.defects == {}

    def test_the_record_is_the_one_the_tables_give(self, hydrogen):
        assert hydrogen.family == "paw-lcao" and hydrogen.symbol == "H"
        assert (hydrogen.xc, hydrogen.relativity) == ("lda", "scalar")
        assert hydrogen.valence_charge == 1.0
        assert hydrogen.norm_deficit == 0.05            # the table of hydrogen
        assert sorted(hydrogen.channels) == [0, 1]       # an empty p channel above the valence s
        assert getattr(hydrogen, "atom", None) is None and hydrogen.spin_orbit == {}
        assert hydrogen.nlcc["applied"] is False

    def test_a_named_option_is_the_callers(self):
        pp = generate_paw("H", ghosts="keep", norm_deficit=0.0, n_bessel=9, q_cut=4.0)
        assert (pp.norm_deficit, pp.q_cut) == (0.0, 4.0)
        assert all(len(c.wavevectors[0]) == 9 for c in pp.channels.values())

    def test_a_unitary_dataset_is_the_zero_deficit_family(self):
        pp = generate_upaw("H")
        assert pp.family == "upaw-lcao" and pp.norm_deficit == 0.0
        # The charge of the scattering reference is not exactly zero (the frozen number of the
        # generator this replaced: -7.639328441744e-05).
        assert pp.channels[0].overlap_correction[1, 1] == pytest.approx(-7.639328441744e-05, abs=1e-12)
        assert abs(pp.channels[0].overlap_correction).max() < 1e-3


class TestSearchModes:
    """Oxygen with its local potential raised by ``-3 Ha`` binds an s ghost at -0.868 Ha."""

    GHOST = ("O: ghost state below the reference (l=0 -0.868 Ha) with local_shift fixed by the "
             "caller.  Leave the cutoffs and the local potential to the generator, or pass "
             "ghosts='keep' to study it.")

    @pytest.mark.parametrize("mode", ["repair", "flag", "refuse"])
    def test_a_ghost_with_a_pinned_local_shift_is_refused_with_the_cause(self, mode):
        with pytest.raises(GhostStateError) as caught:
            generate_paw("O", local_shift=-3.0, ghosts=mode)
        assert str(caught.value) == self.GHOST

    def test_keep_returns_the_ghost_unexamined(self):
        pp = generate_paw("O", local_shift=-3.0, ghosts="keep")
        assert pp.search.label == "first construction" and pp.defects == {}
        assert pp.local_shift == -3.0
