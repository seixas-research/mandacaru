# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_defects.py

# This code is part of Mandacaru.
# MIT License

"""The words of the ``ghosts=`` option (:mod:`mandacaru.pseudopotentials.defects`): the modes, the
reader of a record's ``defects`` block (including the two keys of the native generator,
``missing`` and ``branch_ghosts``) and the text of the warning a defective dataset raises."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from mandacaru.pseudopotentials import defects as d


class TestModes:
    def test_the_modes_are_the_four_of_the_option(self):
        assert d.GHOST_MODES == ("repair", "refuse", "keep", "flag")
        for mode in d.GHOST_MODES:
            assert d.check_mode(mode) == mode

    @pytest.mark.parametrize("mode", ["Repair", "off", None, True])
    def test_another_mode_raises_and_lists_them(self, mode):
        with pytest.raises(ValueError, match=r"ghosts must be one of \('repair', 'refuse', 'keep', 'flag'\)"):
            d.check_mode(mode)

    def test_the_exceptions_have_the_documented_bases(self):
        assert issubclass(d.GhostStateError, RuntimeError)
        assert issubclass(d.DiagnosticError, RuntimeError)
        assert issubclass(d.GhostStateWarning, UserWarning)


class TestReadDefects:
    def test_nothing_recorded_is_an_empty_dict(self):
        assert d.read_defects(None) == {}
        assert d.read_defects({}) == {}
        assert d.read_defects({"ghosts": {}, "phases": {}}) == {}

    def test_keys_come_back_as_numbers(self):
        record = {"ghosts": {"0": -0.131}, "phases": {"1": [0.1, 0.2]},
                  "s_miss": 1.19, "unprojected": {"2": 0.21}, "residuals": {"0": 2.0}}
        got = d.read_defects(record)
        assert got == {"ghosts": {0: -0.131}, "phases": {1: (0.1, 0.2)}, "s_miss": 1.19,
                       "unprojected": {2: 0.21}, "residuals": {0: 2.0}}

    def test_the_native_keys_are_read_and_older_files_lack_them(self):
        got = d.read_defects({"ghosts": {}, "phases": {}, "missing": {"1": 0.0345},
                              "branch_ghosts": {"1:-2": -3.7362}})
        assert got["missing"] == {1: 0.0345}
        assert got["branch_ghosts"] == {"1:-2": -3.7362}
        old = d.read_defects({"ghosts": {"0": -0.1}, "phases": {}})
        assert "missing" not in old and "branch_ghosts" not in old

    def test_a_missing_state_alone_is_a_defect(self):
        assert d.read_defects({"missing": {"0": 0.2}}) == {
            "ghosts": {}, "phases": {}, "missing": {0: 0.2}}

    def test_a_zero_s_miss_is_still_recorded(self):
        assert d.read_defects({"s_miss": 0.0})["s_miss"] == 0.0


class TestMessage:
    def test_a_ghost_message_keeps_its_words(self):
        text = d.defect_message("Ce", "paw-lcao", {"ghosts": {0: -0.131, 1: -0.133}, "phases": {}})
        assert text == (
            "the paw-lcao dataset for Ce has ghost states (l=0 ghost -0.131 Ha; "
            "l=1 ghost -0.133 Ha).  A variational calculation containing Ce can collapse "
            "into a spurious state, so its energies and forces are not reliable.")

    def test_each_defect_has_its_own_headline(self):
        heads = {
            "an incomplete s channel": {"ghosts": {}, "phases": {}, "s_miss": 1.19},
            "a local potential that scatters wrongly": {"ghosts": {}, "phases": {}, "unprojected": {2: 0.3}},
            "divergent projector residuals": {"ghosts": {}, "phases": {}, "residuals": {0: 5.0}},
            "a missing bound state": {"ghosts": {}, "phases": {}, "missing": {1: 0.03}},
            "scattering errors": {"ghosts": {}, "phases": {1: (0.1, 0.4)}},
        }
        for head, defects in heads.items():
            assert f" has {head} (" in d.defect_message("X", "upaw-lcao", defects)

    def test_a_branch_ghost_names_its_branch(self):
        text = d.defect_message("Pb", "paw-lcao", {"ghosts": {}, "phases": {},
                                                    "branch_ghosts": {"1:-2": -3.7362}})
        assert "has ghost states (l=1 kappa=-2 ghost -3.74 Ha)" in text

    def test_a_missing_state_is_worded(self):
        text = d.defect_message("X", "paw-lcao", {"ghosts": {}, "phases": {}, "missing": {1: 0.0345}})
        assert "l=1 does not bind its reference (nearest level +0.0345 Ha from it)" in text


class TestWarning:
    def test_a_defective_dataset_warns_with_the_message(self):
        pp = SimpleNamespace(symbol="Ce", defects={"ghosts": {0: -0.1}, "phases": {}})
        with pytest.warns(d.GhostStateWarning, match="paw-lcao dataset for Ce has ghost states"):
            d.warn_defects(pp, "paw-lcao")

    def test_a_clean_dataset_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d.warn_defects(SimpleNamespace(symbol="O", defects={}), "paw-lcao")
