# -*- coding: utf-8 -*-
# file: test/test_bibliography.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The reference database, the selector, and the ``references.bib`` a run writes.

The valuable tests here are the registry sweeps: every operator pool, mapping,
optimizer and pseudopotential family Mandacaru offers must have a paper behind
it, so adding one without a citation fails rather than shipping a run whose
bibliography quietly omits the method it used.
"""

from __future__ import annotations

import os
import re

import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.circuits.pools import available_pools
from mandacaru.core.mapping import MAPPINGS
from mandacaru.optimizers.optim import NAMED_OPTIMIZERS
from mandacaru.pseudopotentials.families import PSEUDO_FAMILIES
from mandacaru.utils import bibliography as bib
from mandacaru.utils.citations import (DEFAULT_REFERENCES_FILE, _FAMILY_KEYS,
                                       _MAPPING_KEYS, _METHOD_KEYS,
                                       _OPTIMIZER_KEYS, _POOL_KEYS,
                                       citation_keys, resolve_references_path,
                                       write_references)


def h2(cell=8.0):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                  cell=[cell] * 3, pbc=True)
    atoms.center()
    return atoms


# --------------------------------------------------------------------------- #

class TestTheDatabase:
    def test_every_key_matches_the_record_it_labels(self):
        for key, reference in bib.REFERENCES.items():
            head = reference.entry.splitlines()[0]
            assert head.endswith("{" + key + ","), head

    def test_every_record_has_balanced_braces(self):
        for key, reference in bib.REFERENCES.items():
            assert reference.entry.count("{") == reference.entry.count("}"), key
            assert reference.entry.rstrip().endswith("}"), key

    def test_every_record_has_a_type_and_a_note(self):
        for key, reference in bib.REFERENCES.items():
            assert reference.entry.startswith("@"), key
            assert reference.note.strip(), key

    def test_journal_names_are_abbreviated(self):
        """A spelled-out journal name is the mistake this catches."""
        spelled_out = re.compile(
            r"journal\s*=\s*\{[^}]*\b(Physical|Chemical|Journal|Reviews?|"
            r"Letters|Communications|Nature\s+\w|Transactions|Computational|"
            r"Proceedings)\b")
        for key, reference in bib.REFERENCES.items():
            assert not spelled_out.search(reference.entry), key

    def test_unverified_entries_are_flagged_not_hidden(self):
        # The flag exists so a manuscript knows what to check; it must not be
        # empty just because it is inconvenient.
        assert set(bib.UNVERIFIED) <= set(bib.REFERENCES)
        for key in bib.UNVERIFIED:
            assert not bib.REFERENCES[key].verified

    def test_the_adapt_family_was_read_from_local_sources(self):
        for key in ("Grimsley2019", "Tang2021", "Yordanov2021", "Ramoa2025",
                    "Anastasiou2024", "VaqueroSabater2025"):
            assert bib.REFERENCES[key].verified, key

    def test_resolve_keeps_the_database_order(self):
        order = list(bib.REFERENCES)
        keys = [order[5], order[1], order[3]]
        assert [r.key for r in bib.resolve(keys)] == sorted(keys,
                                                            key=order.index)

    def test_an_unknown_key_raises_instead_of_being_dropped(self):
        with pytest.raises(KeyError, match="no bibliography entry"):
            bib.resolve(["Grimsley2019", "NotAPaper2099"])

    def test_bibtex_renders_a_note_per_entry(self):
        text = bib.bibtex(["Grimsley2019"], header="a header")
        assert text.startswith("% a header")
        assert "% ADAPT-VQE" in text
        assert "@article{Grimsley2019," in text

    def test_bibtex_marks_an_unverified_record(self):
        assert "[unverified record]" in bib.bibtex(["Jordan1928"])
        assert "[unverified record]" not in bib.bibtex(["Grimsley2019"])


class TestEveryRegisteredChoiceIsCitable:
    """A method a user can select must have a paper behind it."""

    def test_every_pool(self):
        for pool in available_pools():
            assert _POOL_KEYS.get(pool), pool
            assert citation_keys(pool=pool) != citation_keys()

    def test_every_mapping(self):
        for mapping in MAPPINGS:
            assert _MAPPING_KEYS.get(mapping), mapping

    def test_every_optimizer(self):
        for name in NAMED_OPTIMIZERS:
            assert _OPTIMIZER_KEYS.get(name.lower()), name

    def test_every_pseudopotential_family(self):
        canonical = {spec.name for spec in PSEUDO_FAMILIES.values()}
        for family in canonical:
            assert _FAMILY_KEYS.get(family), family

    def test_every_stable_method(self):
        from mandacaru.algorithms.calculator import STABLE_METHODS
        for method in STABLE_METHODS:
            assert _METHOD_KEYS.get(method), method

    def test_every_key_in_every_table_exists(self):
        tables = (_POOL_KEYS, _MAPPING_KEYS, _METHOD_KEYS, _OPTIMIZER_KEYS,
                  _FAMILY_KEYS)
        for table in tables:
            for name, keys in table.items():
                bib.resolve(keys)             # raises on an unknown key


class TestTheSelector:
    def test_a_bare_run_cites_the_code_and_its_libraries(self):
        keys = citation_keys(has_geometry=False, built_basis=False)
        assert keys == ["Mandacaru", "Harris2020", "Virtanen2020"]

    def test_a_geometry_cites_ase(self):
        assert "Larsen2017" in citation_keys(built_basis=False)

    def test_method_and_pool(self):
        keys = citation_keys(method="adapt-vqe", pool="qubit")
        assert "Grimsley2019" in keys and "Tang2021" in keys
        assert "Yordanov2021" not in keys

    def test_ceo_cites_the_qubit_excitations_it_couples(self):
        keys = citation_keys(pool="ceo-ovp")
        assert "Ramoa2025" in keys and "Yordanov2021" in keys

    def test_method_and_pool_spelling_is_insensitive(self):
        assert citation_keys(method="ADAPT_VQE") == \
            citation_keys(method="adapt-vqe")
        assert citation_keys(pool="CEO") == citation_keys(pool="ceo")

    def test_growth_strategies_are_cited_only_when_on(self):
        assert "Anastasiou2024" not in citation_keys(method="adapt-vqe")
        assert "Anastasiou2024" in citation_keys(method="adapt-vqe",
                                                 tetris=True)
        assert "VaqueroSabater2025" not in citation_keys(method="adapt-vqe")
        assert "VaqueroSabater2025" in citation_keys(method="adapt-vqe",
                                                     prune=True)

    def test_the_taper_adds_a_reference_to_the_parity_mapping(self):
        parity = citation_keys(mapping="parity")
        tapered = citation_keys(mapping="parity_reduced")
        assert set(parity) < set(tapered)
        assert "Bravyi2017" in tapered

    def test_a_pseudopotential_family_cites_its_generator(self):
        assert "Bloechl1994" in citation_keys(family="paw")
        assert "Ivanov2024" in citation_keys(family="upaw")
        assert "Ivanov2024" not in citation_keys(family="paw")
        assert "Hamann2013" in citation_keys(family="oncvpsp")
        assert "Troullier1991" in citation_keys(family="ncpp")

    def test_basis_options_are_cited_only_when_they_did_something(self):
        plain = citation_keys(family="paw", basis_options={"size": "SZ"})
        assert "Junquera2001" not in plain and "Anglada2006" not in plain
        assert "Artacho1999" not in plain
        rich = citation_keys(family="paw",
                             basis_options={"size": "DZP", "filter": True,
                                            "energy_shift": 0.1})
        assert {"Junquera2001", "Sankey1989", "Anglada2006",
                "Artacho1999"} <= set(rich)

    def test_gaussian_family_names(self):
        assert "Dunning1989" in citation_keys(basis="cc-pVTZ")
        assert "Weigend2005" in citation_keys(basis="def2-TZVP")
        assert "Hehre1972" in citation_keys(basis="6-31G(d)")
        assert "Hehre1969" in citation_keys(basis="STO-3G")

    def test_a_plane_wave_basis_cites_the_ewald_sum(self):
        assert "Ewald1921" in citation_keys(basis="PW")

    def test_hao_needs_no_basis_paper(self):
        # Bare hydrogenic orbitals generated here; nothing external to cite.
        assert citation_keys(basis="HAO", built_basis=False) == \
            citation_keys(built_basis=False)

    def test_qiskit_is_cited_only_when_circuits_go_through_it(self):
        assert "Qiskit2024" not in citation_keys(backend_provider="qiskit")
        assert "Qiskit2024" in citation_keys(backend_provider="qiskit",
                                             shots=1024)
        assert "Qiskit2024" in citation_keys(backend_provider="qiskit",
                                             profile=True)

    def test_extras_are_included(self):
        assert "Sim2019" in citation_keys(extras=("Sim2019",))

    def test_the_result_is_ordered_and_unique(self):
        keys = citation_keys(method="adapt-vqe", pool="fermionic",
                             extras=("Grimsley2019", "Grimsley2019"))
        assert len(keys) == len(set(keys))
        order = list(bib.REFERENCES)
        assert keys == sorted(keys, key=order.index)


class TestThePathOption:
    def test_auto_follows_the_log(self, tmp_path):
        log = tmp_path / "run" / "output.txt"
        assert resolve_references_path("auto", log) == \
            os.path.join(str(tmp_path / "run"), DEFAULT_REFERENCES_FILE)

    def test_auto_without_a_log_writes_nothing(self):
        assert resolve_references_path("auto", None) is None

    def test_true_and_false_and_a_path(self):
        assert resolve_references_path(True) == DEFAULT_REFERENCES_FILE
        assert resolve_references_path(False) is None
        assert resolve_references_path(None) is None
        assert resolve_references_path("papers.bib") == "papers.bib"

    def test_an_unknown_value_raises(self):
        with pytest.raises(ValueError, match="is not a path"):
            resolve_references_path(3)

    def test_write_creates_the_parent_directory(self, tmp_path):
        target = tmp_path / "deep" / "refs.bib"
        write_references(target, ["Grimsley2019"])
        assert "Grimsley2019" in target.read_text()

    def test_writing_leaves_no_temporary_behind(self, tmp_path):
        target = tmp_path / "refs.bib"
        write_references(target, ["Grimsley2019"])
        assert [p.name for p in tmp_path.iterdir()] == ["refs.bib"]


class TestARunWritesIt:
    def test_a_log_gets_a_bibliography_beside_it(self, tmp_path):
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis="HAO",
                               h=0.35,
                               txt=str(tmp_path / "output.txt"),
                               trace=False)
        atoms.get_total_energy()
        text = (tmp_path / "references.bib").read_text()
        assert "@article{Grimsley2019," in text        # ADAPT-VQE
        assert "@article{Jordan1928," in text          # the mapping
        assert "@software{Mandacaru," in text

    def test_no_log_means_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.35, trace=False)
        atoms.get_total_energy()
        assert list(tmp_path.iterdir()) == []

    def test_references_true_writes_in_the_working_directory(self, tmp_path,
                                                             monkeypatch):
        monkeypatch.chdir(tmp_path)
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.35, trace=False,
                               references=True)
        atoms.get_total_energy()
        assert (tmp_path / DEFAULT_REFERENCES_FILE).is_file()
        assert atoms.calc.references_written == DEFAULT_REFERENCES_FILE

    def test_an_explicit_path_is_honored(self, tmp_path):
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.35, trace=False,
                               references=str(tmp_path / "papers.bib"))
        atoms.get_total_energy()
        assert (tmp_path / "papers.bib").is_file()

    def test_the_file_describes_the_basis_that_ran_not_the_one_asked_for(self):
        """A PAW basis is filtered and confined by default; both are cited."""
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe",
                               basis={"name": "PAW", "size": "SZ"},
                               h=0.35,
                               trace=False)
        atoms.get_total_energy()
        keys = atoms.calc.citation_keys()
        assert {"Bloechl1994", "Anglada2006", "Junquera2001"} <= set(keys)

    def test_a_bad_value_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="is not a path"):
            Mandacaru(method="vqe", basis="HAO", references=3)

    def test_a_collision_with_the_log_is_refused(self, tmp_path):
        path = str(tmp_path / "same.bib")
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method="adapt-vqe", basis="HAO", txt=path,
                      references=path)

    def test_a_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calc = Mandacaru(method="adapt-vqe", basis="HAO", references=True,
                         dry_run=True)
        calc.dry_run(h2())
        assert list(tmp_path.iterdir()) == []

    def test_excited_states_add_their_own_reference(self, tmp_path):
        target = tmp_path / "papers.bib"
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.35, trace=False,
                               references=str(target))
        atoms.get_total_energy()
        assert "Higgott2019" not in target.read_text()
        atoms.calc.energy_levels(num_states=2)
        assert "Higgott2019" in target.read_text()

    def test_a_failure_to_write_warns_but_does_not_end_the_run(self, tmp_path):
        atoms = h2()
        # A directory where the file should be: opening it for writing fails.
        (tmp_path / "papers.bib").mkdir()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.35, trace=False,
                               references=str(tmp_path / "papers.bib"))
        with pytest.warns(RuntimeWarning, match="could not write"):
            energy = atoms.get_total_energy()
        assert energy < 0.0
