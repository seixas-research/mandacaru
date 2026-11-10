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

import re

import pytest

from mandacaru import _native
from mandacaru.calculation import OPERATOR_POOLS
from mandacaru.core.mapping import MAPPINGS
from mandacaru.optimizers.optim import NAMED_OPTIMIZERS
from mandacaru.pseudopotentials.families import PSEUDO_FAMILIES
from mandacaru.utils import bibliography as bib
from mandacaru.utils.citations import (_ANSATZ_KEYS,
                                       _FAMILY_KEYS, _MAPPING_KEYS,
                                       _METHOD_KEYS,
                                       _OPTIMIZER_KEYS, _POOL_KEYS,
                                       _PROVENANCE_KEYS, _ACTIVE_METHOD_KEYS,
                                       _MEASUREMENT_SCHEME_KEYS, _TWO_BODY_KEYS,
                                       citation_keys,
                                       provenance_keys)


# --------------------------------------------------------------------------- #

class TestTheDatabase:
    def test_every_key_matches_the_record_it_labels(self):
        for key, reference in bib.REFERENCES.items():
            head = reference.entry.splitlines()[0]
            assert head.endswith("{" + key + ","), head

    def test_no_key_is_defined_twice(self):
        """The registry is a dict: a second record under one key silently
        replaced the first (the tetrahedron paper once took the PAW paper's
        ``Bloechl1994``, and every PAW-LCAO run cited it)."""
        records = (bib._ALGORITHMS + bib._POOLS + bib._MAPPINGS + bib._BASES
                   + bib._OPTIMIZERS + bib._CODES)
        keys = [r.key for r in records]
        assert len(keys) == len(set(keys)), sorted(
            k for k in set(keys) if keys.count(k) > 1)
        assert "Projector augmented-wave" in bib.REFERENCES[
            "Bloechl1994"].entry

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
        for pool in OPERATOR_POOLS:
            assert _native.pool_kind(pool), pool
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

    def test_every_ansatz(self):
        from mandacaru.calculation import ANSATZ_NAMES
        for name in ANSATZ_NAMES:
            assert _ANSATZ_KEYS.get(name), name
            assert citation_keys(ansatz=name) != citation_keys()
        assert set(_ANSATZ_KEYS) == set(ANSATZ_NAMES)

    def test_every_two_body_representation(self):
        """The dense default (``None``) needs no paper of its own; the Cholesky
        factors cite theirs."""
        from mandacaru.calculation import TWO_BODY_METHODS
        for name in TWO_BODY_METHODS:
            assert _TWO_BODY_KEYS.get(name), name
            assert set(_TWO_BODY_KEYS[name]) <= set(citation_keys(two_body=name))
        assert set(_TWO_BODY_KEYS) == set(TWO_BODY_METHODS)
        assert citation_keys(two_body=None) == citation_keys()

    def test_every_measurement_scheme(self):
        """The commuting groups are the Pauli terms themselves; the double-factorized
        measurement cites the factorization, its measurement and the Givens networks."""
        from mandacaru.integrations.measurement_scheme import SCHEMES
        assert set(_MEASUREMENT_SCHEME_KEYS) == set(SCHEMES)
        assert citation_keys(measurement_scheme="qwc") == citation_keys()
        cited = citation_keys(measurement_scheme="double-factorized")
        assert set(_MEASUREMENT_SCHEME_KEYS["double-factorized"]) <= set(cited)
        assert cited != citation_keys()

    def test_every_stable_method(self):
        from mandacaru.calculation import STABLE_METHODS
        for method in STABLE_METHODS:
            assert _METHOD_KEYS.get(method), method

    def test_a_crystal_cites_its_supercell_sampling_and_ewald_sum(self):
        """The variational methods solve a crystal as a Born-von Karman supercell."""
        from mandacaru.utils.citations import _PERIODIC_KEYS
        bib.resolve(_PERIODIC_KEYS)
        for method in ("vqe", "adapt-vqe", "vasqa", "valsqa"):
            cited = citation_keys(method=method, periodic=True)
            assert set(_PERIODIC_KEYS) <= set(cited), method
            assert not set(_PERIODIC_KEYS) & set(citation_keys(method=method))

    def test_every_active_space_method(self):
        # "energy" is the exception the table states outright: it is the
        # canonical orbital order the mean field already produced, so it
        # borrows nothing and cites nothing.  The others are constructions.
        from mandacaru._native import ACTIVE_SPACE_METHODS

        for method in ACTIVE_SPACE_METHODS:
            assert method in _ACTIVE_METHOD_KEYS, method
            if method != "energy":
                assert _ACTIVE_METHOD_KEYS[method], method
                assert (citation_keys(active_method=method)
                        != citation_keys())
        assert set(_ACTIVE_METHOD_KEYS) == set(ACTIVE_SPACE_METHODS)

    def test_every_key_in_every_table_exists(self):
        tables = (_POOL_KEYS, _MAPPING_KEYS, _METHOD_KEYS, _ANSATZ_KEYS,
                  _OPTIMIZER_KEYS,
                  _FAMILY_KEYS, _PROVENANCE_KEYS, _ACTIVE_METHOD_KEYS,
                  _MEASUREMENT_SCHEME_KEYS)
        for table in tables:
            for name, keys in table.items():
                bib.resolve(keys)             # raises on an unknown key


class TestHowTheDatasetsWereGenerated:
    """``provenance_keys`` -- the generation options cite their own physics.

    The family key says which construction was used; these say what the
    reference atom was, which is a different question and a different paper.
    """

    class _Dataset:
        """The three fields `provenance_keys` reads, and nothing else."""

        def __init__(self, xc="lda", relativity="none", nlcc=False):
            self.xc = xc
            self.relativity = relativity
            self.nlcc = {"applied": bool(nlcc)}

    def test_the_plain_lda_atom_adds_nothing(self):
        """It is the construction the family key already covers."""
        assert provenance_keys([self._Dataset()]) == []

    def test_a_dataset_predating_the_options_adds_nothing(self):
        """An older dataset carries none of the three fields; reading it must
        not raise, and must not invent a citation for it."""
        assert provenance_keys([object()]) == []

    def test_none_and_empty_are_accepted(self):
        assert provenance_keys(None) == [] and provenance_keys([]) == []

    @pytest.mark.parametrize("kwargs,expected", [
        ({"xc": "pbe"}, {"PBE1996", "PerdewWang1992"}),
        ({"relativity": "scalar"}, {"KoellingHarmon1977"}),
        ({"relativity": "dirac"}, {"KoellingHarmon1977", "Kleinman1980"}),
        ({"nlcc": True}, {"Louie1982"}),
    ])
    def test_each_option_cites_its_paper(self, kwargs, expected):
        assert set(provenance_keys([self._Dataset(**kwargs)])) == expected

    def test_the_shipped_defaults_cite_relativity_and_the_core_correction(self):
        """`relativity="scalar"` and `nlcc=True` are the generation defaults,
        so an ordinary run must carry their references."""
        keys = set(provenance_keys([self._Dataset(relativity="scalar",
                                                  nlcc=True)]))
        assert keys == {"KoellingHarmon1977", "Louie1982"}

    def test_every_key_it_can_emit_resolves(self):
        for kwargs in ({"xc": "pbe"}, {"relativity": "scalar"},
                       {"relativity": "dirac"}, {"nlcc": True}):
            bib.resolve(provenance_keys([self._Dataset(**kwargs)]))

    def test_a_real_generated_dataset_is_read_correctly(self):
        """The duck-typing above is only trustworthy if the real object has
        the same three fields with the same spellings."""
        from mandacaru.pseudopotentials.paw import get_paw
        dataset = get_paw("H")
        for field in ("xc", "relativity", "nlcc"):
            assert hasattr(dataset, field), field
        bib.resolve(provenance_keys([dataset]))


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

    def test_the_taper_option_cites_the_tapering_paper(self):
        assert "Bravyi2017" not in citation_keys(mapping="parity")
        assert "Bravyi2017" in citation_keys(mapping="parity", taper=True)
        assert "Bravyi2017" in citation_keys(mapping="jordan_wigner", taper=True)

    def test_a_pseudopotential_family_cites_its_generator(self):
        assert "Bloechl1994" in citation_keys(family="paw-lcao")
        assert "Ivanov2024" in citation_keys(family="upaw-lcao")
        assert "Ivanov2024" not in citation_keys(family="paw-lcao")
        assert "Hamann2013" in citation_keys(family="paw-lcao")

    def test_basis_options_are_cited_only_when_they_did_something(self):
        plain = citation_keys(family="paw-lcao", basis_options={"size": "SZ"})
        assert "Junquera2001" not in plain and "Anglada2006" not in plain
        assert "Artacho1999" not in plain
        rich = citation_keys(family="paw-lcao",
                             basis_options={"size": "DZP", "filter_cutoff": True,
                                            "energy_shift": 0.1})
        assert {"Junquera2001", "Sankey1989", "Anglada2006",
                "Artacho1999"} <= set(rich)
        assert "Kendall1992" not in rich

    def test_diffuse_shells_cite_the_augmentation(self):
        keys = citation_keys(family="paw-lcao",
                             basis_options={"size": "SZ", "diffuse": True})
        assert {"Kendall1992", "Artacho1999", "Junquera2001"} <= set(keys)
        assert "Kendall1992" not in citation_keys(
            family="paw-lcao", basis_options={"size": "SZ", "diffuse": False})
        assert "Kendall1992" not in citation_keys(
            family="paw-lcao", basis_options={"diffuse": {"O": False}})

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
