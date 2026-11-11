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
from mandacaru.utils.citations import (_ANALYSIS_KEYS, _ANSATZ_KEYS,
                                       _DEVICE_KEYS, _EIGENSOLVER_KEYS,
                                       _EXXDIV_KEYS,
                                       _FAMILY_KEYS, _HUBBARD_KEYS,
                                       _MAPPING_KEYS, _METHOD_KEYS,
                                       _OPTIMIZER_KEYS, _POOL_KEYS,
                                       _PROVENANCE_KEYS, _ACTIVE_METHOD_KEYS,
                                       _MEASUREMENT_SCHEME_KEYS, _TWO_BODY_KEYS,
                                       _FORCE_KEYS, _PARAMETER_SHIFT_KEYS,
                                       _PERIODIC_KEYS, _SMEARING_KEYS,
                                       _TROTTER_KEYS, _VDW_KEYS, _XC_KEYS,
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
        assert "BornKarman1912" in bib.UNVERIFIED    # no DOI to check it against
        assert "[unverified record]" in bib.bibtex(["BornKarman1912"])
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

    def test_every_two_body_representation_of_the_active_space(self):
        """``dense`` is the tensor itself; the double factorization cites its papers."""
        from mandacaru.utils.citations import _TWO_BODY_REPRESENTATION_KEYS
        assert set(_TWO_BODY_REPRESENTATION_KEYS) == {"dense", "double-factorized"}
        keys = _TWO_BODY_REPRESENTATION_KEYS["double-factorized"]
        assert keys and set(keys) <= set(citation_keys(two_body_representation="double-factorized"))
        assert citation_keys(two_body_representation="dense") == citation_keys()

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

    def test_every_functional(self):
        """The functionals ``Mandacaru(xc=...)`` accepts, ``+rvv10`` folded in."""
        names = {"lda", "pbe", "r2scan", "hse06", "pbe+rvv10", "r2scan+rvv10"}
        assert set(_XC_KEYS) == names
        for name in names:
            assert _XC_KEYS[name], name
            assert set(_XC_KEYS[name]) <= set(citation_keys(xc=name)), name

    def test_every_smearing_and_eigensolver_and_exchange_treatment(self):
        from mandacaru.calculation import _EXXDIV_TREATMENTS
        assert set(_SMEARING_KEYS) == {"fermi-dirac", "gaussian",
                                       "methfessel-paxton"}
        assert set(_EIGENSOLVER_KEYS) == {"direct", "davidson", "rmm-diis",
                                          "mixed"}
        assert set(_EXXDIV_KEYS) == set(_EXXDIV_TREATMENTS)
        assert set(_HUBBARD_KEYS) == {"dudarev-fll", "liechtenstein-fll",
                                      "linear-response"}

    def test_every_key_in_every_table_exists(self):
        tables = (_POOL_KEYS, _MAPPING_KEYS, _METHOD_KEYS, _ANSATZ_KEYS,
                  _OPTIMIZER_KEYS,
                  _FAMILY_KEYS, _PROVENANCE_KEYS, _ACTIVE_METHOD_KEYS,
                  _MEASUREMENT_SCHEME_KEYS, _FORCE_KEYS, _XC_KEYS, _VDW_KEYS,
                  _SMEARING_KEYS, _HUBBARD_KEYS, _EIGENSOLVER_KEYS,
                  _EXXDIV_KEYS, _DEVICE_KEYS, _ANALYSIS_KEYS)
        for table in tables:
            for name, keys in table.items():
                bib.resolve(keys)             # raises on an unknown key
        for keys in (_PERIODIC_KEYS, _PARAMETER_SHIFT_KEYS, _TROTTER_KEYS):
            bib.resolve(keys)


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

    def test_variational_forces_cite_the_z_vector_and_its_lagrangian(self):
        lagrangian = {"HandySchaefer1984", "Helgaker1989", "Mitarai2020"}
        assert not lagrangian & set(citation_keys(method="vqe"))
        assert lagrangian <= set(citation_keys(method="vqe", forces="variational"))

    def test_the_taper_option_cites_the_tapering_paper(self):
        assert "Bravyi2017" not in citation_keys(mapping="parity")
        assert "Bravyi2017" in citation_keys(mapping="parity", taper=True)
        assert "Bravyi2017" in citation_keys(mapping="jordan_wigner", taper=True)

    def test_a_pseudopotential_family_cites_its_generator(self):
        assert "Bloechl1994" in citation_keys(family="paw-lcao")
        assert "Ivanov2025" in citation_keys(family="upaw-lcao")
        assert "Ivanov2025" not in citation_keys(family="paw-lcao")
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


class TestWhatTheCodeImplementsIsCited:
    """The configuration choices added by the 2026-11 audit: each one a user
    can select cites the paper the implementation rests on, and the stale
    citation of a method that was refuted and never built is gone."""

    def test_valsqa_cites_its_ranker_and_not_a_gaussian_process(self):
        keys = citation_keys(method="valsqa")
        assert {"BradleyTerry1952", "Gilmer2017"} <= set(keys)
        assert "Rasmussen2006" not in bib.REFERENCES
        assert "BradleyTerry1952" not in citation_keys(method="vasqa")

    def test_variational_forces_cite_their_numerical_machinery(self):
        machinery = {"Kuhn1955", "Munkres1957", "GerrattMills1968", "Pople1979"}
        assert not machinery & set(citation_keys(method="vqe"))
        assert machinery <= set(citation_keys(method="vqe",
                                              forces="variational"))

    def test_analytic_forces_cite_hellmann_feynman_and_pulay(self):
        assert not {"Feynman1939", "Pulay1969"} & set(citation_keys(method="rhf"))
        assert {"Feynman1939", "Pulay1969"} <= set(
            citation_keys(method="rhf", forces="analytic"))

    def test_a_crystal_cites_the_localized_orbital_bloch_sum(self):
        assert "SlaterKoster1954" in _PERIODIC_KEYS
        assert "SlaterKoster1954" in citation_keys(method="vqe", periodic=True)
        assert "SlaterKoster1954" not in citation_keys(method="vqe")

    def test_the_level_shift_is_cited_with_the_scf(self):
        assert "SaundersHillier1973" in citation_keys(method="rhf")
        assert "SaundersHillier1973" not in citation_keys(
            method="rhf", built_basis=False)

    def test_a_circuit_measured_energy_cites_the_parameter_shift(self):
        shift = {"Mitarai2018", "Schuld2019"}
        assert not shift & set(citation_keys(method="vqe"))
        assert shift <= set(citation_keys(method="vqe", shots=1024))
        assert shift <= set(citation_keys(method="adapt-vqe",
                                          execute_circuits=True))
        # A mean-field method ignores the circuit options.
        assert not shift & set(citation_keys(method="rhf", shots=1024))

    def test_the_matrix_product_state_device_cites_its_method(self):
        for spelling in ("AER_mps", "aer-mps", "mps"):
            assert {"Vidal2003", "Schollwoeck2011"} <= set(
                citation_keys(device=spelling)), spelling
        assert not {"Vidal2003", "Schollwoeck2011"} & set(
            citation_keys(device="AER_simulator"))
        assert not {"Vidal2003"} & set(citation_keys(device="no-such-device"))

    def test_trotter_cites_the_product_formula(self):
        assert "Trotter1959" not in citation_keys(method="vqe")
        assert "Trotter1959" in citation_keys(method="vqe", trotter=True)

    def test_functionals_cite_their_papers(self):
        assert "PerdewZunger1981" in citation_keys(xc="lda")
        assert "Furness2020" in citation_keys(xc="r2scan")
        assert {"Heyd2003", "Krukau2006", "Paier2005"} <= set(
            citation_keys(xc="hse06"))
        rvv10 = {"Vydrov2010", "Sabatini2013", "RomanPerez2009"}
        assert rvv10 <= set(citation_keys(xc="pbe+rvv10"))
        assert rvv10 | {"Furness2020", "Ning2022"} <= set(
            citation_keys(xc="r2scan+rvv10"))
        assert not rvv10 & set(citation_keys(xc="pbe"))

    def test_the_xc_spelling_is_insensitive(self):
        assert citation_keys(xc="HSE06") == citation_keys(xc="hse06")

    def test_the_dispersion_correction_cites_its_reference_data(self):
        ts = {"Tkatchenko2009", "ChuDalgarno2004", "Ruiz2012", "Grimme2006",
              "Hirshfeld1977"}
        assert ts <= set(citation_keys(vdw="ts"))
        assert not ts & set(citation_keys())

    def test_smearing_cites_its_broadening(self):
        assert "MethfesselPaxton1989" in citation_keys(smearing="methfessel-paxton")
        assert "FuHo1983" in citation_keys(smearing="Gaussian")
        assert "FuHo1983" not in citation_keys(smearing="fermi-dirac")
        assert "Mermin1965" in citation_keys(smearing="fermi-dirac")

    def test_the_hubbard_forms_and_their_response_cite_their_papers(self):
        assert "Dudarev1998" in citation_keys(hubbard="dudarev-fll")
        assert "Liechtenstein1995" in citation_keys(hubbard="liechtenstein-fll")
        both = citation_keys(hubbard=("dudarev-fll", "linear-response"))
        assert {"Dudarev1998", "Cococcioni2005"} <= set(both)
        assert not {"Dudarev1998", "Liechtenstein1995"} & set(citation_keys())

    def test_the_crystal_solver_options_cite_their_algorithms(self):
        assert "Davidson1975" in citation_keys(eigensolver="davidson")
        assert {"WoodZunger1985", "Pulay1980"} <= set(
            citation_keys(eigensolver="rmm-diis"))
        assert {"Davidson1975", "WoodZunger1985"} <= set(
            citation_keys(eigensolver="mixed"))
        assert not {"Davidson1975", "WoodZunger1985"} & set(
            citation_keys(eigensolver="direct"))
        assert "GygiBaldereschi1986" in citation_keys(
            exxdiv_treatment="gygi-baldereschi")
        assert "GygiBaldereschi1986" not in citation_keys(
            exxdiv_treatment="none")

    def test_the_analyses_cite_their_methods(self):
        assert {"JepsenAndersen1971", "LehmannTaut1972",
                "BloechlJepsenAndersen1994"} <= set(
            citation_keys(analysis="tetrahedron-dos"))
        spectral = {"Lehmann1954", "Lanczos1950", "GaglianoBalseiro1987"}
        assert spectral <= set(citation_keys(analysis=("spectral-function",)))
        assert not spectral & set(citation_keys())
        assert "Spencer2008" in citation_keys(analysis="downfold")
        assert not {"Aryasetiawan2004", "Sasioglu2011"} & set(
            citation_keys(analysis="downfold"))
        assert {"Aryasetiawan2004", "Sasioglu2011", "Spencer2008"} <= set(
            citation_keys(analysis="downfold-screening"))
        assert not {"Aryasetiawan2004", "Sasioglu2011", "Spencer2008"} & set(
            citation_keys())
        assert "SetyawanCurtarolo2010" in citation_keys(analysis="band-path")
        assert "SetyawanCurtarolo2010" not in citation_keys()
        assert "MonkhorstPack1976" in citation_keys(analysis="irreducible-zone")
        echoes = {"Trotter1959", "Strang1968", "Suzuki1991",
                  "GoogleQuantumAI2025"}
        assert echoes <= set(citation_keys(analysis=["nested-otoc"]))
        assert echoes <= set(citation_keys(analysis="quantum-echoes"))
        assert "GoogleQuantumAI2025" not in citation_keys(
            analysis="time-evolution")
        assert {"Kuhn1960", "Doi1991"} <= set(citation_keys(analysis="isosurface"))

    def test_the_records_fixed_by_the_audit(self):
        helgaker = bib.REFERENCES["Helgaker1989"]
        assert helgaker.verified and "10.1007/BF00527713" in helgaker.entry
        assert bib.REFERENCES["Mitarai2020"].verified
        # The first DOI of this record was another paper's.
        assert "10.1137/17M1129696" in bib.REFERENCES["Damle2018"].entry
        assert "Ivanov2025" in bib.REFERENCES and "Ivanov2024" not in bib.REFERENCES
        assert "10.1021/acs.jctc.5c00551" in bib.REFERENCES["Ivanov2025"].entry
