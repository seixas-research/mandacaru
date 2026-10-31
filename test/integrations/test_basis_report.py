# -*- coding: utf-8 -*-
# file: test/integrations/test_basis_report.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``[BASIS]`` fields and tables, assembled from stand-ins for the native
integrals.  The PAW-LCAO block of a real run is pinned by
``test/utils/test_logging.py::TestBasisBlock``."""

from types import SimpleNamespace

import pytest

from mandacaru import _native
from mandacaru.integrations import basis_report as report
from mandacaru.integrations.basis_report import basis_report

H2 = SimpleNamespace(numbers=(1, 1))
LIH = SimpleNamespace(numbers=(3, 1))


@pytest.fixture
def native(monkeypatch):
    """The native report of a basis with two functions on atom 0 and one on atom 1."""
    owners = [0, 0, 1]
    monkeypatch.setattr(_native, "integrals_report",
                        lambda _integrals: dict(atom_of_orbital=owners, n_orbitals=len(owners)))


def driver(basis="HAO", spec=None, **options):
    values = dict(basis=basis, _basis_spec={"name": "HAO"} if spec is None else spec,
                  integrals=object(), _precision="float64", method="rhf")
    values.update(options)
    return SimpleNamespace(**values)


class TestAnAllElectronBasis:
    def test_it_names_the_family_and_counts_the_functions_per_element(self, native):
        fields, tables = basis_report(driver(), LIH)
        assert fields["name"] == "HAO" and fields["family"] == "all-electron"
        assert fields["basis_functions"] == 3
        assert tables["functions"] == [("symbol", "atoms", "functions_per_atom"),
                                       ("Li", 1, 2), ("H", 1, 1)]

    def test_the_options_are_the_resolved_ones_and_absent_when_there_are_none(self, native):
        plain, _ = basis_report(driver(), LIH)
        assert "options" not in plain
        resolved = {"name": "HAO", "virtual_orbitals": 1}
        fields, _ = basis_report(driver(basis={"name": "HAO", "virtual_orbitals": 1},
                                        spec=resolved), LIH)
        assert fields["options"] == "{virtual_orbitals: 1}"

    def test_a_nested_option_is_written_in_a_fixed_order(self):
        assert report._option_text({"b": 1, "a": {"z": 1, "y": 2}}) == \
            "{a: {y: 2, z: 1}, b: 1}"

    def test_a_per_element_basis_is_named_as_such(self, native):
        spec = {"name": "per_element", "elements": {"Li": {"name": "HAO"}}}
        fields, _ = basis_report(driver(basis={"Li": "HAO"}, spec=spec), LIH)
        assert fields["name"] == "per-element"

    def test_the_precision_is_the_last_scalar_field_and_the_only_one(self, native):
        fields, tables = basis_report(driver(_precision="float32"), LIH)
        assert list(fields)[-1] == "precision" and fields["precision"] == "float32"
        assert list(tables) == ["functions"]


class TestACalculationWithoutAGeometry:
    def test_no_basis_was_built_so_only_the_precision_is_stated(self):
        fields, tables = basis_report(driver(integrals=None, _precision="float32"), None)
        assert fields == {"precision": "float32"} and tables == {}
