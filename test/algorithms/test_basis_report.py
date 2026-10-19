# -*- coding: utf-8 -*-
# file: test/algorithms/test_basis_report.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``[BASIS]`` block (:func:`mandacaru.algorithms.basis_report.basis_report`):
what a PAW-LCAO basis reports about itself -- its options, its datasets, its
orbitals and, with ``diffuse=``, every diffuse shell and the box that holds
it."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms._hamiltonian_from_atoms import (
    build_basis_hamiltonian)
from mandacaru.algorithms.basis_report import basis_report
from mandacaru.pseudopotentials import get_paw
from mandacaru.units import BOHR_TO_ANGSTROM

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _library_present() -> bool:
    try:
        get_paw("H")
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_library = pytest.mark.skipif(not _library_present(),
                                   reason="MANDACARU_PAW_PATH is not configured")


def _report(basis, cell=15.4):
    atoms = Atoms("H2", positions=[[cell / 2, cell / 2, cell / 2 - 0.37],
                                   [cell / 2, cell / 2, cell / 2 + 0.37]],
                  cell=[cell] * 3)
    context = build_basis_hamiltonian(atoms, basis, None, 0.5, 0, None)[4]
    return basis_report(basis, context, atoms.get_chemical_symbols())


def _rows(table):
    header, *rows = table
    return [dict(zip(header, row)) for row in rows]


@needs_library
class TestBasisBlock:
    def test_the_ordinary_fields(self):
        fields, tables = _report({"name": "PAW-LCAO"})
        assert fields["name"] == "PAW-LCAO"
        assert fields["size"] == "SZ"
        assert fields["energy_shift"] == "0.1 eV"
        assert fields["diffuse"] == "off"
        assert fields["basis_functions"] == 2
        assert {"datasets", "orbitals", "functions"} <= set(tables)
        assert "diffuse_shells" not in tables and "diffuse_box" not in tables
        assert _rows(tables["functions"]) == [
            {"symbol": "H", "atoms": 2, "functions_per_atom": 1}]

    def test_the_diffuse_tables(self):
        fields, tables = _report({"name": "PAW-LCAO", "diffuse": True})
        assert fields["diffuse"] == ("shells present, count 1, rms_ratio 2.5, "
                                     "cutoff_norm 0.0001")
        assert fields["basis_functions"] == 4
        # The functions table counts the diffuse shells.
        assert _rows(tables["functions"])[0]["functions_per_atom"] == 2
        (shell,) = _rows(tables["diffuse_shells"])
        assert (shell["symbol"], shell["l"], shell["index"]) == ("H", 0, 0)
        assert shell["status"] == "accepted"
        assert shell["reference_l"] == 0
        # Lengths in Bohr and Angstrom, consistent with each other.
        for key in ("rms", "r_cut"):
            assert float(shell[f"{key}_Angstrom"]) == pytest.approx(
                float(shell[f"{key}_Bohr"]) * BOHR_TO_ANGSTROM, abs=1e-4)
        assert float(shell["rms_Bohr"]) == pytest.approx(
            2.5 * float(shell["reference_rms_Bohr"]), abs=1e-3)
        box = _rows(tables["diffuse_box"])
        assert [row["atom"] for row in box] == [0, 1]
        for row in box:
            assert float(row["face_distance_Bohr"]) > float(row["r_cut_Bohr"])
            assert float(row["grid_norm"]) > 1 - 1e-4
        # Cells never contain spaces: a reader splits a row on whitespace.
        for table in tables.values():
            for row in table:
                assert all(" " not in str(cell) for cell in row)

    def test_a_short_box_shows_in_the_table(self):
        _fields, tables = _report({"name": "PAW-LCAO", "diffuse": True},
                                  cell=8.0)
        box = _rows(tables["diffuse_box"])
        assert all(float(row["face_distance_Bohr"]) < float(row["r_cut_Bohr"])
                   for row in box)
        assert np.min([float(row["grid_norm"]) for row in box]) < 1 - 1e-4
