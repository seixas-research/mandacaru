# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_gates.py

# This code is part of Mandacaru.
# MIT License

"""The per-file gates of a dataset through the native entry (``_native.paw_validate``,
``_native.paw_validate_file``; ``paw::validate``): the report's shape, that a file and its record
give the same report, that a damaged record fails its gate, and that the build reports the gates
of the file it wrote.  The thresholds and the gate-by-gate negative controls are in
``crates/mandacaru-core/tests/paw_validate.rs``; none is repeated here."""

from __future__ import annotations

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.pseudopotentials.io import STRIDE
from mandacaru.pseudopotentials.native_context import context

#: Every per-file gate (G12 is a crystal calculation).
GATES = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G13", "G14"]
ROW_KEYS = {"gate", "check", "value", "threshold", "relation", "status", "detail"}


@pytest.fixture(scope="module")
def lithium(tmp_path_factory):
    """Lithium, built in Rust, with the build's own answer."""
    path = str(tmp_path_factory.mktemp("gates") / "Li.parquet")
    answer = _native.paw_build(context(), "Li", {"family": "paw-lcao", "xc": "lda", "relativity": "scalar",
                                                 "ghosts": "repair"}, path, stride=STRIDE)
    return path, answer


def test_the_build_reports_every_gate_of_the_file_it_wrote(lithium):
    _path, answer = lithium
    report = answer["gates"]
    assert report["passed"] and report["symbol"] == "Li"
    assert list(report["gates"]) == GATES
    assert report["gates"]["G1"]["status"] == "pass"     # measured where the generated record is at hand
    assert report["gates"]["G13"]["status"] == "n/a"
    assert all(set(row) == ROW_KEYS for row in report["rows"])


def test_a_file_read_alone_has_nothing_to_reload_and_says_so(lithium):
    path, _ = lithium
    report = _native.paw_validate_file(context(), path)
    assert report["passed"]
    assert report["gates"]["G1"]["status"] == "n/a"
    assert report["gates"]["G10"]["status"] == "n/a"      # needs an external all-electron code
    row, = [r for r in report["rows"] if r["gate"] == "G5" and "core density" in r["check"]]
    assert row["relation"] == "<=" and row["value"] < row["threshold"]


def test_a_file_and_its_record_give_the_same_report(lithium):
    path, _ = lithium
    payload, tables = _native.paw_record_read(context(), path)
    assert _native.paw_validate(context(), payload, tables) == _native.paw_validate_file(context(), path)


def test_a_damaged_record_fails_its_gate_and_the_report_says_which(lithium):
    path, _ = lithium
    payload, tables = _native.paw_record_read(context(), path)
    tables = {name: np.array(values) for name, values in tables.items()}
    tables["smooth_core_density"] = tables["smooth_core_density"] * 1.1
    report = _native.paw_validate(context(), payload, tables)
    assert not report["passed"] and report["gates"]["G9"]["status"] == "FAIL"
    assert [g for g, v in report["gates"].items() if v["status"] == "FAIL"] == ["G9"]


def test_a_twin_with_another_partition_fails_the_partition_gate(lithium):
    path, _ = lithium
    import json
    payload, tables = _native.paw_record_read(context(), path)
    other = json.loads(payload)
    other["semicore_subshells"] = [[1, 0]]
    report = _native.paw_validate(context(), payload, tables, twin=(json.dumps(other), tables))
    row, = [r for r in report["rows"] if r["gate"] == "G8" and "twin" in r["check"]]
    assert row["status"] == "FAIL" and not report["passed"]
