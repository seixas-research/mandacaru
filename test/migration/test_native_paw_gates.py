# -*- coding: utf-8 -*-
# file: test/migration/test_native_paw_gates.py

# This code is part of Mandacaru.
# MIT License

"""The numpy checks the regeneration sweep ran in Python (``paw_regen/worker2.py``, ``validate_all.py``)
against the Rust gates that replaced them (``paw::validate``): grid uniformity (G11), the core
record (G5: charge, orbital density against the stored one, the charge beyond the widest sphere)
and the smooth-core electrons (G9).  The oracle below is that numpy arithmetic, verbatim; the Rust
side must reproduce its numbers.  Lithium is generated here, so no library is read."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.pseudopotentials.io import STRIDE
from mandacaru.pseudopotentials.native_context import context

#: Rust sums in node order, numpy in pairs: agreement far below the thresholds, not bitwise.
AGREE = 1e-12


@pytest.fixture(scope="module")
def lithium(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("parity") / "Li.parquet")
    _native.paw_build(context(), "Li", {"family": "paw-lcao", "xc": "lda", "relativity": "scalar",
                                        "ghosts": "repair"}, path, stride=STRIDE)
    return path


def oracle(path):
    payload, t = _native.paw_record_read(context(), path)
    p = json.loads(payload)
    r = np.asarray(t["r"])
    step = np.diff(r)
    grid = float(max(np.max(np.abs(step - step[0])), abs(r[0] - step[0])) / step[0])
    states = p["core_states"]
    charge = abs(sum(s["occupation"] for s in states) - (p["atomic_number"] - p["valence_charge"]))
    density = np.zeros_like(r)
    for s in states:
        u = np.asarray(t[s["table"]])
        density += s["occupation"] * u * u / (4 * np.pi * r * r)
    core = np.asarray(t["core_density"])
    deviation = float(np.max(np.abs(density - core)) / max(np.max(np.abs(core)), 1e-300))
    radius = max(c["r_cut"] for c in p["channels"].values())
    outside = float(np.trapezoid(np.where(r > radius, 4 * np.pi * r * r * core, 0.0), r))
    smooth = float(np.trapezoid(4 * np.pi * r * r * np.asarray(t["smooth_core_density"]), r))
    return {"grid": grid, "charge": charge, "deviation": deviation, "outside": outside, "smooth": smooth,
            "stored": p["nlcc"]["partial_core_electrons"]}


def value(report, gate, part):
    row, = [r for r in report["rows"] if r["gate"] == gate and part in r["check"]]
    return row["value"]


def test_the_rust_gates_reproduce_the_numpy_numbers(lithium):
    want = oracle(lithium)
    report = _native.paw_validate_file(context(), lithium)
    assert value(report, "G11", "radial grid") == pytest.approx(want["grid"], rel=AGREE, abs=AGREE)
    assert value(report, "G5", "core charge:") == pytest.approx(want["charge"], abs=AGREE)
    assert value(report, "G5", "core density") == pytest.approx(want["deviation"], rel=1e-9, abs=AGREE)
    assert value(report, "G5", "beyond the widest sphere") == pytest.approx(want["outside"], rel=AGREE, abs=AGREE)
    assert value(report, "G9", "integrated") == pytest.approx(want["smooth"], rel=AGREE, abs=AGREE)
    # The consistency row is the distance of that integral from the stored partial-core electrons.
    assert value(report, "G9", "against partial_core_electrons") == pytest.approx(
        abs(want["smooth"] - want["stored"]), rel=1e-9, abs=AGREE)
