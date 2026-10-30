# -*- coding: utf-8 -*-
# file: test/basis/test_filtering_frozen.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The native Fourier filter against the numbers of the Python filter it replaced.

``test/migration/reference/paw/radial_filter.{npz,json}`` were taken on
2026-10-30 from the NumPy + SciPy implementation of ``basis/filtering.py``
(``spherical_jn`` kernel matrices) before it was deleted: PAW-LCAO pseudo waves of
H, C, O, Si, Cu, Ag, Au and Pt at several cutoffs, a diffuse function, and
analytic functions through every method and option.  Each case records its
``info`` numbers and a fingerprint of the output (sum, sum of squares, maximum, eight
samples); the small ones also keep the whole array.  The tolerance is
``1e-9`` relative to the function's peak (the residuals, which are
``1 - (a number near one)``, carry an absolute floor of ``1e-13``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mandacaru.basis.filtering import filter_radial

REFERENCE = Path(__file__).parents[1] / "migration" / "reference" / "paw"
FROZEN = json.loads((REFERENCE / "radial_filter.json").read_text())
ARRAYS = np.load(REFERENCE / "radial_filter.npz")
TOLERANCE = FROZEN["tolerance_relative"]

MESHES = {
    "uniform": lambda: np.linspace(0.02, 25.0, 700),
    "log": lambda: 0.01 * np.exp(np.linspace(0.0, np.log(25.0 / 0.01), 600)),
    "zero": lambda: np.linspace(0.0, 20.0, 500),
}


def inputs(case):
    """``(r, values)`` of a case: stored arrays, the shipped library, or a formula."""
    l = case["l"]
    if case["source"] == "analytic":
        r = MESHES[case["grid"]]()
        return r, r ** l * np.exp(-case["a"] * r * r)
    if case["input_keys"] is not None:
        r_key, v_key = case["input_keys"]
        return ARRAYS[r_key], ARRAYS[v_key]
    from mandacaru.pseudopotentials.paw import get_paw

    pp = get_paw(case["symbol"])
    r = np.asarray(pp.r, dtype=float)
    values = np.asarray(pp.channels[l].pseudo_waves[0], dtype=float)
    # the library must still hold the table the reference was taken on
    assert float(values.sum()) == pytest.approx(
        case["input_fingerprint"]["sum"], rel=1e-12)
    return r, values


def run(case):
    r, values = inputs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return filter_radial(r, values, case["l"], case["k_c"], **case["options"])


def close(a, b, floor=0.0):
    return abs(a - b) <= TOLERANCE * abs(b) + floor


@pytest.mark.parametrize("case", FROZEN["cases"], ids=lambda c: c["tag"])
def test_the_native_filter_reproduces_the_frozen_python_numbers(case):
    if case["source"] == "library" and case["input_keys"] is None:
        from mandacaru.pseudopotentials.paw import get_paw
        try:
            get_paw(case["symbol"])
        except Exception as error:               # library not installed
            pytest.skip(f"PAW library unavailable: {error}")
    out, info = run(case)
    frozen = case["info"]
    for name in ("support_radius", "mask_radius", "norm_kept", "max_change"):
        assert close(info[name], frozen[name]), (name, info[name], frozen[name])
    for name in ("residual_before", "residual_after"):
        # ``1 - inside``: the two sums differ in order, so the noise floor is
        # an absolute 1e-13 (measured: 7e-15 on a residual of 2.7e-8)
        assert close(info[name], frozen[name], floor=1e-13), (
            name, info[name], frozen[name])
    assert info["masked"] == frozen["masked"]
    assert info["method"] == frozen["method"] and info["l"] == frozen["l"]
    assert info["cutoff"] == frozen["cutoff"]

    fingerprint = case["output"]
    peak = fingerprint["max_abs"]
    assert out.size == case["n"]
    assert close(float(out.sum()), fingerprint["sum"], floor=1e-12 * peak)
    assert close(float((out * out).sum()), fingerprint["sum_squares"])
    for index, expected in zip(fingerprint["indices"], fingerprint["samples"]):
        assert abs(out[index] - expected) <= TOLERANCE * peak, index
    assert float(np.abs(out).max()) == pytest.approx(peak, rel=TOLERANCE)

    if case["output_key"] is not None:
        assert np.abs(out - ARRAYS[case["output_key"]]).max() <= TOLERANCE * peak
