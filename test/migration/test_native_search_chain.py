"""Seeded VASQA chains stay the same chain when the native search is restructured.

``reference/search_chain.json`` was written by ``freeze_search_chain.py`` from the extension
installed before the VASQA plan's classes of same-chain changes (A1 uniform proposals skip the pool
screen and rows stop copying four vectors when nothing records, A2 the seed-cache fix of the cold
chain, A3 the algebraic detailed-balance test). Every option is explicit, so a later default does
not move it, and every chain was identical at one and eight threads when it was frozen. The moves,
sequences and accept flags are compared exactly, the energies and recorded vectors to
``TOLERANCE``.

``transfer_cold`` is the one case a planned change may move: with ``warm_start=False`` the seed
(the transferred ansatz) was evaluated from its transferred angles and cached under its sequence,
while every other state is evaluated from zero. The seed-cache fix evaluates the transferred start
uncached for ``kept`` and for ``best`` and installs ``current`` from a zero-start evaluation, so the
seed's cost and importances move by the optimizer's tolerance (expected about 1e-9); its tolerance
is ``COLD_TRANSFER_TOLERANCE``, and its ``num_evaluations`` (optimizer calls) is not compared because
the zero-start evaluation of the seed adds calls. Its moves, sequences and accept flags must still
match exactly; if only a float fails, read the actual difference before loosening anything: the
``log_acceptance`` is a cost difference over a temperature down to 3.7e-5 Ha, which amplifies an
energy shift about 3e4 times, and the angles of a flat manifold only agree to the square root of the
energy tolerance. The reference stays the pre-change build's. The other six cases, the default warm-start paths among them, must
hold bit for bit.
"""

import json
import math
from pathlib import Path

import pytest

from .search_chain_reference import CASES, run, run_h2

REFERENCE = json.loads((Path(__file__).parent / "reference" / "search_chain.json").read_text())

TOLERANCE = 1e-10
COLD_TRANSFER_TOLERANCE = 1e-7
THREADS = (1, 8)


def assert_same(got, want, path, tol):
    if isinstance(want, dict):
        assert set(got) == set(want), path
        for key in want:
            assert_same(got[key], want[key], f"{path}.{key}", tol)
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            assert_same(g, w, f"{path}[{i}]", tol)
    elif isinstance(want, float):
        assert math.isclose(got, want, rel_tol=0, abs_tol=tol), f"{path}: {got} != {want}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("case", list(CASES))
def test_a_frozen_chain_is_reproduced(case, threads, tmp_path):
    got, want = run(case, threads, tmp_path), dict(REFERENCE[case])
    tol = TOLERANCE
    if case == "transfer_cold":
        tol = COLD_TRANSFER_TOLERANCE
        got.pop("num_evaluations")
        want.pop("num_evaluations")
    assert_same(got, want, case, tol)


def test_the_recording_and_the_screening_leave_the_uniform_chain_alone():
    off = REFERENCE["uniform_warm_record_off"]
    on = REFERENCE["uniform_warm_record_on"]
    assert off["steps"] == on["steps"]
    assert len(on["records"]) == len(on["steps"])
    screened = REFERENCE["uniform_cold_screened"]
    assert any(row["screened"] for row in screened["records"])
    assert [row["accepted"] for row in screened["steps"]].count(True) > 0
    assert REFERENCE["gradient_warm"]["steps"] != off["steps"]


@pytest.mark.parametrize("case, copies", [("uniform_warm_record_off", False),
                                          ("gradient_warm", False),
                                          ("uniform_warm_record_on", True)])
def test_step_rows_carry_the_source_vectors_only_when_the_chain_records(case, copies, tmp_path):
    fields = ("source_gradients", "source_probabilities", "source_importances", "source_removal")
    present = run_h2(case, 1, tmp_path, view=lambda result: [
        [hasattr(step, name) for name in fields] for step in result.steps])
    assert all(all(row) == copies and any(row) == copies for row in present)
