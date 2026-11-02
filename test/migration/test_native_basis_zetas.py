"""Native split-valence zetas against the frozen scipy construction, on grids from 0 and from h.

``reference/basis_zetas/`` holds what ``multizeta.zeta_tables`` (``split_radius``, ``split_valence_tail``,
cumulative trapezoid and ``CubicSpline``) returned before it became a forwarder to ``_native.radial_zetas``
(``freeze_basis_zetas.py``, stamped): the synthetic grids below and the O and Fe PAW-LCAO grids."""
import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.basis.multizeta import zeta_tables
from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.environment import LibraryPathError

FROZEN_DIR = Path(__file__).resolve().parent / "reference" / "basis_zetas"
META = json.loads((FROZEN_DIR / "zeta_tables.json").read_text())
FROZEN = np.load(FROZEN_DIR / "zeta_tables.npz")

STEP = 0.02
TAILS = np.array([0.15, 0.31, 0.54])
#: Both sides run the same not-a-knot spline and trapezoid on the same knots, so
#: they differ by round-off only. The worst difference over every case below and
#: the O and Fe PAW grids, relative to max|table|, was 4.6e-15 (measured with the
#: Rust function against the Python one before the binding existed); 1e-12 leaves
#: two orders of margin and no physical slack.
RTOL = 1e-12


def grid(start):
    return start + STEP * np.arange(700)


def radial(r, l):
    """A bound-state shape that vanishes smoothly at the last node."""
    return r**l * np.exp(-1.3 * r) * np.clip(1 - (r / r[-1]) ** 2, 0, None) ** 2


def native(r, values, l, count, split, tails=None, fractions=False):
    ctx = _native.Context(threads=1)
    out = _native.radial_zetas(ctx, r, values, l, count, split, tails, fractions)
    assert ctx.memory_used_bytes == 0
    return out


def assert_close(got, expected):
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() <= RTOL * np.abs(expected).max()


@pytest.mark.parametrize("index", range(len(META["synthetic"])))
def test_the_frozen_synthetic_tables(index):
    case = META["synthetic"][index]
    r = grid(case["start"])
    tails = None if case["tails"] is None else np.array(case["tails"])
    got = native(r, radial(r, case["l"]), case["l"], case["count"], case["split"], tails)
    assert len(got) == case["count_out"]
    for k, table in enumerate(got):
        assert_close(table, FROZEN[f"syn{index}_{k}"])
    # The forwarder in multizeta hands the same tables over.
    forwarded = zeta_tables(r, radial(r, case["l"]), case["l"] + 1, case["l"], case["count"], case["split"],
                            tail_norms=None if tails is None else tuple(tails))
    assert [t.zeta for t in forwarded] == list(range(1, case["count_out"] + 1))
    for k, table in enumerate(forwarded):
        assert_close(table.values, FROZEN[f"syn{index}_{k}"])


def test_the_grid_origin_is_part_of_the_answer():
    """The same table on a grid from 0 and from h splits at different radii."""
    values = radial(grid(STEP), 1)
    shifted = native(grid(STEP), values, 1, 2, 0.15)[1]
    unshifted = native(grid(0.0), values, 1, 2, 0.15)[1]
    assert np.abs(shifted - unshifted).max() > 1e-3 * np.abs(shifted).max()


@pytest.mark.parametrize("case", META["paw"], ids=lambda c: c["key"])
def test_the_frozen_library_grid_of_a_paw_dataset(case):
    try:
        pp = get_paw(case["symbol"])
    except (FileNotFoundError, LibraryPathError) as error:
        pytest.skip(f"external PAW dataset library is required: {error}")
    r = np.asarray(pp.r)
    assert 0.0 < r[0] < 0.01
    values = np.asarray(pp.channels[case["l"]].pseudo_waves[0], dtype=float)
    tails = TAILS if case["scheme"] == "first_zeta" else None
    got = native(r, values, case["l"], 3, 0.15, tails)
    assert len(got) == case["count_out"]
    for k, table in enumerate(got[1:], start=1):
        assert_close(table, FROZEN[f"{case['key']}_{k}"])


def test_tail_fractions_are_the_squares_of_tail_norms():
    r = grid(STEP)
    values = radial(r, 1)
    by_norm = native(r, values, 1, 4, 0.15, TAILS)
    by_fraction = native(r, values, 1, 4, 0.15, TAILS**2, fractions=True)
    assert len(by_norm) == len(by_fraction) == 4
    for a, b in zip(by_norm, by_fraction):
        np.testing.assert_array_equal(a, b)


def test_unsupported_requests_raise():
    r = grid(STEP)
    v = radial(r, 0)
    for count, split in [(0, 0.15), (5, 0.15), (2, 0.0), (2, 1.0), (2, np.nan),
                         (2, np.inf), (2, -0.1)]:
        with pytest.raises(ValueError):
            native(r, v, 0, count, split)
    with pytest.raises(ValueError):
        native(r, v, 0, 3, 0.15, TAILS[:1])
    with pytest.raises(ValueError):
        native(r, v, 0, 2, 0.15, np.array([np.nan]))
    bad = v.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        native(r, bad, 0, 2, 0.15)
    with pytest.raises(ValueError):
        native(r[:-1], v, 0, 2, 0.15)
    with pytest.raises(ValueError):
        native(r[::-1].copy(), v, 0, 2, 0.15)
    with pytest.raises(ValueError):
        native(r, np.zeros_like(v), 0, 2, 0.15)
    with pytest.raises(ValueError):
        native(r - 1.0, v, 0, 2, 0.15)
