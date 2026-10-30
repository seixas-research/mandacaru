# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_orbitals.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The valence basis of a pseudopotential: ``pseudo_basis`` and the states it is
built on (``pseudopotentials/orbitals.py``).

``test/migration/reference/paw/orbitals.npz`` (inputs in ``orbitals.json``) holds
the radial tables the Python implementation built for SZ, DZ, DZP and TZP on
H, C, N, O, Si, Cu and Au (confined at 0.1 eV, Gaussian polarization, Fourier
filter on), the unconfined and orbital-polarization variants, and the bound
levels and orbitals of the empty channels.  The confined orbitals sit behind
the cutoff-radius root search, so the tables agree to 1e-8 of their largest
entry (measured: 3e-11); the levels to 1e-9.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.families import pseudo_basis_arguments
from mandacaru.pseudopotentials.orbitals import (
    BOUND_LEVEL, KBProjector, PseudoAtomicOrbital, _RadialTabulated,
    basis_states, bound_radial,
    occupied_states, pseudo_basis, valence_electrons, _lowest_level)

REFERENCE = Path(__file__).resolve().parents[1] / "migration" / "reference" / "paw"
META = json.loads((REFERENCE / "orbitals.json").read_text())
FROZEN = np.load(REFERENCE / "orbitals.npz")
TABLE_META = json.loads((REFERENCE / "radial_table.json").read_text())
TABLE_FROZEN = np.load(REFERENCE / "radial_table.npz")

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _library_present() -> bool:
    try:
        get_paw("H")
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_library = pytest.mark.skipif(not _library_present(),
                                   reason="MANDACARU_PAW_PATH is not configured")


def close(value, expected, tol=1e-8):
    value, expected = np.asarray(value, float), np.asarray(expected, float)
    assert value.shape == expected.shape
    scale = max(float(np.abs(expected).max()), 1e-300)
    assert float(np.abs(value - expected).max()) <= tol * scale


def strided(array):
    return array[::-(-len(array) // META["max_points"])]


def unique_tables(functions):
    """One ``(l, kind, zeta, n, r, values)`` per distinct radial table, in
    basis order (the m components of a shell share their table)."""
    seen = {}
    for f in functions:
        table = getattr(f, "table", None)
        if table is None:
            r, values, kind, zeta, n = f._r, f._values, "orbital", 1, f.n
        else:
            r, values, kind, zeta, n = (table.r, table.values, f.kind, f.zeta,
                                        f.n)
        seen.setdefault(id(values), (f.l, kind, zeta, n, r, values))
    return list(seen.values())


def frozen_table(tag):
    """The ``(r, values)`` table a ``radial_table`` case was frozen on."""
    kind = tag.split("_")[0]
    l = int(tag.split("_")[2 if tag.split("_")[1] == "w" else -1])
    if kind.startswith("syn"):
        if kind == "syn0":
            r = np.linspace(0.0, 9.0, 901)
            return r, r ** l * np.exp(-r)
        if kind == "synr":
            r = np.linspace(0.02, 9.0, 450)
            return r, r ** l * np.exp(-r) * np.cos(r)
        r = np.geomspace(1e-3, 12, 300)
        return r, np.sin(r) * np.exp(-r / 3)
    pp = get_paw(kind)
    channel = pp.channels[l]
    if tag.split("_")[1] == "w":
        return np.asarray(pp.r), np.asarray(channel.pseudo_waves[0])
    return np.asarray(pp.r), np.asarray(channel.projectors[0])


class TestFrozenRadialTable:
    """``_RadialTabulated.radial``, ``radial_slope`` and the spline coefficients
    against what the SciPy ``CubicSpline`` version produced (``radial_table.npz``:
    shipped H, O, Cu tables, origin-started, offset and geometric grids; the
    nodes reach below the first point, past the last one, NaN and -inf)."""

    CASES = [c for c in TABLE_META["cases"]
             if c["tag"].startswith("syn") or _library_present()]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c["tag"])
    def test_the_values_and_slopes(self, case):
        tag = case["tag"]
        r, values = frozen_table(tag)
        assert [r[0], r[-1], len(r)] == pytest.approx(
            [case["r0"], case["r_max"], case["n"]], rel=1e-14)
        function = _RadialTabulated(r, values, case["l"], 0, real=True)
        nodes = TABLE_FROZEN[f"{tag}_nodes"]
        close(function.radial(nodes), TABLE_FROZEN[f"{tag}_radial"], 1e-12)
        close(function.radial_slope(nodes), TABLE_FROZEN[f"{tag}_slope"], 1e-11)
        scalar = TABLE_FROZEN[f"{tag}_scalar"]
        assert function.radial(0.37).shape == ()
        close([function.radial(0.37), function.radial_slope(0.37)], scalar, 1e-12)
        if f"{tag}_c" in TABLE_FROZEN.files:
            close(function._spline.x, TABLE_FROZEN[f"{tag}_x"], 1e-14)
            # The leading coefficient is the curvature jump over the interval
            # width: on the geometric grid (first interval 3e-5) it carries
            # the solver round-off 1e-11 of the curvature, divided by it.
            tol = 2e-5 if tag.startswith("syng") else 1e-9
            close(function._spline.c, TABLE_FROZEN[f"{tag}_c"], tol)

    def test_a_shape_is_kept_for_arrays(self):
        r = np.linspace(0.1, 5.0, 60)
        function = _RadialTabulated(r, np.exp(-r), 1, 0, real=True)
        nodes = np.linspace(0.0, 6.0, 12).reshape(3, 4)
        assert function.radial(nodes).shape == (3, 4)
        assert function.radial_slope(nodes).shape == (3, 4)

    def test_a_table_that_is_not_increasing_is_refused(self):
        with pytest.raises(ValueError):
            _RadialTabulated(np.array([0.0, 1.0, 1.0, 2.0, 3.0]), np.ones(5),
                             0, 0, real=True)


@needs_library
class TestFrozenPythonOutputs:
    @pytest.mark.parametrize("case", META["cases"], ids=lambda c: c["tag"])
    def test_the_radial_tables(self, case):
        symbol = case["symbol"]
        functions, owners = pseudo_basis(
            [symbol], np.zeros((1, 3)), {symbol: get_paw(symbol)},
            filter_wavevector=case["k_c"],
            **pseudo_basis_arguments(
                "paw-lcao", dict(case["options"], size=case["size"])))
        assert len(functions) == case["n_functions"]
        assert owners == [0] * len(functions)
        tables = unique_tables(functions)
        assert len(tables) == len(case["tables"])
        for i, ((l, kind, zeta, n, r, values), meta) in enumerate(
                zip(tables, case["tables"])):
            assert [l, kind, zeta, n, len(r)] == meta
            close(strided(r), FROZEN[f"{case['tag']}_{i}_r"], 1e-13)
            close(strided(values), FROZEN[f"{case['tag']}_{i}_v"])

    @pytest.mark.parametrize("key", META["levels"])
    def test_the_bound_level_and_orbital_of_an_empty_channel(self, key):
        symbol, l = key.split("_")
        pp = get_paw(symbol)
        assert _lowest_level(pp, int(l)) == pytest.approx(
            FROZEN[f"level_{key}"][0], rel=1e-9)
        radial = bound_radial(pp, int(l))
        assert len(radial) == int(FROZEN[f"bound_{key}_n"][0])
        close(strided(radial), FROZEN[f"bound_{key}"], 1e-9)
        # Zero past the box it was solved in.
        assert radial[-1] == 0.0


@needs_library
class TestStates:
    @pytest.mark.parametrize("symbol", ["H", "Li", "C", "N", "O", "Na", "Si",
                                        "Ca", "Cu", "Au"])
    def test_the_states_match_the_reference(self, symbol):
        frozen = json.loads(
            (REFERENCE / "confinement.json").read_text())["states"][symbol]
        states = [[s.l, s.index, s.n, s.semicore, s.empty]
                  for s in basis_states(get_paw(symbol))]
        assert states == frozen

    def test_a_semicore_channel_has_two_states(self):
        states = [(s.l, s.index, s.n, s.semicore)
                  for s in occupied_states(get_paw("Na"))]
        assert states == [(0, 0, 2, True), (0, 1, 3, False), (1, 0, 2, True)]

    def test_an_empty_channel_that_binds_joins_the_basis(self):
        states = basis_states(get_paw("Ca"))
        d = [s for s in states if s.l == 2]
        assert len(d) == 1 and d[0].empty and d[0].level < BOUND_LEVEL
        assert not any(s.empty for s in occupied_states(get_paw("Ca")))

    def test_scattering_only_channels_stay_out(self):
        assert {s.l for s in basis_states(get_paw("Li"))} == {0}

    def test_the_valence_charge(self):
        potentials = {s: get_paw(s) for s in ("H", "O")}
        assert valence_electrons(["O", "H", "H"], potentials) == \
            pytest.approx(potentials["O"].valence_charge
                          + 2 * potentials["H"].valence_charge)


@needs_library
class TestFunctions:
    def test_a_minimal_basis_keeps_the_stored_partial_wave(self):
        pp = get_paw("O")
        functions, owners = pseudo_basis(
            ["O", "O"], np.array([[0.0, 0, 0], [0, 0, 1.2]]), {"O": pp},
            **pseudo_basis_arguments("paw-lcao", {"size": "SZ",
                                                  "energy_shift": None}))
        assert owners == [0] * 4 + [1] * 4
        assert all(isinstance(f, PseudoAtomicOrbital) for f in functions)
        np.testing.assert_array_equal(functions[0]._values,
                                      pp.channels[0].pseudo_waves[0])

    def test_the_orbital_holds_its_arrays_and_evaluates_them(self):
        pp = get_paw("H")
        orbital = PseudoAtomicOrbital(pp, 0, 0, center=[0.0, 0.0, 0.0])
        r = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            orbital.radial(r), np.interp(r, pp.r, pp.channels[0].pseudo_waves[0]),
            rtol=1e-4)
        assert orbital.radial(np.array([pp.r[-1] + 1.0]))[0] == 0.0
        assert orbital.state == (1, 0, 0)

    def test_a_projector_carries_its_labels(self):
        pp = get_paw("O")
        projector = KBProjector(pp, 1, 0, atom_index=3, index=1,
                                radial=pp.projectors[1][1], kb_energy=0.25)
        assert projector.block_key == (3, 1, 0)
        assert projector.channel == (1, 0)
        assert projector.kb_energy == 0.25

    def test_a_size_mapping_needs_every_element(self):
        pp = get_paw("H")
        with pytest.raises(ValueError, match="no pseudo-basis size"):
            pseudo_basis(["H"], np.zeros((1, 3)), {"H": pp},
                         size={"O": "SZ"})
