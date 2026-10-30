# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_diffuse.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``diffuse`` option of a PAW-LCAO / UPAW-LCAO basis.

``basis={"name": "PAW-LCAO", "diffuse": True}`` appends quasi-Gaussian shells,
sized from the free atom and cut at a tail norm, to each atom's ordinary
basis.  These tests pin the option's spellings, the radial function, the
one-center Gram-Schmidt, the shell counts per element, the default recipe and
the molecule path (ghosts, the box warning, the per-element form).
"""

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru, _native
from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.diffuse import (
    DEFAULT_CUTOFF_NORM, DEFAULT_DIFFUSE, DEFAULT_RMS_RATIO,
    DEFAULT_SHELL_RATIO, LINEAR_DEPENDENCE_TOL, PRESENT_L_MAX, DiffuseRecipe,
    box_diagnostic, box_table, diffuse_label, diffuse_of, diffuse_parameters,
    diffuse_radial, free_valence_rms, refuse_crystal, validate_diffuse)
from mandacaru.pseudopotentials.families import pseudo_basis_arguments
from mandacaru.pseudopotentials.orbitals import pseudo_basis

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

REFERENCE = Path(__file__).resolve().parents[1] / "migration" / "reference" / "paw"
META = json.loads((REFERENCE / "diffuse.json").read_text())
FROZEN = np.load(REFERENCE / "diffuse.npz")


def close(value, expected, tol=1e-9):
    """Largest difference relative to the largest entry of ``expected``."""
    value, expected = np.asarray(value, float), np.asarray(expected, float)
    assert value.shape == expected.shape
    scale = max(float(np.abs(expected[np.isfinite(expected)]).max(initial=0)),
                1e-300)
    assert float(np.nanmax(np.abs(value - expected), initial=0)) <= tol * scale


def strided(array):
    return array[::-(-len(array) // META["max_points"])]


def _library_present() -> bool:
    try:
        get_paw("H")
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_library = pytest.mark.skipif(not _library_present(),
                                   reason="MANDACARU_PAW_PATH is not configured")


class TestOption:
    @pytest.mark.parametrize("spec", [None, False, 0, {"H": False}])
    def test_off_spellings(self, spec):
        assert validate_diffuse(spec) is None
        assert diffuse_label(spec) == "no diffuse"

    def test_true_is_the_default_recipe(self):
        assert validate_diffuse(True) is DEFAULT_DIFFUSE
        assert validate_diffuse({}) == DEFAULT_DIFFUSE

    def test_an_int_is_the_count(self):
        assert validate_diffuse(2) == DiffuseRecipe(count=2)

    @pytest.mark.parametrize("shells, expected", [
        ("present", "present"), ("sp", (0, 1)), (["s", "p"], (0, 1)),
        ([1, 0], (0, 1)), ("spdf", (0, 1, 2, 3)), (["d"], (2,))])
    def test_shell_spellings(self, shells, expected):
        assert validate_diffuse({"shells": shells}).shells == expected

    def test_a_recipe_dict(self):
        recipe = validate_diffuse({"shells": "sp", "count": 2,
                                   "rms_ratio": 2.4, "ratio": 2.0,
                                   "cutoff_norm": 1e-6})
        assert recipe == DiffuseRecipe(shells=(0, 1), count=2, rms_ratio=2.4,
                                       ratio=2.0, cutoff_norm=1e-6)
        assert validate_diffuse(recipe) is recipe          # idempotent

    def test_an_element_mapping(self):
        table = validate_diffuse({"h": True, "O": {"shells": "p"},
                                  "*": False})
        assert table == {"H": DEFAULT_DIFFUSE,
                         "O": DiffuseRecipe(shells=(1,)), "*": None}
        assert diffuse_of(table, "H") is DEFAULT_DIFFUSE
        assert diffuse_of(table, "C") is None
        assert diffuse_of({"*": 1}, "C") == DEFAULT_DIFFUSE

    @pytest.mark.parametrize("spec, match", [
        ({"radius_scale": 2.0}, "unknown diffuse key"),
        ({"shells": "sx"}, "shells"),
        ({"shells": [5]}, "shells"),
        ({"rms_ratio": 1.0}, "rms_ratio"),
        ({"ratio": 0.5}, "ratio"),
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"rms_radius": -1.0}, "rms_radius"),
        ({"cutoff_norm": 0.1}, "cutoff_norm"),
        ({"H": True, "count": 2}, "not a mix"),
        ({"H": {"O": True}}, "cannot nest"),
        (-1, "count"),
        ("yes", "diffuse must be")])
    def test_refusals(self, spec, match):
        with pytest.raises(ValueError, match=match):
            validate_diffuse(spec)

    def test_a_crystal_is_refused(self):
        refuse_crystal(None)
        with pytest.raises(ValueError, match="molecules"):
            refuse_crystal(True)


class TestFrozenPythonOutputs:
    """The Rust radial functions against the Python ones they replace
    (``test/migration/reference/paw/diffuse.npz``, inputs in the JSON).  The
    closed forms agree to 1e-9 (measured 3e-11); the tables behind the
    Fourier filter and the projection too."""

    def test_the_width_and_the_support(self):
        for l, rms, cutoff, alpha, r_cut, cut, unit in FROZEN["parameters"]:
            got_alpha, got_cut = diffuse_parameters(int(l), rms, cutoff)
            assert got_alpha == pytest.approx(alpha, rel=1e-9)
            assert got_cut == pytest.approx(r_cut, rel=1e-9)
            # c_l = r_cut sqrt(alpha) and the unit-alpha rms are the factors.
            assert got_cut * np.sqrt(got_alpha) == pytest.approx(cut, rel=1e-9)
            assert rms * np.sqrt(got_alpha) == pytest.approx(unit, rel=1e-9)

    @pytest.mark.parametrize("l", [0, 1, 2])
    @pytest.mark.parametrize("cutoff_norm", [1e-4, 1e-6])
    def test_the_radial_function(self, l, cutoff_norm):
        values, alpha, r_cut = diffuse_radial(
            FROZEN["radial_r"], l, 4.0, cutoff_norm)
        close(values, FROZEN[f"radial_{l}_{cutoff_norm}"], 1e-10)
        close([alpha, r_cut], FROZEN[f"radial_{l}_{cutoff_norm}_ar"], 1e-10)

    @needs_library
    def test_the_free_valence_radii(self):
        for symbol, expected in META["free_valence_rms"].items():
            got = free_valence_rms(get_paw(symbol))
            assert {str(k) for k in got} == set(expected)
            for l, value in got.items():
                assert value == pytest.approx(expected[str(l)], rel=1e-9)

    @needs_library
    @pytest.mark.parametrize("case", [c for c in META["cases"]
                                      if "error" not in c],
                             ids=lambda c: c["tag"])
    def test_the_shells_and_their_tables(self, case):
        symbol = case["symbol"]
        record = {}
        functions, _owners = pseudo_basis(
            [symbol], np.zeros((1, 3)), {symbol: get_paw(symbol)},
            filter_wavevector=case["k_c"],
            **pseudo_basis_arguments("paw-lcao", {
                "size": "DZP", "energy_shift": 0.1, "diffuse": case["spec"]},
                diffuse=record))
        assert len(functions) == case["n_functions"]
        shells = record.get(symbol, [])
        assert len(shells) == len(case["shells"])
        for got, (l, index, n, ref_l, ref_rms, rms, alpha, r_cut, residual,
                  accepted, r_max) in zip(shells, case["shells"]):
            assert (got.l, got.index, got.n, got.reference_l,
                    got.accepted) == (l, index, n, ref_l, accepted)
            for value, expected in ((got.reference_rms, ref_rms),
                                    (got.rms, rms), (got.alpha, alpha),
                                    (got.r_cut, r_cut), (got.r_max, r_max)):
                assert value == pytest.approx(expected, rel=1e-9)
            assert got.residual == pytest.approx(residual, rel=1e-8, abs=1e-12)
        tables = {}
        for f in functions:
            if f.diffuse:
                tables.setdefault(id(f.table.values), f.table)
        assert len(tables) == len(case["tables"])
        for i, (table, (n, l, length, r_end)) in enumerate(
                zip(tables.values(), case["tables"])):
            assert (table.n, table.l, len(table.r)) == (n, l, length)
            assert float(table.r[-1]) == pytest.approx(r_end, rel=1e-12)
            close(strided(table.r), FROZEN[f"{case['tag']}_{i}_r"], 1e-13)
            close(strided(table.values), FROZEN[f"{case['tag']}_{i}_v"], 1e-8)


class TestBoxDiagnostic:
    """The cell that cuts a diffuse tail, on the synthetic grid of the
    reference (a sheared 4 x 5 x 6 grid, five functions, three diffuse)."""

    @staticmethod
    def _inputs():
        info = META["box"]
        grid = SimpleNamespace(
            X=np.zeros(info["shape"]), Y=np.zeros(info["shape"]),
            Z=np.zeros(info["shape"]), step=FROZEN["box_step"],
            shape=tuple(info["shape"]))
        grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0] = FROZEN["box_origin"]
        psi = FROZEN["box_psi"]
        norms = [float(np.vdot(row, row).real) * info["dV"] for row in psi]
        functions = [
            SimpleNamespace(diffuse=flag, center=center,
                            table=SimpleNamespace(r=FROZEN["box_tables_r"],
                                                  values=values))
            for flag, center, values in zip(
                info["diffuse"], FROZEN["box_centers"],
                FROZEN["box_tables_v"])]
        shells = {s: [SimpleNamespace(r_cut=r_cut, accepted=True)]
                  for s, r_cut in info["r_cut"].items()}
        return norms, functions, grid, shells

    def test_the_ratios_and_the_faces_match_the_reference(self):
        info = META["box"]
        norms, functions, grid, shells = self._inputs()
        with pytest.warns(RuntimeWarning, match="the cell cuts") as caught:
            ratios, rows = box_diagnostic(norms, functions, info["owners"],
                                          info["symbols"], grid, shells)
        assert len(caught) == info["warnings"]
        close(ratios, FROZEN["box_ratios"], 1e-12)
        expected = FROZEN["box_rows"]
        assert [row[0] for row in rows] == [int(a) for a in expected[:, 0]]
        close([row[2] for row in rows], expected[:, 1], 1e-12)
        close([row[3] for row in rows], expected[:, 2], 1e-12)
        close([row[4] for row in rows], expected[:, 3], 1e-12)

    def test_only_diffuse_functions_have_a_ratio(self):
        info = META["box"]
        norms, functions, grid, shells = self._inputs()
        with pytest.warns(RuntimeWarning):
            ratios, _rows = box_diagnostic(norms, functions, info["owners"],
                                           info["symbols"], grid, shells)
        flags = info["diffuse"]
        assert all(np.isnan(r) != f for r, f in zip(ratios, flags))

    def test_a_cell_that_holds_the_tail_is_silent(self):
        info = META["box"]
        _norms, functions, grid, shells = self._inputs()
        # The grid norm of each function is its radial norm: ratio one.
        norms = [_native.diffuse_radial_norm(f.table.r, f.table.values)
                 for f in functions]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ratios, rows = box_diagnostic(norms, functions, info["owners"],
                                          info["symbols"], grid, shells)
        assert all(abs(r - 1.0) < 1e-12 for r in ratios if r == r)
        table = box_table(rows)
        assert table[0][0] == "atom" and len(table) == 1 + len(rows)


class TestTheDefaultRecipe:
    def test_the_constants(self):
        # The recipe diffuse=True selects, chosen from the measurements in
        # HISTORY.md ("Diffuse functions, Phases 1-4").  A change here is a
        # change of every diffuse calculation's basis.
        assert DEFAULT_DIFFUSE == DiffuseRecipe(
            shells="present", count=1, rms_ratio=DEFAULT_RMS_RATIO,
            ratio=DEFAULT_SHELL_RATIO, rms_radius=None,
            cutoff_norm=DEFAULT_CUTOFF_NORM)
        assert DEFAULT_RMS_RATIO == 2.5
        assert DEFAULT_CUTOFF_NORM == 1e-4
        assert DEFAULT_SHELL_RATIO == pytest.approx(np.sqrt(3.5))
        assert PRESENT_L_MAX == 2
        assert LINEAR_DEPENDENCE_TOL == 1e-4


class TestRadialFunction:
    @pytest.mark.parametrize("l", [0, 1, 2])
    @pytest.mark.parametrize("cutoff_norm", [1e-4, 1e-6])
    def test_shape(self, l, cutoff_norm):
        r = np.linspace(0.0, 40.0, 40001)
        values, alpha, r_cut = diffuse_radial(r, l, 4.0, cutoff_norm)
        density = values * values * r * r
        # Unit norm and the requested rms radius (the *cut* function's).
        assert np.trapezoid(density, r) == pytest.approx(1.0, abs=1e-10)
        rms = np.sqrt(np.trapezoid(density * r * r, r))
        assert rms == pytest.approx(4.0, rel=1e-3)
        # r^l at the origin.
        small = r[1:6]
        ratio = values[1:6] / small ** l
        assert np.ptp(ratio) < 1e-3 * abs(ratio[0])
        # Value and slope vanish at the cutoff, and nothing lies beyond it.
        inside = r < r_cut
        edge = np.flatnonzero(inside)[-1]
        slope = np.gradient(values, r)
        assert abs(values[edge]) < 1e-6 * np.abs(values).max()
        assert abs(slope[edge]) < 1e-4 * np.abs(slope).max()
        assert np.all(values[~inside] == 0.0)

    @pytest.mark.parametrize("l", [0, 1, 2])
    def test_the_cutoff_is_the_gaussian_tail(self, l):
        from scipy.special import gammaincc

        alpha, r_cut = diffuse_parameters(l, 3.0, 1e-5)
        assert gammaincc(l + 1.5, 2 * alpha * r_cut ** 2) == \
            pytest.approx(1e-5, rel=1e-8)
        # Width and support scale together: twice the radius, twice r_cut.
        alpha2, r_cut2 = diffuse_parameters(l, 6.0, 1e-5)
        assert r_cut2 == pytest.approx(2 * r_cut)
        assert alpha2 == pytest.approx(alpha / 4)

    def test_a_tighter_tail_is_a_shorter_support(self):
        # The cost lever: 1e-4 instead of 1e-6 shortens r_cut by about 15 %.
        _a, long = diffuse_parameters(0, 4.0, 1e-6)
        _a, short = diffuse_parameters(0, 4.0, 1e-4)
        assert 0.80 < short / long < 0.90


def _atom_basis(symbol, size="DZP", diffuse=True, k_c=None,
                energy_shift=0.1):
    pp = {symbol: get_paw(symbol)}
    record = {}
    functions, owners = pseudo_basis(
        [symbol], np.zeros((1, 3)), pp, filter_wavevector=k_c,
        **pseudo_basis_arguments("paw-lcao", {
            "size": size, "energy_shift": energy_shift, "diffuse": diffuse},
            diffuse=record))
    return pp[symbol], functions, record.get(symbol, [])


def _radial_overlap(a, b, r):
    return float(np.trapezoid(a.radial(r) * b.radial(r) * r * r, r))


@needs_library
class TestShells:
    def test_free_atom_reference(self):
        # The scale the widths are measured in: the dataset's own free
        # valence orbitals (Bohr).  Hydrogen's 1.974 became 1.967 with the
        # 1.0 Bohr sphere (2026-10-25).
        assert free_valence_rms(get_paw("H"))[0] == pytest.approx(1.967,
                                                                  abs=2e-3)
        reference = free_valence_rms(get_paw("O"))
        assert reference[0] == pytest.approx(1.316, abs=2e-3)
        assert reference[1] == pytest.approx(1.539, abs=2e-3)
        assert set(free_valence_rms(get_paw("Li"))) == {0}

    @pytest.mark.parametrize("symbol, size, shells", [
        ("H", "SZ", [0]), ("H", "DZP", [0, 1]), ("O", "SZ", [0, 1]),
        ("O", "DZP", [0, 1, 2]), ("Li", "DZP", [0, 1]),
        ("O", "TZ2P", [0, 1, 2])])
    def test_present_shells(self, symbol, size, shells):
        _pp, plain, _r = _atom_basis(symbol, size, diffuse=False)
        _pp, functions, record = _atom_basis(symbol, size)
        assert [s.l for s in record] == shells
        assert all(s.accepted for s in record)
        added = functions[len(plain):]
        assert len(added) == sum(2 * l + 1 for l in shells)
        assert all(f.kind == "diffuse" and f.diffuse for f in added)
        assert not any(getattr(f, "diffuse", False) for f in plain)

    def test_an_l_without_a_valence_state_takes_the_outermost_scale(self):
        _pp, _f, record = _atom_basis("O", "DZP")
        by_l = {s.l: s for s in record}
        assert by_l[2].reference_l == 1                  # O d from O 2p
        assert by_l[2].rms == pytest.approx(DEFAULT_RMS_RATIO * 1.539,
                                            abs=5e-3)
        _pp, _f, record = _atom_basis("Li", "DZP")
        assert {s.l: s.reference_l for s in record} == {0: 0, 1: 0}

    def test_the_ordinary_basis_is_untouched(self):
        r = np.linspace(0.0, 20.0, 801)
        _pp, plain, _r = _atom_basis("O", "DZP", diffuse=False, k_c=6.0)
        _pp, functions, _r = _atom_basis("O", "DZP", k_c=6.0)
        for a, b in zip(plain, functions):
            np.testing.assert_array_equal(a.radial(r), b.radial(r))

    def test_off_is_byte_identical(self):
        r = np.linspace(0.0, 20.0, 801)
        for size in ("SZ", "DZP"):
            pp = {"O": get_paw("O")}
            plain, _o = pseudo_basis(
                ["O"], np.zeros((1, 3)), pp, filter_wavevector=6.0,
                **pseudo_basis_arguments("paw-lcao", {"size": size,
                                                      "energy_shift": 0.1}))
            for spec in (False, None, 0):
                _pp, functions, record = _atom_basis("O", size, diffuse=spec,
                                                     k_c=6.0)
                assert record == [] and len(functions) == len(plain)
                for a, b in zip(plain, functions):
                    assert type(a) is type(b)
                    np.testing.assert_array_equal(a.radial(r), b.radial(r))

    @pytest.mark.parametrize("k_c", [None, 6.0])
    def test_one_center_gram_schmidt(self, k_c):
        r = np.linspace(0.0, 30.0, 6001)
        _pp, functions, _r = _atom_basis("O", "DZP", diffuse=2, k_c=k_c)
        for f in functions:
            if not f.diffuse:
                continue
            assert _radial_overlap(f, f, r) == pytest.approx(1.0, abs=1e-6)
            for g in functions:
                if g is not f and g.l == f.l and g.m == f.m and (
                        not g.diffuse or g.n < f.n):
                    assert abs(_radial_overlap(f, g, r)) < 1e-6

    def test_a_redundant_shell_is_rejected(self):
        # A second shell practically equal to the first leaves nothing after
        # the projection: recorded as rejected, never built.
        _pp, plain, _r = _atom_basis("H", "SZ", diffuse=False)
        _pp, functions, record = _atom_basis(
            "H", "SZ", diffuse={"count": 2, "ratio": 1.0001})
        assert [s.accepted for s in record] == [True, False]
        assert record[1].residual < LINEAR_DEPENDENCE_TOL
        assert len(functions) == len(plain) + 1

    def test_a_shell_beyond_the_table_gets_its_own_grid(self):
        # Lithium's s shell reaches past the 30-Bohr dataset table.
        pp, functions, record = _atom_basis("Li", "DZP", k_c=None,
                                            diffuse={"rms_ratio": 3.0})
        s_shell = next(s for s in record if s.l == 0)
        assert s_shell.r_cut > float(pp.r[-1])
        diffuse = next(f for f in functions if f.diffuse and f.l == 0)
        assert float(diffuse.table.r[-1]) > s_shell.r_cut
        assert abs(float(diffuse.radial(np.array([31.0]))[0])) > 1e-6

    def test_an_explicit_radius_is_angstrom(self):
        from mandacaru.units import ANGSTROM_TO_BOHR

        _pp, _f, record = _atom_basis("H", "SZ", diffuse={"rms_radius": 2.5})
        assert record[0].rms == pytest.approx(2.5 * ANGSTROM_TO_BOHR)


def h2(cell, distance=0.74):
    center = np.array([cell / 2 + 0.013, cell / 2 - 0.021, cell / 2 + 0.007])
    half = np.array([0.0, 0.0, distance / 2])
    return Atoms("H2", positions=[center - half, center + half],
                 cell=[cell] * 3)


@needs_library
class TestMolecule:
    def _calculator(self, basis, **kwargs):
        return Mandacaru(method="rhf", basis=basis, grid_spacing=0.35,
                         **kwargs)

    def test_it_lowers_the_energy_in_a_box_that_holds_it(self):
        atoms = h2(15.4)
        atoms.calc = self._calculator({"name": "PAW-LCAO"})
        plain = atoms.get_potential_energy()
        atoms = h2(15.4)
        atoms.calc = self._calculator({"name": "PAW-LCAO", "diffuse": True})
        diffuse = atoms.get_potential_energy()
        assert diffuse < plain

    def test_a_small_box_is_warned_about(self):
        atoms = h2(7.0)
        atoms.calc = self._calculator({"name": "PAW-LCAO", "diffuse": True})
        with pytest.warns(RuntimeWarning, match="the cell cuts"):
            atoms.get_potential_energy()

    def test_ghosts_carry_their_diffuse_shells(self):
        atoms = h2(15.4)
        calc = Mandacaru(method="uhf", grid_spacing=0.35, ghosts=[1],
                         basis={"name": "PAW-LCAO", "diffuse": True})
        atoms.calc = calc
        atoms.get_potential_energy()
        context = calc.solver._gradient_context
        owners = np.asarray(context["atom_of_orbital"])
        assert np.sum(owners == 1) == 2              # 1s and its diffuse s
        assert context["ghosts"] == (1,)

    def test_upaw_has_it(self):
        estimate = Mandacaru(method="rhf", basis={
            "name": "UPAW-LCAO", "diffuse": True}).dry_run(h2(15.4))
        assert estimate.n_basis_functions == 4

    def test_the_per_element_form(self):
        water = Atoms("OH2", positions=[[8, 8, 8], [8, 8.77, 8.59],
                                        [8, 7.23, 8.59]], cell=[16.0] * 3)
        calc = Mandacaru(method="rhf",
                         basis={"O": {"name": "PAW-LCAO", "diffuse": True},
                                "H": "PAW-LCAO"})
        estimate = calc.dry_run(water)
        assert estimate.n_basis_functions == 4 + 4 + 1 + 1   # O sp + s p
        assert "diffuse ({O: shells present" in estimate.basis
        with pytest.raises(ValueError, match="recipe"):
            Mandacaru(method="rhf",
                      basis={"O": {"name": "PAW-LCAO",
                                   "diffuse": {"O": True}},
                             "H": "PAW-LCAO"}).dry_run(water)

    def test_a_bad_value_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="diffuse"):
            Mandacaru(method="rhf", basis={"name": "PAW-LCAO",
                                           "diffuse": {"width": 2}})
