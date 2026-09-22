# -*- coding: utf-8 -*-
# file: test/test_basis_filtering.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Fourier filtering of the pseudopotential basis (:mod:`mandacaru.basis.filtering`).

Four things are pinned here, in this order:

1. **the transform** -- the spherical Bessel pair round-trips, Parseval holds,
   an already band-limited function is left alone, a filtered one really has
   no weight above :math:`k_c`, the norm is preserved and the :math:`r^l`
   behavior at the origin survives (the property whose loss broke the forces
   once already in the polarization shell);
2. **the option** -- ``True`` / ``"auto"`` / a number in eV / ``False``
   accepted, everything else refused with a clear message, in the single-family
   and the per-element form and in the dry run;
3. **the default** -- on for PAW and UPAW, off for NCPP and ONCVPSP, declared
   in the registry, and ``filter=False`` reproducing the unfiltered basis
   **byte for byte** (this is what keeps every pinned NCPP/ONCVPSP number
   still valid and makes the PAW flip auditable);
4. **what it buys** -- the rigid-shift energy ripple shrinks, and the analytic
   force is still the derivative of the calculator's own energy.

Everything that runs a solver goes through ``Mandacaru(method=...)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import (
    build_basis_hamiltonian, grid_from_cell, resolve_basis,
    resolve_pseudo_basis)
from mandacaru.algorithms.dry_run import count_basis_functions, estimate_qubits
from mandacaru.basis.filtering import (DEFAULT_FILTER_METHOD, FILTER_METHODS,
                                       FILTER_NYQUIST_FRACTION,
                                       ROLLOFF_FRACTION, bessel_matrix,
                                       cutoff_energy_ev, cutoff_window,
                                       filter_cutoff, filter_label,
                                       filter_radial, filter_table,
                                       inverse_spherical_bessel_transform,
                                       radial_norm, residual_weight,
                                       spherical_bessel_transform,
                                       support_radius, validate_filter)
from mandacaru.basis.multizeta import RadialTable, zeta_tables
from mandacaru.pseudopotentials.families import PSEUDO_FAMILIES, FamilySpec
from mandacaru.pseudopotentials.orbitals import pseudo_basis
from mandacaru.pseudopotentials.paw import get_paw
from mandacaru.units import ANGSTROM_TO_BOHR, EV_TO_HARTREE

#: A uniform radial grid like the pseudopotential library's (0.02 Bohr out to
#: 30 Bohr), so the transform is exercised on the grid it really runs on.
R_GRID = np.linspace(0.005, 30.0, 1500)


def gaussian(r, l, alpha):
    r"""``r^l exp(-alpha r^2)`` -- its Bessel transform is analytic and it is
    as close to band-limited as a localized function gets, which is what makes
    it the right probe for "the filter leaves a smooth function alone"."""
    return r ** l * np.exp(-alpha * r * r)


def h2(d=0.74, cell=8.0):
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, d)], cell=(cell,) * 3)
    atoms.center()
    return atoms


def water(cell=10.0):
    return Atoms("OH2", cell=(cell,) * 3,
                 positions=[[5.0, 5.0, 5.0], [5.0, 6.0, 5.0], [5.0, 5.0, 6.0]])


def build(atoms, basis, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_basis_hamiltonian(atoms, basis, None, h, 0, None)


def paw_tables(symbol="O"):
    """Every radial table a DZP basis of ``symbol`` is built from."""
    pp = get_paw(symbol)
    r = np.asarray(pp.r)
    out = []
    for l in sorted(pp.channels):
        channel = pp.channels[l]
        for table in zeta_tables(r, np.asarray(channel.pseudo_radial),
                                 int(channel.n), l, 2, 0.15):
            out.append(table)
    l_max = max(pp.channels)
    shape = r * np.asarray(pp.channels[l_max].pseudo_radial)
    shape = shape / np.sqrt(np.trapezoid(shape * shape * r * r, r))
    out.extend(zeta_tables(r, shape, l_max + 2, l_max + 1, 1, 0.15))
    return out


# --------------------------------------------------------------------------- #
# (1) The transform.
# --------------------------------------------------------------------------- #

class TestTransform:
    @pytest.mark.parametrize("l", [0, 1, 2])
    def test_round_trip(self, l):
        """Forward then inverse returns the function.

        ``k_max = 10`` is not laziness: the quadrature runs on the *radial*
        table (0.02 Bohr here, as in the shipped library) and ``j_l(kr)``
        has only ``2 pi / (k dr)`` points per oscillation, so the trapezoid
        error grows as ``(k dr)^2``.  Measured on this grid, the round trip
        is 8.0e-5 at k_max = 10, 2.7e-4 at 15, 6.4e-4 at 20 and 1.2e-3 at
        25, and refining the *k* grid changes none of them -- it is the r
        grid that limits.  The cutoffs the filter actually uses are 6.6-8.3
        Bohr^-1 (h = 0.25-0.20), i.e. inside the 1e-4 regime.
        """
        values = gaussian(R_GRID, l, 1.0)
        k = np.linspace(0.0, 10.0, 400)
        matrix = bessel_matrix(l, k, R_GRID)
        F = spherical_bessel_transform(R_GRID, values, l, k, matrix)
        back = inverse_spherical_bessel_transform(k, F, l, R_GRID, matrix)
        assert np.abs(back - values).max() < 2e-4 * np.abs(values).max()

    @pytest.mark.parametrize("l", [0, 1, 2])
    def test_parseval(self, l):
        """``int R^2 r^2 dr == (2/pi) int F^2 k^2 dk`` -- the identity that
        makes "the weight above k_c" a number rather than a figure of speech."""
        values = gaussian(R_GRID, l, 1.0)
        k = np.linspace(0.0, 25.0, 900)
        F = spherical_bessel_transform(R_GRID, values, l, k)
        spectral = (2.0 / np.pi) * float(np.trapezoid(F * F * k * k, k))
        assert spectral == pytest.approx(radial_norm(R_GRID, values), rel=1e-5)

    def test_a_band_limited_function_is_left_alone(self):
        """A smooth Gaussian already lives below the cutoff (3.3e-7 of its
        norm above k_c = 8), so filtering it moves it by 2e-5 of its peak."""
        values = gaussian(R_GRID, 0, 0.3)
        assert residual_weight(R_GRID, values, 0, 8.0) < 1e-6
        out, info = filter_radial(R_GRID, values, 0, 8.0)
        assert info["max_change"] < 1e-4
        assert np.abs(out - values).max() < 1e-4 * np.abs(values).max()

    @pytest.mark.parametrize("l", [0, 1])
    def test_the_filtered_function_has_no_weight_above_the_cutoff(self, l):
        """A compact split zeta starts with 1e-3 of its norm above k_c and
        keeps ~1e-4 after the switch-off (which re-introduces a little);
        both are far below what it started with."""
        table = [t for t in paw_tables("O") if t.l == l and t.zeta == 2][0]
        k_c = 7.5
        before = residual_weight(table.r, table.values, l, k_c)
        out, info = filter_radial(table.r, table.values, l, k_c)
        assert before > 1e-4                       # there is something to remove
        assert info["residual_after"] < 0.1 * before
        assert residual_weight(table.r, out, l, k_c) == pytest.approx(
            info["residual_after"], abs=1e-12)

    def test_plain_filtering_is_exactly_band_limited(self):
        """Without a real-space window the result is band-limited to machine
        noise -- the switch-off is the only thing that puts weight back."""
        table = [t for t in paw_tables("O") if t.l == 0 and t.zeta == 1][0]
        out, info = filter_radial(table.r, table.values, 0, 7.5, method="plain")
        assert abs(info["residual_after"]) < 1e-6
        assert out.shape == np.asarray(table.values).shape

    def test_normalization_is_preserved(self):
        """Rescaled to the table's *own* norm, not to 1: the tables do not all
        arrive normalized (PAW hydrogen 1s carries 0.979) and forcing them to
        1 would make the filter change a function even as k_c -> infinity."""
        for table in paw_tables("O"):
            out, _info = filter_radial(table.r, table.values, table.l, 7.5)
            assert radial_norm(table.r, out) == pytest.approx(
                radial_norm(table.r, table.values), rel=1e-7)

    @pytest.mark.parametrize("l", [1, 2])
    def test_the_origin_keeps_its_r_to_the_l(self, l):
        """``R(r) ~ r^l`` as ``r -> 0``.

        Free, and not a detail: the filter is done *inside* the angular
        channel and ``j_l(kr) ~ (kr)^l``, so the reconstruction cannot help
        behaving this way.  A radial function that does not vanish as r^l at
        its own nucleus blew up the displaced-sampling derivative once
        already (the polarization-shell bug), so it is asserted:
        ``R/r^l`` is flat over the first points, and the first two points are
        in the ratio ``(r_1/r_0)^l``.
        """
        values = gaussian(R_GRID, l, 1.0)
        out, _info = filter_radial(R_GRID, values, l, 7.5)
        head = out[:10] / R_GRID[:10] ** l
        assert np.all(np.isfinite(head))
        assert np.abs(head - head[0]).max() < 0.05 * abs(head[0])
        assert out[1] / out[0] == pytest.approx((R_GRID[1] / R_GRID[0]) ** l,
                                                rel=0.01)

    def test_the_cutoff_window_is_smooth_and_compact(self):
        k = np.linspace(0.0, 10.0, 501)
        w = cutoff_window(k, 8.0, ROLLOFF_FRACTION)
        assert np.all(w[k <= 8.0 * (1.0 - ROLLOFF_FRACTION)] == 1.0)
        assert np.all(w[k >= 8.0] == 0.0)
        assert np.all(np.diff(w) <= 1e-12)          # monotonically decreasing

    def test_the_support_radius_finds_the_range(self):
        """``support_radius`` is where the remaining *tail norm* falls below
        1e-6, so a little amplitude survives beyond it (0.15 % of the peak on
        this split zeta) -- which is the point: the switch-off is placed a
        further 50 % out, well clear of anything that matters."""
        table = [t for t in paw_tables("O") if t.l == 0 and t.zeta == 2][0]
        r, values = np.asarray(table.r), np.asarray(table.values)
        r_s = support_radius(r, values)
        assert 0.5 < r_s < 5.0
        beyond = np.abs(values[r > r_s])
        assert beyond.max() < 0.01 * np.abs(values).max()
        # a split zeta really is compact: exactly zero past its split radius
        last = r[np.nonzero(np.abs(values) > 0)[0][-1]]
        assert last < 1.1 * r_s
        assert np.all(values[r > last] == 0.0)

    def test_filter_table_copies_the_record(self):
        table = paw_tables("O")[0]
        out, _info = filter_table(table, 7.5)
        assert isinstance(out, RadialTable)
        assert out is not table
        assert np.asarray(table.values) is not np.asarray(out.values)
        assert (out.n, out.l, out.zeta) == (table.n, table.l, table.zeta)

    def test_the_mask_method_is_worse_here_and_that_is_why_it_is_not_default(self):
        """The evidence behind ``DEFAULT_FILTER_METHOD``.

        The classic mask trick divides by a window that decays *inside* the
        function's support.  These orbitals are bound states still decaying
        there, so the division amplifies the tail and the result ends up with
        **more** weight above k_c than it started with -- the method's own
        documented caveat, measured.  ``"switch"`` matches ``"plain"`` on a
        long-ranged function and keeps a compact one compact.

        Measured on the PAW hydrogen 1s at k_c = 7.5: 1.20e-6 of the norm
        above the cutoff to begin with, 4.89e-7 after ``"plain"`` or
        ``"switch"`` (they coincide -- the switch-off lands where the function
        is already 1e-7 of its peak) and **5.28e-5 after ``"mask"``**, 44x
        worse than doing nothing.
        """
        assert DEFAULT_FILTER_METHOD == "switch"
        assert set(FILTER_METHODS) == {"plain", "switch", "mask"}
        table = [t for t in paw_tables("H") if t.l == 0 and t.zeta == 1][0]
        k_c = 7.5
        before = residual_weight(table.r, table.values, table.l, k_c)
        results = {}
        for method in FILTER_METHODS:
            _out, info = filter_radial(table.r, table.values, table.l, k_c,
                                       method=method, warn=False)
            results[method] = info["residual_after"]
        assert results["switch"] < before
        assert results["switch"] == pytest.approx(results["plain"], rel=0.05)
        assert results["mask"] > 10 * before
        assert results["mask"] > 10 * results["switch"]
        with pytest.raises(ValueError, match="unknown filter method"):
            filter_radial(table.r, table.values, table.l, k_c, method="nope")


# --------------------------------------------------------------------------- #
# (2) The option.
# --------------------------------------------------------------------------- #

class TestOption:
    @pytest.mark.parametrize("spec, expected", [
        (None, False), (False, False), (True, True), ("auto", True),
        ("AUTO", True), (" auto ", True), (500.0, 500.0), (500, 500.0),
    ])
    def test_accepted_spellings(self, spec, expected):
        assert validate_filter(spec) == expected

    @pytest.mark.parametrize("spec", ["yes", "on", "nyquist", "", 0.0, -1.0,
                                      [1], {"k": 1}, np.nan, np.inf, object()])
    def test_refused_values(self, spec):
        with pytest.raises(ValueError, match="filter"):
            validate_filter(spec)

    def test_true_is_not_a_one_electronvolt_cutoff(self):
        """``isinstance(True, int)`` is True in Python, so the bool has to be
        caught before the number or ``filter=True`` would mean 1 eV."""
        assert validate_filter(True) is True
        assert validate_filter(1) == 1.0

    def test_the_cutoff_is_the_nyquist_wave_vector(self):
        spacing = 0.20 * ANGSTROM_TO_BOHR
        assert filter_cutoff(False, spacing) is None
        assert filter_cutoff(True, spacing) == pytest.approx(
            FILTER_NYQUIST_FRACTION * np.pi / spacing)
        # A number is a kinetic-energy cutoff in eV: k_c = sqrt(2E).
        assert filter_cutoff(500.0, spacing) == pytest.approx(
            np.sqrt(2.0 * 500.0 * EV_TO_HARTREE))
        assert cutoff_energy_ev(filter_cutoff(500.0, spacing)) == \
            pytest.approx(500.0)
        with pytest.raises(ValueError, match="needs the grid spacing"):
            filter_cutoff(True, None)
        with pytest.raises(ValueError, match="must be positive"):
            filter_cutoff(True, -0.1)

    def test_the_default_removes_exactly_the_unrepresentable_band(self):
        """``FILTER_NYQUIST_FRACTION == 1`` is the whole principle: what the
        grid cannot carry goes, what it can stays.  The measurement that
        chose it is tabulated on the constant."""
        assert FILTER_NYQUIST_FRACTION == 1.0

    @pytest.mark.parametrize("family", ["PAW", "UPAW", "NCPP", "ONCVPSP"])
    def test_every_family_accepts_the_option(self, family):
        spec = PSEUDO_FAMILIES[family.lower()]
        assert "filter" in spec.options
        _name, options = resolve_basis({"name": family, "filter": True})
        assert options["filter"] is True

    def test_an_unknown_option_is_still_refused(self):
        with pytest.raises(ValueError, match="unknown option"):
            Mandacaru(method="adapt-vqe",
                      basis={"name": "PAW", "filtre": True})

    @pytest.mark.parametrize("bad", ["yes", -1.0, 0.0, [1]])
    def test_a_bad_value_is_refused_at_construction(self, bad):
        with pytest.raises(ValueError, match="filter"):
            Mandacaru(method="adapt-vqe", basis={"name": "PAW",
                                                 "filter": bad})

    @pytest.mark.parametrize("bad", ["yes", -1.0])
    def test_a_bad_value_in_the_per_element_form_is_refused_too(self, bad):
        """Which element gets which family needs a geometry; an option
        *value* does not, so it is checked in the constructor either way."""
        with pytest.raises(ValueError, match="filter"):
            Mandacaru(method="adapt-vqe",
                      basis={"O": {"name": "PAW", "filter": bad},
                             "H": {"name": "PAW", "filter": bad}})

    def test_the_per_element_form_merges_one_shared_filter(self):
        family, options = resolve_pseudo_basis(
            "per-element", {"O": {"name": "PAW", "size": "DZ", "filter": 700.0},
                            "H": {"name": "PAW", "filter": 700.0}},
            ["O", "H", "H"])
        assert family is PSEUDO_FAMILIES["paw"]
        assert options == {"size": {"O": "DZ", "H": "SZ"}, "filter": 700.0}

    @pytest.mark.parametrize("mapping", [
        {"O": {"name": "PAW", "filter": True},
         "H": {"name": "PAW", "filter": False}},
        {"O": {"name": "PAW", "filter": True}, "H": "PAW"},
    ])
    def test_a_per_element_filter_must_agree(self, mapping):
        """The cutoff is a property of the grid, which every atom shares."""
        with pytest.raises(ValueError, match="same for every element"):
            resolve_pseudo_basis("per-element", mapping, ["O", "H"])

    def test_a_family_cannot_default_an_option_it_does_not_accept(self):
        with pytest.raises(ValueError, match="does not accept"):
            FamilySpec(name="bogus", description="", generate=None, get=None,
                       build=None, options=("size",),
                       default_options={"filter": True})


# --------------------------------------------------------------------------- #
# (3) The default, and byte-identity with it off.
# --------------------------------------------------------------------------- #

class TestDefault:
    @pytest.mark.parametrize("family, expected", [
        ("paw", True), ("upaw", True), ("ncpp", None), ("oncvpsp", None),
    ])
    def test_declared_in_the_registry(self, family, expected):
        """The default lives in one place -- the family spec -- so a new
        family declares its own with no driver edit.  (Only the filter is
        looked at: PAW's spec also declares its confinement.)"""
        spec = PSEUDO_FAMILIES[family]
        assert spec.default_options.get("filter") is expected
        assert spec.resolved_options().get("filter") is expected
        assert spec.resolved_options({"filter": False})["filter"] is False

    @pytest.mark.parametrize("family", ["paw", "upaw"])
    def test_the_default_really_filters(self, family):
        context = build(h2(), family, 0.25)[4]
        assert context["filter_cutoff"] is not None
        assert context["filter_cutoff"] == pytest.approx(
            FILTER_NYQUIST_FRACTION * np.pi / (0.25 * ANGSTROM_TO_BOHR),
            rel=0.05)              # the realized spacing, not exactly h

    @pytest.mark.parametrize("family", ["ncpp", "oncvpsp"])
    def test_the_norm_conserving_families_stay_opt_in(self, family):
        assert build(h2(), family, 0.25)[4]["filter_cutoff"] is None
        assert build(h2(), {"name": family, "filter": True},
                     0.25)[4]["filter_cutoff"] is not None

    @pytest.mark.parametrize("family", ["paw", "upaw", "ncpp", "oncvpsp"])
    def test_filter_off_is_byte_identical_to_an_unfiltered_basis(self, family):
        """``filter=False`` must reproduce the historical basis **exactly**.

        Not against another ``filter=False`` run -- that would compare a thing
        to itself.  Against what the construction was *before* the option
        existed: the SZ first zeta is the dataset's own ``pseudo_radial``, and
        the DZP tables are exactly what ``zeta_tables`` returns from it.  This
        is what keeps every pinned NCPP / ONCVPSP number valid and makes the
        PAW flip a decision rather than a drift.
        """
        spec = PSEUDO_FAMILIES[family]
        symbols, positions = ["H", "H"], np.array([[0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.4]])
        pp = spec.get("H")
        potentials = {"H": pp}
        # Sampled at the table's OWN nodes: an interpolating cubic spline
        # reproduces its data there exactly, so this is byte-identity of the
        # table and not of a resampling.  The *last* node is dropped -- it is
        # the one point `radial` treats with `r >= r_max`, and the spline's
        # boundary evaluation there lands a few ULP away (8e-28 on a value of
        # 4e-12 for NCPP hydrogen); everywhere else the match is exact.
        r = np.asarray(pp.r)[:-1]

        # SZ: the basis function IS the dataset's tabulated partial wave.
        functions, _o = pseudo_basis(symbols, positions, potentials, size="SZ")
        reference = np.asarray(pp.channels[0].pseudo_radial)
        assert len(functions) == 2
        for fn in functions:
            assert np.array_equal(fn.radial(r), reference[:-1])

        # DZP: every table is exactly what the split-valence construction gives.
        functions, _o = pseudo_basis(symbols, positions, potentials, size="DZP")
        expected = zeta_tables(np.asarray(pp.r), reference,
                               int(pp.channels[0].n), 0, 2, 0.15)
        built = [fn.table for fn in functions
                 if fn.l == 0 and fn.center[2] == 0.0]
        assert len(built) == len(expected) == 2
        for got, want in zip(built, expected):
            assert np.array_equal(np.asarray(got.values),
                                  np.asarray(want.values))

    @pytest.mark.parametrize("family", ["paw", "upaw"])
    def test_filter_false_restores_the_unfiltered_hamiltonian(self, family):
        """End to end: ``filter=False`` builds the Hamiltonian from the raw
        tables, and the default builds a different one.

        The 'raw' side is assembled here from the dataset's own radial
        functions rather than from a second ``filter=False`` call, so the
        comparison has independent content.
        """
        atoms = h2()
        # The filter alone is under test, so the confinement PAW also applies
        # by default is switched off on both sides: the raw table below is the
        # dataset's free-atom partial wave.
        free = {"name": family, "energy_shift": None}
        off = build(atoms, {**free, "filter": False}, 0.25)
        on = build(atoms, free, 0.25)
        assert off[4]["filter_cutoff"] is None
        assert on[4]["filter_cutoff"] is not None

        pp = PSEUDO_FAMILIES[family].get("H")
        r = np.asarray(pp.r)[:-1]                   # the table's own nodes
        raw = np.asarray(pp.channels[0].pseudo_radial)[:-1]
        assert np.array_equal(off[4]["integrals"].basis[0].radial(r), raw)
        filtered = on[4]["integrals"].basis[0].radial(r)
        assert not np.array_equal(filtered, raw)
        # 1.1e-2 at h = 0.25 -- 0.9 % of the peak, small but not nothing.
        assert np.abs(filtered - raw).max() > 1e-3

        # The Hamiltonians differ, and by little on H2: max|dH| = 2.4e-3 Ha
        # (PAW) / 3.9e-3 (UPAW), a ground-state shift of -2.5e-5 / +8.8e-4 Ha.
        h_off = off[0].map_to_qubits("jordan_wigner").to_matrix()
        h_on = on[0].map_to_qubits("jordan_wigner").to_matrix()
        assert not np.allclose(h_off, h_on)         # the default does something
        assert np.abs(h_off - h_on).max() < 1e-2    # ...but not a lot, on H2

    def test_the_dry_run_names_the_filter_and_counts_the_same(self):
        """Filtering reshapes functions; it never adds or removes one."""
        on = estimate_qubits(water(), basis="PAW")
        off = estimate_qubits(water(), basis={"name": "PAW", "filter": False})
        assert on.n_qubits == off.n_qubits and on.per_atom == off.per_atom
        assert "filtered (auto: 1 x Nyquist)" in on.basis
        assert "unfiltered" in off.basis
        assert "unfiltered" in estimate_qubits(water(), basis="NCPP").basis
        explicit = count_basis_functions(water(),
                                         {"name": "PAW", "filter": 700.0})[1]
        assert "filtered (700 eV)" in explicit

    def test_the_dry_run_refuses_a_bad_value(self):
        with pytest.raises(ValueError, match="filter"):
            count_basis_functions(water(), {"name": "PAW", "filter": "yes"})

    @pytest.mark.parametrize("spec, text", [
        (False, "unfiltered"), (True, "auto"), (500.0, "500 eV")])
    def test_filter_label(self, spec, text):
        assert text in filter_label(spec)


# --------------------------------------------------------------------------- #
# (4) What it buys: less egg-box, and forces that still differentiate E.
# --------------------------------------------------------------------------- #

def rigid_shift_ripple(basis, h, fractions=(0.0, 0.25, 0.5, 0.75)):
    """Peak-to-peak RHF energy of water rigidly shifted on a FROZEN grid.

    The grid is built once and passed explicitly, which is exactly what the
    force path freezes; without that the grid re-centers on the molecule and
    the egg-box is invisible by construction.
    """
    atoms = water()
    grid = grid_from_cell(atoms, h)
    energies = []
    for f in fractions:
        moved = atoms.copy()
        moved.positions += f * h
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _H, _p, _n, _pr, context = build_basis_hamiltonian(
                moved, basis, grid, h, 0, None)
        ints = context["integrals"]
        rhf = ints.hartree_fock(context["n_electrons"])
        energies.append(rhf.electronic_energy + ints.nuclear_repulsion
                        + ints.constant_energy)
    return float(max(energies) - min(energies))


class TestWhatItBuys:
    def test_the_rigid_shift_ripple_shrinks(self):
        """Water / PAW-SZ at h = 0.25: 268.7 meV of ripple unfiltered against
        27.1 meV with the default, a factor 9.9 (22x at h = 0.20).  The margin
        asked for here is a factor 3 -- well inside what was measured, and
        loose enough to survive a change in the compensation quadrature."""
        off = rigid_shift_ripple({"name": "PAW", "filter": False}, 0.25)
        on = rigid_shift_ripple("PAW", 0.25)
        assert off > 2e-3                       # there is an egg-box to fix
        assert on < off / 3.0

    def test_the_kinetic_term_stops_aliasing(self):
        """The cleanest single signal, and it needs no density: ``tr(T)`` in
        the raw atomic-orbital basis depends on where the nuclei fall between
        grid nodes and on nothing else, so its rigid-shift ripple *is* the
        kinetic aliasing.  Measured at h = 0.25: **197.3 -> 0.19 meV**, a
        factor of 1041.

        (``resolution_ratios`` was tried here first and is a poor proxy --
        0.0767 -> 0.0741, barely moving -- because it compares the grid
        kinetic energy against the *exact* one for the same function, which
        the filter also changes.  This compares one function against itself
        at four grid offsets, which is the property that matters.)
        """
        def kinetic_ripple(basis):
            atoms = water()
            grid = grid_from_cell(atoms, 0.25)
            traces = []
            for f in (0.0, 0.25, 0.5, 0.75):
                moved = atoms.copy()
                moved.positions += f * 0.25
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ints = build_basis_hamiltonian(
                        moved, basis, grid, 0.25, 0, None)[4]["integrals"]
                T, _V = ints._engine.one_body(ints.external_potential(),
                                              energy_units="Ha",
                                              kinetic=ints.kinetic)
                traces.append(float(np.trace(T).real))
            return max(traces) - min(traces)

        off = kinetic_ripple({"name": "PAW", "filter": False})
        on = kinetic_ripple("PAW")
        assert off > 1e-3                       # ~7 mHa of aliasing to remove
        assert on < off / 100.0

    def test_the_analytic_force_is_the_derivative_of_the_energy(self):
        """With the filter on, ``forces == -dE/dR`` of the calculator's own
        energy on its frozen grid -- the check of
        ``test_paw_forces.test_forces_are_the_derivative_of_the_energy``,
        repeated for the filtered basis because the filter changes every
        sampled function the gradient differentiates.

        Read from ``unprojected``: the finite difference is of the discretized
        energy, which is not translation invariant, while the reported force
        has had that component projected out.
        """
        atoms = h2(0.75)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="PAW", h=0.25,
                               pool="fermionic", profile=False, trace=False,
                               project_translation=False)
        step = 0.005
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms.get_forces()
            raw = atoms.calc.force_result.unprojected
            energies = []
            for sign in (1, -1):
                moved = atoms.copy()
                moved.positions[1, 2] += sign * step
                moved.calc = atoms.calc
                energies.append(moved.get_potential_energy())
        numerical = -(energies[0] - energies[1]) / (2 * step)
        assert raw[1, 2] == pytest.approx(numerical, abs=5e-3)

    def test_the_geometry_barely_moves(self):
        """The variational shift is nearly a constant offset, so it cancels in
        the differences that matter.  Measured on LiH / PAW-SZ: d_eq 1.6536 ->
        1.6538 Angstrom and D_e 1.8794 -> 1.8764 eV; on OH the total energy
        moves 379 meV while d_eq moves 0.0018 Angstrom.  Here the cheap
        version: the H2 bond force at a fixed distance keeps its sign and its
        magnitude.
        """
        forces = {}
        for label, basis in (("off", {"name": "PAW", "filter": False}),
                             ("on", "PAW")):
            atoms = h2(0.90)
            atoms.calc = Mandacaru(method="adapt-vqe", basis=basis, h=0.25,
                                   pool="fermionic", profile=False,
                                   trace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = atoms.get_forces()
            forces[label] = float((f[0, 2] - f[1, 2]) / 2)
        assert np.sign(forces["on"]) == np.sign(forces["off"])
        # Absolute, not relative: 0.90 Angstrom is 0.015 Angstrom from the
        # minimum, where the bond force crosses zero, so a *relative* tolerance
        # is ill-conditioned there (it failed at 9 % on a 0.032 eV/Angstrom
        # difference).  0.05 eV/Angstrom on a ~20 eV/Angstrom^2 bond is a
        # 0.0025 Angstrom shift of the minimum; measured across the filter and
        # the exact local potential, H2's d_eq stays within 0.9132-0.9147.
        assert forces["on"] == pytest.approx(forces["off"], abs=0.05)
