# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_confinement.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The confined orbitals of ``pseudopotentials/confinement.py``, now solved in
Rust, against the numbers the Python solver gave.

``test/migration/reference/paw/confinement.npz`` (inputs in ``confinement.json``)
holds the Python outputs on the shipped ``lda-sr`` datasets of H, Li, C, N, O,
Na, Si, Ca, Cu and Au: the free levels, the confined levels at fixed walls, the
eigenstates, the cutoff radii and orbitals at 0.1 and 0.3 eV, the Gaussian
polarization shells and the confining potential and quasi-Gaussian on fixed
grids.  Quantities at a fixed wall agree to 1e-9 (measured: 5e-11); the ones
that sit behind the root search of the radius (an ``xtol`` of 1e-4 Bohr) are
compared at 1e-8.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.confinement import (
    confined_energy, confined_eigenstate, confined_orbital, confinement_potential,
    confinement_radius, free_energy, gaussian_polarization, quasi_gaussian)

REFERENCE = Path(__file__).resolve().parents[1] / "migration" / "reference" / "paw"
META = json.loads((REFERENCE / "confinement.json").read_text())
FROZEN = np.load(REFERENCE / "confinement.npz")

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _library_present() -> bool:
    try:
        get_paw("H")
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_library = pytest.mark.skipif(not _library_present(),
                                   reason="MANDACARU_PAW_PATH is not configured")


def close(value, expected, tol=1e-9):
    """Largest difference relative to the largest entry of ``expected``."""
    value, expected = np.asarray(value, float), np.asarray(expected, float)
    assert value.shape == expected.shape
    scale = max(float(np.abs(expected).max()), 1e-300)
    assert float(np.abs(value - expected).max()) <= tol * scale


def strided(array):
    """The entries the reference kept (``ceil(n / max_points)`` apart)."""
    return array[::-(-len(array) // META["max_points"])]


class TestFixedGrids:
    @pytest.mark.parametrize("tag", ["a", "b"])
    def test_the_confining_potential(self, tag):
        r_c, amplitude, inner = META[f"pot_{tag}"]
        expected = FROZEN[f"pot_{tag}"]
        value = confinement_potential(FROZEN["pot_r"], r_c, amplitude, inner)
        wall = np.isinf(expected)
        assert np.array_equal(np.isinf(value), wall)
        close(value[~wall], expected[~wall], 1e-13)
        assert np.all(value[FROZEN["pot_r"] <= inner * r_c] == 0.0)

    @pytest.mark.parametrize("tag", ["a", "b"])
    def test_the_quasi_gaussian(self, tag):
        alpha, r_cut = META[f"qg_{tag}"]
        close(quasi_gaussian(FROZEN["qg_r"], alpha, r_cut),
              FROZEN[f"qg_{tag}"], 1e-13)

    def test_the_quasi_gaussian_vanishes_with_its_slope_at_the_cutoff(self):
        r = np.linspace(0.0, 8.0, 80001)
        values = quasi_gaussian(r, 0.5, 6.0)
        edge = np.flatnonzero(r < 6.0)[-1]
        assert abs(values[edge]) < 1e-8
        assert abs((values[edge] - values[edge - 1]) / (r[1] - r[0])) < 1e-3
        assert np.all(values[r >= 6.0] == 0.0)


@needs_library
class TestFrozenPythonOutputs:
    @pytest.mark.parametrize("symbol", META["elements"])
    def test_the_levels_and_the_eigenstates(self, symbol):
        pp = get_paw(symbol)
        for l, index, *_rest in META["states"][symbol]:
            key = f"{symbol}_{l}_{index}"
            assert free_energy(pp, l, index) == pytest.approx(
                FROZEN[f"{key}_free"][0], rel=1e-9)
            for r_c, expected in zip(META[f"{key}_radii"],
                                     FROZEN[f"{key}_energy"]):
                assert confined_energy(pp, l, r_c, state=index) == \
                    pytest.approx(expected, rel=1e-9)
            r, u, energy = confined_eigenstate(pp, l, 6.0, state=index)
            first, last, count = FROZEN[f"{key}_eig_r"]
            assert (r[0], r[-1], len(r)) == pytest.approx((first, last, count),
                                                          rel=1e-12)
            assert energy == pytest.approx(FROZEN[f"{key}_eig_e"][0], rel=1e-9)
            close(u, FROZEN[f"{key}_eig_u"])

    @pytest.mark.parametrize("symbol", META["elements"])
    def test_the_radii_and_the_orbitals(self, symbol):
        pp = get_paw(symbol)
        for l, index, *_rest in META["states"][symbol]:
            for shift in META["shifts_eV"]:
                tag = f"{symbol}_{l}_{index}_s{shift}"
                orbital = confined_orbital(pp, l, shift, None, index)
                scalars = FROZEN[f"{tag}_scalars"]
                close([orbital.r_c, orbital.energy, orbital.free_energy,
                       orbital.achieved_shift], scalars, 1e-8)
                close(strided(orbital.radial), FROZEN[f"{tag}_radial"], 1e-8)
                assert orbital.radial[pp.r >= orbital.r_c].max(initial=0) == 0

    @pytest.mark.parametrize("symbol", META["elements"])
    def test_a_custom_confining_potential(self, symbol):
        l, index, shift, amplitude, inner = META[f"{symbol}_custom"]
        orbital = confined_orbital(get_paw(symbol), l, shift,
                                   (amplitude, inner), index)
        close([orbital.r_c, orbital.energy, orbital.free_energy,
               orbital.achieved_shift], FROZEN[f"{symbol}_custom_scalars"], 1e-8)
        close(strided(orbital.radial), FROZEN[f"{symbol}_custom_radial"], 1e-8)

    def test_the_radius_of_the_copper_d_channel(self):
        radius = confinement_radius(get_paw("Cu"), 2, 0.1, tightest=True)
        assert radius == pytest.approx(FROZEN["Cu_d_tightest"][0], rel=1e-8)

    def test_a_shift_beyond_the_sphere_takes_the_tightest_wall(self):
        pp = get_paw("O")
        radius = confinement_radius(pp, 0, 5.0, tightest=True)
        assert radius == pytest.approx(pp.channels[0].r_cut / 0.6, rel=1e-12)
        orbital = confined_orbital(pp, 0, 5.0)
        assert orbital.r_c == pytest.approx(radius, rel=1e-12)
        assert 0.0 < orbital.achieved_shift < 5.0

    def test_a_shift_too_large_is_refused_with_the_largest_usable(self):
        with pytest.raises(ValueError) as caught:
            confinement_radius(get_paw("O"), 0, 5.0)
        assert str(caught.value) == META["errors"]["large"]

    @pytest.mark.parametrize("symbol", META["polarization_elements"])
    @pytest.mark.parametrize("shift", [0.1, 0.05])
    def test_the_gaussian_polarization_shell(self, symbol, shift):
        shell = gaussian_polarization(get_paw(symbol), shift)
        key = f"gp_{symbol}_{shift}"
        close([shell.l, shell.base_l, shell.r_cut, shell.r_char,
               shell.reference_radius], FROZEN[f"{key}_scalars"], 1e-8)
        close(strided(shell.radial), FROZEN[f"{key}_radial"], 1e-8)
        table = get_paw(symbol).r
        assert np.trapezoid(shell.radial ** 2 * table ** 2,
                            table) == pytest.approx(1.0, abs=1e-12)


@needs_library
class TestTheNativeRoutinesRefuse:
    def test_a_wall_outside_the_table(self):
        with pytest.raises(ValueError, match="radial table"):
            confined_energy(get_paw("H"), 0, 31.0)

    def test_a_bad_confining_potential(self):
        with pytest.raises(ValueError, match="confining potential"):
            confined_energy(get_paw("H"), 0, 6.0, amplitude=-1.0)
        with pytest.raises(ValueError, match="confining potential"):
            confined_energy(get_paw("H"), 0, 6.0, inner_fraction=1.0)

    def test_a_missing_channel(self):
        with pytest.raises(KeyError):
            confined_energy(get_paw("H"), 5, 6.0)

    def test_a_nonpositive_shift(self):
        with pytest.raises(ValueError, match="energy_shift"):
            confinement_radius(get_paw("H"), 0, 0.0)
