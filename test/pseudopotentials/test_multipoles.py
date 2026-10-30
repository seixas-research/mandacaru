# -*- coding: utf-8 -*-
# file: test_multipoles.py

"""The PAW-LCAO compensation multipoles (:mod:`mandacaru.pseudopotentials.multipoles`).

Copper is the witness: its 3d channel is cut at 0.77 Bohr and its 4s at 2.78,
so every s-d moment integrates the d partial waves well past their own cutoff.
Until 2026-09-27 the smooth d wave there was the stored Bessel expansion, which
diverges outside ``r_cut``; the s-d moments came out ~100x too large and the
isolated copper atom collapsed to -16,000 Ha as the grid was refined.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from mandacaru.integrals import reciprocal as rc
from mandacaru.pseudopotentials import get_paw, multipoles
from mandacaru.pseudopotentials.multipoles import (partial_waves,
                                                    radial_moments,
                                                    radial_remainder,
                                                    shape_function)


@pytest.fixture(scope="module")
def copper():
    return get_paw("Cu")


class TestPartialWaves:
    @pytest.mark.parametrize("l", [0, 2])
    def test_the_smooth_wave_is_the_all_electron_one_beyond_its_cutoff(
            self, copper, l):
        r = np.asarray(copper.r)
        outside = r > copper.channels[l].r_cut
        ae, smooth = partial_waves(copper, l)
        for a, s in zip(ae, smooth):
            assert np.array_equal(s[outside], a[outside])

    @pytest.mark.parametrize("l", [0, 2])
    def test_they_join_continuously_at_the_cutoff(self, copper, l):
        r = np.asarray(copper.r)
        k = int(np.searchsorted(r, copper.channels[l].r_cut, side="right"))
        ae, smooth = partial_waves(copper, l)
        for a, s in zip(ae, smooth):
            assert s[k - 1] == pytest.approx(a[k - 1], abs=1e-4)


class TestCrossMoments:
    def test_s_d_quadrupoles_are_of_the_size_of_the_d_moments(self, copper):
        """The divergent expansion put them at 20-40; the correct ones are
        a few tenths, like the d-d moments of the same shell."""
        cross = np.abs(radial_moments(copper, 0, 2, 2)).max()
        own = np.abs(radial_moments(copper, 2, 2, 2)).max()
        assert cross < 1.0
        assert cross < 20.0 * max(own, 1e-2)


class TestTheRemainderBeyondTheMultipoles:
    """What the compensation shape leaves of the all-electron minus smooth
    pair density, ``E^(L)_ij(q)``: its transform less the shape that carries
    the moment."""

    @staticmethod
    def _shape(dataset, L, q):
        r_g = float(dataset.compensation_radius)
        r = np.linspace(0.0, r_g, 1201)
        return rc.radial_transform(r, shape_function(r, r_g, L), L, q)

    @pytest.mark.parametrize("L", [0, 2, 4])
    def test_it_has_no_moment(self, copper, L):
        """The shape restores the moment, so the remainder vanishes with
        the wavevector (as ``q^(L+2)``): at 0.05 Bohr^-1 it is below 2e-3
        of the largest transform of the pair.  (1e-3 -> 2e-3, 2026-10-21:
        the semicore copper's compensation radius grew from 1.12 to
        1.50 Bohr, so the L = 0 ratio went from 7.3e-4 to 1.03e-3; L = 2 and
        4 stay at 6e-7 and 3e-10.)"""
        q = np.array([0.0, 0.05, 2.0])
        E = radial_remainder(copper, 2, 2, L, q, self._shape(copper, L, q))
        assert np.abs(E[0]).max() < 1e-6
        assert np.abs(E[1]).max() < 2e-3 * np.abs(E[2]).max()

    def test_it_is_the_transform_less_the_shape(self, copper):
        """At a large wavevector the shape's transform has fallen away and
        the remainder is the pair density's own transform."""
        q = np.array([6.0])
        E = radial_remainder(copper, 2, 2, 2, q, self._shape(copper, 2, q))
        r = np.asarray(copper.r)
        ae, smooth = partial_waves(copper, 2)
        inside = r <= copper.channels[2].r_cut
        from scipy.special import spherical_jn
        density = ((ae[0] * ae[0] - smooth[0] * smooth[0])
                   * r ** 2)[inside]
        direct = np.trapezoid(spherical_jn(2, q[0] * r[inside]) * density,
                              r[inside])
        shape = self._shape(copper, 2, q)[0] * copper.multipole_moments(
            2, 2, 2, "dual")[0, 0]
        assert E[0, 0, 0] == pytest.approx(direct - shape, abs=1e-6)

    def test_it_vanishes_without_an_augmentation(self, copper, monkeypatch):
        """Smooth waves equal to the all-electron ones leave nothing."""
        monkeypatch.setattr(multipoles, "partial_waves",
                            lambda dataset, l: (lambda a: (a, a))(
                                partial_waves(dataset, l)[0]))
        q = np.array([0.5, 2.0, 4.0])
        E = radial_remainder(copper, 2, 2, 2, q, np.zeros(3))
        assert np.abs(E).max() < 1e-12




class TestRealAngularTables:
    """The real-harmonic Gaunt coefficients and multipole Coulomb matrix of a
    molecule are the complex ones (a crystal's) transformed shell by shell."""

    def test_every_gaunt_coefficient_up_to_d(self):
        from mandacaru.basis._angular import real_harmonic_transform
        from mandacaru.pseudopotentials.multipoles import gaunt

        U = {l: real_harmonic_transform(l) for l in range(5)}
        for l1 in range(3):
            for l2 in range(3):
                for L in range(abs(l1 - l2), l1 + l2 + 1):
                    for M in range(-L, L + 1):
                        for m1 in range(-l1, l1 + 1):
                            for m2 in range(-l2, l2 + 1):
                                total = 0.0 + 0.0j
                                for K in range(-L, L + 1):
                                    for k1 in range(-l1, l1 + 1):
                                        for k2 in range(-l2, l2 + 1):
                                            total += (
                                                np.conj(U[L][M + L, K + L])
                                                * np.conj(U[l1][m1 + l1,
                                                                k1 + l1])
                                                * U[l2][m2 + l2, k2 + l2]
                                                * gaunt(L, K, l1, k1, l2, k2,
                                                        real=False))
                                value = gaunt(L, M, l1, m1, l2, m2, real=True)
                                assert isinstance(value, float)
                                assert abs(value - total) < 1e-12


REFERENCE = (Path(__file__).resolve().parents[1] / "migration" / "reference"
             / "paw" / "multipoles.npz")


@pytest.fixture(scope="module")
def frozen():
    """Outputs of the Python implementation, frozen before the port to Rust."""
    return np.load(REFERENCE)


def _scaled_error(value, reference):
    return (np.abs(np.asarray(value) - reference).max()
            / max(np.abs(reference).max(), 1e-300))


class TestFrozenPythonOutputs:
    """The Rust routines reproduce the numbers the Python ones gave (the
    shipped lda-sr datasets and fixed radii; ``test/migration/reference/paw``)."""

    X = np.concatenate([[0.0, 1e-9, 1e-3], np.linspace(0.01, 6.0, 300)])

    @pytest.mark.parametrize("r_g", [1.2, 2.5])
    @pytest.mark.parametrize("L", range(5))
    def test_the_shape_and_its_potential(self, frozen, r_g, L):
        from mandacaru.pseudopotentials.multipoles import shape_potential
        assert _scaled_error(shape_function(self.X, r_g, L),
                             frozen[f"shape_{r_g}_{L}"]) < 1e-14
        assert _scaled_error(shape_potential(self.X, r_g, L),
                             frozen[f"shapepot_{r_g}_{L}"]) < 1e-12

    @pytest.mark.parametrize("real", [True, False])
    def test_every_gaunt_coefficient_up_to_f(self, frozen, real):
        from mandacaru.pseudopotentials.multipoles import gaunt
        keys, values = frozen["gaunt_keys"], frozen["gaunt_values"]
        for (flag, *index), value in zip(keys, values):
            if bool(flag) is real:
                assert abs(gaunt(*map(int, index), real=real) - value) < 1e-13

    def test_the_angular_rule(self, frozen):
        from mandacaru.pseudopotentials.multipoles import angular_grid
        for polar, azimuthal, tag in ((24, 48, "angular"), (10, 14, "angular2")):
            theta, phi, weight = angular_grid(polar, azimuthal)
            assert np.abs(theta - frozen[f"{tag}_theta"]).max() < 1e-14
            assert np.abs(phi - frozen[f"{tag}_phi"]).max() < 1e-14
            assert np.abs(weight - frozen[f"{tag}_w"]).max() < 1e-14

    @pytest.mark.parametrize("symbol", ["H", "O", "Cu"])
    def test_the_smooth_waves_and_their_moments(self, frozen, symbol):
        dataset = get_paw(symbol)
        q, shape = frozen["remainder_q"], frozen["remainder_shape_transform"]
        for l in dataset.channels:
            _ae, smooth = partial_waves(dataset, l)
            assert _scaled_error(np.array(smooth)[:, ::7],
                                 frozen[f"{symbol}_pw_sm_{l}"]) < 1e-11
        for l1, l2 in itertools.product(dataset.channels, repeat=2):
            for L in multipoles.multipole_range(l1, l2):
                tag = f"{symbol}_{{}}_{l1}{l2}{L}"
                assert _scaled_error(radial_moments(dataset, l1, l2, L),
                                     frozen[tag.format("mom")]) < 1e-9
                assert _scaled_error(radial_remainder(dataset, l1, l2, L, q, shape),
                                     frozen[tag.format("rem")]) < 1e-10

    def test_the_selection_rule_of_the_multipole_range(self, frozen):
        for a, b, count in frozen["range_keys"]:
            levels = multipoles.multipole_range(int(a), int(b))
            assert len(levels) == count
            assert all(abs(a - b) <= L <= a + b and (L + a + b) % 2 == 0
                       for L in levels)
