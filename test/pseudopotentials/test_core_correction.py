# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_core_correction.py

"""The nonlinear core correction (Louie, Froyen and Cohen).

Two things have to hold for the partial core density to be usable: it must
**match the true core in value and slope** at the matching radius, so the
unscreened potential has no kink there, and it must be **nodeless and finite**
inside it, so it is softer than what it replaces rather than harder.  Both are
properties of the ``A sin(Br)/r`` form with ``Br_c`` kept inside ``(0, pi)``,
and both are checked directly.
"""

import numpy as np
import pytest

from mandacaru.pseudopotentials.core_correction import (crossover_radius,
                                                        partial_core_density,
                                                        smooth_core_density)


def grid(points=4000, r_max=20.0):
    return np.arange(1, points + 1) * (r_max / (points + 1))


def core_and_valence(r, core_scale=6.0, valence_scale=1.0):
    """A tight core and a diffuse valence, as an atom has."""
    core = 2.0 * (core_scale ** 3 / np.pi) * np.exp(-2 * core_scale * r)
    valence = 6.0 * (valence_scale ** 3 / np.pi) * np.exp(-2 * valence_scale * r)
    return core, valence


class TestTheMatchingRadius:
    def test_it_is_where_the_two_densities_cross(self):
        r = grid()
        core, valence = core_and_valence(r)
        radius = crossover_radius(r, core, valence)
        assert radius is not None
        index = int(np.argmin(np.abs(r - radius)))
        # The crossing is interpolated between nodes, and these densities
        # change their ratio by ~5 % per node, so the nearest node is only
        # that close.
        assert core[index] == pytest.approx(valence[index], rel=5e-2)

    def test_a_larger_ratio_moves_it_inward(self):
        """rho_c = 2 rho_v happens closer in than rho_c = rho_v."""
        r = grid()
        core, valence = core_and_valence(r)
        assert crossover_radius(r, core, valence, 2.0) < crossover_radius(
            r, core, valence, 1.0)

    def test_an_atom_whose_core_never_dominates_has_none(self):
        r = grid()
        _core, valence = core_and_valence(r)
        assert crossover_radius(r, np.zeros_like(r), valence) is None


class TestTheSmoothCore:
    def test_it_matches_the_true_core_in_value_and_slope(self):
        r = grid()
        core, valence = core_and_valence(r)
        radius = crossover_radius(r, core, valence)
        smooth, A, B = smooth_core_density(r, core, radius)
        index = int(np.argmin(np.abs(r - radius)))
        r_m = r[index]
        assert smooth[index] == pytest.approx(core[index], rel=1e-10)
        # The slope is compared analytically.  `smooth` is piecewise -- the
        # sin form inside, the true core outside -- so a centered difference
        # *at* the junction straddles both branches and measures the stencil
        # rather than the match.
        d_sin = A * (B * np.cos(B * r_m) / r_m - np.sin(B * r_m) / r_m ** 2)
        d_core = np.gradient(core, r, edge_order=2)[index]
        assert d_sin == pytest.approx(d_core, rel=1e-9)

    def test_it_is_the_true_core_outside_the_radius(self):
        r = grid()
        core, valence = core_and_valence(r)
        radius = crossover_radius(r, core, valence)
        smooth, _A, _B = smooth_core_density(r, core, radius)
        outside = r >= radius
        assert np.array_equal(smooth[outside], core[outside])

    def test_it_is_nodeless_positive_and_finite_inside(self):
        r = grid()
        core, valence = core_and_valence(r)
        radius = crossover_radius(r, core, valence)
        smooth, _A, B = smooth_core_density(r, core, radius)
        inside = r < radius
        assert np.all(smooth[inside] > 0.0)
        assert np.all(np.isfinite(smooth))
        assert 0.0 < B * radius < np.pi          # no node inside r_nlcc

    def test_it_is_softer_than_what_it_replaces(self):
        """The point of the correction: less charge piled at the nucleus."""
        r = grid()
        core, valence = core_and_valence(r)
        radius = crossover_radius(r, core, valence)
        smooth, _A, _B = smooth_core_density(r, core, radius)
        inside = r < radius
        assert smooth[inside].max() < core[inside].max()

    def test_a_radius_where_the_core_is_not_decreasing_is_refused(self):
        r = grid(500, 10.0)
        rising = np.exp(r)                       # increasing everywhere
        with pytest.raises(ValueError, match="not decreasing"):
            smooth_core_density(r, rising, 2.0)

    def test_a_radius_where_the_core_has_vanished_is_refused(self):
        r = grid(500, 10.0)
        core = np.where(r < 1.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="vanishes"):
            smooth_core_density(r, core, 5.0)


class TestThePartialCoreDensity:
    def test_it_reports_how_much_charge_it_kept(self):
        r = grid()
        core, valence = core_and_valence(r)
        smooth, details = partial_core_density(r, core, valence)
        assert details["applied"]
        shell = 4.0 * np.pi * r * r
        assert details["core_electrons"] == pytest.approx(2.0, rel=1e-4)
        assert details["partial_core_electrons"] == pytest.approx(
            float(np.trapezoid(smooth * shell, r)), rel=1e-12)
        # Smoothing removes charge from the peak, so it holds less.
        assert (details["partial_core_electrons"]
                < details["core_electrons"])

    def test_an_explicit_radius_is_used_as_given(self):
        r = grid()
        core, valence = core_and_valence(r)
        _smooth, details = partial_core_density(r, core, valence, r_nlcc=0.5)
        assert details["r_nlcc"] == pytest.approx(0.5, abs=r[1] - r[0])

    def test_an_atom_with_no_core_gets_nothing_and_is_told_why(self):
        r = grid()
        _core, valence = core_and_valence(r)
        smooth, details = partial_core_density(r, np.zeros_like(r), valence)
        assert not details["applied"]
        assert not np.any(smooth)
        assert "never exceeds" in details["reason"]


@pytest.mark.slow
class TestInTheGenerators:
    """What the correction changes about a pseudopotential, end to end."""

    @pytest.fixture(scope="class")
    def oxygen(self):
        from mandacaru.pseudopotentials.oncv import generate_oncv

        with_nlcc = generate_oncv("O", points=4000, r_max=20.0)
        without = generate_oncv("O", points=4000, r_max=20.0, nlcc=False)
        return with_nlcc, without

    def test_it_is_on_by_default_and_recorded(self, oxygen):
        with_nlcc, without = oxygen
        assert with_nlcc.has_core_correction
        assert not without.has_core_correction
        assert with_nlcc.nlcc["r_nlcc"] > 0.0

    def test_the_partial_core_holds_less_than_the_true_core(self, oxygen):
        with_nlcc, _without = oxygen
        assert 0.0 < with_nlcc.core_charge() < 2.0

    def test_it_changes_the_ionic_local_potential(self, oxygen):
        """The correction acts in the unscreening; if it did not move
        v_local it would not be doing anything."""
        with_nlcc, without = oxygen
        assert np.max(np.abs(with_nlcc.v_local - without.v_local)) > 1e-3

    def test_switching_it_off_reproduces_the_uncorrected_potential(self):
        from mandacaru.pseudopotentials.oncv import generate_oncv

        a = generate_oncv("O", points=4000, r_max=20.0, nlcc=False)
        b = generate_oncv("O", points=4000, r_max=20.0, nlcc=False)
        assert np.array_equal(a.v_local, b.v_local)

    def test_hydrogen_has_no_core_to_correct(self):
        from mandacaru.pseudopotentials.oncv import generate_oncv

        pp = generate_oncv("H", points=3000, r_max=20.0)
        assert not pp.has_core_correction
        assert pp.core_charge() == 0.0
