# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_paw_relativity.py

"""Relativity, GGA and spin-orbit coupling in a PAW-LCAO dataset.

PAW-LCAO's relativistic story differs from ONCVPSP's in one respect worth stating
plainly: a ``relativity="dirac"`` **PAW-LCAO** dataset is scalar-relativistic
partial waves plus a spin-orbit *term*, not a j-resolved augmentation sphere.
The reason is the overlap operator.  ``S = 1 + sum |p~> q <p~|`` is the metric
of the generalized eigenproblem, and a j-dependent ``q`` would give that
metric an ``L.S`` structure that every consumer of ``S`` would have to learn
about.  Keeping one partial-wave set per ``l`` leaves ``q``, ``Delta T`` and
the compensation charges untouched.

The two identities this file leans on are the ones that broke first when the
reference atom became relativistic, and they are checked at their real
tolerances rather than loosened:

* ``D^scr = Delta T + Delta V`` (``consistency_error``), and
* ``D^scr`` symmetric (``asymmetry``),

which close together only when the **coupling** is built from the conserved
(M-weighted) norm.  ``q`` itself is not: it is the charge the smooth density
is missing, so the overlap operator, the compensation charge and the L = 0
multipole all take the plain inner product.  The two matrices are identical
non-relativistically and differ at ``O(c^-2)`` when they are not.
"""

import warnings

import numpy as np
import pytest

from mandacaru.basis.relativity import spin_orbit_radial
from mandacaru.pseudopotentials.paw import (from_payload, generate_paw,
                                            spin_orbit_blocks, to_payload)

HARTREE_EV = 27.211386245988

#: Every test here builds a dataset from scratch; the module-scoped
#: fixtures share that cost but it is still ~50 s.  `-m slow` runs it.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def oxygen_scalar():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_paw("O", points=4000, r_max=20.0)


@pytest.fixture(scope="module")
def oxygen_dirac():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_paw("O", points=4000, r_max=20.0, relativity="dirac")


class TestTheOneCenterIdentitiesStillClose:
    """The two that break if the conserved norm and the overlap are confused."""

    @pytest.mark.parametrize("symbol", ["H", "Li", "O"])
    def test_consistency_and_symmetry(self, symbol):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = generate_paw(symbol, points=4000, r_max=20.0)
        for l, channel in pp.channels.items():
            assert channel.consistency_error < 1e-10, f"l={l}"
            assert channel.asymmetry < 1e-4, f"l={l}"

    def test_they_close_without_relativity_too(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = generate_paw("O", points=4000, r_max=20.0,
                              relativity="none", nlcc=False)
        for channel in pp.channels.values():
            assert channel.consistency_error < 1e-10
            assert channel.asymmetry < 1e-4

    def test_the_conserved_norm_differs_from_the_overlap(self):
        """If it did not, none of the above would be saying anything."""
        from mandacaru.pseudopotentials.oncv import (inner_overlaps,
                                                     norm_targets)
        from mandacaru.basis.atomic_solver import solve_atom
        from mandacaru.pseudopotentials.oncv import bound_state, scattering_wave

        atom = solve_atom(8, points=4000, r_max=20.0, tolerance=1e-6,
                          relativity="scalar")
        r, v = atom.r, atom.v_effective
        u, e1 = bound_state(r, v, 1, atom.eigenvalues[(2, 1)], 8.0,
                            treatment="scalar")
        e2 = e1 + 1.0
        u2 = scattering_wave(r, v, 1, e2, 8.0, "scalar")
        r_cut = 1.45
        inside = r <= r_cut
        u2 = u2 / np.sqrt(np.trapezoid(u2[inside] ** 2, r[inside]))
        waves = [u / r, u2 / r]
        targets = norm_targets(r, waves, [e1, e2], r_cut, v, "scalar",
                               z_eff=8.0)
        overlaps = inner_overlaps(r, waves, r_cut)
        shift = np.max(np.abs(targets - overlaps))
        assert shift > 1e-6        # there is something to get wrong
        assert shift < 1e-2        # and it is an O(c^-2) correction


class TestTheOptions:
    def test_the_defaults_are_scalar_relativistic_with_a_core_correction(
            self, oxygen_scalar):
        assert oxygen_scalar.relativity == "scalar"
        assert oxygen_scalar.xc == "lda"
        assert oxygen_scalar.nlcc.get("applied")

    def test_none_and_no_nlcc_reproduce_the_old_construction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = generate_paw("O", points=4000, r_max=20.0,
                             relativity="none", nlcc=False)
            b = generate_paw("O", points=4000, r_max=20.0,
                             relativity="none", nlcc=False)
        assert np.array_equal(a.v_local, b.v_local)
        assert a.relativity == "none" and not a.nlcc.get("applied")

    def test_relativity_moves_the_dataset(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = generate_paw("O", points=4000, r_max=20.0,
                                 relativity="none", nlcc=False)
            scalar = generate_paw("O", points=4000, r_max=20.0,
                                  relativity="scalar", nlcc=False)
        assert np.max(np.abs(scalar.v_local - plain.v_local)) > 1e-4

    def test_a_pbe_dataset_is_built_and_labelled(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = generate_paw("O", points=4000, r_max=20.0, xc="pbe")
        assert pp.xc == "pbe"
        for channel in pp.channels.values():
            assert channel.consistency_error < 1e-10

    def test_an_extra_channel_conserves_norm_exactly(self):
        """It has no bound state, so its norm deficit is zero by construction.

        Zero *in which matrix* is the whole point.  ``norm_deficit = 0`` is
        imposed on the norm the Vanderbilt condition conserves -- relativistically
        the M-weighted one, :attr:`PAWChannel.norm_correction` -- and that is
        what vanishes identically.  The plain charge
        :attr:`~PAWChannel.overlap_correction` is the same quantity only when
        the reference atom is non-relativistic; under the shipped
        scalar-relativistic default the two part company at O(c^-2), and
        asserting the deficit on the charge instead measures that difference.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = generate_paw("O", points=4000, r_max=20.0, extra_l=1)
        assert sorted(pp.channels) == [0, 1, 2]
        assert np.abs(pp.norm_correction[2]).max() < 1e-10
        # ... while the occupied channels keep theirs.
        assert np.abs(pp.norm_correction[1]).max() > 1e-3

    def test_the_extra_channel_s_charge_follows_only_without_relativity(self):
        """The O(c^-2) gap between the two norms, pinned from both sides.

        Measured on oxygen: 1.4e-14 with ``relativity="none"`` -- the same
        quantity -- against 9.6e-5 with the default.  A change that made these
        agree relativistically would mean one of the two had stopped being
        what it claims to be.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            free = generate_paw("O", points=4000, r_max=20.0, extra_l=1,
                                relativity="none")
            scalar = generate_paw("O", points=4000, r_max=20.0, extra_l=1)
        assert np.abs(free.overlap_correction[2]).max() < 1e-10
        assert np.abs(free.norm_correction[2]).max() < 1e-10
        assert np.abs(scalar.norm_correction[2]).max() < 1e-10
        assert 1e-6 < np.abs(scalar.overlap_correction[2]).max() < 1e-3


class TestSpinOrbitCoupling:
    def test_it_is_off_unless_asked_for(self, oxygen_scalar):
        assert not oxygen_scalar.has_spin_orbit
        assert oxygen_scalar.spin_orbit == {}

    def test_a_dirac_dataset_carries_it(self, oxygen_dirac):
        assert oxygen_dirac.has_spin_orbit
        assert oxygen_dirac.relativity == "dirac"
        assert 1 in oxygen_dirac.spin_orbit

    def test_an_s_channel_has_none(self, oxygen_dirac):
        assert 0 not in oxygen_dirac.spin_orbit

    def test_the_blocks_are_symmetric_and_the_right_size(self, oxygen_dirac):
        for l, D in oxygen_dirac.spin_orbit.items():
            n = len(oxygen_dirac.channels[l].ae_waves)
            assert D.shape == (n, n)
            assert np.allclose(D, D.T, atol=1e-14)

    def test_the_partial_waves_stay_one_set_per_l(self, oxygen_dirac,
                                                  oxygen_scalar):
        """A Dirac PAW-LCAO dataset is scalar partial waves plus a term, so its
        projector count -- and therefore its overlap operator -- is unchanged.

        The overlap correction is *nearly*, not exactly, the same: asking for
        ``"dirac"`` also solves the reference **atom** with the Dirac equation,
        and its (2j+1) average differs from what Koelling-Harmon gives at
        ``O(c^-4)``.  The partial waves are built in that slightly different
        potential, which moves ``q`` by 6e-6.  What must not change is the
        *structure*.
        """
        for l in oxygen_scalar.channels:
            assert (len(oxygen_dirac.channels[l].projectors)
                    == len(oxygen_scalar.channels[l].projectors))
            assert np.allclose(oxygen_dirac.overlap_correction[l],
                               oxygen_scalar.overlap_correction[l], atol=1e-4)

    def test_the_all_electron_term_reproduces_the_atom_splitting(self,
                                                                 oxygen_dirac):
        """<xi> (2l+1)/2 must be the Dirac atom's own 2p splitting."""
        pp = oxygen_dirac
        xi = spin_orbit_radial(pp.r, pp.atom.v_effective,
                               atomic_number=pp.atomic_number)
        phi = pp.channels[1].ae_waves[0]
        mean = np.trapezoid(xi * phi * phi * pp.r * pp.r, pp.r)
        predicted = mean * (2 * 1 + 1) / 2.0
        assert predicted * HARTREE_EV == pytest.approx(
            pp.atom.spin_orbit_splitting(2, 1) * HARTREE_EV, rel=0.01)

    def test_most_of_it_lives_inside_the_augmentation_sphere(self,
                                                             oxygen_dirac):
        """Which is why a one-center term captures it at all."""
        pp = oxygen_dirac
        xi = spin_orbit_radial(pp.r, pp.atom.v_effective,
                               atomic_number=pp.atomic_number)
        phi = pp.channels[1].ae_waves[0]
        weight = xi * phi * phi * pp.r * pp.r
        inside = pp.r <= pp.channels[1].r_cut
        fraction = (np.trapezoid(weight[inside], pp.r[inside])
                    / np.trapezoid(weight, pp.r))
        assert fraction > 0.95

    def test_the_smooth_part_is_subtracted_and_is_small(self, oxygen_dirac):
        """Small, but not assumed away: leaving it in would double-count."""
        pp = oxygen_dirac
        with_smooth = pp.spin_orbit[1]
        ae_only = spin_orbit_blocks(pp.r, pp.channels, pp.atom.v_effective,
                                    np.zeros_like(pp.r), pp.atomic_number)[1]
        difference = np.abs(ae_only - with_smooth).max()
        assert difference > 0.0
        assert difference < 0.05 * np.abs(ae_only).max()

    def test_it_survives_a_payload_round_trip(self, oxygen_dirac):
        back = from_payload(to_payload(oxygen_dirac))
        assert back.has_spin_orbit and back.relativity == "dirac"
        for l, D in oxygen_dirac.spin_orbit.items():
            assert np.allclose(back.spin_orbit[l], D)

    def test_a_dataset_without_the_fields_is_not_relabelled(self):
        """A payload written before these options existed is what it was."""
        payload = to_payload(generate_paw("H", points=3000, r_max=20.0))
        for key in ("xc", "relativity", "nlcc", "spin_orbit", "extra_l"):
            payload.pop(key, None)
        back = from_payload(payload)
        assert back.relativity == "none" and back.xc == "lda"
        assert not back.nlcc.get("applied") and not back.has_spin_orbit
