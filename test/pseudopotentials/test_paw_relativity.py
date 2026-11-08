# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_paw_relativity.py

"""Relativity, GGA and spin-orbit coupling in a PAW-LCAO dataset.

One fact about PAW-LCAO's relativistic story is worth stating plainly: a
``relativity="dirac"`` **PAW-LCAO** dataset is scalar-relativistic
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

import json
import warnings

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.pseudopotentials.generation import generate_paw
from mandacaru.pseudopotentials.native_context import context
from mandacaru.pseudopotentials.record import record_of

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
        # The construction itself: anchored at the 2p energy, oxygen's empty
        # d channel binds a spurious level at -0.33 Ha that the atom does not
        # have, which the acceptance now refuses (HISTORY.md, 2026-10-11,
        # "The PAW audit report checked").
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = generate_paw("O", points=4000, r_max=20.0, extra_l=1,
                              ghosts="keep")
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
                                relativity="none", ghosts="keep")
            scalar = generate_paw("O", points=4000, r_max=20.0, extra_l=1,
                                  ghosts="keep")
        assert np.abs(free.overlap_correction[2]).max() < 1e-10
        assert np.abs(free.norm_correction[2]).max() < 1e-10
        assert np.abs(scalar.norm_correction[2]).max() < 1e-10
        assert 1e-6 < np.abs(scalar.overlap_correction[2]).max() < 1e-3


class TestSpinOrbitCoupling:
    """The j-resolved term: two unitary branches per l, stored as their
    (2j+1) average and L.S difference on the union of their projectors."""

    def test_it_is_off_unless_asked_for(self, oxygen_scalar):
        assert not oxygen_scalar.has_spin_orbit
        assert oxygen_scalar.spin_orbit == {}

    def test_a_dirac_dataset_carries_it(self, oxygen_dirac):
        assert oxygen_dirac.has_spin_orbit
        assert oxygen_dirac.relativity == "dirac"
        assert 1 in oxygen_dirac.spin_orbit

    def test_an_s_channel_has_none(self, oxygen_dirac):
        assert 0 not in oxygen_dirac.spin_orbit

    def test_the_union_holds_both_branches(self, oxygen_dirac):
        for l, entry in oxygen_dirac.spin_orbit.items():
            n = 2 * len(oxygen_dirac.channels[l].projectors)
            assert len(entry["projectors"]) == n
            for key in ("average", "average_screened", "coupling",
                        "coupling_screened", "overlap_average",
                        "overlap_coupling"):
                assert entry[key].shape == (n, n)
                assert np.allclose(entry[key], entry[key].T, atol=1e-10)

    def test_each_j_is_the_dirac_atoms_level(self, oxygen_dirac):
        """The level of each branch, solved by the generator from the stored terms,
        against the Dirac atom's: the error is stored with the entry."""
        for entry in oxygen_dirac.spin_orbit.values():
            assert np.all(np.abs(entry["level_errors"]) < 1e-4)

    def test_the_overlap_stays_spin_free(self, oxygen_dirac):
        """Unitary branches: a j-dependent q would give the metric an L.S
        structure."""
        for entry in oxygen_dirac.spin_orbit.values():
            assert np.abs(entry["overlap_average"]).max() < 1e-3
            assert np.abs(entry["overlap_coupling"]).max() < 1e-3

    def test_the_scalar_channels_are_untouched(self, oxygen_dirac,
                                              oxygen_scalar):
        """The spin-free part -- density, compensation, overlap -- is the
        scalar construction's; the Dirac atom's (2j+1) average differs from
        Koelling-Harmon's at O(c^-4), which moves q by 6e-6."""
        for l in oxygen_scalar.channels:
            assert (len(oxygen_dirac.channels[l].projectors)
                    == len(oxygen_scalar.channels[l].projectors))
            assert np.allclose(oxygen_dirac.overlap_correction[l],
                               oxygen_scalar.overlap_correction[l], atol=1e-4)

    def test_it_survives_a_record_round_trip(self, oxygen_dirac, tmp_path):
        from mandacaru.pseudopotentials.record import read_dataset, write_dataset

        back = read_dataset(write_dataset(oxygen_dirac, tmp_path / "O.parquet"))
        assert back.has_spin_orbit and back.relativity == "dirac"
        for l, entry in oxygen_dirac.spin_orbit.items():
            for key, value in entry.items():
                if key == "projectors":
                    # Tables are stored to ten digits.
                    assert all(np.allclose(a, b, rtol=1e-9, atol=1e-12) for a, b in
                               zip(back.spin_orbit[l][key], value))
                else:
                    assert np.allclose(back.spin_orbit[l][key], value)

    def test_the_first_order_format_is_refused(self, oxygen_dirac, tmp_path):
        payload, tables = record_of(oxygen_dirac)
        record = json.loads(payload)
        record["spin_orbit"] = {"1": [[0.01, 0.0], [0.0, 0.01]]}
        with pytest.raises(ValueError, match="first-order spin-orbit"):
            _native.paw_record_write(context(), str(tmp_path / "O.parquet"),
                                     json.dumps(record), tables)

    def test_a_strided_record_keeps_the_projectors_on_its_grid(self, oxygen_dirac,
                                                               tmp_path):
        """The union projectors are radial tables like every other, stored on the
        dataset's stride; they were once written at full length."""
        from mandacaru.pseudopotentials.record import read_dataset, write_dataset

        back = read_dataset(write_dataset(oxygen_dirac, tmp_path / "O.parquet", stride=4))
        for entry in back.spin_orbit.values():
            assert all(p.size == back.r.size for p in entry["projectors"])

    def test_projectors_off_the_grid_are_refused(self, oxygen_dirac, tmp_path):
        payload, tables = record_of(oxygen_dirac)
        record = json.loads(payload)
        tables = {k: np.ascontiguousarray(v[3::4]) for k, v in tables.items()}      # the grid at stride 4 ...
        # ... with the union projectors left at full length.
        record["spin_orbit"]["1"]["projectors"] = [
            list(p) for p in oxygen_dirac.spin_orbit[1]["projectors"]]
        with pytest.raises(ValueError, match="radial grid"):
            _native.paw_record_write(context(), str(tmp_path / "O.parquet"),
                                     json.dumps(record), tables)
