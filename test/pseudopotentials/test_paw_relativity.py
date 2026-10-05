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

import warnings

import numpy as np
import pytest

from mandacaru.pseudopotentials.paw import (from_payload, generate_paw,
                                            paw_spin_orbit_blocks,
                                            paw_spin_orbit_projectors,
                                            to_payload)

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
        from mandacaru.basis.atomic_solver import solve_atom
        from mandacaru.pseudopotentials.partial_waves import (
            bound_state, inner_overlaps, norm_targets, scattering_wave)

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


def _level_with_spin_orbit(pp, l, lz_s):
    """Lowest level of channel ``l`` with ``<L.S> = lz_s``, built only from
    what the dataset stores: the union projectors, the average plus the
    L.S difference, and the overlap split the same way."""
    from scipy.interpolate import CubicSpline

    from mandacaru.pseudopotentials.paw import _generalized_spectrum

    entry = pp.spin_orbit[l]
    h = 0.004
    r = np.arange(1, int(22.0 / h)) * h
    v = CubicSpline(pp.r, pp.v_local_screened)(r) + l * (l + 1) / (2 * r * r)
    p_u = [CubicSpline(pp.r, p)(r) * r for p in entry["projectors"]]
    D = entry["average_screened"] + lz_s * entry["coupling_screened"]
    q = entry["overlap_average"] + lz_s * entry["overlap_coupling"]
    return float(_generalized_spectrum(r, v, p_u, D, q, 1)[0])


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
        """Solved per j from the stored terms, not from the branches."""
        pp = oxygen_dirac
        for l, entry in pp.spin_orbit.items():
            down, up = entry["reference_energies"]
            assert _level_with_spin_orbit(pp, l, -(l + 1) / 2) == \
                pytest.approx(down[0], abs=1e-4)
            assert _level_with_spin_orbit(pp, l, l / 2) == \
                pytest.approx(up[0], abs=1e-4)
            assert np.all(np.abs(entry["level_errors"]) < 1e-4)

    def test_the_splitting_is_the_dirac_atoms(self, oxygen_dirac):
        """0.28 % here; the first-order term it replaced was 0.8 % off for
        oxygen and 19-20 % for the 6p of Tl-Bi."""
        pp = oxygen_dirac
        split = (_level_with_spin_orbit(pp, 1, 0.5)
                 - _level_with_spin_orbit(pp, 1, -1.0))
        assert split == pytest.approx(pp.atom.spin_orbit_splitting(2, 1),
                                      rel=0.01)

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

    def test_it_survives_a_payload_round_trip(self, oxygen_dirac):
        back = from_payload(to_payload(oxygen_dirac))
        assert back.has_spin_orbit and back.relativity == "dirac"
        for l, entry in oxygen_dirac.spin_orbit.items():
            for key, value in entry.items():
                if key == "projectors":
                    assert all(np.allclose(a, b) for a, b in
                               zip(back.spin_orbit[l][key], value))
                else:
                    assert np.allclose(back.spin_orbit[l][key], value)

    def test_the_first_order_format_is_refused(self, oxygen_dirac):
        payload = to_payload(oxygen_dirac)
        payload["spin_orbit"] = {"1": [[0.01, 0.0], [0.0, 0.01]]}
        with pytest.raises(ValueError, match="first-order spin-orbit"):
            from_payload(payload)

    def test_a_strided_payload_keeps_the_projectors_on_its_grid(
            self, oxygen_dirac):
        """The union projectors are radial tables like every other, stored
        on the dataset's stride; they were once written at full length."""
        back = from_payload(to_payload(oxygen_dirac, stride=4))
        for entry in back.spin_orbit.values():
            assert all(p.size == back.r.size for p in entry["projectors"])

    def test_projectors_off_the_grid_are_refused(self, oxygen_dirac):
        payload = to_payload(oxygen_dirac, stride=4)
        entry = payload["spin_orbit"]["1"]
        entry["projectors"] = [p + p for p in entry["projectors"]]
        with pytest.raises(ValueError, match="radial grid"):
            from_payload(payload)

    def test_the_blocks_act_through_the_union_projectors(self, oxygen_dirac):
        position = np.zeros((1, 3))
        projectors = paw_spin_orbit_projectors(["O"], position,
                                               {"O": oxygen_dirac})
        n = len(oxygen_dirac.spin_orbit[1]["projectors"])
        assert len(projectors) == 3 * n and {p.l for p in projectors} == {1}
        blocks = paw_spin_orbit_blocks(projectors, ["O"], {"O": oxygen_dirac})
        assert set(blocks) == {(0, 1)}
        assert np.allclose(blocks[(0, 1)],
                           oxygen_dirac.spin_orbit[1]["coupling"])


    def test_a_dataset_without_the_fields_is_not_relabelled(self):
        """A payload written before these options existed is what it was."""
        payload = to_payload(generate_paw("H", points=3000, r_max=20.0))
        for key in ("xc", "relativity", "nlcc", "spin_orbit", "extra_l"):
            payload.pop(key, None)
        back = from_payload(payload)
        assert back.relativity == "none" and back.xc == "lda"
        assert not back.nlcc.get("applied") and not back.has_spin_orbit


class TestSpinOrbitInTheMolecularHamiltonian:
    """Option (a) of DIRAC.md: the scalar Hamiltonian plus the L.S term
    through the union projectors, on an O atom."""

    @pytest.fixture(scope="class")
    def integrals(self, oxygen_dirac):
        from ase import Atoms

        from mandacaru.pseudopotentials.paw import build_paw

        atoms = Atoms("O", positions=[[0.0, 0.0, 0.0]], cell=[6.0] * 3)
        atoms.center()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _h, _n, _m, _p, context = build_paw(
                atoms, None, 0.25, 0, None, {"size": "SZ"},
                loader=lambda symbol, directory: oxygen_dirac)
        return context["integrals"]

    def _p_block(self, integrals, matrix):
        l = np.array([f.l for f in integrals.basis])
        p = np.where(l == 1)[0]
        index = np.concatenate([p, p + l.size])
        return index, matrix[np.ix_(index, index)]

    def test_the_term_conserves_j_z(self, integrals):
        m = np.array([f.m for f in integrals.basis], dtype=float)
        index, block = self._p_block(integrals,
                                     integrals.spin_orbit_matrix())
        jz = np.concatenate([m + 0.5, m - 0.5])[index]
        commutator = block * (jz[None, :] - jz[:, None])
        # Only the cubic grid mixes m (6e-5 here at h = 0.25 Angstrom); a
        # wrong L.S or basis would break it at order one.
        assert np.linalg.norm(commutator) < 1e-3 * np.linalg.norm(block)

    def test_the_p_shell_splits_into_its_two_j_levels(self, integrals):
        """Doublet below quartet, centered on the scalar level: the term is
        in the basis the scalar one-body matrix is in (Loewdin)."""
        h = integrals.one_body()
        M = h.shape[0]
        full = np.zeros((2 * M, 2 * M), dtype=complex)
        full[:M, :M] = full[M:, M:] = h
        full += integrals.spin_orbit_matrix()
        index, block = self._p_block(integrals, full)
        levels = np.linalg.eigvalsh(block)
        assert np.allclose(levels[:2], levels[0], atol=1e-9)
        assert np.allclose(levels[2:], levels[2], atol=1e-9)
        assert levels[2] > levels[0]
        _i, scalar = self._p_block(integrals, full - integrals.spin_orbit_matrix())
        assert levels.mean() == pytest.approx(
            np.linalg.eigvalsh(scalar).mean(), abs=1e-9)



class TestKramersPairActiveSpaces:
    """A spin-orbit Hamiltonian is written in GHF spinor pairs, and its
    active space freezes and deletes pairs.  HI with the installed Dirac
    library (SZ): 18 electrons in 10 pairs; relations, not absolute values."""

    @staticmethod
    def _exact(active_space):
        from ase import Atoms

        from mandacaru.core.sector import ParticleSector
        from mandacaru.pseudopotentials.environment import library_folder
        from mandacaru.pseudopotentials.paw import FAMILY, build_paw

        atoms = Atoms("HI", positions=[[0, 0, 0], [0, 0, 1.609]],
                      cell=[9.0] * 3)
        atoms.center()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H, particles, pairs, _p, context = build_paw(
                atoms, None, 0.25, 0, None,
                {"size": "SZ",
                 "directory": library_folder(FAMILY, "lda-dirac")},
                active_space=active_space)
        sector = ParticleSector(2 * pairs, tuple(particles), "jordan_wigner",
                                spin_conserving=False)
        matrix = sector.restrict(H.map_to_qubits("jordan_wigner",
                                                 n_modes=2 * pairs))
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        return np.linalg.eigvalsh(matrix)[0], context["integrals"]

    def test_freezing_and_truncating_pairs_is_variational(self):
        full, integrals = self._exact(None)
        assert integrals.spinor_basis
        assert integrals.mo_coefficients.shape == (20, 20)
        one_frozen, _i = self._exact({"frozen": 1})
        two_frozen, _i = self._exact({"frozen": 2})
        determinant, _i = self._exact({"orbitals": 9})
        assert full < one_frozen < two_frozen < determinant
        assert one_frozen - full < 5e-4            # Hartree

    def test_a_ranking_of_spatial_orbitals_is_refused(self):
        with pytest.raises(NotImplementedError, match="Kramers pairs"):
            self._exact({"method": "mp2", "orbitals": 9})
