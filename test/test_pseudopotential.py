# -*- coding: utf-8 -*-
# file: test/test_pseudopotential.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Norm-conserving pseudopotentials: atomic solver, Troullier-Martins, KB.

The tests follow the physics that defines a valid pseudopotential:

* the all-electron reference reproduces published LDA eigenvalues;
* the pseudo-orbital **conserves norm** inside the cutoff (what makes the
  potential transferable) and is identical to the all-electron orbital outside;
* it is **nodeless**, and the potential that supports it is **finite at the
  origin** -- the property the whole exercise exists for;
* re-solving in the pseudopotential returns the all-electron eigenvalue;
* the Kleinman-Bylander separable form reproduces the semilocal channel.

The last class measures the payoff: how much the grid sensitivity that blocks
geometry optimization (see :mod:`carcara.algorithms.forces`) actually improves.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcara.basis.atomic_solver import (hartree_potential, lda_xc, solve_atom,
                                         solve_radial)
from carcara.basis.pseudopotential import (TM_POWERS, PseudoPotential,
                                           check_channel,
                                           generate_pseudopotential,
                                           pseudize_channel, report)
from carcara.units import BOHR_TO_ANGSTROM

#: NIST LSD reference eigenvalues (Hartree) for closed-shell atoms.
NIST_EIGENVALUES = {
    2: {(1, 0): -0.570425},
    4: {(1, 0): -3.856411, (2, 0): -0.205719},
}


@pytest.fixture(scope="module")
def oxygen_atom():
    return solve_atom(8, points=6000, r_max=30.0, tolerance=1e-7, mixing=0.25)


@pytest.fixture(scope="module")
def oxygen_pp(oxygen_atom):
    return generate_pseudopotential("O", atom=oxygen_atom)


# --------------------------------------------------------------------------- #
# All-electron atom.
# --------------------------------------------------------------------------- #

class TestAtomicSolver:
    @pytest.mark.parametrize("atomic_number", sorted(NIST_EIGENVALUES))
    def test_matches_published_lda_eigenvalues(self, atomic_number):
        """Closed-shell atoms, where spin-restricted LDA is the right model."""
        atom = solve_atom(atomic_number, points=6000, r_max=30.0,
                          tolerance=1e-7, mixing=0.25)
        assert atom.converged
        for state, reference in NIST_EIGENVALUES[atomic_number].items():
            assert atom.eigenvalues[state] == pytest.approx(reference, abs=5e-3)

    def test_valence_eigenvalue_is_accurate(self, oxygen_atom):
        """The valence shell is what a pseudopotential is built from."""
        assert oxygen_atom.eigenvalues[(2, 1)] == pytest.approx(-0.338381,
                                                                abs=2e-3)

    def test_density_integrates_to_the_electron_count(self, oxygen_atom):
        r = oxygen_atom.r
        electrons = np.trapezoid(4.0 * np.pi * oxygen_atom.density * r * r, r)
        assert electrons == pytest.approx(8.0, rel=2e-3)

    def test_orbitals_have_the_right_node_count(self, oxygen_atom):
        for (n, l), u in oxygen_atom.orbitals.items():
            interior = u[(oxygen_atom.r > 0.05) & (oxygen_atom.r < 12.0)]
            nodes = int(np.sum(np.diff(np.sign(interior)) != 0))
            assert nodes == n - l - 1, f"state {(n, l)} has {nodes} nodes"

    def test_hartree_potential_of_a_point_charge(self):
        """A tight spherical blob must look like ``1/r`` outside itself."""
        r = np.linspace(1e-3, 20.0, 4000)
        width = 0.2
        rho = np.exp(-(r / width) ** 2) / (np.pi ** 1.5 * width ** 3)
        potential = hartree_potential(r, rho)
        far = r > 2.0
        assert np.allclose(potential[far], 1.0 / r[far], rtol=1e-4)

    def test_lda_xc_is_negative(self):
        e_xc, v_xc = lda_xc(np.array([0.01, 0.1, 1.0]))
        assert np.all(e_xc < 0) and np.all(v_xc < 0)


# --------------------------------------------------------------------------- #
# Troullier-Martins construction.
# --------------------------------------------------------------------------- #

class TestTroullierMartins:
    @pytest.mark.parametrize("symbol", ["H", "Li", "Be", "C", "N", "O", "F"])
    def test_generates_across_the_first_rows(self, symbol):
        pp = generate_pseudopotential(symbol)
        assert pp.channels
        assert np.isfinite(pp.v_local).all()

    def test_norm_is_conserved(self, oxygen_pp):
        """The defining property: transferability rests on it."""
        for l in oxygen_pp.channels:
            assert check_channel(oxygen_pp, l)["norm_error"] < 1e-8

    def test_tail_matches_the_all_electron_orbital(self, oxygen_pp):
        """Outside r_c the pseudo-orbital *is* the all-electron orbital."""
        for l in oxygen_pp.channels:
            assert check_channel(oxygen_pp, l)["tail_error"] < 1e-12

    def test_pseudo_orbitals_are_nodeless(self, oxygen_pp):
        """A node inside r_c would defeat the purpose -- it needs a fine grid."""
        for l in oxygen_pp.channels:
            assert check_channel(oxygen_pp, l)["nodes"] == 0

    def test_eigenvalue_is_reproduced(self, oxygen_pp):
        """Re-solving in the screened pseudopotential returns the AE eigenvalue."""
        for l in oxygen_pp.channels:
            error = check_channel(oxygen_pp, l)["eigenvalue_error"]
            assert abs(error) < 1e-3, f"l={l}: {error}"

    def test_potential_is_finite_at_the_origin(self, oxygen_pp):
        """The entire point: no -Z/r singularity left to under-resolve."""
        assert np.isfinite(oxygen_pp.v_local[0])
        assert abs(oxygen_pp.v_local[0]) < 100.0
        # The all-electron potential at the same radius is far deeper.
        all_electron = -8.0 / oxygen_pp.r[0]
        assert abs(oxygen_pp.v_local[0]) < 0.1 * abs(all_electron)

    def test_potential_is_flat_at_the_origin(self, oxygen_pp):
        """The curvature condition c2^2 + (2l+5)c4 = 0 makes V'(0) vanish."""
        for l, channel in oxygen_pp.channels.items():
            c2, c4 = channel.coefficients[1], channel.coefficients[2]
            assert c2 ** 2 + (2 * l + 5) * c4 == pytest.approx(0.0, abs=1e-8)

    def test_local_potential_decays_to_the_ionic_tail(self, oxygen_pp):
        """Far out the pseudopotential must be -Z_ion/r, not -Z/r."""
        r = oxygen_pp.r
        far = (r > 6.0) & (r < 12.0)
        expected = -oxygen_pp.valence_charge / r[far]
        assert np.allclose(oxygen_pp.v_local[far], expected, atol=2e-2)

    def test_valence_charge_excludes_the_core(self, oxygen_pp):
        assert oxygen_pp.valence_charge == 6.0        # O: 1s^2 frozen out
        assert oxygen_pp.atomic_number == 8

    def test_rejects_an_impossible_cutoff(self, oxygen_atom):
        """A cutoff inside the core cannot satisfy the seven conditions."""
        with pytest.raises(RuntimeError, match="Troullier-Martins fit failed"):
            pseudize_channel(oxygen_atom, 2, 1, r_cut=0.02)

    def test_report_is_informative(self, oxygen_pp):
        text = report(oxygen_pp)
        assert "norm err" in text and "E_KB" in text


# --------------------------------------------------------------------------- #
# Kleinman-Bylander separable form.
# --------------------------------------------------------------------------- #

class TestKleinmanBylander:
    def test_projectors_exist_for_the_non_local_channels(self, oxygen_pp):
        assert oxygen_pp.local_l in oxygen_pp.channels
        assert oxygen_pp.local_l not in oxygen_pp.projectors
        assert set(oxygen_pp.projectors) == set(oxygen_pp.channels) - {
            oxygen_pp.local_l}

    def test_kb_energies_are_finite(self, oxygen_pp):
        for l, energy in oxygen_pp.kb_energies.items():
            assert np.isfinite(energy) and energy != 0.0

    def test_projector_is_localized(self, oxygen_pp):
        """chi_l = dV_l R_ps vanishes beyond r_c, since dV_l does."""
        for l, chi in oxygen_pp.projectors.items():
            outside = oxygen_pp.r > oxygen_pp.channels[l].r_cut * 1.5
            assert np.max(np.abs(chi[outside])) < 1e-10

    def test_kb_reproduces_the_semilocal_expectation(self, oxygen_pp):
        r"""``<R|chi> E_KB <chi|R> == <R|dV|R>`` for the reference orbital.

        This is the identity the separable form is built on: acting on the
        orbital it was constructed from, KB is exact.
        """
        r = oxygen_pp.r
        for l, chi in oxygen_pp.projectors.items():
            channel = oxygen_pp.channels[l]
            radial = channel.pseudo_radial
            delta_v = channel.v_ionic - oxygen_pp.v_local

            overlap = np.trapezoid(radial * chi * r * r, r)
            separable = overlap * oxygen_pp.kb_energies[l] * overlap
            semilocal = np.trapezoid(radial * delta_v * radial * r * r, r)
            assert separable == pytest.approx(semilocal, rel=1e-8)

    def test_interpolation_helpers(self, oxygen_pp):
        probe = np.array([0.5, 1.0, 3.0])
        assert np.isfinite(oxygen_pp.local_potential(probe)).all()
        # Far outside, the local potential is the ionic tail.
        assert oxygen_pp.local_potential(np.array([40.0]))[0] == \
            pytest.approx(-oxygen_pp.valence_charge / 40.0, rel=1e-9)
        for l in oxygen_pp.projectors:
            assert oxygen_pp.projector(l, np.array([50.0]))[0] == 0.0

    def test_local_channel_choice(self, oxygen_atom):
        pp = generate_pseudopotential("O", local_l=0, atom=oxygen_atom)
        assert pp.local_l == 0 and set(pp.projectors) == {1}

    def test_invalid_local_channel_rejected(self, oxygen_atom):
        with pytest.raises(ValueError, match="not among the channels"):
            generate_pseudopotential("O", local_l=3, atom=oxygen_atom)


# --------------------------------------------------------------------------- #
# The payoff: grid sensitivity.
# --------------------------------------------------------------------------- #

class TestGridSensitivity:
    """Does the pseudopotential actually cure what blocked geometry optimization?

    The all-electron oxygen core produces an egg-box error -- the energy changes
    when the atom is translated rigidly across the grid -- that swamps chemistry
    and does not converge with refinement.  These tests measure the same quantity
    for the pseudopotential.
    """

    @staticmethod
    def _egg_box(pp, spacing_angstrom, all_electron=False):
        """Range of ``int rho V`` over sub-grid offsets; exact answer is zero."""
        spacing = spacing_angstrom / BOHR_TO_ANGSTROM
        axis = (np.arange(31) - 15) * spacing
        X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
        values = []
        for fraction in np.linspace(0.0, 1.0, 5)[:-1]:
            shift = spacing * fraction
            radius = np.sqrt((X - shift) ** 2 + (Y - shift) ** 2
                             + (Z - shift) ** 2)
            if all_electron:
                density = np.interp(radius, pp.atom.r, pp.atom.density,
                                    left=pp.atom.density[0], right=0.0)
                # The engine's regularization: -Z/max(r, half a grid step).
                potential = -pp.atomic_number / np.maximum(radius,
                                                           0.5 * spacing)
            else:
                density = np.interp(radius, pp.r, pp.valence_density,
                                    left=pp.valence_density[0], right=0.0)
                potential = pp.local_potential(radius)
            values.append(float(np.sum(density * potential) * spacing ** 3))
        return max(values) - min(values)

    def test_far_smaller_than_all_electron(self, oxygen_pp):
        pseudo = self._egg_box(oxygen_pp, 0.20)
        all_electron = self._egg_box(oxygen_pp, 0.20, all_electron=True)
        assert pseudo < all_electron / 100.0, (pseudo, all_electron)

    def test_converges_with_the_grid(self, oxygen_pp):
        """Unlike the all-electron case, refining the grid now *helps*."""
        coarse = self._egg_box(oxygen_pp, 0.20)
        fine = self._egg_box(oxygen_pp, 0.15)
        assert fine < coarse / 2.0, (coarse, fine)

    def test_potential_is_bounded_independently_of_the_grid(self, oxygen_pp):
        """|V| at the nearest node saturates, instead of growing as 1/h."""
        depths = []
        for spacing_angstrom in (0.30, 0.20, 0.15, 0.10):
            spacing = spacing_angstrom / BOHR_TO_ANGSTROM
            depths.append(abs(float(
                oxygen_pp.local_potential(np.array([0.5 * spacing]))[0])))
        assert max(depths) < 2.0 * min(depths)
