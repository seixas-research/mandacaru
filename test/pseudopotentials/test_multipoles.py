# -*- coding: utf-8 -*-
# file: test_multipoles.py

"""The PAW-LCAO compensation multipoles (:mod:`mandacaru.pseudopotentials.multipoles`).

Copper is the witness: its 3d channel is cut at 0.77 Bohr and its 4s at 2.78,
so every s-d moment integrates the d partial waves well past their own cutoff.
Until 2026-09-27 the smooth d wave there was the stored Bessel expansion, which
diverges outside ``r_cut``; the s-d moments came out ~100x too large and the
isolated copper atom collapsed to -16,000 Ha as the grid was refined.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
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


def test_compensation_keeps_the_copper_coulomb_tensor_positive():
    """``(n~ + n^ | n~ + n^)`` is a Coulomb energy: the augmentation may not
    push the tensor's spectrum below what the grid part alone gives."""
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[7.0] * 3)
    atoms.center()
    _H, _particles, n, _profile, context = build_basis_hamiltonian(
        atoms, {"name": "PAW-LCAO", "size": "SZ"}, None, 0.25, 0, None,
        spin=True)
    integrals = context["integrals"]
    grid = np.asarray(integrals._engine.two_body(energy_units="Ha"))
    total = grid + integrals.two_body_augmentation().dense()

    def lowest(tensor):
        matrix = tensor.transpose(0, 2, 1, 3).reshape(n * n, n * n)
        return np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min()

    assert lowest(total) > lowest(grid) - 0.05
    # The copper atom's 19 valence electrons (3s 3p 3d 4s since 2026-10-21;
    # (6, 5) was the eleven of the valence-only dataset): (10, 9) fills the
    # ten functions of this basis but one.  Window -60..-30 -> -215..-175
    # Ha (measured -195.9 Ha).
    energy = integrals.open_shell_hartree_fock(10, 9).electronic_energy
    assert -215.0 < energy < -175.0


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

    def test_the_multipole_coulomb_matrix(self):
        from mandacaru.basis._angular import real_harmonic_transform
        from mandacaru.pseudopotentials.multipoles import \
            multipole_coulomb_matrix

        levels = [(L, M) for L in range(3) for M in range(-L, L + 1)]
        displacement = np.array([0.9, -1.4, 2.1])
        real = multipole_coulomb_matrix(1.1, levels, 0.9, levels,
                                        displacement, real=True)
        complex_ = multipole_coulomb_matrix(1.1, levels, 0.9, levels,
                                            displacement, real=False)
        T = np.zeros((len(levels), len(levels)), dtype=complex)
        offset = 0
        for L in range(3):
            T[offset:offset + 2 * L + 1, offset:offset + 2 * L + 1] = \
                real_harmonic_transform(L)
            offset += 2 * L + 1
        assert np.isrealobj(real)
        assert np.abs(T @ complex_ @ T.T - real).max() < 1e-10
