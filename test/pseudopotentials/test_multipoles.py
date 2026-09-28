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
from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.multipoles import partial_waves, radial_moments


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


def test_compensation_keeps_the_copper_coulomb_tensor_positive():
    """``(n~ + n^ | n~ + n^)`` is a Coulomb energy: the augmentation may not
    push the tensor's spectrum below what the grid part alone gives."""
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[7.0] * 3)
    atoms.center()
    _H, _particles, n, _profile, context = build_basis_hamiltonian(
        atoms, {"name": "PAW-LCAO", "size": "SZ"}, None, 0.25, 0, None,
        spin=True)
    integrals = context["integrals"]
    grid = np.asarray(integrals._engine.two_body(method="fft",
                                                 energy_units="Ha"))
    total = grid + np.asarray(integrals.two_body_augmentation())

    def lowest(tensor):
        matrix = tensor.transpose(0, 2, 1, 3).reshape(n * n, n * n)
        return np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min()

    assert lowest(total) > lowest(grid) - 0.05
    energy = integrals.open_shell_hartree_fock(6, 5).electronic_energy
    assert -60.0 < energy < -30.0
