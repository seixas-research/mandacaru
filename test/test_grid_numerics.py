# -*- coding: utf-8 -*-
# file: test/test_grid_numerics.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Numerics that keep a real-space calculation variational.

Four things are pinned here, each found by chasing an interaction energy that
would not converge with the grid spacing:

* the Loewdin transform of the two-electron tensor must keep the tensor's
  physicists'-notation symmetries even when the overlap is complex (any
  molecule with p or d functions off a symmetry plane);
* the SCF must converge, and to the lowest of the stationary points it can
  reach (DIIS, level shift, two starting points);
* the spectral kinetic operator must be exact for resolved functions and
  never *under*-estimate an unresolved one (the finite-difference stencil
  does, which lets compact functions collapse into deep potentials);
* extra zetas are normalized, and the grid-resolution check flags functions
  the grid cannot represent.
"""

import warnings

import numpy as np
import pytest
from ase.build import molecule

from carcara.algorithms import RHF, UHF
from carcara.algorithms._hamiltonian_from_atoms import (build_basis_hamiltonian,
                                                        coherent_positions)
from carcara.basis import BasisSet, GaussianOrbital
from carcara.basis.multizeta import zeta_tables
from carcara.basis.nao import solve_confined_radial
from carcara.core import MolecularIntegrals
from carcara.integrals import Grid, IntegralEngine


def _water_integrals(h=0.30, orthogonalize=True, **kw):
    water = molecule("H2O"); water.set_cell([8.0, 8.0, 8.0]); water.center()
    pos = coherent_positions(water); bset = BasisSet.build("FAO")
    fns, nuclei = [], []
    for Z, s, p in zip(water.get_atomic_numbers(), water.get_chemical_symbols(), pos):
        fns += bset.atom(s, center=p); nuclei.append((float(Z), p))
    grid = Grid(center=water.cell.diagonal() / 2, box_size=0.0, h=h,
                units="angstrom", cell=water.cell)
    return MolecularIntegrals(nuclei, fns, grid, softening=0.3,
                              orthogonalize=orthogonalize, **kw)


def _symmetry_defects(t):
    """(swap, hermiticity) defects of a physicists' <pq|rs> tensor."""
    return (float(np.abs(t - t.transpose(1, 0, 3, 2)).max()),
            float(np.abs(t.conj() - t.transpose(2, 3, 0, 1)).max()))


# --------------------------------------------------------------------------- #
# The Loewdin transform with a complex overlap.
# --------------------------------------------------------------------------- #

class TestLoewdinTransform:
    def test_water_overlap_is_complex(self):
        mi = _water_integrals(orthogonalize=False)
        assert np.abs(mi.overlap().imag).max() > 1e-3
        assert np.abs(mi._lowdin_x().imag).max() > 1e-3

    def test_transformed_tensor_keeps_its_symmetries(self):
        mi = _water_integrals()
        swap, herm = _symmetry_defects(mi.two_body())
        assert swap < 1e-10 and herm < 1e-10

    def test_wrong_conjugation_pattern_would_break_them(self):
        # The historical (X*, X, X*, X) pattern: documented so the regression
        # is recognizable if it ever comes back.
        mi = _water_integrals(orthogonalize=False)
        X = mi._lowdin_x()
        wrong = np.einsum("ap,bq,cr,ds,abcd->pqrs", X.conj(), X, X.conj(), X,
                          mi.two_body(), optimize=True)
        assert _symmetry_defects(wrong)[0] > 1e-3

    def test_scf_energy_equals_the_raw_ao_energy_of_its_density(self):
        """The decisive check: the orthonormal-basis SCF energy must equal
        the energy of the same density evaluated directly with the raw
        (non-orthogonal) AO matrices, with no transform in between.  Both
        latent bugs -- the transform's conjugation pattern and the Fock
        build's transposed density -- broke this for complex orbitals."""
        raw = _water_integrals(orthogonalize=False)
        orth = _water_integrals()
        r = orth.hartree_fock(10)
        c = orth._lowdin_x() @ r.mo_coefficients[:, :5]        # AO coefficients
        S = raw.overlap()
        assert np.allclose(c.conj().T @ S @ c, np.eye(5), atol=1e-10)
        P = c @ c.conj().T
        h, eri = raw.one_body(), raw.two_body()
        e_ao = (2 * np.einsum("ab,ba", h, P)
                + 2 * np.einsum("abcd,ca,db", eri, P, P)
                - np.einsum("abcd,da,cb", eri, P, P))
        assert np.real(e_ao) == pytest.approx(r.electronic_energy, abs=1e-8)

    def test_scf_energy_is_invariant_under_a_complex_unitary(self):
        """Mixing the orthonormal basis with a random complex unitary must
        leave the Hartree-Fock energy unchanged."""
        mi = _water_integrals()
        h, eri = mi.one_body(), mi.two_body()
        e0 = RHF(h, eri, 10).run().electronic_energy
        rng = np.random.default_rng(1)
        A = rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)
        U, _ = np.linalg.qr(A)
        h2 = U.conj().T @ h @ U
        eri2 = np.einsum("ap,bq,cr,ds,abcd->pqrs", U.conj(), U.conj(), U, U,
                         eri, optimize=True)
        assert RHF(h2, eri2, 10).run().electronic_energy == pytest.approx(e0, abs=1e-7)


# --------------------------------------------------------------------------- #
# SCF convergence.
# --------------------------------------------------------------------------- #

class TestSCF:
    def test_diis_converges_and_agrees_with_plain_iteration(self):
        mi = _water_integrals()
        h, eri = mi.one_body(), mi.two_body()
        plain = RHF(h, eri, 10).run(diis=False, guesses=1)
        diis = RHF(h, eri, 10).run()
        assert diis.converged
        assert diis.electronic_energy <= plain.electronic_energy + 1e-8
        if plain.converged:
            assert diis.electronic_energy == pytest.approx(
                plain.electronic_energy, abs=1e-6)

    def test_second_guess_never_raises_the_energy(self):
        mi = _water_integrals()
        h, eri = mi.one_body(), mi.two_body()
        one = RHF(h, eri, 10).run(guesses=1)
        two = RHF(h, eri, 10).run(guesses=2)
        assert two.electronic_energy <= one.electronic_energy + 1e-10

    def test_uhf_matches_rhf_for_a_closed_shell(self):
        mi = _water_integrals()
        h, eri = mi.one_body(), mi.two_body()
        rhf = RHF(h, eri, 10).run()
        uhf = UHF(h, eri, 5, 5).solve()
        assert uhf.converged
        assert uhf.electronic_energy == pytest.approx(rhf.electronic_energy,
                                                      abs=1e-6)

    def test_occupied_orbital_energies_are_physical(self):
        # Water's valence levels sit between -1.5 and -0.2 Ha; the broken
        # transform produced a positive occupied level here.
        r = _water_integrals(h=0.25).hartree_fock(10)
        assert np.all(r.mo_energies[:4] < 0.0)


# --------------------------------------------------------------------------- #
# Spectral kinetic operator.
# --------------------------------------------------------------------------- #

class TestSpectralKinetic:
    @staticmethod
    def _ratios(alpha, h, box=6.0):
        g = GaussianOrbital(0, 0, [alpha], [1.0])
        grid = Grid(center=[0, 0, 0], box_size=box, h=h)
        eng = IntegralEngine([g], grid)
        zero = lambda x, y, z: np.zeros_like(x)
        exact = 1.5 * alpha
        fd = eng.one_body(zero, "Ha", "fd")[0][0, 0].real / exact
        sp = eng.one_body(zero, "Ha", "spectral")[0][0, 0].real / exact
        return fd, sp

    def test_exact_for_a_resolved_function(self):
        fd, sp = self._ratios(0.5, 0.20)
        assert sp == pytest.approx(1.0, abs=2e-3)
        assert fd < 0.99                              # the stencil is biased low

    def test_never_underestimates_an_unresolved_function(self):
        for alpha, h in ((2.0, 0.30), (8.0, 0.20), (8.0, 0.30)):
            fd, sp = self._ratios(alpha, h)
            assert sp >= 1.0 - 1e-6
            assert sp >= fd

    def test_hermitian_and_agrees_with_fd_on_smooth_functions(self):
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.25)
        fns = [GaussianOrbital(0, 0, [0.4], [1.0], center=[0, 0, 0]),
               GaussianOrbital(1, 0, [0.6], [1.0], center=[0.3, 0, 0])]
        eng = IntegralEngine(fns, grid)
        zero = lambda x, y, z: np.zeros_like(x)
        T_fd = eng.one_body(zero, "Ha", "fd")[0]
        T_sp = eng.one_body(zero, "Ha", "spectral")[0]
        assert np.allclose(T_sp, T_sp.conj().T)
        assert np.allclose(T_sp, T_fd, rtol=0.1, atol=0.02)

    def test_requires_an_orthogonal_grid_and_known_name(self):
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.4)
        eng = IntegralEngine([GaussianOrbital(0, 0, [0.5], [1.0])], grid)
        with pytest.raises(ValueError):
            eng.one_body(lambda x, y, z: np.zeros_like(x), "Ha", "fft")
        with pytest.raises(ValueError):
            MolecularIntegrals([(1.0, np.zeros(3))],
                               [GaussianOrbital(0, 0, [0.5], [1.0])], grid,
                               kinetic="fourier")

    def test_drivers_accept_the_option(self):
        from ase import Atoms
        from carcara.algorithms import Carcara
        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
        h2.calc = Carcara(method="vqe", basis="FAO", h=0.35, kinetic="spectral", optimizer="L-BFGS-B")
        e_sp = h2.get_potential_energy()
        h2.calc = Carcara(method="vqe", basis="FAO", h=0.35, kinetic="fd", optimizer="L-BFGS-B")
        e_fd = h2.get_potential_energy()
        assert abs(e_sp - e_fd) < 2.0                # same physics, finite grid
        from carcara.algorithms import VQE
        with pytest.raises(ValueError):
            VQE(kinetic="fourier", verbose=False)


# --------------------------------------------------------------------------- #
# Normalized zetas and the resolution check.
# --------------------------------------------------------------------------- #

class TestZetasAndResolution:
    def test_extra_zetas_are_normalized(self):
        r, R, _e = solve_confined_radial(1, 0, 1.0, 5.0)
        for table in zeta_tables(r, R, 1, 0, 3):
            assert np.trapezoid(table.values ** 2 * r * r, r) == pytest.approx(
                1.0, rel=1e-6)

    def test_resolution_ratios_flag_a_compact_function(self):
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.30)
        fns = [GaussianOrbital(0, 0, [0.4], [1.0]),
               GaussianOrbital(0, 0, [12.0], [1.0])]          # width 0.1 A
        eng = IntegralEngine(fns, grid)
        ratios = eng.resolution(kinetic="fd")
        assert abs(ratios[0] - 1.0) < 0.1
        assert abs(ratios[1] - 1.0) > 0.25
        mi = MolecularIntegrals([(1.0, np.zeros(3))], fns, grid)
        bad_fns, bad_proj = mi.unresolved()
        assert bad_fns == [1] and bad_proj == []

    def test_builder_warns_about_unresolved_functions(self):
        from ase import Atoms
        # A bare oxygen 1s (FAO, Z = 8) on a 0.4 A grid is unresolvable.
        atoms = Atoms("O", positions=[[0, 0, 0]], cell=[6.0] * 3)
        with pytest.warns(RuntimeWarning, match="does not resolve"):
            build_basis_hamiltonian(atoms, "FAO", None, 0.40, 0, None)

    def test_resolved_basis_raises_no_warning(self):
        from ase import Atoms
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            build_basis_hamiltonian(atoms, "FAO", None, 0.30, 0, None)
