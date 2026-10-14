# -*- coding: utf-8 -*-
# file: test/integrals/test_two_center.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Analytic two-center integrals (``integrals/two_center.py``).

The module is checked against what it replaces on the real-space grid, on
functions where the grid is exact (smooth Gaussians sampled finely) and on a
real PAW-LCAO basis, and its gradient against central differences of its own
matrices.  The calculator-level behavior of ``two_center="analytic"`` (energy,
forces, refusals) is in ``test/algorithms/test_dft.py``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.basis._angular import spherical_harmonic
from mandacaru.basis.base import BasisFunction
from mandacaru.integrals import Grid, IntegralEngine
from mandacaru.integrals import two_center as tc
from mandacaru.integrals.two_center import (TwoCenter, multipoles, real_gaunt,
                                            validate_two_center)

#: Gaussian exponent of the test functions: negligible (1e-16) beyond 5 Bohr,
#: and so smooth that a 0.12 Bohr grid integrates them to round-off.
EXPONENT = 1.5
CUT = 5.0


class Gaussian(BasisFunction):
    """``r^l exp(-a r^2) S_lm``, real, cut at :data:`CUT` Bohr."""

    real = True

    def __init__(self, l, m, center, exponent=EXPONENT):
        self.l, self.m = l, m
        self.center = np.asarray(center, dtype=float)
        self.exponent = exponent
        self.r_c = CUT

    def radial(self, r):
        r = np.asarray(r, dtype=float)
        return np.where(r < self.r_c, r ** self.l * np.exp(-self.exponent * r * r),
                        0.0)

    def evaluate(self, x, y, z):
        d = [np.asarray(c, dtype=float) - o
             for c, o in zip((x, y, z), self.center)]
        r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        safe = np.where(r < 1e-15, 1e-15, r)
        theta = np.arccos(np.clip(d[2] / safe, -1.0, 1.0))
        phi = np.arctan2(d[1], d[0])
        return self.radial(r) * spherical_harmonic(self.l, self.m, theta, phi,
                                                   real=True)


def shell(l, center, exponent=EXPONENT):
    return [Gaussian(l, m, center, exponent) for m in range(-l, l + 1)]


def trimer(shift=0.0):
    """s, p, d and f shells on three centers (one displaced by ``shift``)."""
    return (shell(0, [0.0, 0.0, 0.0]) + shell(1, [0.3, -0.2, 1.1 + shift])
            + shell(2, [-1.2, 0.9, 0.4]) + shell(3, [0.5, 1.0, -0.8])
            + shell(1, [0.0, 0.0, 0.0], 1.1))


class TestGaunt:
    def test_the_monopole_coupling_is_the_orthonormality_of_the_harmonics(self):
        for l in (0, 1, 2, 3):
            G = real_gaunt(l, l, 0)[:, :, 0]
            assert np.allclose(G, np.eye(2 * l + 1) / np.sqrt(4 * np.pi),
                               atol=1e-13)

    def test_a_closed_form(self):
        # int S_10 S_10 S_20 = 1/sqrt(5 pi)
        assert real_gaunt(1, 1, 2)[1, 1, 2] == pytest.approx(
            1.0 / np.sqrt(5.0 * np.pi), abs=1e-13)

    def test_it_is_symmetric_and_obeys_the_selection_rules(self):
        G = real_gaunt(1, 2, 3)
        assert np.allclose(G, real_gaunt(2, 1, 3).transpose(1, 0, 2))
        assert not np.any(real_gaunt(1, 1, 1))           # odd l1 + l2 + L
        assert not np.any(real_gaunt(0, 1, 3))           # triangle rule
        assert multipoles(1, 2) == [1, 3]


def _grid_matrices(functions, spacing=0.12, half_edge=7.5):
    grid = Grid(center=[0.3, 0.2, 0.3], box_size=half_edge, grid_spacing=spacing,
                units="bohr")
    engine = IntegralEngine(functions, grid)
    T, _ = engine.one_body(lambda x, y, z: np.zeros_like(x), energy_units="Ha",
                           kinetic="spectral")
    S = engine._psi @ engine._psi.T * grid.dV
    return S, np.real(T)


class TestAgainstTheGrid:
    """Smooth functions, a grid fine enough to be exact: the analytic
    integrals must be the grid's to the tables' own accuracy."""

    def test_overlap_and_kinetic_energy(self):
        functions = trimer()
        S, T = _grid_matrices(functions)
        tables = TwoCenter(functions)
        assert np.abs(tables.matrix("overlap") - S).max() < 1e-9
        assert np.abs(tables.matrix("kinetic") - T).max() < 1e-8

    def test_they_are_symmetric(self):
        tables = TwoCenter(trimer())
        for kind in tc.KINDS:
            matrix = tables.matrix(kind)
            assert np.allclose(matrix, matrix.T, atol=1e-13)

    def test_bra_and_ket_sets_are_independent(self):
        functions = trimer()
        full = TwoCenter(functions).matrix("overlap")
        part = TwoCenter(functions[:4], functions[4:]).matrix("overlap")
        assert np.allclose(part, full[:4, 4:], atol=1e-13)

    def test_functions_out_of_reach_do_not_overlap(self):
        far = shell(1, [0.0, 0.0, 2.0 * CUT + 0.5])
        matrix = TwoCenter(shell(0, [0.0, 0.0, 0.0]) + far).matrix("overlap")
        assert np.all(matrix[0, 1:] == 0.0)


def _fd_gradient(build, kind, step=1e-4):
    """Central difference of the matrix with the displacement of the second
    center (``build(z)`` places it at ``z`` along z)."""
    return (TwoCenter(build(step)).matrix(kind)
            - TwoCenter(build(-step)).matrix(kind)) / (2.0 * step)


class TestGradient:
    @pytest.mark.parametrize("kind", tc.KINDS)
    def test_it_is_the_derivative_of_the_matrix(self, kind):
        """``gradient`` is d/d(R_ket - R_bra): moving the p shell's center
        along z changes its rows by minus it and its columns by plus it."""
        def build(z):
            return (shell(0, [0.0, 0.0, 0.0]) + shell(1, [0.3, -0.2, 1.1 + z])
                    + shell(2, [-1.2, 0.9, 0.4]))

        functions = build(0.0)
        slope = TwoCenter(functions).gradient(kind)[2]
        moved = np.zeros(len(functions), dtype=bool)
        moved[1:4] = True                       # the p shell
        analytic = (np.where(moved[:, None], -slope, 0.0)
                    + np.where(moved[None, :], slope, 0.0))
        assert np.abs(analytic - _fd_gradient(build, kind)).max() < 1e-8

    def test_a_coincident_dipole_pair_has_a_gradient(self):
        """An s and a p function on one center: their overlap is zero but
        moving the s function alone is not -- the dipole term at zero
        separation (`origin`), which a rigid translation cancels."""
        def build(z):
            return shell(0, [0.0, 0.0, z]) + shell(1, [0.0, 0.0, 0.0], 1.1)

        slope = TwoCenter(build(0.0)).gradient("overlap")
        assert abs(slope[2, 0, 2]) > 0.1              # p_z (M = 0)
        assert abs(slope[2, 0, 1]) < 1e-12            # p_y does not couple
        step = 1e-4
        fd = (TwoCenter(build(step)).matrix("overlap")[0, 1:]
              - TwoCenter(build(-step)).matrix("overlap")[0, 1:]) / (2 * step)
        # the s center is the bra; moving it by z changes d = R_ket - R_bra by -z
        assert np.allclose(-slope[2, 0, 1:], fd, atol=1e-8)


class TestTheFunctionSets:
    def test_a_complex_function_is_refused(self):
        f = Gaussian(0, 0, [0, 0, 0])
        f.real = False
        with pytest.raises(ValueError, match="real"):
            TwoCenter([f])

    def test_a_function_without_a_radial_part_is_refused(self):
        class Bare(BasisFunction):
            real = True
            center = np.zeros(3)

            def evaluate(self, x, y, z):
                return np.zeros_like(x)

        with pytest.raises(ValueError, match="radial"):
            TwoCenter([Bare()])

    def test_an_unbounded_function_is_refused(self):
        f = Gaussian(0, 0, [0, 0, 0])
        f.r_c = None
        with pytest.raises(ValueError, match="finite support"):
            TwoCenter([f])

    def test_an_incomplete_shell_is_refused(self):
        with pytest.raises(ValueError, match="complete shells"):
            TwoCenter(shell(1, [0, 0, 0])[:2])

    def test_the_tables_are_cached_by_content(self):
        TwoCenter(trimer()).matrix("overlap")
        known = len(tc._TABLES)
        TwoCenter(trimer(shift=0.4)).matrix("overlap")     # a new geometry
        assert len(tc._TABLES) == known


class TestTheOption:
    def test_it_validates(self):
        assert validate_two_center(None) == "grid"
        assert validate_two_center(" Analytic ") == "analytic"
        with pytest.raises(ValueError, match="two_center"):
            validate_two_center("exact")


    def test_the_integrals_refuse_a_crystal_and_an_explicit_kinetic_operator(self):
        from mandacaru.core import MolecularIntegrals

        grid = Grid(center=[0, 0, 0], box_size=2.0, grid_spacing=0.5)
        nuclei = [(1.0, [0.0, 0.0, 0.0])]
        with pytest.raises(NotImplementedError, match="molecular"):
            MolecularIntegrals(nuclei, [], grid, two_center="analytic",
                               periodic=True)
        with pytest.raises(ValueError, match="kinetic"):
            MolecularIntegrals(nuclei, [], grid, two_center="analytic",
                               kinetic="fd")


class TestPAWLCAO:
    """The real thing: a filtered PAW-LCAO DZP basis and its projectors."""

    @pytest.fixture(scope="class")
    def h2(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import grid_from_cell
        from mandacaru.pseudopotentials.families import (
            pseudo_basis_arguments, resolve_family)
        from mandacaru.pseudopotentials.orbitals import pseudo_basis
        from mandacaru.pseudopotentials.paw import get_paw, paw_projectors

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.03, 0.02, 0.74]],
                      cell=[8.0] * 3)
        atoms.center()
        options = resolve_family("paw-lcao").resolved_options({"size": "DZP"})
        potentials = {"H": get_paw("H", None)}
        symbols, positions = atoms.get_chemical_symbols(), atoms.get_positions()
        # Filtered at the cutoff of a 0.2 Angstrom grid, whatever grid is
        # compared against.
        functions, _owners = pseudo_basis(
            symbols, positions, potentials, filter_wavevector=8.3123,
            **pseudo_basis_arguments("paw-lcao", options))
        projectors = paw_projectors(symbols, positions, potentials)
        return atoms, functions, projectors, grid_from_cell(atoms, 0.15)

    def test_overlap_and_kinetic_energy_are_the_grid_s(self, h2):
        _atoms, functions, _projectors, grid = h2
        engine = IntegralEngine(functions, grid)
        T, _ = engine.one_body(lambda x, y, z: np.zeros_like(x),
                               energy_units="Ha", kinetic="spectral")
        S = engine._psi @ engine._psi.T * grid.dV
        tables = TwoCenter(functions)
        assert np.abs(tables.matrix("overlap") - S).max() < 5e-7
        assert np.abs(tables.matrix("kinetic") - np.real(T)).max() < 5e-7

    def test_the_projections_are_the_quadrature_s(self, h2):
        from mandacaru.pseudopotentials.paw import (
            atom_centered_projection_gradients, atom_centered_projections)

        _atoms, functions, projectors, _grid = h2
        tables = TwoCenter(functions, projectors)
        C = tables.matrix("overlap")
        assert np.abs(C - atom_centered_projections(functions, projectors)
                      ).max() < 1e-6
        # <d phi/d R| p> is minus the gradient with respect to the
        # separation, projector minus function
        G = np.moveaxis(-tables.gradient("overlap"), 0, -1)
        assert np.abs(G - atom_centered_projection_gradients(
            functions, projectors)).max() < 1e-5

    def test_the_tables_converge_in_their_numerical_parameters(self, h2):
        """Doubling the wave-number range and halving the table spacing
        moves nothing above the parameters' own accuracy."""
        _atoms, functions, projectors, _grid = h2
        reference = [TwoCenter(functions).matrix("kinetic"),
                     TwoCenter(functions, projectors).matrix("overlap")]
        saved = (tc.K_MAX, tc.TABLE_STEP, tc._K, tc._KW)
        try:
            tc.K_MAX, tc.TABLE_STEP = 2 * saved[0], 0.5 * saved[1]
            tc._K, tc._KW = tc._wave_numbers()
            for cache in (tc._BESSEL, tc._TRANSFORMS, tc._TABLES):
                cache.clear()
            refined = [TwoCenter(functions).matrix("kinetic"),
                       TwoCenter(functions, projectors).matrix("overlap")]
        finally:
            tc.K_MAX, tc.TABLE_STEP, tc._K, tc._KW = saved
            for cache in (tc._BESSEL, tc._TRANSFORMS, tc._TABLES):
                cache.clear()
        for a, b in zip(reference, refined):
            assert np.abs(a - b).max() < 1e-7
