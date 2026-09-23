# -*- coding: utf-8 -*-
# file: test/algorithms/test_kpath.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Band paths over a mesh: :mod:`mandacaru.algorithms.kpath`.

The invariant these tests exist to protect is that **a path never invents a
k-point**.  The spectral function is defined only at the commensurate k-points
of the Born-von Karman supercell, so a path is a *selection*, and a
high-symmetry point the mesh does not carry has to be reported rather than
interpolated or silently dropped.

Nothing here builds a Hamiltonian, so the whole file is a fraction of a second.
"""

import numpy as np
import pytest
from ase.dft.kpoints import monkhorst_pack

from mandacaru.algorithms.kpath import band_path, resolve_path

#: A square lattice in a tall box -- the 2-D case the path machinery exists for.
SQUARE = np.diag([2.5, 2.5, 12.0])
SQUARE_PBC = [True, True, False]

#: Lattices that are neither cubic nor orthogonal, which is where a fractional
#: metric would quietly differ from the k-space one.
HEXAGONAL = np.array([[2.5, 0.0, 0.0],
                      [-1.25, 2.5 * np.sqrt(3) / 2, 0.0],
                      [0.0, 0.0, 12.0]])
FCC = 0.5 * 4.0 * np.array([[0.0, 1.0, 1.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0]])
TRICLINIC = np.array([[3.0, 0.0, 0.0], [0.9, 2.8, 0.0], [0.5, 0.4, 3.2]])


def mesh(size):
    """A Gamma-centered Monkhorst-Pack mesh, as the drivers build it."""
    offset = np.array([0.5 / n if n % 2 == 0 else 0.0 for n in size])
    return monkhorst_pack(size) + offset


class TestThePathOnlySelects:
    """It returns mesh points; it never makes one up."""

    @pytest.mark.parametrize("size", [(2, 2, 1), (4, 4, 1), (3, 3, 1)])
    def test_every_returned_point_is_a_mesh_point(self, size):
        full = mesh(size)
        kpath = band_path(SQUARE, full, path="GXMG", pbc=SQUARE_PBC, warn=False)
        for point, index in zip(kpath.points, kpath.indices):
            assert point == pytest.approx(full[index], abs=1e-12)

    def test_a_coarse_mesh_gives_fewer_points_not_interpolated_ones(self):
        two = band_path(SQUARE, mesh((2, 2, 1)), path="GXMG", pbc=SQUARE_PBC,
                        warn=False)
        four = band_path(SQUARE, mesh((4, 4, 1)), path="GXMG", pbc=SQUARE_PBC,
                         warn=False)
        assert len(two) < len(four)

    def test_indices_gather_the_matching_rows(self):
        """``indices`` is the contract a caller uses on a per-k array."""
        full = mesh((4, 4, 1))
        kpath = band_path(SQUARE, full, path="GXMG", pbc=SQUARE_PBC, warn=False)
        payload = np.arange(len(full)) * 10.0
        assert payload[kpath.indices] == pytest.approx(
            np.array([10.0 * i for i in kpath.indices]))


class TestAMissingHighSymmetryPointIsReported:
    """The failure mode is an x-tick naming a point nobody evaluated."""

    def test_a_three_by_three_mesh_has_no_x_or_m(self):
        """X = (0, 1/2, 0) needs an even division; a 3x3 mesh has thirds."""
        kpath = band_path(SQUARE, mesh((3, 3, 1)), path="GXMG",
                          pbc=SQUARE_PBC, warn=False)
        assert set(kpath.missing) == {"X", "M"}
        assert not kpath.complete

    @pytest.mark.parametrize("size", [(2, 2, 1), (4, 4, 1)])
    def test_an_even_mesh_carries_them(self, size):
        kpath = band_path(SQUARE, mesh(size), path="GXMG", pbc=SQUARE_PBC)
        assert kpath.missing == []
        assert kpath.complete

    def test_it_warns(self):
        with pytest.warns(RuntimeWarning, match="not on this"):
            band_path(SQUARE, mesh((3, 3, 1)), path="GXMG", pbc=SQUARE_PBC)

    def test_the_labels_are_kept_even_when_missing(self):
        """The tick belongs on the axis; what is absent is the *data*."""
        kpath = band_path(SQUARE, mesh((3, 3, 1)), path="GXMG",
                          pbc=SQUARE_PBC, warn=False)
        assert kpath.labels == ["G", "X", "M", "G"]


class TestTheLinearAxis:
    def test_it_matches_ase(self):
        """A Mandacaru axis must overlay an ASE one, same 2 pi convention."""
        from ase.cell import Cell

        kpath = band_path(SQUARE, mesh((2, 2, 1)), path="GXMG",
                          pbc=SQUARE_PBC, warn=False)
        # npoints must be large enough for ASE to place the ticks: its axis
        # is measured along the k-points it generated, so npoints=0 collapses
        # every tick onto the few it has.
        ase_path = Cell(SQUARE).bandpath("GXMG", npoints=100, pbc=SQUARE_PBC)
        _x, ticks, labels = ase_path.get_linear_kpoint_axis()
        assert list(labels) == kpath.labels
        assert kpath.label_distances == pytest.approx(ticks, abs=1e-9)

    def test_it_increases_along_the_path(self):
        kpath = band_path(SQUARE, mesh((4, 4, 1)), path="GXMG",
                          pbc=SQUARE_PBC, warn=False)
        assert np.all(np.diff(kpath.distances) > 0)

    def test_gamma_appears_twice_at_two_distances(self):
        """``GXMG`` returns to Gamma; it is one mesh point at two path spots."""
        kpath = band_path(SQUARE, mesh((2, 2, 1)), path="GXMG",
                          pbc=SQUARE_PBC, warn=False)
        gamma = [i for i, k in zip(kpath.indices, kpath.points)
                 if np.allclose(k, 0)]
        assert len(gamma) == 2
        assert gamma[0] == gamma[1]          # the same mesh point
        assert kpath.distances[0] == pytest.approx(0.0)
        assert kpath.distances[-1] > 0.0

    def test_a_shared_endpoint_is_not_emitted_twice(self):
        kpath = band_path(SQUARE, mesh((4, 4, 1)), path="GXMG",
                          pbc=SQUARE_PBC, warn=False)
        assert len(np.unique(np.round(kpath.distances, 10))) == len(kpath)


class TestNonOrthogonalLattices:
    """Collinearity is affine-invariant, but the *metric* is not.

    A hexagonal or triclinic cell has ``b_i . b_j != 0``, so a fractional dot
    product is not the k-space one; the projection is done in Cartesian
    reciprocal space for exactly this reason.
    """

    def test_the_hexagonal_default_path(self):
        path, special = resolve_path(HEXAGONAL, pbc=[True, True, False])
        assert path == "GMKG"
        assert special["K"] == pytest.approx([1 / 3, 1 / 3, 0.0], abs=1e-9)

    def test_a_hexagonal_mesh_carries_k_but_not_m(self):
        """K is at thirds, M at halves: a 3x3 mesh has the first, not the second."""
        kpath = band_path(HEXAGONAL, mesh((3, 3, 1)), path="GMKG",
                          pbc=[True, True, False], warn=False)
        assert kpath.missing == ["M"]

    def test_hexagonal_segment_lengths_use_the_real_metric(self):
        r"""The Gamma-M-K triangle: ``2 pi / (a sqrt 3)``, ``2 pi / 3a``, ``4 pi / 3a``.

        Every one of these comes out wrong if the path length is taken on
        fractional coordinates, since ``b_1 . b_2 = -|b|^2 / 2`` here rather
        than zero.  Note the three points are **not collinear** -- the triangle
        has its right angle at M -- so ``|MK|`` is not ``|GK| - |GM|``.
        """
        a = 2.5
        kpath = band_path(HEXAGONAL, mesh((3, 3, 1)), path="GMKG",
                          pbc=[True, True, False], warn=False)
        ticks = dict(zip(kpath.labels, kpath.label_distances))
        assert ticks["M"] == pytest.approx(2 * np.pi / (a * np.sqrt(3)),
                                           rel=1e-9)
        assert ticks["K"] - ticks["M"] == pytest.approx(2 * np.pi / (3 * a),
                                                        rel=1e-9)
        # The closing leg back to Gamma is |GK| itself.
        assert kpath.label_distances[-1] - ticks["K"] == pytest.approx(
            4 * np.pi / (3 * a), rel=1e-9)

    @pytest.mark.parametrize("cell, pbc", [
        (FCC, [True, True, True]),
        (TRICLINIC, [True, True, True]),
        (HEXAGONAL, [True, True, False]),
    ])
    def test_a_three_dimensional_path_resolves_and_selects(self, cell, pbc):
        kpath = band_path(cell, mesh((2, 2, 2)), pbc=pbc, warn=False)
        assert len(kpath) > 0
        assert len(kpath.labels) == len(kpath.label_distances)
        assert np.all(np.diff(kpath.label_distances) >= -1e-12)

    def test_fcc_special_points_outside_the_first_zone_are_found(self):
        """``W = (1/2, 1/4, 3/4)`` needs unfolding by a reciprocal vector.

        A mesh point folded to ``[-1/2, 1/2)`` can only land on it after being
        displaced by an integer vector, which is why the image search exists.
        """
        full = mesh((4, 4, 4))
        kpath = band_path(FCC, full, path="GXW", pbc=[True] * 3, warn=False)
        assert "W" not in kpath.missing
        assert "X" not in kpath.missing


class TestPathStringsAreValidated:
    def test_an_unknown_label_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown high-symmetry point"):
            band_path(SQUARE, mesh((2, 2, 1)), path="GQM", pbc=SQUARE_PBC)

    def test_a_one_point_segment_is_refused(self):
        with pytest.raises(ValueError, match="at least two points"):
            band_path(SQUARE, mesh((2, 2, 1)), path="G", pbc=SQUARE_PBC)

    def test_a_comma_separates_disconnected_pieces(self):
        kpath = band_path(SQUARE, mesh((4, 4, 1)), path="GX,MG",
                          pbc=SQUARE_PBC, warn=False)
        assert kpath.labels == ["G", "X", "M", "G"]

    def test_a_malformed_mesh_is_refused(self):
        with pytest.raises(ValueError, match=r"shape \(nk, 3\)"):
            band_path(SQUARE, np.zeros((4, 2)), path="GXMG", pbc=SQUARE_PBC)


class TestTheDefaultPath:
    def test_pbc_selects_the_two_dimensional_walk(self):
        """A square lattice in a tall box is tetragonal in 3-D.

        Without ``pbc`` ASE walks through Z, R and A -- points along the vacuum
        direction, where there is no dispersion and no k-point but Gamma.
        """
        flat, _ = resolve_path(SQUARE, pbc=SQUARE_PBC)
        solid, _ = resolve_path(SQUARE, pbc=[True, True, True])
        assert flat == "MGXM"
        assert set("ZRA") & set(solid)
        assert not set("ZRA") & set(flat)
