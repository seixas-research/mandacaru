# -*- coding: utf-8 -*-
# file: test/utils/test_viewer.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``utils/viewer.py``: the point cloud, the isosurface, the camera path and
the PNG / GIF files.

Almost everything runs on an analytic hydrogen 2p_z orbital, so the geometry
is known exactly and no solver is involved; one class draws a real ADAPT-VQE
state end to end.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru, Viewer3D
from mandacaru.utils.viewer import enclosing_level, marching_tetrahedra

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")








class TestMarchingTetrahedra:
    """The surface of ``|r| = R`` is a sphere: its vertices lie on it and its
    area converges to ``4 pi R^2``."""

    @pytest.fixture(scope="class")
    def sphere(self):
        g = np.linspace(-2.0, 2.0, 41)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        return marching_tetrahedra(np.sqrt(X**2 + Y**2 + Z**2), X, Y, Z, 1.3)

    def test_vertices_lie_on_the_sphere(self, sphere):
        assert np.isfinite(sphere).all()
        assert np.abs(np.linalg.norm(sphere, axis=2) - 1.3).max() < 0.1

    def test_area_is_that_of_the_sphere(self, sphere):
        edges = np.cross(sphere[:, 1] - sphere[:, 0], sphere[:, 2] - sphere[:, 0])
        area = 0.5 * np.linalg.norm(edges, axis=1).sum()
        assert area == pytest.approx(4 * np.pi * 1.3 ** 2, rel=0.01)

    def test_a_level_outside_the_range_gives_no_surface(self):
        g = np.linspace(-1.0, 1.0, 5)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        assert marching_tetrahedra(X, X, Y, Z, 5.0).shape == (0, 3, 3)


class TestIsovalue:
    def test_enclosing_level_holds_the_fraction(self):
        w = np.random.default_rng(1).random(10000)
        level = enclosing_level(w, 0.85)
        assert w[w >= level].sum() / w.sum() == pytest.approx(0.85, abs=1e-3)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
    def test_fraction_must_be_open_interval(self, fraction):
        with pytest.raises(ValueError, match="enclosed"):
            enclosing_level(np.ones(4), fraction)













class TestContracts:


    def test_anything_else_is_refused(self):
        with pytest.raises(TypeError, match="VolumetricField"):
            Viewer3D(np.zeros((3, 3, 3)))


class TestFromTheCalculator:
    """A real ADAPT-VQE state, drawn end to end."""

    @pytest.fixture(scope="class")
    def calc(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[6, 6, 6])
        atoms.center()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.30,
                               trace=False)
        atoms.get_potential_energy()
        return atoms.calc

    def test_the_field_is_the_calculators(self, calc):
        viewer = Viewer3D(calc, "natural_orbital", 1)
        expected = calc.volumetric_field("natural_orbital", 1)
        assert np.array_equal(viewer.field.data, expected.data)
        assert viewer.nuclei == pytest.approx(
            calc.atoms.get_positions(), abs=1e-6)

    def test_h_resamples_on_a_finer_grid_of_the_same_box(self, calc, tmp_path):
        coarse = calc.volumetric_field("natural_orbital", 1)
        viewer = Viewer3D(calc, "natural_orbital", 1, grid_spacing=0.2, mode="isosurface",
                          figsize=(2, 2))
        assert viewer.field.data.size > coarse.data.size
        assert viewer.field.norm == pytest.approx(1.0, abs=0.05)
        # The antibonding orbital has one nodal plane between the atoms.
        _, signs = viewer.surface()
        assert set(np.unique(signs)) == {-1, 1}
        viewer.save(tmp_path / "no1.png", dpi=40)
        viewer.animate(tmp_path / "no1.gif", frames=3, dpi=30)
        assert (tmp_path / "no1.png").stat().st_size > 0
        assert (tmp_path / "no1.gif").stat().st_size > 0
