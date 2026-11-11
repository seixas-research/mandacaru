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

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru, Viewer3D
from mandacaru.utils.viewer import Camera, enclosing_level, marching_tetrahedra

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


def antisymmetric_field():
    """``f = x`` on a 5 x 5 x 5 grid (Bohr), two hydrogen nuclei: a signed amplitude."""
    g = np.linspace(-2.0, 2.0, 5)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    grid = SimpleNamespace(X=X, Y=Y, Z=Z, step=np.diag([1.0, 1.0, 1.0]))
    return SimpleNamespace(data=X.copy(), grid=grid, quantity="natural_orbital",
                           component="auto", numbers=np.array([1, 1]),
                           positions=np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
                           label=lambda: "x")


class TestCameraPath:
    """The path is built from the viewer's pose, the trajectory, the spin and the zoom."""

    @pytest.fixture
    def viewer(self):
        return Viewer3D(antisymmetric_field(), "natural_orbital", points=200,
                        camera=Camera(elev=10.0, azim=20.0, zoom=1.5))

    def test_one_frame_is_the_viewers_pose(self, viewer):
        (pose,) = viewer.camera_path(1, rotate=None)
        assert (pose.elev, pose.azim) == (10.0, 20.0)
        assert pose.zoom == pytest.approx(1.5, rel=1e-14)
        assert pose.focus == pytest.approx(viewer.nuclei.mean(axis=0), abs=1e-15)

    def test_a_spin_is_whole_turns_without_a_seam(self, viewer):
        path = viewer.camera_path(8, turns=2.0)
        assert [c.azim for c in path] == pytest.approx([20.0 + 90.0 * i for i in range(8)])
        assert {c.elev for c in path} == {10.0}
        swing = viewer.camera_path(4, rotate="elevation", turns=1.0)
        assert [c.elev for c in swing] == pytest.approx([10.0, 100.0, 190.0, 280.0])

    def test_zoom_keyframes_are_geometric_and_multiply_the_pose_zoom(self, viewer):
        path = viewer.camera_path(5, rotate=None, zoom=(1.0, 4.0))
        assert path[0].zoom == pytest.approx(1.5)
        assert path[2].zoom == pytest.approx(3.0)
        assert path[4].zoom == pytest.approx(6.0)

    def test_a_trajectory_passes_through_its_keyframes(self, viewer):
        keys = [Camera(elev=0, azim=0), Camera(elev=60, azim=90, zoom=2.0, focus=1),
                Camera(elev=0, azim=180, focus=(0.0, 0.0, 0.3))]
        path = viewer.camera_path(5, rotate=None, trajectory=keys)
        assert (path[0].elev, path[0].azim) == (0.0, 0.0)
        assert (path[2].elev, path[2].azim) == pytest.approx((60.0, 90.0))
        assert path[2].zoom == pytest.approx(2.0)
        assert path[2].focus == pytest.approx(viewer.nuclei[1])
        assert path[4].focus == pytest.approx((0.0, 0.0, 0.3))

    def test_a_bad_request_is_refused(self, viewer):
        with pytest.raises(ValueError, match="frames"):
            viewer.camera_path(0)
        with pytest.raises(ValueError, match="rotate"):
            viewer.camera_path(3, rotate="roll")
        with pytest.raises(ValueError, match="zoom must be positive"):
            viewer.camera_path(3, zoom=(1.0, -2.0))
        with pytest.raises(ValueError, match="trajectory"):
            viewer.camera_path(3, trajectory=[])

    def test_the_extent_fits_the_nuclei_and_what_is_drawn(self, viewer):
        explicit = Viewer3D(antisymmetric_field(), "natural_orbital", extent=2.5)
        assert explicit.extent == 2.5
        # The nuclei alone reach 0.7 A from the center plus the 0.5 A margin.
        assert viewer.extent >= 1.05 * (0.37 + 0.5)
        assert Viewer3D(antisymmetric_field(), "natural_orbital", mode="isosurface").extent > 0.0


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
