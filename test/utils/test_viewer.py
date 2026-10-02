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
from PIL import Image

from mandacaru import Camera, Mandacaru, Viewer3D
from mandacaru.algorithms.volumetric import (DENSITY_UNITS, ORBITAL_UNITS,
                                             VolumetricField)
from mandacaru.integrals import Grid
from mandacaru.units import BOHR_TO_ANGSTROM
from mandacaru.utils.viewer import enclosing_level, marching_tetrahedra

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _analytic(quantity="natural_orbital", positions=((0.0, 0.0, 0.0),)):
    """A hydrogen 2p_z orbital (or its square, as a density) on a Bohr grid."""
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=12.0, h=0.5, units="bohr")
    r = np.sqrt(grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2)
    orbital = grid.Z * np.exp(-r / 2.0) / np.sqrt(32.0 * np.pi)
    density = quantity not in ("natural_orbital", "molecular_orbital")
    data = orbital ** 2 if density else orbital
    positions = np.asarray(positions, dtype=float)
    return VolumetricField(
        quantity=quantity, data=np.ascontiguousarray(data), grid=grid,
        numbers=np.ones(len(positions), dtype=int), positions=positions,
        charges=np.ones(len(positions)),
        units=DENSITY_UNITS if density else ORBITAL_UNITS,
        integral=float(data.sum() * grid.dV),
        norm=float((data ** 2).sum() * grid.dV),
        index=None if density else 0)


@pytest.fixture(scope="module")
def p_orbital():
    return _analytic()


@pytest.fixture(scope="module")
def p_density():
    return _analytic("density")


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

    def test_orbital_weight_is_the_probability(self, p_orbital):
        viewer = Viewer3D(p_orbital, mode="isosurface", enclosed=0.9)
        assert viewer.signed and viewer.power == 2
        p = p_orbital.data ** 2
        inside = p[np.abs(p_orbital.data) >= viewer.isovalue].sum() / p.sum()
        assert inside == pytest.approx(0.9, abs=1e-3)

    def test_density_is_unsigned_and_linear(self, p_density):
        viewer = Viewer3D(p_density, mode="isosurface")
        assert not viewer.signed and viewer.power == 1

    def test_explicit_isovalue_wins(self, p_orbital):
        assert Viewer3D(p_orbital, isovalue=-0.02).isovalue == 0.02


class TestCloud:
    def test_same_seed_same_cloud(self, p_orbital):
        a, _ = Viewer3D(p_orbital, points=500, seed=7).cloud()
        b, _ = Viewer3D(p_orbital, points=500, seed=7).cloud()
        c, _ = Viewer3D(p_orbital, points=500, seed=8).cloud()
        assert a.shape == (500, 3)
        assert np.array_equal(a, b) and not np.array_equal(a, c)

    def test_points_are_in_angstrom_inside_the_box(self, p_orbital):
        positions, _ = Viewer3D(p_orbital, points=2000).cloud()
        half = (12.0 + 0.5) * BOHR_TO_ANGSTROM
        assert np.abs(positions).max() <= half

    def test_phase_follows_the_lobes(self, p_orbital):
        positions, values = Viewer3D(p_orbital, points=4000).cloud()
        # A point is jittered by at most half a cell, so away from the nodal
        # plane its side of z = 0 is the sign of the orbital.
        clear = np.abs(positions[:, 2]) > 0.5 * 0.5 * BOHR_TO_ANGSTROM
        assert np.all(np.sign(positions[clear, 2]) == np.sign(values[clear]))
        assert 0.4 < np.mean(values > 0) < 0.6

    def test_cloud_follows_the_probability(self, p_orbital):
        # <z^2> of 2p_z is 18 Bohr^2 analytically; the box clips a little.
        positions, _ = Viewer3D(p_orbital, points=30000, seed=3).cloud()
        z2 = np.mean((positions[:, 2] / BOHR_TO_ANGSTROM) ** 2)
        assert z2 == pytest.approx(18.0, rel=0.1)


class TestSurface:
    def test_orbital_has_two_lobes_on_their_sides(self, p_orbital):
        triangles, signs = Viewer3D(p_orbital, mode="isosurface").surface()
        assert set(np.unique(signs)) == {-1, 1}
        z = triangles[:, :, 2].mean(axis=1)
        assert np.all(z[signs > 0] > 0) and np.all(z[signs < 0] < 0)

    def test_density_has_one_sign(self, p_density):
        _, signs = Viewer3D(p_density, mode="isosurface").surface()
        assert len(signs) and np.all(signs == 1)


class TestCameraPath:
    def test_rotation_is_periodic(self, p_orbital):
        path = Viewer3D(p_orbital).camera_path(8)
        azim = np.array([c.azim for c in path])
        assert np.allclose(np.diff(azim), 45.0)
        assert azim[-1] + 45.0 - azim[0] == pytest.approx(360.0)
        assert all(c.elev == Camera().elev for c in path)

    def test_elevation_rotation(self, p_orbital):
        path = Viewer3D(p_orbital).camera_path(4, rotate="elevation", turns=0.5)
        assert [c.elev for c in path] == pytest.approx(
            [15.0, 60.0, 105.0, 150.0])

    def test_zoom_keyframes(self, p_orbital):
        path = Viewer3D(p_orbital).camera_path(5, rotate=None, zoom=(1, 2, 1))
        assert [c.zoom for c in path] == pytest.approx([1, np.sqrt(2), 2,
                                                        np.sqrt(2), 1])

    def test_trajectory_hits_its_keyframes(self):
        field = _analytic(positions=((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)))
        keyframes = [Camera(10, 0, 1.0), Camera(50, 90, 3.0, focus=1),
                     Camera(10, 180, 1.0)]
        path = Viewer3D(field).camera_path(9, rotate=None,
                                           trajectory=keyframes)
        assert (path[0].elev, path[0].azim) == (10, 0)
        assert (path[4].elev, path[4].azim, path[4].zoom) == pytest.approx(
            (50, 90, 3.0))
        assert path[4].focus == pytest.approx((0, 0, BOHR_TO_ANGSTROM))
        assert path[0].focus == pytest.approx((0, 0, 0))
        assert path[-1].azim == pytest.approx(180)

    def test_bad_paths_are_refused(self, p_orbital):
        viewer = Viewer3D(p_orbital)
        with pytest.raises(ValueError, match="rotate"):
            viewer.camera_path(4, rotate="roll")
        with pytest.raises(ValueError, match="zoom"):
            viewer.camera_path(4, zoom=(1, 0))
        with pytest.raises(ValueError, match="trajectory"):
            viewer.camera_path(4, trajectory=[(15, 30)])
        with pytest.raises(ValueError, match="zoom > 0"):
            viewer.camera_path(4, trajectory=[Camera(zoom=0)])
        with pytest.raises(ValueError, match="names no atom"):
            viewer.camera_path(4, trajectory=[Camera(focus=3)])


class TestFiles:
    @pytest.mark.parametrize("mode", ["scatter", "isosurface"])
    def test_png_has_the_requested_size_and_alpha(self, p_orbital, tmp_path,
                                                  mode):
        path = Viewer3D(p_orbital, mode=mode, points=2000,
                        figsize=(2, 2)).save(tmp_path / "p.png", dpi=50)
        with Image.open(path) as image:
            assert image.size == (100, 100)
            assert image.mode == "RGBA"
            assert image.getextrema()[3][0] == 0           # transparent corner

    def test_gif_has_every_frame_and_loops(self, p_orbital, tmp_path):
        viewer = Viewer3D(p_orbital, mode="isosurface", figsize=(2, 2),
                          background="black")
        path = viewer.animate(tmp_path / "p.gif", frames=4, fps=10, dpi=40,
                              zoom=(1, 1.5))
        with Image.open(path) as image:
            assert image.n_frames == 4
            assert image.info["loop"] == 0
            assert image.info["duration"] == 100

    def test_transparent_gif_clears_every_frame(self, p_orbital, tmp_path):
        path = Viewer3D(p_orbital, points=2000, figsize=(2, 2)).animate(
            tmp_path / "t.gif", frames=3, dpi=40)
        with Image.open(path) as image:
            assert "transparency" in image.info
            for frame in range(image.n_frames):
                image.seek(frame)
                assert image.disposal_method == 2
                # The corner is background in every frame: no trail.
                assert image.convert("RGBA").getpixel((0, 0))[3] == 0

    def test_animation_must_be_a_gif(self, p_orbital, tmp_path):
        with pytest.raises(ValueError, match=".gif"):
            Viewer3D(p_orbital).animate(tmp_path / "p.mp4", frames=2)


class TestContracts:
    def test_unknown_mode(self, p_orbital):
        with pytest.raises(ValueError, match="mode"):
            Viewer3D(p_orbital, mode="volume")

    def test_a_field_cannot_be_resampled(self, p_orbital):
        with pytest.raises(ValueError, match="calculator"):
            Viewer3D(p_orbital, h=0.1)

    def test_anything_else_is_refused(self):
        with pytest.raises(TypeError, match="VolumetricField"):
            Viewer3D(np.zeros((3, 3, 3)))


class TestFromTheCalculator:
    """A real ADAPT-VQE state, drawn end to end."""

    @pytest.fixture(scope="class")
    def calc(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[6, 6, 6])
        atoms.center()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.30,
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
        viewer = Viewer3D(calc, "natural_orbital", 1, h=0.2, mode="isosurface",
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
