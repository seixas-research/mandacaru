# -*- coding: utf-8 -*-
# file: utils/viewer.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Three-dimensional pictures of orbitals and densities: PNG stills and GIFs.

:class:`Viewer3D` draws one real-space quantity of a finished run -- a natural
or reference orbital, the electron density, the spin density or the
correlation (difference) density -- as a point cloud or as an isosurface, and
renders it either as a still image or as an animation whose camera rotates,
zooms and travels along a keyframed path.

What is drawn is a **one-particle reduction** of the variational state, never
"the wavefunction": the many-electron state is a function of :math:`3N`
coordinates and has no picture in three dimensions.  The field comes from
:meth:`~mandacaru.algorithms.calculator.Mandacaru.volumetric_field`, which
contracts the converged ansatz (its operators and optimized parameters) into
the one-particle reduced density matrix and expands the result in the
calculation's own basis functions; see :mod:`mandacaru.algorithms.volumetric`.

Two modes
---------
``"scatter"``
    A Monte Carlo cloud.  Points are drawn with probability proportional to
    :math:`|\phi|^2` for an orbital and to :math:`|n|` for a density, so the
    cloud's local density *is* the probability density.  The sampling is done
    on the grid: a node is picked with probability proportional to its weight
    and the point is placed uniformly inside the node's cell, which is exact
    rejection sampling of the piecewise-constant field.  Each point is colored
    by the sign of the field (cyan positive, orange negative) with a brightness
    that grows with :math:`|f|`.
``"isosurface"``
    The surface :math:`f = \pm f_0`, extracted by marching tetrahedra and lit
    by a light that follows the camera.  The isovalue :math:`f_0` is chosen,
    unless given, so that the region inside the surface holds a fraction
    ``enclosed`` of :math:`\int|\phi|^2` (orbital) or :math:`\int|n|`
    (density) -- the usual convention for drawing an orbital "at 85%".

Signed quantities (orbitals, spin density, difference density) draw both
signs; the electron density and its spin channels are non-negative and draw one.

Units
-----
Everything the viewer takes or reports -- ``focus``, ``extent``, ``h`` and the
sampled coordinates -- is in **Angstrom**, like the rest of Mandacaru's user
surface.  The field itself stays in the units it was computed in.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import numpy as np

from ..units import BOHR_TO_ANGSTROM, to_bohr

#: The drawing modes :class:`Viewer3D` knows.
MODES = ("scatter", "isosurface")

#: The camera motions :meth:`Viewer3D.animate` can superimpose on a path.
ROTATIONS = ("azimuth", "elevation", None)

#: Quantities whose sign carries meaning and is drawn in two colors.
SIGNED_QUANTITIES = ("natural_orbital", "molecular_orbital", "spin_density",
                     "difference_density")

#: Quantities that are amplitudes: their probability weight is ``|f|^2``.
AMPLITUDE_QUANTITIES = ("natural_orbital", "molecular_orbital")

#: Positive / negative phase: point-cloud colormaps (dark to bright) and the
#: flat surface colors.
POSITIVE_COLORS = ("#0b6f86", "#00c8ee", "#00f0ff", "#b8fbff")
NEGATIVE_COLORS = ("#c2410c", "#f59e0b", "#ffd23f", "#fff3b0")
POSITIVE_SURFACE = "#00b4d8"
NEGATIVE_SURFACE = "#f59e0b"

#: The Kuhn decomposition of a cube into six tetrahedra.  Corner ``c`` of a
#: cell is node ``(i, j, k) = (c >> 2, (c >> 1) & 1, c & 1)`` relative to the
#: cell's first node.  Every tetrahedron runs along the main diagonal 0 -> 7,
#: so neighboring cells cut their shared faces the same way and the surface
#: has no cracks.
_KUHN_TETRAHEDRA = np.array([[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7],
                             [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]])


# --------------------------------------------------------------------------- #
# Geometry kernels, free of any plotting.
# --------------------------------------------------------------------------- #

def marching_tetrahedra(values, X, Y, Z, level: float) -> np.ndarray:
    """Triangles of the surface ``values == level`` on a structured grid.

    ``values``, ``X``, ``Y`` and ``Z`` are ``(nx, ny, nz)`` arrays: the field
    and the Cartesian coordinates of every node, so a skewed grid works as
    well as an orthogonal one.  Returns an ``(n, 3, 3)`` array -- triangle,
    vertex, coordinate -- in the units of ``X``.  The winding is not
    consistent; :class:`Viewer3D` lights both sides of every face.
    """
    values = np.asarray(values, dtype=float)
    nx, ny, nz = values.shape
    if min(nx, ny, nz) < 2:
        return np.zeros((0, 3, 3))
    flat = values.ravel()
    corner_offsets = np.array([(c >> 2) * ny * nz + ((c >> 1) & 1) * nz + (c & 1)
                               for c in range(8)])
    i, j, k = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1),
                          np.arange(nz - 1), indexing="ij")
    base = (i * ny * nz + j * nz + k).ravel()

    # Keep only the cells the surface crosses before touching coordinates:
    # on a fine grid that is a few percent of them.
    corner_values = flat[base[:, None] + corner_offsets]
    above = corner_values > level
    crossed = above.any(axis=1) & ~above.all(axis=1)
    nodes = base[crossed][:, None] + corner_offsets[_KUHN_TETRAHEDRA.ravel()]
    nodes = nodes.reshape(-1, 4)                                # (T, 4)

    tet_values = flat[nodes]
    inside = tet_values > level
    count = inside.sum(axis=1)
    keep = (count > 0) & (count < 4)
    nodes, tet_values, inside, count = (nodes[keep], tet_values[keep],
                                        inside[keep], count[keep])
    points = np.stack([np.asarray(X, dtype=float).ravel(),
                       np.asarray(Y, dtype=float).ravel(),
                       np.asarray(Z, dtype=float).ravel()], axis=1)[nodes]

    # Put the corners above the level first; then each case has a fixed shape.
    order = np.argsort(~inside, axis=1, kind="stable")
    v = np.take_along_axis(tet_values, order, axis=1)
    p = np.take_along_axis(points, order[:, :, None], axis=1)

    def edge(rows, a, b):
        t = (level - v[rows, a]) / (v[rows, b] - v[rows, a])
        return p[rows, a] + t[:, None] * (p[rows, b] - p[rows, a])

    triangles = []
    one = np.flatnonzero(count == 1)          # corner 0 alone above
    triangles.append(np.stack([edge(one, 0, 1), edge(one, 0, 2),
                               edge(one, 0, 3)], axis=1))
    three = np.flatnonzero(count == 3)        # corner 3 alone below
    triangles.append(np.stack([edge(three, 3, 0), edge(three, 3, 1),
                               edge(three, 3, 2)], axis=1))
    two = np.flatnonzero(count == 2)          # corners 0, 1 above; 2, 3 below
    quad = [edge(two, 0, 2), edge(two, 0, 3), edge(two, 1, 3), edge(two, 1, 2)]
    triangles.append(np.stack([quad[0], quad[1], quad[2]], axis=1))
    triangles.append(np.stack([quad[0], quad[2], quad[3]], axis=1))
    return np.concatenate(triangles, axis=0)


def enclosing_level(weights, fraction: float) -> float:
    """Smallest weight ``w0`` such that the nodes with ``w >= w0`` hold ``fraction``.

    ``weights`` are non-negative.  This is the isovalue convention for
    orbitals ("the surface enclosing 85% of the probability"), applied to the
    weight the field is sampled with.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"enclosed must lie strictly between 0 and 1, "
                         f"got {fraction!r}")
    w = np.sort(np.asarray(weights, dtype=float).ravel())[::-1]
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("the field is zero everywhere; there is nothing to draw")
    cumulative = np.cumsum(w) / total
    return float(w[min(int(np.searchsorted(cumulative, fraction)), len(w) - 1)])


def _ease(t):
    """Cosine ease-in-out on ``[0, 1]``: a keyframe is approached and left smoothly."""
    return 0.5 - 0.5 * np.cos(np.pi * np.asarray(t, dtype=float))


def _keyframe_values(keyframes, t, log=False):
    """Interpolate evenly spaced ``keyframes`` at times ``t`` in ``[0, 1]``.

    Each segment is eased, so the camera slows into and out of every keyframe
    instead of turning a corner.  ``log=True`` interpolates geometrically,
    which is how a zoom is perceived.
    """
    keys = np.asarray(keyframes, dtype=float)
    if keys.ndim == 1:
        keys = keys[:, None]
    if log:
        keys = np.log(keys)
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    if len(keys) == 1:
        out = np.repeat(keys, len(t), axis=0)
    else:
        position = t * (len(keys) - 1)
        segment = np.minimum(position.astype(int), len(keys) - 2)
        local = _ease(position - segment)[:, None]
        out = keys[segment] + local * (keys[segment + 1] - keys[segment])
    return np.exp(out) if log else out


# --------------------------------------------------------------------------- #
# The camera.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Camera:
    """One camera pose; a list of them is a keyframed path for :meth:`Viewer3D.animate`.

    Attributes
    ----------
    elev, azim : float
        Elevation and azimuth in degrees, as in Matplotlib's ``view_init``.
        Keyframed azimuths are interpolated as given, so ``0 -> 720`` turns
        twice.
    zoom : float
        Magnification relative to the view that frames the whole picture
        (``2`` halves the visible width).
    focus : None, int or (3,) array_like
        The point the camera looks at: ``None`` for the center of the
        molecule, an integer for that atom, or a position in Angstrom.
    """

    elev: float = 15.0
    azim: float = -60.0
    zoom: float = 1.0
    focus: object = None


# --------------------------------------------------------------------------- #
# The viewer.
# --------------------------------------------------------------------------- #

class Viewer3D:
    """Draw an orbital or a density of a finished run in three dimensions.

    .. code-block:: python

        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.30)
        atoms.get_potential_energy()

        viewer = Viewer3D(atoms.calc, quantity="natural_orbital", index=1,
                          h=0.12)
        viewer.save("no1.png")
        viewer.animate("no1.gif", frames=120, zoom=(1.0, 1.8, 1.0))
        Viewer3D(atoms.calc, quantity="difference_density",
                 mode="isosurface").save("difference.png")

    Parameters
    ----------
    source : Mandacaru or ~mandacaru.algorithms.volumetric.VolumetricField
        A calculator that has been solved -- the field is then built from its
        converged state -- or a field already in hand.
    quantity, index, state, component :
        Passed to
        :meth:`~mandacaru.algorithms.calculator.Mandacaru.volumetric_field`;
        see :mod:`mandacaru.algorithms.volumetric` for what each quantity is.
        Ignored when ``source`` is a field.
    h : float, optional
        Re-evaluate the basis functions on a grid of this spacing (Angstrom)
        over the same box, for a smoother picture than the calculation's own
        grid gives.  Nothing is interpolated.  Needs a calculator as
        ``source``.  Rendering time of an isosurface grows as ``1/h^2``.
    grid : Grid, optional
        An explicit grid to sample on instead (exclusive with ``h``).
    mode : {"scatter", "isosurface"}
        How the field is drawn; see the module documentation.
    points : int
        Size of the point cloud (scatter mode).
    isovalue : float, optional
        Absolute isovalue, in the field's units (isosurface mode).  Overrides
        ``enclosed``.
    enclosed : float
        Fraction of the probability (orbital) or of the charge (density)
        inside the isosurface, when ``isovalue`` is not given.
    camera : Camera
        The pose of a still image, and the starting pose of an animation.
    extent : float, optional
        Half-width of the view at ``zoom=1``, in Angstrom.  By default it is
        fitted to the cloud or the surface and the nuclei.
    background : None or color
        ``None`` (the default) for a transparent background, or any Matplotlib
        color.
    figsize : (float, float)
        Figure size in inches; the pixel size is ``figsize * dpi``.
    point_size, alpha :
        Marker area (points^2) and opacity of the cloud or the surface.
    atoms, bonds : bool
        Draw the nuclei (Jmol colors, covalent radii) and the bonds between
        them.
    seed : int or None
        Seed of the point-cloud sampler; the same seed draws the same cloud.
    """

    def __init__(self, source, quantity: str = "density", index: int = 0, *,
                 state=0, component: str = "auto", h=None, grid=None,
                 mode: str = "scatter", points: int = 30000, isovalue=None,
                 enclosed: float = 0.85, camera: Camera | None = None,
                 extent=None, background=None, figsize=(6.0, 6.0),
                 point_size: float = 2.0, alpha=None, atoms: bool = True,
                 bonds: bool = True, seed: int | None = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if h is not None and grid is not None:
            raise ValueError("give either h= or grid=, not both")
        self.field = self._build_field(source, quantity, index, state,
                                       component, h, grid)
        self.mode = mode
        self.points = int(points)
        if self.points < 1:
            raise ValueError(f"points must be positive, got {points!r}")
        self._isovalue = None if isovalue is None else abs(float(isovalue))
        self.enclosed = float(enclosed)
        self.camera = Camera() if camera is None else camera
        self._extent = None if extent is None else float(extent)
        if self._extent is not None and self._extent <= 0.0:
            raise ValueError(f"extent must be positive, got {extent!r}")
        self.background = background
        self.figsize = tuple(figsize)
        self.point_size = float(point_size)
        self.alpha = (0.7 if mode == "scatter" else 0.8) if alpha is None \
            else float(alpha)
        self.show_atoms = bool(atoms)
        self.show_bonds = bool(bonds)
        self.seed = seed
        self._cloud = None
        self._surface = None

    # -- the field ---------------------------------------------------------- #

    @staticmethod
    def _build_field(source, quantity, index, state, component, h, grid):
        if hasattr(source, "data") and hasattr(source, "grid") \
                and hasattr(source, "positions"):
            if h is not None or grid is not None:
                raise ValueError(
                    "h= and grid= re-evaluate the basis functions, which needs "
                    "the calculator; a VolumetricField can only be drawn on "
                    "the grid it was sampled on")
            return source
        if not hasattr(source, "volumetric_field"):
            raise TypeError(
                f"Viewer3D draws a Mandacaru calculator or a VolumetricField, "
                f"not {type(source).__name__}")
        if h is not None:
            h = float(h)
            if h <= 0.0:
                raise ValueError(f"h must be positive, got {h!r}")
            own = source.volumetric_field(quantity, index, state=state,
                                          component=component).grid
            grid = dataclasses.replace(
                own, h=to_bohr(h, "angstrom") / to_bohr(1.0, own.units))
        return source.volumetric_field(quantity, index, state=state,
                                       component=component, grid=grid)

    @property
    def signed(self) -> bool:
        """Whether the field's sign carries meaning and two colors are drawn."""
        return (self.field.quantity in SIGNED_QUANTITIES
                and self.field.component != "modulus")

    @property
    def power(self) -> int:
        """Exponent of the probability weight: 2 for an orbital, 1 for a density."""
        return 2 if self.field.quantity in AMPLITUDE_QUANTITIES else 1

    def _weights(self) -> np.ndarray:
        return np.abs(self.field.data) ** self.power

    @property
    def isovalue(self) -> float:
        """The isovalue in the field's units, given or fitted to ``enclosed``."""
        if self._isovalue is None:
            level = enclosing_level(self._weights(), self.enclosed)
            self._isovalue = float(level ** (1.0 / self.power))
        return self._isovalue

    def _nodes(self) -> np.ndarray:
        grid = self.field.grid
        return np.stack([np.asarray(grid.X, dtype=float).ravel(),
                         np.asarray(grid.Y, dtype=float).ravel(),
                         np.asarray(grid.Z, dtype=float).ravel()],
                        axis=1) * BOHR_TO_ANGSTROM

    @property
    def nuclei(self) -> np.ndarray:
        """Nuclear positions in Angstrom."""
        return np.asarray(self.field.positions, dtype=float) * BOHR_TO_ANGSTROM

    # -- what is drawn ------------------------------------------------------ #

    def cloud(self):
        """``(positions, values)`` of the point cloud: ``(points, 3)`` in Angstrom
        and the field's value at each point."""
        if self._cloud is None:
            weights = self._weights().ravel()
            total = float(weights.sum())
            if total <= 0.0:
                raise ValueError("the field is zero everywhere; there is "
                                 "nothing to sample")
            rng = np.random.default_rng(self.seed)
            picked = rng.choice(weights.size, size=self.points,
                                p=weights / total)
            step = np.asarray(self.field.grid.step, dtype=float) \
                * BOHR_TO_ANGSTROM
            jitter = rng.uniform(-0.5, 0.5, size=(self.points, 3)) @ step.T
            self._cloud = (self._nodes()[picked] + jitter,
                           np.asarray(self.field.data, dtype=float)
                           .ravel()[picked])
        return self._cloud

    def surface(self):
        """``(triangles, signs)`` of the isosurface: ``(n, 3, 3)`` in Angstrom and
        ``+1`` / ``-1`` per triangle for the lobe it belongs to."""
        if self._surface is None:
            grid = self.field.grid
            data = np.asarray(self.field.data, dtype=float)
            X, Y, Z = (np.asarray(a, dtype=float) * BOHR_TO_ANGSTROM
                       for a in (grid.X, grid.Y, grid.Z))
            level = self.isovalue
            lobes = [marching_tetrahedra(data, X, Y, Z, level)]
            if self.signed:
                lobes.append(marching_tetrahedra(-data, X, Y, Z, level))
            signs = np.concatenate([np.full(len(t), s, dtype=int)
                                    for t, s in zip(lobes, (1, -1))])
            self._surface = (np.concatenate(lobes, axis=0), signs)
        return self._surface

    def _focus_point(self, focus) -> np.ndarray:
        if focus is None:
            return self.nuclei.mean(axis=0)
        if isinstance(focus, (int, np.integer)) and not isinstance(focus, bool):
            n = len(self.nuclei)
            if not -n <= int(focus) < n:
                raise ValueError(f"focus={focus} names no atom: there are {n}")
            return self.nuclei[int(focus)]
        point = np.asarray(focus, dtype=float).reshape(-1)
        if point.shape != (3,):
            raise ValueError(f"focus must be None, an atom index or a point "
                             f"(x, y, z) in Angstrom, got {focus!r}")
        return point

    @property
    def extent(self) -> float:
        """Half-width of the view at ``zoom=1``, in Angstrom."""
        if self._extent is None:
            center = self._focus_point(None)
            if self.mode == "scatter":
                shown = self.cloud()[0]
            else:
                shown = self.surface()[0].reshape(-1, 3)
            reach = [np.abs(self.nuclei - center).max() + 0.5]
            if len(shown):
                # A percentile, not the maximum: a handful of points in a
                # far tail must not shrink the whole picture.
                reach.append(np.percentile(np.abs(shown - center).max(axis=1),
                                           99.5 if self.mode == "scatter"
                                           else 100.0))
            self._extent = 1.05 * float(max(reach))
        return self._extent

    # -- drawing ------------------------------------------------------------ #

    def figure(self, camera: Camera | None = None):
        """A Matplotlib figure of the picture at ``camera`` (default: the viewer's)."""
        fig, ax, update = self._scene()
        update(self.camera if camera is None else camera)
        return fig

    def _scene(self):
        """Build the figure once; return it with a function that poses the camera."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        facecolor = "none" if self.background is None else self.background
        fig = Figure(figsize=self.figsize, facecolor=facecolor)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_facecolor(facecolor)
        ax.set_axis_off()
        # Draw in call order: the nuclei stay visible on top of the cloud,
        # and the surface is depth-sorted within its own single collection.
        ax.computed_zorder = False
        ax.set_box_aspect((1, 1, 1), zoom=1.25)

        relight = (self._draw_cloud(ax) if self.mode == "scatter"
                   else self._draw_surface(ax))
        self._draw_nuclei(ax)
        extent = self.extent

        def update(camera: Camera, focus_point=None):
            center = (self._focus_point(camera.focus) if focus_point is None
                      else focus_point)
            half = extent / float(camera.zoom)
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
            ax.view_init(elev=camera.elev, azim=camera.azim)
            if relight is not None:
                relight(camera.elev, camera.azim)

        return fig, ax, update

    def _draw_cloud(self, ax):
        from matplotlib.colors import LinearSegmentedColormap

        positions, values = self.cloud()
        weights = np.abs(values) ** self.power
        brightness = np.sqrt(weights / weights.max())
        positive = LinearSegmentedColormap.from_list("positive", POSITIVE_COLORS)
        negative = LinearSegmentedColormap.from_list("negative", NEGATIVE_COLORS)
        colors = np.where((values >= 0.0)[:, None], positive(brightness),
                          negative(brightness))
        # Both phases in one scatter, so they are depth-sorted together.
        ax.scatter(*positions.T, c=colors, s=self.point_size, alpha=self.alpha,
                   edgecolors="none", depthshade=False, zorder=1)
        return None

    def _draw_surface(self, ax):
        from matplotlib.colors import to_rgb
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        triangles, signs = self.surface()
        if not len(triangles):
            return None
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-300)[:, None]
        base = np.where((signs > 0)[:, None], to_rgb(POSITIVE_SURFACE),
                        to_rgb(NEGATIVE_SURFACE))
        # Both lobes in one collection: Matplotlib depth-sorts the faces of a
        # collection, but not one collection against another.
        surface = Poly3DCollection(triangles, edgecolors="none", zorder=1)
        ax.add_collection3d(surface)
        alpha = self.alpha

        def relight(elev, azim):
            # A headlight slightly above and to the left of the eye; both sides
            # of a face are lit, since the winding is not consistent.
            e, a = np.radians(elev + 25.0), np.radians(azim - 30.0)
            light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                              np.sin(e)])
            shade = 0.35 + 0.65 * np.abs(normals @ light)
            rgba = np.concatenate([base * shade[:, None],
                                   np.full((len(shade), 1), alpha)], axis=1)
            surface.set_facecolor(rgba)

        return relight

    def _draw_nuclei(self, ax):
        if not self.show_atoms:
            return
        from ase.data import covalent_radii
        from ase.data.colors import jmol_colors

        numbers = np.asarray(self.field.numbers, dtype=int)
        nuclei = self.nuclei
        radii = covalent_radii[numbers]
        if self.show_bonds and len(nuclei) > 1:
            for a in range(len(nuclei)):
                for b in range(a + 1, len(nuclei)):
                    if np.linalg.norm(nuclei[a] - nuclei[b]) \
                            < 1.2 * (radii[a] + radii[b]):
                        ax.plot(*np.stack([nuclei[a], nuclei[b]]).T,
                                color="#6b7280", linewidth=2.5, zorder=2)
        ax.scatter(*nuclei.T, color=jmol_colors[numbers],
                   s=60.0 * (radii / radii.max()) ** 2 + 20.0,
                   edgecolors="#1f2937", linewidths=0.6, depthshade=False,
                   zorder=3)

    # -- output ------------------------------------------------------------- #

    def save(self, path, *, camera: Camera | None = None, dpi: int = 150) -> str:
        """Write a still image and return its path.

        The format follows the extension (``.png``; any format Matplotlib
        writes works).  A transparent background stays transparent in PNG.
        """
        path = os.fspath(path)
        fig = self.figure(camera)
        fig.savefig(path, dpi=dpi, transparent=self.background is None,
                    facecolor=fig.get_facecolor())
        return path

    def camera_path(self, frames: int = 120, *, rotate="azimuth",
                    turns: float = 1.0, zoom=None, trajectory=None):
        """The camera of every frame of :meth:`animate`, as a list of :class:`Camera`.

        The path is built in three layers:

        1. ``trajectory`` -- a sequence of :class:`Camera` keyframes, evenly
           spaced in time from the first frame to the last and joined with
           eased interpolation of elevation, azimuth, zoom and focus.  Without
           one, the camera holds the viewer's pose.
        2. ``rotate`` -- ``"azimuth"`` spins the camera around the vertical
           axis, ``"elevation"`` swings it over the top, ``turns`` times, on
           top of the trajectory.  The spin is periodic: the frame after the
           last one would be the first, so a looping GIF has no seam.
        3. ``zoom`` -- a number, or a sequence of keyframes such as
           ``(1, 2, 1)`` (zoom in, then back out), multiplied into the
           trajectory's own zoom.
        """
        frames = int(frames)
        if frames < 1:
            raise ValueError(f"frames must be positive, got {frames!r}")
        if rotate not in ROTATIONS:
            raise ValueError(f"rotate must be one of {ROTATIONS}, got {rotate!r}")
        keyframes = [self.camera] if trajectory is None else list(trajectory)
        if not keyframes or not all(isinstance(k, Camera) for k in keyframes):
            raise ValueError("trajectory must be a non-empty sequence of Camera")
        if any(not k.zoom > 0.0 for k in keyframes):
            raise ValueError("every Camera of a trajectory needs zoom > 0")
        t = np.linspace(0.0, 1.0, frames) if frames > 1 else np.zeros(1)
        angles = _keyframe_values([[k.elev, k.azim] for k in keyframes], t)
        zooms = _keyframe_values([k.zoom for k in keyframes], t, log=True)[:, 0]
        focus = _keyframe_values([self._focus_point(k.focus)
                                  for k in keyframes], t)
        if zoom is not None:
            factors = np.atleast_1d(np.asarray(zoom, dtype=float))
            if np.any(factors <= 0.0):
                raise ValueError(f"zoom must be positive, got {zoom!r}")
            zooms = zooms * _keyframe_values(factors, t, log=True)[:, 0]
        if rotate is not None:
            sweep = 360.0 * float(turns) * np.arange(frames) / frames
            angles[:, 0 if rotate == "elevation" else 1] += sweep
        return [Camera(elev=float(e), azim=float(a), zoom=float(z),
                       focus=tuple(float(c) for c in f))
                for (e, a), z, f in zip(angles, zooms, focus)]

    def animate(self, path, frames: int = 120, *, fps: float = 12.0,
                rotate="azimuth", turns: float = 1.0, zoom=None,
                trajectory=None, dpi: int = 80) -> str:
        """Write an animated GIF along a camera path and return its path.

        ``frames``, ``rotate``, ``turns``, ``zoom`` and ``trajectory`` define
        the path (see :meth:`camera_path`); ``fps`` sets the playback speed,
        so ``frames / fps`` is the length in seconds.  Every frame clears the
        previous one, so a transparent background leaves no trail.

        .. code-block:: python

            # One slow turn while zooming in and back out.
            viewer.animate("turn.gif", frames=180, zoom=(1.0, 2.0, 1.0))

            # Fly from the whole molecule to atom 0, from above, and back.
            viewer.animate("fly.gif", frames=150, rotate=None, trajectory=[
                Camera(elev=15, azim=-60),
                Camera(elev=60, azim=30, zoom=2.5, focus=0),
                Camera(elev=15, azim=300)])
        """
        from PIL import Image

        path = os.fspath(path)
        if not path.lower().endswith(".gif"):
            raise ValueError(f"animate writes a GIF; give a path ending in "
                             f".gif, not {path!r}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        cameras = self.camera_path(frames, rotate=rotate, turns=turns,
                                   zoom=zoom, trajectory=trajectory)
        fig, _ax, update = self._scene()
        fig.set_dpi(dpi)
        transparent = self.background is None
        images = []
        for camera in cameras:
            update(camera, focus_point=np.asarray(camera.focus))
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            image = Image.fromarray(rgba.copy(), "RGBA")
            images.append(image if transparent else image.convert("RGB"))
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=max(int(round(1000.0 / fps)), 20), loop=0,
                       disposal=2)
        return path

    def __repr__(self) -> str:
        return (f"Viewer3D({self.field.label()!r}, mode={self.mode!r}, "
                f"grid={'x'.join(str(n) for n in self.field.data.shape)})")
