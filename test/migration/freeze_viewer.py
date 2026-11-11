"""Freeze the viewer's geometry kernels before they move to Rust (MG-V1), and the viewer's
derived values before they move (VQ-8).

Run once, on the Python route, from the repository root:
``python test/migration/freeze_viewer.py``. It writes
``test/migration/reference/viewer.json``, which ``test_native_viewer.py`` reads.

The cases: a signed p-like field on a skewed grid (both lobes, as the viewer draws them), the
sphere ``|r| = R`` on an orthogonal grid, a level outside the range, and the enclosing level of a
fixed weight list (ties included) at several fractions.

``python test/migration/freeze_viewer.py --views`` writes ``reference/viewer_views.json`` instead:
what a ``Viewer3D`` of a fixed analytic field computes before any plotting (the isovalue, the
extent, the camera paths, the cloud and the surface) and what it hands Matplotlib (face colors,
marker sizes, bonds, axis limits), recorded on the Python route that the native route replaced.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mandacaru.utils.viewer import Camera, Viewer3D, enclosing_level, marching_tetrahedra

OUT = Path(__file__).parent / "reference" / "viewer.json"
VIEWS = Path(__file__).parent / "reference" / "viewer_views.json"


def skewed():
    shape = (7, 8, 9)
    step = np.array([[0.35, 0.0, 0.0], [0.05, 0.3, 0.0], [0.02, -0.04, 0.28]])
    origin = np.array([-1.1, -1.05, -1.2])
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    nodes = origin + np.stack([i, j, k], axis=-1) @ step
    X, Y, Z = (nodes[..., a] for a in range(3))
    r = np.sqrt(X**2 + Y**2 + Z**2)
    return (Z + 0.3 * X) * np.exp(-r), X, Y, Z


def sphere():
    g = np.linspace(-2.0, 2.0, 9)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return np.sqrt(X**2 + Y**2 + Z**2), X, Y, Z


def case(name, values, X, Y, Z, level):
    triangles = marching_tetrahedra(values, X, Y, Z, level)
    return {
        "name": name,
        "shape": list(values.shape),
        "values": values.ravel().tolist(),
        "x": X.ravel().tolist(),
        "y": Y.ravel().tolist(),
        "z": Z.ravel().tolist(),
        "level": level,
        "triangles": triangles.reshape(-1).tolist(),
        "count": len(triangles),
    }


def main():
    field, X, Y, Z = skewed()
    surfaces = [
        case("skewed positive lobe", field, X, Y, Z, 0.08),
        case("skewed negative lobe", -field, X, Y, Z, 0.08),
        case("sphere", *sphere(), 1.3),
        case("level above the range", *sphere(), 5.0),
    ]
    weights = [0.5, 0.1, 0.1, 0.3, 0.05, 0.2, 0.2, 0.0, 0.7, 0.15, 0.05, 0.4]
    levels = [{"fraction": f, "level": enclosing_level(weights, f)}
              for f in (0.05, 0.25, 0.5, 0.85, 0.999)]
    OUT.write_text(json.dumps({"surfaces": surfaces, "weights": weights, "levels": levels}))
    for s in surfaces:
        print(f"{s['name']}: {s['count']} triangles")
    print("levels", [(x["fraction"], x["level"]) for x in levels])


def analytic_field(quantity, seed=3):
    """A fixed field on a skewed grid (Bohr) around a water-like set of nuclei."""
    shape = (11, 12, 13)
    step = np.array([[0.62, 0.0, 0.0], [0.07, 0.58, 0.0], [0.03, -0.05, 0.55]])
    origin = np.array([-3.4, -3.3, -3.2])
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    nodes = origin + np.stack([i, j, k], axis=-1) @ step
    X, Y, Z = (np.ascontiguousarray(nodes[..., a]) for a in range(3))
    r = np.sqrt(X**2 + Y**2 + Z**2)
    rng = np.random.default_rng(seed)
    data = (Z + 0.3 * X) * np.exp(-0.6 * r) + 0.01 * rng.standard_normal(shape)
    if quantity == "density":
        data = np.abs(data)
    grid = SimpleNamespace(X=X, Y=Y, Z=Z, step=step, dV=1.0, shape=shape)
    positions = np.array([[0.0, 0.0, 0.2], [1.43, 0.0, -0.9], [-1.43, 0.1, -0.9]])
    return SimpleNamespace(data=data, grid=grid, positions=positions,
                           numbers=np.array([8, 1, 1]), quantity=quantity, component="auto",
                           label=lambda: quantity)


def listed(value):
    """The values of an array as they are: lists up to 40 numbers, else a digest of every bit."""
    array = np.ascontiguousarray(value, dtype=float)
    if array.size <= 40:
        return array.tolist()
    return {"shape": list(array.shape), "head": array.ravel()[:4].tolist(),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def scene(viewer, cameras):
    """What the viewer hands Matplotlib at each camera, read off the artists."""
    fig, ax, update = viewer._scene()
    out = []
    for camera in cameras:
        update(camera)
        record = {"limits": [list(ax.get_xlim()), list(ax.get_ylim()), list(ax.get_zlim())],
                  "collections": [], "lines": []}
        for collection in ax.collections:
            record["collections"].append({
                "facecolor": listed(collection.get_facecolor()),
                # What the viewer set, before Matplotlib projects it at draw time (a surface's
                # lighting is only visible here).
                "facecolor3d": listed(getattr(collection, "_facecolor3d", collection.get_facecolor())),
                "sizes": listed(collection.get_sizes()) if hasattr(collection, "get_sizes") else None,
            })
        for line in ax.lines:
            record["lines"].append(listed(np.asarray(line.get_data_3d())))
        out.append(record)
    return out


def views(Viewer3D=Viewer3D, Camera=Camera):
    """Everything ``--views`` freezes, as a JSON-ready dictionary (``Viewer3D`` and ``Camera`` name the
    implementation: the package's, or the Python route's module when the references are retaken)."""
    cameras = [Camera(), Camera(elev=50.0, azim=33.0), Camera(elev=-20.0, azim=200.0, zoom=2.0)]
    trajectory = [Camera(elev=15, azim=-60), Camera(elev=60, azim=30, zoom=2.5, focus=0),
                  Camera(elev=15, azim=300, focus=(0.4, -0.2, 0.1)), Camera(zoom=0.7, focus=-1)]
    paths = [dict(frames=1), dict(frames=7), dict(frames=120, zoom=(1.0, 2.0, 1.0)),
             dict(frames=5, rotate="elevation", turns=1.7), dict(frames=9, rotate=None, zoom=1.5),
             dict(frames=33, trajectory=trajectory), dict(frames=18, trajectory=trajectory[:1]),
             dict(frames=11, trajectory=trajectory[1:3], zoom=[0.5, 3.0], turns=0.25)]
    out = {}
    for quantity, mode in (("natural_orbital", "scatter"), ("natural_orbital", "isosurface"),
                           ("density", "scatter"), ("density", "isosurface"),
                           ("spin_density", "isosurface"), ("spin_density", "scatter")):
        viewer = Viewer3D(analytic_field(quantity), quantity, mode=mode, points=2500, seed=11)
        key = f"{quantity}-{mode}"
        record = {"isovalue": viewer.isovalue, "extent": viewer.extent}
        if mode == "scatter":
            positions, values = viewer.cloud()
            record["cloud"] = {"positions": listed(positions), "values": listed(values)}
        else:
            triangles, signs = viewer.surface()
            record["surface"] = {"triangles": listed(triangles), "signs": listed(signs)}
        record["scene"] = scene(viewer, cameras)
        record["paths"] = [listed([[c.elev, c.azim, c.zoom, *c.focus] for c in viewer.camera_path(**p)])
                           for p in paths]
        out[key] = record
    for name, kwargs in (("explicit isovalue", dict(isovalue=0.05, mode="isosurface")),
                         ("explicit extent", dict(extent=3.5)),
                         ("enclosed 0.5", dict(enclosed=0.5, mode="isosurface")),
                         ("no atoms", dict(atoms=False)),
                         ("no bonds", dict(bonds=False, mode="isosurface"))):
        viewer = Viewer3D(analytic_field("natural_orbital"), "natural_orbital", points=1500,
                          seed=5, **kwargs)
        out[name] = {"isovalue": viewer.isovalue, "extent": viewer.extent,
                     "scene": scene(viewer, cameras[:2])}
    return out


if __name__ == "__main__":
    if "--views" in sys.argv[1:]:
        VIEWS.write_text(json.dumps(views()))
        print(f"wrote {VIEWS}")
    else:
        main()
