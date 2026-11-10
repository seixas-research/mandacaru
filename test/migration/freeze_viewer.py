"""Freeze the viewer's geometry kernels before they move to Rust (MG-V1).

Run once, on the Python route, from the repository root:
``python test/migration/freeze_viewer.py``. It writes
``test/migration/reference/viewer.json``, which ``test_native_viewer.py`` reads.

The cases: a signed p-like field on a skewed grid (both lobes, as the viewer draws them), the
sphere ``|r| = R`` on an orthogonal grid, a level outside the range, and the enclosing level of a
fixed weight list (ties included) at several fractions.
"""

import json
from pathlib import Path

import numpy as np

from mandacaru.utils.viewer import enclosing_level, marching_tetrahedra

OUT = Path(__file__).parent / "reference" / "viewer.json"


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


if __name__ == "__main__":
    main()
